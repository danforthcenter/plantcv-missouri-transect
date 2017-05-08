#!/usr/bin/env python

import argparse
import os
import cv2
import numpy as np
import plantcv as pcv


# Parse command-line arguments
def options():
    parser = argparse.ArgumentParser(description="Imaging processing with opencv")
    parser.add_argument("-i", "--image", help="Input image file.", required=True)
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=False)
    parser.add_argument("-r", "--result", help="result file.", required=False)
    parser.add_argument("-r2", "--coresult", help="result file.", required=False)
    parser.add_argument("-p", "--pdfs", help="Naive Bayes PDF file.", required=True)
    parser.add_argument("-w", "--writeimg", help="write out images.", default=False, action="store_true")
    parser.add_argument("-D", "--debug", help="Turn on debug, prints intermediate images.", default=None)
    args = parser.parse_args()
    return args


def main():
    # Get options
    args = options()

    # Initialize device counter
    device = 0

    # Read in the input image
    vis, path, filename = pcv.readimage(filename=args.image, debug=args.debug)

    # Parse camera metadata
    metadata = filename.split("_")
    camera = metadata[1]
    if camera == "SV":
        zoom = metadata[3]
    elif camera == "TV":
        zoom = metadata[2]
    else:
        pcv.fatal_error("Unknown camera type: {0}".format(camera))

    # Classify each pixel as plant or background (background and system components)
    device, masks = pcv.naive_bayes_classifier(img=vis, pdf_file=args.pdfs, device=device, debug=args.debug)

    # Fill in small contours
    device, mask_filled = pcv.fill(img=np.copy(masks["plant"]), mask=np.copy(masks["plant"]), size=50, device=device,
                                   debug=args.debug)

    device, roi, roi_hierarchy = pcv.define_roi(img=vis, shape="rectangle", device=device, roi=None,
                                                roi_input="default", debug=args.debug, adjust=True, x_adj=500,
                                                y_adj=250, w_adj=-500, h_adj=-250)

    # Find contours
    device, obj, obj_hierarchy = pcv.find_objects(img=vis, mask=mask_filled, device=device, debug=args.debug)

    # Keep contours that overlap the ROI
    device, roi_obj, roi_obj_hierarchy, obj_mask, obj_area = pcv.roi_objects(img=vis, roi_type="partial",
                                                                             roi_contour=roi,
                                                                             roi_hierarchy=roi_hierarchy,
                                                                             object_contour=obj,
                                                                             obj_hierarchy=obj_hierarchy, device=device,
                                                                             debug=args.debug)

    # Combine remaining contours into a single object (the plant)
    device, plant_obj, plant_mask = pcv.object_composition(img=vis, contours=roi_obj, hierarchy=roi_obj_hierarchy,
                                                           device=device, debug=args.debug)

    # Analyze the shape features of the plant object
    if args.writeimg:
        outfile = os.path.join(args.outdir, filename)
    else:
        outfile = False
    device, shape_header, shape_data, shape_img = pcv.analyze_object(img=vis, imgname=filename, obj=plant_obj,
                                                                     mask=plant_mask, device=device, debug=args.debug,
                                                                     filename=outfile)

    # Write data to results file
    results = open(args.result, "a")
    # Write shapes results
    results.write("\t".join(map(str, shape_header)) + "\n")
    results.write("\t".join(map(str, shape_data)) + "\n")
    for row in shape_img:
        results.write("\t".join(map(str, row)) + "\n")

    # If this is a side-view image, calculate boundary-line results
    # The boundary line position depends on the camera zoom level
    if camera == "SV":
        if zoom == "z300":
            device, boundary_header, boundary_data, boundary_image = pcv.analyze_bound(img=vis, imgname=filename,
                                                                                       obj=plant_obj, mask=plant_mask,
                                                                                       line_position=680, device=device,
                                                                                       debug=args.debug,
                                                                                       filename=outfile)
            results.write("\t".join(map(str, boundary_header)) + "\n")
            results.write("\t".join(map(str, boundary_data)) + "\n")
            for row in boundary_image:
                results.write("\t".join(map(str, row)) + "\n")
        elif zoom == "z1":
            device, boundary_header, boundary_data, boundary_image = pcv.analyze_bound(img=vis, imgname=filename,
                                                                                       obj=plant_obj, mask=plant_mask,
                                                                                       line_position=670, device=device,
                                                                                       debug=args.debug,
                                                                                       filename=outfile)
            results.write("\t".join(map(str, boundary_header)) + "\n")
            results.write("\t".join(map(str, boundary_data)) + "\n")
            for row in boundary_image:
                results.write("\t".join(map(str, row)) + "\n")

    # Analyze color
    device, color_headers, color_data, analysis_images = pcv.analyze_color(img=vis, imgname=filename, mask=plant_mask,
                                                                           bins=256, device=device, debug=args.debug,
                                                                           hist_plot_type=None, pseudo_channel="v",
                                                                           pseudo_bkg="img", resolution=300,
                                                                           filename=outfile)
    results.write("\t".join(map(str, color_headers)) + "\n")
    results.write("\t".join(map(str, color_data)) + "\n")
    for row in analysis_images:
        results.write("\t".join(map(str, row)) + "\n")
    results.close()

    # Find the corresponding NIR image
    device, nirpath = pcv.get_nir(path=path, filename=filename, device=device, debug=args.debug)
    nir, nir_path, nir_filename = pcv.readimage(filename=nirpath, debug=args.debug)
    device, nir = pcv.rgb2gray(img=nir, device=device, debug=args.debug)
    if camera == "TV":
        # The top-view camera needs to be rotated
        device, nir = pcv.flip(img=nir, direction="vertical", device=device, debug=args.debug)
        device, nir = pcv.flip(img=nir, direction="horizontal", device=device, debug=args.debug)

    # Rescale the size of the VIS plant mask to fit on the smaller NIR image
    device, nir_mask = pcv.resize(img=plant_mask, resize_x=0.278, resize_y=0.278, device=device, debug=args.debug)

    # Map the plant mask onto the NIR image
    # Settings depend on the camera and zoom level
    if camera == "TV":
        device, newmask = pcv.crop_position_mask(img=nir, mask=nir_mask, device=device, x=3, y=7, v_pos="bottom",
                                                 h_pos="right", debug=args.debug)
    elif camera == "SV":
        if zoom == "z300":
            device, newmask = pcv.crop_position_mask(img=nir, mask=nir_mask, device=device, x=43, y=6, v_pos="top",
                                                     h_pos="right", debug=args.debug)
        elif zoom == "z1":
            device, newmask = pcv.crop_position_mask(img=nir, mask=nir_mask, device=device, x=39, y=6, v_pos="top",
                                                     h_pos="right", debug=args.debug)
    # Identify contours
    device, nir_objects, nir_hierarchy = pcv.find_objects(img=cv2.cvtColor(nir, cv2.COLOR_GRAY2BGR), mask=newmask,
                                                          device=device, debug=args.debug)

    # Combine contours into a single object (plant)
    device, nir_combined, nir_combinedmask = pcv.object_composition(img=cv2.cvtColor(nir, cv2.COLOR_GRAY2BGR),
                                                                    contours=nir_objects, hierarchy=nir_hierarchy,
                                                                    device=device, debug=args.debug)

    # Measure the NIR contour shape properties
    device, nir_shape_header, nir_shape_data, nir_shape_img = pcv.analyze_object(
        img=cv2.cvtColor(nir, cv2.COLOR_GRAY2BGR), imgname=nir_filename, obj=nir_combined, mask=nir_combinedmask,
        device=device, debug=args.debug, filename=outfile)

    # Write data to results file
    results = open(args.coresult, "a")
    # Write shapes results
    results.write("\t".join(map(str, nir_shape_header)) + "\n")
    results.write("\t".join(map(str, nir_shape_data)) + "\n")
    for row in nir_shape_img:
        results.write("\t".join(map(str, row)) + "\n")

    # Analyze NIR signal
    device, nhist_header, nhist_data, nir_imgs = pcv.analyze_NIR_intensity(img=nir,
                                                                           rgbimg=cv2.cvtColor(nir, cv2.COLOR_GRAY2BGR),
                                                                           mask=nir_combinedmask, bins=256,
                                                                           device=device, histplot=False,
                                                                           debug=args.debug, filename=outfile)
    results.write("\t".join(map(str, nhist_header)) + "\n")
    results.write("\t".join(map(str, nhist_data)) + "\n")
    for row in nir_imgs:
        results.write("\t".join(map(str, row)) + "\n")
    results.close()


if __name__ == '__main__':
    main()
