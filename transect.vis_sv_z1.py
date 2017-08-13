#!/usr/bin/env python

import argparse
import os
import plantcv as pcv
import numpy as np
import cv2


# Parse command-line arguments
def options():
    parser = argparse.ArgumentParser(description="Imaging processing with opencv")
    parser.add_argument("-i", "--image", help="Input image file.", required=True)
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=False)
    parser.add_argument("-r", "--result", help="result file.", required=False)
    parser.add_argument("-b", "--bgimg", help="Background image file.", required=True)
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

    # Read in the background image
    bg_img, bg_path, bg_filename = pcv.readimage(filename=args.bgimg, debug=args.debug)

    # Background subtraction
    device, fgmask = pcv.background_subtraction(foreground_image=vis, background_image=bg_img, device=device,
                                                debug=args.debug)

    # The background subtraction MOG method misses plant areas that overlap dark background areas
    # Use the MOG2 method to capture these areas
    bgsub = cv2.BackgroundSubtractorMOG2()
    _ = bgsub.apply(bg_img)
    fgmask2 = bgsub.apply(vis)

    # Threshold the MOG2 image to remove pixels labeled as shadow
    device, fgmask2_thresh = pcv.binary_threshold(img=fgmask2, threshold=254, maxValue=255, object_type="light",
                                                  device=device, debug=args.debug)

    # Calculate the input image size
    img_size = np.shape(vis)
    # Apply a rectangle mask to remove the corrugated lines on the pots of the MOG2 image
    device, fgmask2_masked, _, _, _ = pcv.rectangle_mask(img=fgmask2_thresh, p1=(0, 1300),
                                                         p2=(img_size[1], img_size[0]), device=device, debug=args.debug,
                                                         color="black")

    # Add the MOG and MOG2 masks together
    device, fgmask_merged = pcv.logical_or(img1=fgmask2_masked, img2=fgmask, device=device, debug=args.debug)

    # Mask the top of the pot (it's hard to get rid of otherwise)
    device, pot_masked, _, _, _ = pcv.rectangle_mask(img=fgmask_merged, p1=(1100, 1356), p2=(1400, 1500), device=device,
                                                     debug=args.debug, color="black")

    # Use median blur to remove the vertical pot lines
    device, fgmask_blurred = pcv.median_blur(img=pot_masked, ksize=11, device=device, debug=args.debug)

    # Fill in small background objects
    device, plant_mask = pcv.fill(img=np.copy(fgmask_blurred), mask=np.copy(fgmask_blurred), size=100, device=device,
                                  debug=args.debug)

    # Define ROI
    device, roi, roi_hierarchy = pcv.define_roi(img=vis, shape='rectangle', device=device, roi=None,
                                                roi_input='default', debug=args.debug, adjust=True, x_adj=600,
                                                y_adj=250, w_adj=-600, h_adj=-750)

    # Find contours
    device, objects, obj_hierarchy = pcv.find_objects(img=vis, mask=plant_mask, device=device, debug=args.debug)

    # ROI objects
    device, roi_objects, hierarchy, kept_mask, obj_area = pcv.roi_objects(img=vis, roi_type='partial', roi_contour=roi,
                                                                          roi_hierarchy=roi_hierarchy,
                                                                          object_contour=objects,
                                                                          obj_hierarchy=obj_hierarchy, device=device,
                                                                          debug=args.debug)

    # Object composition
    device, obj, mask = pcv.object_composition(img=vis, contours=roi_objects, hierarchy=hierarchy, device=device,
                                               debug=args.debug)

    # Analyze the shape features of the plant object
    if args.writeimg:
        outfile = os.path.join(args.outdir, filename)
    else:
        outfile = False
    device, shape_header, shape_data, shape_img = pcv.analyze_object(img=vis, imgname=args.image, obj=obj, mask=mask,
                                                                     device=device, debug=args.debug, filename=outfile)
    # Write data to results file
    results = open(args.result, "a")
    # Write shapes results
    results.write("\t".join(map(str, shape_header)) + "\n")
    results.write("\t".join(map(str, shape_data)) + "\n")
    for row in shape_img:
        results.write("\t".join(map(str, row)) + "\n")

    # Boundary line tool
    device, boundary_header, boundary_data, boundary_img = pcv.analyze_bound(img=vis, imgname=args.image, obj=obj,
                                                                             mask=mask, line_position=700,
                                                                             device=device, debug=args.debug,
                                                                             filename=outfile)
    # Write boundary results
    results.write("\t".join(map(str, boundary_header)) + "\n")
    results.write("\t".join(map(str, boundary_data)) + "\n")
    for row in boundary_img:
        results.write("\t".join(map(str, row)) + "\n")

    # Analyze color
    device, color_header, color_data, analysis_images = pcv.analyze_color(img=vis, imgname=args.image, mask=mask,
                                                                          bins=256, device=device, debug=args.debug,
                                                                          hist_plot_type=None, pseudo_channel='v',
                                                                          pseudo_bkg='img', resolution=300,
                                                                          filename=outfile)

    results.write("\t".join(map(str, color_header)) + "\n")
    results.write("\t".join(map(str, color_data)) + "\n")
    for row in analysis_images:
        results.write("\t".join(map(str, row)) + "\n")
    results.close()


if __name__ == '__main__':
    main()
