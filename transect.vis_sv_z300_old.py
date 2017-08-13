#!/usr/bin/env python

import argparse
import os
import plantcv as pcv


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

    # Define ROI
    device, roi, roi_hierarchy = pcv.define_roi(img=vis, shape='rectangle', device=device, roi=None,
                                                roi_input='default', debug=args.debug, adjust=True, x_adj=600,
                                                y_adj=250, w_adj=-600, h_adj=-750)

    # Find contours
    device, objects, obj_hierarchy = pcv.find_objects(img=vis, mask=fgmask, device=device, debug=args.debug)

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
                                                                             mask=mask, line_position=690,
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
