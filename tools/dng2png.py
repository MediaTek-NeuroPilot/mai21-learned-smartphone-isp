##############
# DNG to PNG #
##############

import numpy as np
import imageio
import rawpy
import sys
import os


if __name__ == "__main__":
    input_dir = sys.argv[1]
    if not os.path.isdir(input_dir):
        print("The folder doesn't exist!")
        sys.exit()
    input_dng = [f for f in os.listdir(input_dir) if os.path.isfile(input_dir + f)]
    input_dng.sort()

    output_dir = sys.argv[2]
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for dng_images in input_dng:

        print("Converting file " + dng_images)

        if not os.path.isfile(input_dir + dng_images):
            print("The file doesn't exist!")
            sys.exit()

        raw = rawpy.imread(input_dir + dng_images)
        raw_image = raw.raw_image
        del raw

        # convert format
        raw_image = raw_image.astype(np.float32)
        png_image = raw_image.astype(np.uint16)
        new_name = dng_images.replace(".dng", ".png") # save to output_dir
        imageio.imwrite(output_dir + new_name, png_image)
