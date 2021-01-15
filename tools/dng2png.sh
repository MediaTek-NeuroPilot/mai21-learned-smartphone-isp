#!/bin/bash         

input_dir=../raw_images/test/full_resolution_raw_images_for_results_visualization/
output_dir=../raw_images/test/fujifilm_full_resolution/

# run code
python dng2png.py $input_dir $output_dir
