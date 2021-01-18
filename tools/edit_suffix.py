##########################
# Count model statistics #
##########################

import sys
import os


if __name__ == "__main__":
    input_dir = sys.argv[1]
    if not os.path.isdir(input_dir):
        print("The folder doesn't exist!")
        sys.exit()
    input_names = [f for f in os.listdir(input_dir) if os.path.isfile(input_dir + f)]
    input_names.sort()

    output_dir = sys.argv[2]
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    input_suffix = sys.argv[3]
    output_suffix = sys.argv[4]

    for name_in in input_names:

        print("Converting file name:" + name_in)

        if not os.path.isfile(input_dir + name_in):
            print("The file doesn't exist!")
            sys.exit()

        name_out = name_in.replace(input_suffix, output_suffix) # change file name suffix
        os.system('mv ' + input_dir + name_in + ' ' + output_dir + name_out)
