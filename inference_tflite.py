#################################
# Inferening using TFLite model #
#################################

import os
from os import path
import glob
import itertools

import time
import argparse

import numpy as np
import imageio
import tqdm
import math

import tensorflow as tf
from load_dataset import extract_bayer_channels


def preprocess(image_path):
    I = np.asarray(imageio.imread(image_path))
    I = extract_bayer_channels(I)
    I = I[0:256//2, 0:256//2, :]
    I = np.reshape(I, [1, I.shape[0], I.shape[1], 4])
    return I

def psnr(x, y):
    diff = x.astype(np.float32) - y.astype(np.float32)
    mse = np.mean(diff**2)
    ret = 20 * math.log10(1.0/math.sqrt(mse))
    return ret

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='raw_images')
    parser.add_argument('--dir_type', type=str, default='test')
    parser.add_argument('--phone_dir', type=str, default='mediatek_raw')
    parser.add_argument('--dslr_dir', type=str, default=None)
    parser.add_argument('--model_file', type=str, default='models/original/punet_pretrained.tflite')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_dir', type=str, default='results')
    args = parser.parse_args()

    # initialization of TFLite interpreter
    interpreter = tf.lite.Interpreter(
        model_path=args.model_file,
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    # set input directory
    input_dir = path.join(args.dataset_dir, args.dir_type, args.phone_dir)
    target_dir = path.join(args.dataset_dir, args.dir_type, args.dslr_dir) if args.dslr_dir is not None else None
    save_dir = path.join(args.save_dir, args.dir_type)

    if args.save_results:
        os.makedirs(save_dir, exist_ok=True)

    # prepare data list
    scan = lambda x, y: glob.glob(path.join(x, y))
    input_list = [scan(input_dir, d) for d in os.listdir(input_dir)]
    input_list = sorted([it for it in itertools.chain(*input_list)])
    if target_dir is None:
        target_list = [None for _ in input_list]
    else:
        target_list = [scan(target_dir, d) for d in os.listdir(target_dir)]
        target_list = sorted([it for it in itertools.chain(*target_list)])

    # pass each input image to the TFLite model
    psnr_sum = 0.0
    tq = tqdm.tqdm(zip(input_list, target_list), total=len(input_list))
    for input_path, target_path in tq:
        # print(input_path)
        input_img = preprocess(input_path)
        if target_path is not None:
            target_img = np.asarray(imageio.imread(target_path))
            target_img = target_img.astype(np.float32) / 255

        interpreter.set_tensor(input_details[0]['index'], input_img)
        interpreter.invoke()
        output_img = interpreter.get_tensor(output_details[0]['index'])
        output_img = np.clip(output_img, 0., 1.)
        output_img = np.squeeze(output_img)
  
        # calculate PSNR if ground truth is available
        if target_path is not None:
            psnr_sum += psnr(output_img, target_img)

        # save results
        if args.save_results:
            save_as = input_path.replace(input_dir, save_dir)
            imageio.imwrite(save_as, (output_img*255).astype(np.uint8))
    
    if target_path is not None:
        print('Avg. PSNR: {:.2f}'.format(psnr_sum / len(input_list)))

if __name__ == '__main__':
    main()

