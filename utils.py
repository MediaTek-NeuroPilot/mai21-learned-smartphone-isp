#####################
# Utility functions #
#####################

from functools import reduce
import tensorflow as tf
import numpy as np
import sys
import os

NUM_DEFAULT_TRAIN_ITERS = [100000, 35000, 20000, 20000, 5000, 5000]


def process_command_args(arguments):

    # Specifying the default parameters for training/validation

    # --- data path ---
    dataset_dir = 'raw_images/'
    model_dir = 'models/'
    result_dir = 'results/'
    vgg_dir = 'vgg_pretrained/imagenet-vgg-verydeep-19.mat'
    dslr_dir = 'fujifilm/'
    phone_dir = 'mediatek_raw/'
    # --- architecture ---
    arch = "punet"
    level = 0
    inst_norm = False
    num_maps_base = 16
    # --- model weights ---
    restore_iter = None
    # --- input size ---
    patch_w = 256 # default size for MAI dataset
    patch_h = 256 # default size for MAI dataset
    # --- training options ---
    batch_size = 32
    train_size = 5000
    learning_rate = 5e-5
    eval_step = 1000
    num_train_iters = None
    # --- more options ---
    save_mid_imgs = False

    for args in arguments:

        # --- data path ---
        if args.startswith("dataset_dir"):
            dataset_dir = args.split("=")[1]

        if args.startswith("model_dir"):
            model_dir = args.split("=")[1]

        if args.startswith("result_dir"):
            result_dir = args.split("=")[1]

        if args.startswith("vgg_dir"):
            vgg_dir = args.split("=")[1]

        if args.startswith("dslr_dir"):
            dslr_dir = args.split("=")[1]

        if args.startswith("phone_dir"):
            phone_dir = args.split("=")[1]

        # --- architecture ---
        if args.startswith("arch"):
            arch = args.split("=")[1]

        if args.startswith("level"):
            level = int(args.split("=")[1])

        if args.startswith("inst_norm"):
            inst_norm = eval(args.split("=")[1])
        
        if args.startswith("num_maps_base"):
            num_maps_base = int(args.split("=")[1])

        # --- model weights ---
        if args.startswith("restore_iter"):
            restore_iter = int(args.split("=")[1])

        # --- input size ---
        if args.startswith("patch_w"):
            patch_w = int(args.split("=")[1])

        if args.startswith("patch_h"):
            patch_h = int(args.split("=")[1])

        # --- training options ---
        if args.startswith("batch_size"):
            batch_size = int(args.split("=")[1])

        if args.startswith("train_size"):
            train_size = int(args.split("=")[1])

        if args.startswith("learning_rate"):
            learning_rate = float(args.split("=")[1])

        if args.startswith("eval_step"):
            eval_step = int(args.split("=")[1])

        if args.startswith("num_train_iters"):
            num_train_iters = int(args.split("=")[1])

        # --- more options ---
        if args.startswith("save_mid_imgs"):
            save_mid_imgs = eval(args.split("=")[1])

    # choose architecture
    if arch == "punet":
        name_model = "punet"

    # obtain restore iteration info
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    
    if restore_iter is None: # no need to get the last iteration if specified
        restore_iter = get_last_iter(model_dir, name_model)

    if num_train_iters is None:
        num_train_iters = NUM_DEFAULT_TRAIN_ITERS[level]

    print("The following parameters will be applied for training:")

    print("Model architecture: " + arch)
    print("Restore Iteration: " + str(restore_iter))
    print("Batch size: " + str(batch_size))
    print("Training size: " + str(train_size))
    print("Learning rate: " + str(learning_rate))
    print("Training iterations: " + str(num_train_iters))
    print("Evaluation step: " + str(eval_step))
    print("Path to the dataset: " + dataset_dir)
    print("Path to Raw-to-RGB model network: " + model_dir)
    print("Path to result images: " + result_dir)
    print("Path to VGG-19 network: " + vgg_dir)
    print("Path to RGB data from DSLR: " + dslr_dir)
    print("Path to Raw data from phone: " + phone_dir)

    return dataset_dir, model_dir, result_dir, vgg_dir, dslr_dir, phone_dir,\
        arch, level, inst_norm, num_maps_base, restore_iter, patch_w, patch_h,\
            batch_size, train_size, learning_rate, eval_step, num_train_iters, save_mid_imgs


def process_test_model_args(arguments):

    # Specifying the default parameters for testing

    # --- data path ---
    dataset_dir = 'raw_images/'
    test_dir = 'fujifilm_full_resolution/'
    model_dir = 'models/'
    result_dir = 'results/'
    # --- architecture ---
    arch = "punet"
    level = 0
    inst_norm = False
    num_maps_base = 16
    # --- model weights ---
    orig_model = False
    rand_param = False
    restore_iter = None
    # --- input size ---
    img_h = 2976 # default size
    img_w = 3968 # default size
    # --- more options ---
    use_gpu = True
    save_model = True
    test_image = False

    for args in arguments:
        
        # --- data path ---
        if args.startswith("dataset_dir"):
            dataset_dir = args.split("=")[1]

        if args.startswith("test_dir"):
            test_dir = args.split("=")[1]

        if args.startswith("model_dir"):
            model_dir = args.split("=")[1]

        if args.startswith("result_dir"):
            result_dir = args.split("=")[1]
        
        # --- architecture ---
        if args.startswith("arch"):
            arch = args.split("=")[1]

        if args.startswith("level"):
            level = 0 if arch == "pynet_0" else int(args.split("=")[1])

        if args.startswith("inst_norm"):
            inst_norm = eval(args.split("=")[1])

        if args.startswith("num_maps_base"):
            num_maps_base = int(args.split("=")[1])

        # --- model weights ---
        if args.startswith("orig"):
            orig_model = eval(args.split("=")[1])

        if args.startswith("rand"):
            rand_param = eval(args.split("=")[1])

        if args.startswith("restore_iter"):
            restore_iter = int(args.split("=")[1])

        # --- input size ---
        if args.startswith("img_h"):
            img_h = int(args.split("=")[1])

        if args.startswith("img_w"):
            img_w = int(args.split("=")[1])
        
        # --- more options ---        
        if args.startswith("use_gpu"):
            use_gpu = eval(args.split("=")[1])

        if args.startswith("save"):
            save_model = eval(args.split("=")[1])

        if args.startswith("test_image"):
            test_image = eval(args.split("=")[1])

    # choose architecture
    if arch == "punet":
        name_model = "punet"

    # obtain restore iteration info (necessary if no pre-trained model or not random weights)
    if restore_iter is None and not orig_model and not rand_param: # need to restore a model

        restore_iter = get_last_iter(model_dir, name_model)
        if restore_iter == -1:
            print("Error: Cannot find any pre-trained models for " + name_model + ".")
            sys.exit()

    print("The following parameters will be applied for testing:")

    print("Model architecture: " + arch)
    print("Restore Iteration: " + str(restore_iter))
    print("Path to the dataset: " + dataset_dir)
    print("Path to Raw-to-RGB model network: " + model_dir)
    print("Path to result images: " + result_dir)
    print("Path to testing data: " + test_dir)

    return dataset_dir, test_dir, model_dir, result_dir,\
        arch, level, inst_norm, num_maps_base, orig_model, rand_param, restore_iter,\
            img_h, img_w, use_gpu, save_model, test_image


def get_last_iter(model_dir, name_model):

    saved_models = [int(model_file.split(".")[0].split("_")[-1])
                    for model_file in os.listdir(model_dir)
                    if model_file.startswith(name_model)]

    if len(saved_models) > 0:
        return np.max(saved_models)
    else:
        return -1


def log10(x):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)


def export_pb(sess, output_node_name, output_dir='.', export_pb_name='test.pb'):
    gd = sess.graph.as_graph_def()

    # replace variables in a graph with constants for exporting pb
    output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
        sess, gd, [output_node_name]
    )

    tf.io.write_graph(output_graph_def, output_dir, export_pb_name, as_text=False)
    print(f'Save {export_pb_name} in {output_dir} Done')