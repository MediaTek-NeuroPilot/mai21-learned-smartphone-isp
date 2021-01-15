#################################
# RAW-to-RGB Model architecture #
#################################

import tensorflow as tf
import numpy as np


def PUNET(input, instance_norm=False, instance_norm_level_1=False, num_maps_base=16):

    with tf.compat.v1.variable_scope("generator"):

        # -----------------------------------------
        # Downsampling layers
        conv_l1_d1 = _conv_multi_block(input, 3, num_maps=num_maps_base, instance_norm=False)              # 128 -> 128
        pool1 = max_pool(conv_l1_d1, 2)                                                         # 128 -> 64

        conv_l2_d1 = _conv_multi_block(pool1, 3, num_maps=num_maps_base*2, instance_norm=instance_norm)      # 64 -> 64
        pool2 = max_pool(conv_l2_d1, 2)                                                         # 64 -> 32

        conv_l3_d1 = _conv_multi_block(pool2, 3, num_maps=num_maps_base*4, instance_norm=instance_norm)     # 32 -> 32
        pool3 = max_pool(conv_l3_d1, 2)                                                         # 32 -> 16

        conv_l4_d1 = _conv_multi_block(pool3, 3, num_maps=num_maps_base*8, instance_norm=instance_norm)     # 16 -> 16
        pool4 = max_pool(conv_l4_d1, 2)                                                         # 16 -> 8

        # -----------------------------------------
        # Processing: Level 5,  Input size: 8 x 8
        conv_l5_d1 = _conv_multi_block(pool4, 3, num_maps=num_maps_base*16, instance_norm=instance_norm)
        conv_l5_d2 = _conv_multi_block(conv_l5_d1, 3, num_maps=num_maps_base*16, instance_norm=instance_norm) + conv_l5_d1
        conv_l5_d3 = _conv_multi_block(conv_l5_d2, 3, num_maps=num_maps_base*16, instance_norm=instance_norm) + conv_l5_d2
        conv_l5_d4 = _conv_multi_block(conv_l5_d3, 3, num_maps=num_maps_base*16, instance_norm=instance_norm)

        conv_t4b = _conv_tranpose_layer(conv_l5_d4, num_maps_base*8, 3, 2)      # 8 -> 16

        # -----------------------------------------
        # Processing: Level 4,  Input size: 16 x 16
        conv_l4_d6 = conv_l4_d1
        conv_l4_d7 = stack(conv_l4_d6, conv_t4b)
        conv_l4_d8 = _conv_multi_block(conv_l4_d7, 3, num_maps=num_maps_base*8, instance_norm=instance_norm)

        conv_t3b = _conv_tranpose_layer(conv_l4_d8, num_maps_base*4, 3, 2)      # 16 -> 32

        # -----------------------------------------
        # Processing: Level 3,  Input size: 32 x 32
        conv_l3_d6 = conv_l3_d1
        conv_l3_d7 = stack(conv_l3_d6, conv_t3b)
        conv_l3_d8 = _conv_multi_block(conv_l3_d7, 3, num_maps=num_maps_base*4, instance_norm=instance_norm)

        conv_t2b = _conv_tranpose_layer(conv_l3_d8, num_maps_base*2, 3, 2)       # 32 -> 64

        # -------------------------------------------
        # Processing: Level 2,  Input size: 64 x 64
        conv_l2_d7 = conv_l2_d1
        conv_l2_d8 = stack(_conv_multi_block(conv_l2_d7, 3, num_maps=num_maps_base*2, instance_norm=instance_norm), conv_t2b)
        conv_l2_d9 = _conv_multi_block(conv_l2_d8, 3, num_maps=num_maps_base*2, instance_norm=instance_norm)

        conv_t1b = _conv_tranpose_layer(conv_l2_d9, num_maps_base, 3, 2)       # 64 -> 128

        # -------------------------------------------
        # Processing: Level 1,  Input size: 128 x 128
        conv_l1_d9 = conv_l1_d1
        conv_l1_d10 = stack(_conv_multi_block(conv_l1_d9, 3, num_maps=num_maps_base, instance_norm=False), conv_t1b)
        conv_l1_d11 = stack(conv_l1_d10, conv_l1_d1)
        conv_l1_d12 = _conv_multi_block(conv_l1_d11, 3, num_maps=num_maps_base, instance_norm=False)

        # ----------------------------------------------------------
        # Processing: Level 0 (x2 upscaling),  Input size: 128 x 128
        conv_l0 = _conv_tranpose_layer(conv_l1_d12, num_maps_base//4, 3, 2)        # 128 -> 256
        conv_l0_out = _conv_layer(conv_l0, 3, 3, 1, relu=False, instance_norm=False)

        output_l0 = tf.nn.tanh(conv_l0_out) * 0.58 + 0.5
        
    output_l0 = tf.identity(output_l0, name='output_l0')

    return output_l0


def _conv_multi_block(input, max_size, num_maps, instance_norm):

    conv_3a = _conv_layer(input, num_maps, 3, 1, relu=True, instance_norm=instance_norm)
    conv_3b = _conv_layer(conv_3a, num_maps, 3, 1, relu=True, instance_norm=instance_norm)

    output_tensor = conv_3b

    if max_size >= 5:

        conv_5a = _conv_layer(input, num_maps, 5, 1, relu=True, instance_norm=instance_norm)
        conv_5b = _conv_layer(conv_5a, num_maps, 5, 1, relu=True, instance_norm=instance_norm)

        output_tensor = stack(output_tensor, conv_5b)

    if max_size >= 7:

        conv_7a = _conv_layer(input, num_maps, 7, 1, relu=True, instance_norm=instance_norm)
        conv_7b = _conv_layer(conv_7a, num_maps, 7, 1, relu=True, instance_norm=instance_norm)

        output_tensor = stack(output_tensor, conv_7b)

    if max_size >= 9:

        conv_9a = _conv_layer(input, num_maps, 9, 1, relu=True, instance_norm=instance_norm)
        conv_9b = _conv_layer(conv_9a, num_maps, 9, 1, relu=True, instance_norm=instance_norm)

        output_tensor = stack(output_tensor, conv_9b)

    return output_tensor


def stack(x, y):
    return tf.concat([x, y], 3)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def _conv_layer(net, num_filters, filter_size, strides, relu=True, instance_norm=False, padding='SAME'):

    weights_init = _conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    bias = tf.Variable(tf.constant(0.01, shape=[num_filters]))

    net = tf.nn.conv2d(net, weights_init, strides_shape, padding=padding) + bias

    if instance_norm:
        net = _instance_norm(net)

    if relu:
        net = tf.compat.v1.nn.leaky_relu(net)

    return net


def _instance_norm(net):

    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]

    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)

    return scale * normalized + shift


def _conv_init_vars(net, out_channels, filter_size, transpose=False):

    _, rows, cols, in_channels = [i.value for i in net.get_shape()]

    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.random.truncated_normal(weights_shape, stddev=0.01, seed=1), dtype=tf.float32)
    return weights_init


def _conv_tranpose_layer(net, num_filters, filter_size, strides):
    weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True)

    batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
    new_rows, new_cols = int(rows * strides), int(cols * strides)

    new_shape = [batch_size, new_rows, new_cols, num_filters]
    tf_shape = tf.stack(new_shape)

    strides_shape = [1, strides, strides, 1]
    net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')

    return tf.compat.v1.nn.leaky_relu(net)


def max_pool(x, n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')
