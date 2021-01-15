##################################################
# Train a RAW-to-RGB model using training images #
##################################################

import tensorflow as tf
import imageio
import numpy as np
import sys
from datetime import datetime

from load_dataset import load_train_patch, load_val_data
from model import PUNET
import utils
import vgg

# Processing command arguments
dataset_dir, model_dir, result_dir, vgg_dir, dslr_dir, phone_dir,\
    arch, LEVEL, inst_norm, num_maps_base, restore_iter, patch_w, patch_h,\
        batch_size, train_size, learning_rate, eval_step, num_train_iters, save_mid_imgs = \
            utils.process_command_args(sys.argv)

# Defining the size of the input and target image patches
PATCH_WIDTH, PATCH_HEIGHT = patch_w//2, patch_h//2

DSLR_SCALE = float(1) / (2 ** (max(LEVEL,0) - 1))
TARGET_WIDTH = int(PATCH_WIDTH * DSLR_SCALE)
TARGET_HEIGHT = int(PATCH_HEIGHT * DSLR_SCALE)
TARGET_DEPTH = 3
TARGET_SIZE = TARGET_WIDTH * TARGET_HEIGHT * TARGET_DEPTH

np.random.seed(0)

# Defining the model architecture
with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
    time_start = datetime.now()

    # determine model name
    if arch == "punet":
        name_model = "punet"
    
    # Placeholders for training data
    phone_ = tf.compat.v1.placeholder(tf.float32, [batch_size, PATCH_HEIGHT, PATCH_WIDTH, 4])
    dslr_ = tf.compat.v1.placeholder(tf.float32, [batch_size, TARGET_HEIGHT, TARGET_WIDTH, TARGET_DEPTH])

    # Get the processed enhanced image
    if arch == "punet":
        enhanced = PUNET(phone_, instance_norm=inst_norm, instance_norm_level_1=False, num_maps_base=num_maps_base)

    # Losses
    enhanced_flat = tf.reshape(enhanced, [-1, TARGET_SIZE])
    dslr_flat = tf.reshape(dslr_, [-1, TARGET_SIZE])

    # MSE loss
    loss_mse = tf.reduce_sum(tf.pow(dslr_flat - enhanced_flat, 2))/(TARGET_SIZE * batch_size)

    # PSNR loss
    loss_psnr = 20 * utils.log10(1.0 / tf.sqrt(loss_mse))

    # SSIM loss
    loss_ssim = tf.reduce_mean(tf.image.ssim(enhanced, dslr_, 1.0))

    # MS-SSIM loss
    loss_ms_ssim = tf.reduce_mean(tf.image.ssim_multiscale(enhanced, dslr_, 1.0))

    # Content loss
    CONTENT_LAYER = 'relu5_4'

    enhanced_vgg = vgg.net(vgg_dir, vgg.preprocess(enhanced * 255))
    dslr_vgg = vgg.net(vgg_dir, vgg.preprocess(dslr_ * 255))

    content_size = utils._tensor_size(dslr_vgg[CONTENT_LAYER]) * batch_size
    loss_content = 2 * tf.nn.l2_loss(enhanced_vgg[CONTENT_LAYER] - dslr_vgg[CONTENT_LAYER]) / content_size

    # Final loss function
    loss_generator = loss_mse * 20 + loss_content + (1 - loss_ssim) * 20

    # Optimize network parameters
    generator_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith("generator")]
    train_step_gen = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss_generator, var_list=generator_vars)

    # Initialize and restore the variables
    print("Initializing variables...")
    sess.run(tf.compat.v1.global_variables_initializer())

    saver = tf.compat.v1.train.Saver(var_list=generator_vars, max_to_keep=100)

    if restore_iter > 0: # restore the variables/weights
        name_model_restore = name_model

        name_model_restore_full = name_model_restore + "_iteration_" + str(restore_iter)
        print("Restoring Variables from:", name_model_restore_full)
        saver.restore(sess, model_dir + name_model_restore_full + ".ckpt")

    # Loading training and validation data
    print("Loading validation data...")
    val_data, val_answ = load_val_data(dataset_dir, dslr_dir, phone_dir, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE)
    print("Validation data was loaded\n")

    print("Loading training data...")
    train_data, train_answ = load_train_patch(dataset_dir, dslr_dir, phone_dir, train_size, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE)
    print("Training data was loaded\n")

    VAL_SIZE = val_data.shape[0]
    num_val_batches = int(val_data.shape[0] / batch_size)

    if save_mid_imgs:
        visual_crops_ids = np.random.randint(0, VAL_SIZE, batch_size)
        visual_val_crops = val_data[visual_crops_ids, :]
        visual_target_crops = val_answ[visual_crops_ids, :]


    print("Training network...")

    iter_start = restore_iter+1 if restore_iter > 0 else 0
    logs = open(model_dir + "logs_" + str(iter_start) + "-" + str(num_train_iters) + ".txt", "w+")
    logs.close()

    training_loss = 0.0

    name_model_save = name_model
    
    for i in range(iter_start, num_train_iters + 1):
        name_model_save_full = name_model_save + "_iteration_" + str(i)

        # Train model
        idx_train = np.random.randint(0, train_size, batch_size)

        phone_images = train_data[idx_train]
        dslr_images = train_answ[idx_train]

        # Data augmentation: random flips and rotations
        for k in range(batch_size):

            random_rotate = np.random.randint(1, 100) % 4
            phone_images[k] = np.rot90(phone_images[k], random_rotate)
            dslr_images[k] = np.rot90(dslr_images[k], random_rotate)
            random_flip = np.random.randint(1, 100) % 2

            if random_flip == 1:
                phone_images[k] = np.flipud(phone_images[k])
                dslr_images[k] = np.flipud(dslr_images[k])

        # Training step
        [loss_temp, temp] = sess.run([loss_generator, train_step_gen], feed_dict={phone_: phone_images, dslr_: dslr_images})
        training_loss += loss_temp / eval_step

        if i % eval_step == 0:

            # Evaluate model
            val_losses = np.zeros((1, 5))

            for j in range(num_val_batches):

                be = j * batch_size
                en = (j+1) * batch_size

                phone_images = val_data[be:en]
                dslr_images = val_answ[be:en]

                losses = sess.run([loss_generator, loss_content, loss_mse, loss_psnr, loss_ms_ssim], \
                                    feed_dict={phone_: phone_images, dslr_: dslr_images})

                val_losses += np.asarray(losses) / num_val_batches

            logs_gen = "step %d | training: %.4g, validation: %.4g | content: %.4g, mse: %.4g, psnr: %.4g, " \
                           "ms-ssim: %.4g\n" % (i, training_loss, val_losses[0][0], val_losses[0][1],
                                                val_losses[0][2], val_losses[0][3], val_losses[0][4])
            print(logs_gen)

            # Save the results to log file
            logs = open(model_dir + "logs_" + str(iter_start) + "-" + str(num_train_iters) + ".txt", "a")
            logs.write(logs_gen)
            logs.write('\n')
            logs.close()

            # Optional: save visual results for several validation image crops
            if save_mid_imgs:
                enhanced_crops = sess.run(enhanced, feed_dict={phone_: visual_val_crops, dslr_: dslr_images})

                idx = 0
                for crop in enhanced_crops:
                    if idx < 4:
                        before_after = np.hstack((crop,
                                        np.reshape(visual_target_crops[idx], [TARGET_HEIGHT, TARGET_WIDTH, TARGET_DEPTH])))
                        imageio.imwrite(result_dir + name_model_save_full + "_img_" + str(idx) + ".jpg",
                                        before_after)
                    idx += 1

            # Saving the model that corresponds to the current iteration
            saver.save(sess, model_dir + name_model_save_full + ".ckpt", write_meta_graph=False)

            training_loss = 0.0

        # Loading new training data
        if i % 1000 == 0:

            del train_data
            del train_answ
            train_data, train_answ = load_train_patch(dataset_dir, dslr_dir, phone_dir, train_size, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE)

    print('total train/eval time:', datetime.now() - time_start)

