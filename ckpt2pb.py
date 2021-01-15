#################################################
# Convert checkpoint to frozen graph (protobuf) #
#################################################

import argparse
import tensorflow as tf


def freeze_graph(input_checkpoint,output_graph,output_node_names):
    """Freeze model weights to get the pb file

    Args:
        input_checkpoint: path to input checkpoint.
        output_graph: path to output pb file.

    """

    # output name in the model graph (may need to check it using tensorboard)
    saver = tf.compat.v1.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
 
    with tf.compat.v1.Session() as sess:
        saver.restore(sess, input_checkpoint) # restore the model parameters
        output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(  # freeze the parameters
            sess=sess,
            input_graph_def=sess.graph_def,
            output_node_names=output_node_names.split(",")) # seperate multiple output names using ","
 
        with tf.io.gfile.GFile(output_graph, "wb") as f: # save the model
            f.write(output_graph_def.SerializeToString()) 
        print("%d ops in the final graph." % len(output_graph_def.node)) # obtain node #


def _parse_argument():
    """Return arguments for Model Freezer for NeuroPilot Model Hub."""
    parser = argparse.ArgumentParser(
        description='Model Freezer for NeuroPilot Model Hub.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--in_path', help='Path to input checkpoint.', type=str, default='model.ckpt', required=True)
    parser.add_argument(
        '--out_path', help='Path to the output pb.', type=str, default='model.pb', required=True)
    parser.add_argument(
        '--out_nodes', help='Output node names.', type=str, default='generator/add_308', required=True)
    return parser.parse_args()


def main(args):
    """Entry point of Model Freezer Top Level Wrapper for NeuroPilot Model Hub.

    Args:
        args: A `argparse.ArgumentParser` includes arguments for processing.

    Raises:
        ValueError: If process type is wrong.
    """
    freeze_graph(args.in_path, args.out_path, args.out_nodes)

if __name__ == '__main__':
    arguments = _parse_argument()
    main(arguments)
