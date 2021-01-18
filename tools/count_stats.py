##########################
# Count model statistics #
##########################

import argparse

from millify import millify, prettify
import tensorflow.compat.v1 as tf


def load_pb(pb_fpath):
    with tf.gfile.GFile(pb_fpath, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph


def profile_using_pb(pb_fpath, cmd_args):
    graph = load_pb(pb_fpath)
    with graph.as_default():
        if cmd_args.MAC:
            # [Profiling for flops]
            prof_opts = tf.profiler.ProfileOptionBuilder.float_operation()
            prof_opts['trim_name_regexes'] = [
                '.*BatchNorm.*', '.*Initializer.*', '.*Regularizer.*', '.*BiasAdd.*'
            ]
            flops = tf.profiler.profile(graph, options=prof_opts)
            print(
                f'FLOPs:\n\t{prettify(flops.total_float_ops)}, {millify(flops.total_float_ops, precision=2)}'
            )
            mac_value = round(flops.total_float_ops / 2)
            print(f'MACs (FLOPs / 2):\n\t{prettify(mac_value)}, {millify(mac_value, precision=2)}')
        if cmd_args.Param:
            raise NotImplementedError('Functionality unavailable when using pb.')


def parse_args():
    parser = argparse.ArgumentParser(description="Tensorflow (meta) graph flops and param counter")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pb_fpath', help='Path to fix-shaped pb file (.pb).')
    group = parser.add_argument_group('stats', 'What to count.')
    group.add_argument('--MAC', action='store_true', help='Count number of MACs.')
    group.add_argument('--Param', action='store_true', help='Count parameter size.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    profile_using_pb(args.pb_fpath, args)


if __name__ == '__main__':
    main()
