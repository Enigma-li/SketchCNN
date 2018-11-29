#
# Project SketchCNN
#
#   Author: Changjian Li (chjili2011@gmail.com),
#   Copyright (c) 2018. All Rights Reserved.
#
# ==============================================================================
"""Freeze graph to be used in production.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from SketchCNN.script.loader import SketchReader
import os
import argparse

# Hyper parameters
hyper_parameters = {
    'naiveNet': "SASDNNet/output_d,SASDNNet/output_n",
    'baselineNet': 'SASBLNet/output_d,SASBLNet/output_n,SASBLNet/output_c',
    'fullNet': 'SASFieldNet/output_f,SASMFGeoNet/output_d,SASMFGeoNet/output_n,SASMFGeoNet/output_c',
    'output_dir': '',
    'ckpt_dir': '',
    'input_graph_name': '',
    'output_graph_name': '',
}

# predefined network type
net_type = ''


def convert_model():
    input_checkpoint_path = tf.train.latest_checkpoint(hyper_parameters['ckpt_dir'])
    input_graph_path = os.path.join(hyper_parameters['output_dir'], hyper_parameters['input_graph_name'])
    input_saver_def_path = ""
    input_binary = False
    output_node_names = hyper_parameters[net_type]
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_graph_path = os.path.join(hyper_parameters['output_dir'], hyper_parameters['output_graph_name'])
    clear_devices = False

    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path, input_binary, input_checkpoint_path,
                              output_node_names, restore_op_name, filename_tensor_name, output_graph_path,
                              clear_devices, initializer_nodes='', variable_names_blacklist='')


if __name__ == '__main__':
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True, help='output path for frozen network', type=str)
    parser.add_argument('--ckpt_dir', required=True, help='checkpoint path', type=str)
    parser.add_argument('--ckpt_name', required=True, help='input checkpoint name - ckpt_name.pbtxt', type=str)
    parser.add_argument('--graph_name', required=True, help='frozen graph name - ckpt_frozen.pb', type=str)
    parser.add_argument('--net_type', required=True, help='network type to selection output nodes', type=int)

    args = parser.parse_args()
    hyper_parameters['output_dir'] = args.output_dir
    hyper_parameters['ckpt_dir'] = args.ckpt_dir
    hyper_parameters['input_graph_name'] = args.ckpt_name
    hyper_parameters['output_graph_name'] = args.graph_name
    if args.net_type == 0:
        net_type = 'naiveNet'
    elif args.net_type == 1:
        net_type = 'baselineNet'
    elif args.net_type == 2:
        net_type = 'fullNet'

    convert_model()
