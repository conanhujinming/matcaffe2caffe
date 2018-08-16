# -*- coding: utf-8 -*-

"""
This is a tool for converting a matlab caffe(column major) model to a pycaffe(row major) model 
"""

from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
import math, copy
import sys,os
import caffe
import caffe.proto.caffe_pb2 as caffe_pb2
import time
from google.protobuf import text_format


def parse_args():
    parser = argparse.ArgumentParser(
        description='convert a matcaffe model to a pycaffe model(column major to row major')
    parser.add_argument('--proto', dest='proto',
                        help="path to deploy prototxt.", type=str)
    parser.add_argument('--model', dest='model',
                        help='path to pretrained weights', type=str)
    parser.add_argument('--output', dest='output',
                        help='path to output model', type=str, default='pycaffe.caffemodel')

    args = parser.parse_args()
    return args, parser


global args, parser
args, parser = parse_args()


def usage_info():
    """
    usage info
    """
    print("Input params is illegal...")
    print("try it again:\n python matcaffe2pycaffe.py -h")


def main():
    """
    main function
    """
    print(args)

    if args.proto == None or args.model == None:
        usage_info()
        return None

    # deploy caffe prototxt path
    net_file = args.proto

    # trained caffemodel path
    caffe_model = args.model

    # the output caffemodel file
    output_path = args.output

    caffe.set_mode_cpu()
    net = caffe.Net(net_file,caffe_model,caffe.TEST)
    last_shape = None
    for param_name in net.params.keys():
        idx = list(net._layer_names).index(param_name)
        if net.layers[idx].type in ['Convolution', 'InnerProduct']:
            # for typical conv filter, we need to transpose its weight
            if len(net.params[param_name][0].data.shape) == 4:
                trans = net.params[param_name][0].data.transpose((0, 1, 3, 2))
            # for innerproduct filter
            elif len(net.params[param_name][0].data.shape) == 2:
                ori_shape = net.params[param_name][0].data.shape
                # if the previous layer is typical conv filter, then we still need to transpose its weight
                if last_shape and last_shape == 4:
                    trans = net.params[param_name][0].data.reshape(ori_shape[0], -1, 3, 3).\
                                                     transpose((0, 1, 3, 2)).reshape(*ori_shape)
                # if the previous layer is not typical conv filter, we do nothing
                else:
                    trans = net.params[param_name][0].data
            # net.params[param_name][0].reshape(*trans.shape)
            net.params[param_name][0].data[...] = trans
            last_shape = len(net.params[param_name][0].data.shape)

    net.save(output_path)
    print("\nConversion complete!")

if __name__ == "__main__":
    main()
