#*******************************************************************************
# Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
# Notified per clause 4(b) of the license.
#*******************************************************************************

#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import argparse
import tensorflow as tf
from google.protobuf import text_format

from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.python.framework import dtypes

#OPTIMIZATION = 'strip_unused_nodes remove_nodes(op=Identity, op=CheckNumerics) fold_constants(ignore_errors=true) fold_batch_norms fold_old_batch_norms'

def load_graph(model_file, inputs, outputs, opt_graph):
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()
    file_ext = os.path.splitext(model_file)[1]
    with open(model_file, "rb") as f:
        if file_ext == '.pbtxt':
            text_format.Merge(f.read(), graph_def)
        else:
            graph_def.ParseFromString(f.read())
    with graph.as_default():
        if opt_graph:
            graph_def = optimize_for_inference(graph_def, [inputs], [outputs],
                                               dtypes.float32.as_datatype_enum, False)
        tf.import_graph_def(graph_def, name='')
    return graph

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        help="name of model")
    parser.add_argument("--model_file", required=True,
                        help="graph/model to be executed")
    parser.add_argument("--input_height", required=True,
                        type=int, help="input height")
    parser.add_argument("--input_width", required=True,
                        type=int, help="input width")
    parser.add_argument("--batch_size", default=640,
                        type=int, help="batch size")
    parser.add_argument("--input_layer", required=True,
                        help="name of input layer")
    parser.add_argument("--output_layer", required=True,
                        help="name of output layer")
    parser.add_argument("--num_inter_threads", default=1,
                        type=int, help="number threads across operators")
    parser.add_argument("--num_intra_threads", default=24,
                        type=int, help="number threads for an operator")
    parser.add_argument("--opt_graph", default=False,
                        action="store_true", help="Enable or disable demo mode")
    parser.add_argument("--warmup_steps", default=10,
                        type=int, help="number of warmup steps")
    parser.add_argument("--steps", default=100,
                        type=int, help="number of steps")
    parser.add_argument("--data_location", default=None,
                         help='location of validation data.')
    parser.add_argument("--data_resize_method", default='crop',
                         help='Specify the image preprocessing method - crop/bilinear.')
    parser.add_argument('--data_num_inter_threads', default=1,
                        type=int, help='number threads across operators')
    parser.add_argument('--data_num_intra_threads', default=24,
                        type=int, help='number threads for data layer operator')
    parser.add_argument('--num_cores', default=24,
                        type=int, help='number of cores')
    args = parser.parse_args()
    return args
