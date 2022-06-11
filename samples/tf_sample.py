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

import os
import time

import numpy as np
import tensorflow as tf

import utils
from preprocessing import FolderInputPreprocessor

if __name__ == "__main__":
    args = utils.get_arguments()

    with open("imagenet_classes.txt") as f:
        label_name_list = np.asarray(
            [line.strip() for line in f.readlines()])

    label_offset = 0
    if(args.data_resize_method!='cropwithmeansub'):
        label_offset = -1

    igraph = utils.load_graph(args.model_file, args.input_layer, args.output_layer, args.opt_graph)

    input_tensor = igraph.get_tensor_by_name(args.input_layer + ":0")
    output_tensor = igraph.get_tensor_by_name(args.output_layer + ":0")

    iconfig = tf.compat.v1.ConfigProto()
    iconfig.inter_op_parallelism_threads = args.num_inter_threads
    iconfig.intra_op_parallelism_threads = args.num_intra_threads
    infer_sess = tf.compat.v1.Session(graph=igraph, config=iconfig)

    ds = FolderInputPreprocessor(args.data_location,
                                 args.input_height, args.input_width,
                                 args.batch_size, args.data_resize_method)

    total_time = 0
    for np_images, filenames in ds:
        start = time.time()
        predictions = infer_sess.run(output_tensor,
                                    {input_tensor: np_images})
        total_time += time.time() - start
        for img_count in range(len(np_images)):

            top_1 = np.argsort(predictions[img_count])[-1] + label_offset
            top_5 = np.argsort(predictions[img_count])[-5:]
            top_5 = [x + label_offset for x in top_5]

            print("\nFilename: {}".format(filenames[img_count]))
            print("Top 1 Class: {}".format(label_name_list[top_1]))
            print("Top 5 classes: {}".format("; ".join(label_name_list[top_5][::-1])))

    avg_throughput = ds.filepath_len / total_time
    print("\nAverage Throughput for {} images: {:.2f} images/s".format(ds.filepath_len, avg_throughput))