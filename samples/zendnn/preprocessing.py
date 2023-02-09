#!/usr/bin/env python

#*******************************************************************************
# Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
# Notified per clause 4(b) of the license.
#*******************************************************************************

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

"""Image pre-processing utilities.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob

import numpy as np
import tensorflow as tf

def _mean_image_subtraction(image, means):
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(axis=2, values=channels)

def eval_image(image, height, width, resize_method,
               central_fraction=0.875, scope=None):

    with tf.compat.v1.name_scope('eval_image'):
        if resize_method == 'crop' or resize_method == 'cropwithmeansub':
            shape = tf.shape(input=image)
            image = tf.cond(pred=tf.less(shape[0], shape[1]),
                            true_fn=lambda: tf.image.resize(image,
                                                           tf.convert_to_tensor(value=[256, 256 * shape[1] / shape[0]],
                                                                                dtype=tf.int32)),
                            false_fn=lambda: tf.image.resize(image,
                                                           tf.convert_to_tensor(value=[256 * shape[0] / shape[1], 256],
                                                                                dtype=tf.int32)))
            shape = tf.shape(input=image)
            y0 = (shape[0] - height) // 2
            x0 = (shape[1] - width) // 2
            distorted_image = tf.image.crop_to_bounding_box(image, y0, x0, height, width)
            distorted_image.set_shape([height, width, 3])
            if resize_method == 'cropwithmeansub':
                distorted_image = tf.compat.v1.to_float(distorted_image)
                distorted_image = _mean_image_subtraction(distorted_image, [123.68, 116.78, 103.94])
            return distorted_image
        else:  # bilinear
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            # Crop the central region of the image with an area containing 87.5% of
            # the original image.
            if central_fraction:
                image = tf.image.central_crop(image, central_fraction=central_fraction)

            if height and width:
                # Resize the image to the specified height and width.
                image = tf.expand_dims(image, 0)
                #image = tf.image.resize(image, [height, width],
                #                                 method=tf.image.ResizeMethod.BILINEAR)
                image = tf.image.resize(image, [height, width],
                                                 method=tf.image.ResizeMethod.BILINEAR)
                image = tf.squeeze(image, [0])
            image = tf.subtract(image, 0.5)
            image = tf.multiply(image, 2.0)
            return image



class FolderInputPreprocessor(tf.keras.Sequential):

    def __init__(self,
                 data_location,
                 height,
                 width,
                 batch_size,
                 resize_method='cropwithmeansub'):

        self.data_location = data_location
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.resize_method = resize_method

        self.index = -1
        self.file_paths = self.__parse_data(self.data_location)
        self.filepath_len = len(self.file_paths)

    def __parse_data(self, data_location):

        file_paths = glob.glob(data_location+'/*')
        if(not len(file_paths)):
            raise ValueError("The location given is empty")
        return file_paths

    def __get_input(self, filepath, height, width, resize_method):

        image = tf.io.read_file(filepath)
        image = tf.io.decode_jpeg(image, channels=3,
                                  fancy_upscaling=False, dct_method='INTEGER_FAST')
        image = eval_image(image, height, width, resize_method)
        return image

    def __get_data(self, filepath_batch):

        return np.asarray(
            [self.__get_input(filepath, self.height, self.width, self.resize_method) for filepath in filepath_batch]
        )

    def __iter__(self):
        return self

    def __next__(self):
        self.index += 1
        start = self.index * self.batch_size
        end = (self.index + 1) * self.batch_size
        if(end > self.filepath_len):
            end = self.filepath_len
        filepath_batch = self.file_paths[start:end]

        if(not len(filepath_batch)):
            raise StopIteration
        return self.__get_data(filepath_batch), filepath_batch
