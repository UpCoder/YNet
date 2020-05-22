# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Contains utilities for downloading and converting datasets."""

import tensorflow as tf
import numpy as np

slim = tf.contrib.slim
import util


def get_split(split_name, dataset_dir, file_pattern, num_samples, reader=None):
    dataset_dir = util.io.get_absolute_path(dataset_dir)

    if util.str.contains(file_pattern, '%'):
        file_pattern = util.io.join_path(dataset_dir, file_pattern % split_name)
    else:
        file_pattern = util.io.join_path(dataset_dir, file_pattern)
    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader
    keys_to_features = {
        # 'image/encoded': tf.FixedLenFeature((512, 512, 3), tf.float32, default_value=0.0),
        # 'maskimage/encoded': tf.FixedLenFeature((), tf.int64),
        # 'fullAnnTumorMask/encoded': tf.FixedLenFeature((), tf.int64),
        # 'livermask/encoded': tf.FixedLenFeature((), tf.int64),
        'image/encoded': tf.VarLenFeature(dtype=tf.float32),
        'maskimage/encoded': tf.VarLenFeature(dtype=tf.int64),
        'fullAnnTumorMask/encoded': tf.VarLenFeature(dtype=tf.int64),
        'livermask/encoded': tf.VarLenFeature(dtype=tf.int64),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Tensor('image/encoded'),
        'mask_image': slim.tfexample_decoder.Tensor('maskimage/encoded'),
        'liver_mask': slim.tfexample_decoder.Tensor('livermask/encoded'),
        'full_mask': slim.tfexample_decoder.Tensor('fullAnnTumorMask/encoded'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
            ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/oriented_bbox/x1': slim.tfexample_decoder.Tensor('image/object/bbox/xmin'),
        'object/oriented_bbox/x2': slim.tfexample_decoder.Tensor('image/object/bbox/xmax'),
        'object/oriented_bbox/x3': slim.tfexample_decoder.Tensor('image/object/bbox/xmax'),
        'object/oriented_bbox/x4': slim.tfexample_decoder.Tensor('image/object/bbox/xmin'),
        'object/oriented_bbox/y1': slim.tfexample_decoder.Tensor('image/object/bbox/ymin'),
        'object/oriented_bbox/y2': slim.tfexample_decoder.Tensor('image/object/bbox/ymin'),
        'object/oriented_bbox/y3': slim.tfexample_decoder.Tensor('image/object/bbox/ymax'),
        'object/oriented_bbox/y4': slim.tfexample_decoder.Tensor('image/object/bbox/ymax'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label')
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    labels_to_names = {0: 'background', 1: 'text'}
    items_to_descriptions = {
        'image': 'A color image of varying height and width.',
        'mask_image': 'A binary label map, mask',
    }
    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples,
        items_to_descriptions=items_to_descriptions,
        num_classes=2,
        labels_to_names=labels_to_names)
