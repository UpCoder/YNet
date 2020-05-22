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
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'maskimage/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'fullAnnTumorMask/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'livermask/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'mask_image': slim.tfexample_decoder.Image('maskimage/encoded', 'image/format'),
        'liver_mask': slim.tfexample_decoder.Image('livermask/encoded', 'image/format'),
        'full_mask': slim.tfexample_decoder.Image('fullAnnTumorMask/encoded', 'image/format')
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
