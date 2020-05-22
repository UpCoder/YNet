# Copyright 2015 Paul Balanca. All Rights Reserved.
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
"""Converts Pascal VOC data to TFRecords file format with Example protos.

The raw Pascal VOC data set is expected to reside in JPEG files located in the
directory 'JPEGImages'. Similarly, bounding box annotations are supposed to be
stored in the 'Annotation directory'

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of 1024 and 128 TFRecord files, respectively.

Each validation TFRecord file contains ~500 records. Each training TFREcord
file contains ~1000 records. Each record within the TFRecord file is a
serialized Example proto. The Example proto contains the following fields:

    image/encoded: string containing JPEG encoded image in RGB colorspace
    image/height: integer, image height in pixels
    image/width: integer, image width in pixels
    image/channels: integer, specifying the number of channels, always 3
    image/format: string, specifying the format, always'JPEG'


    image/object/bbox/xmin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/xmax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/label: list of integer specifying the classification index.
    image/object/bbox/label_text: list of string descriptions.

Note that the length of xmin is identical to the length of xmax, ymin and ymax
for each example.
"""
import os
import sys
import random

import numpy as np
import tensorflow as tf
from glob import glob
from tqdm import tqdm
import cv2
import xml.etree.ElementTree as ET

from datasets.dataset_utils import int64_feature, float_feature, bytes_feature

# Original dataset organisation.
# TFRecords convertion parameters.
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 200

MEDICAL_LABELS = {
    'none': (0, 'Background'),
    'CYST': (1, 'Begin'),
    'FNH': (1, 'Begin'),
    'HCC': (1, 'Begin'),
    'HEM': (1, 'Begin'),
    'METS': (1, 'Begin'),
}
MEDICAL_LABELS_multi_category = {
    'none': (0, 'Background'),
    'CYST': (1, 'Begin'),
    'FNH': (2, 'Begin'),
    'HCC': (3, 'Begin'),
    'HEM': (4, 'Begin'),
    'METS': (5, 'Begin'),
}


def preprocessing_dir(dataset_dir):
    case_names = os.listdir(dataset_dir)
    for case_name in tqdm(case_names):
        mask_dir = os.path.join(dataset_dir, case_name, 'masks')
        mask_paths = glob(os.path.join(mask_dir, '*.png'))
        mask = None
        for mask_path in mask_paths:
            cur_mask = cv2.imread(mask_path)
            if mask is None:
                mask = np.asarray(cur_mask, np.uint8)
            else:
                mask += cur_mask
        mask = np.asarray(mask != 0, np.uint8)
        cv2.imwrite(os.path.join(dataset_dir, case_name, 'whole_mask.png'), mask)
        cv2.imwrite(os.path.join(dataset_dir, case_name, 'whole_mask_show.png'), mask * 250)


def _process_image(img_dir, label_dir, weakly_label_dir, liver_mask_dir, case_name):
    """Process a image and annotation file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    filename = os.path.join(img_dir, case_name)
    print('filename is ', filename)
    image_data = tf.gfile.FastGFile(filename, 'rb').read()

    mask_path = os.path.join(weakly_label_dir, case_name)
    mask_data = tf.gfile.FastGFile(mask_path,'rb').read()

    tumor_fully_mask_path = os.path.join(label_dir, case_name)
    tumor_fully_mask_data = tf.gfile.FastGFile(tumor_fully_mask_path, 'rb').read()

    liver_mask_path = os.path.join(liver_mask_dir, case_name)
    liver_mask_data = tf.gfile.FastGFile(liver_mask_path, 'rb').read()

    return image_data, tumor_fully_mask_data, mask_data, liver_mask_data


def _convert_to_example(image_data, tumor_fully_mask_data, mask_data, liver_mask_data):
    """Build an Example proto for an image example.

    Args:
      image_data: string, JPEG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """
    image_format = b'PNG'
    example = tf.train.Example(features=tf.train.Features(feature={
        # 'image/height': int64_feature(512),
        # 'image/width': int64_feature(512),
        # 'image/channels': int64_feature(3),
        # 'image/shape': int64_feature([512, 512, 3]),
        'image/format': bytes_feature(image_format),
        'image/encoded': bytes_feature(image_data),
        'livermask/encoded': bytes_feature(liver_mask_data),
        'fullAnnTumorMask/encoded': bytes_feature(tumor_fully_mask_data),
        'maskimage/encoded': bytes_feature(mask_data)}))
    return example


def _add_to_tfrecord(img_dir, label_dir, weakly_label_dir, liver_mask_dir, case_name, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """

    image_data, tumor_fully_mask_data, mask_data, liver_mask_data = _process_image(img_dir, label_dir, weakly_label_dir,
                                                                                   liver_mask_dir, case_name)

    example = _convert_to_example(image_data, tumor_fully_mask_data, mask_data, liver_mask_data)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def run(dataset_dir, output_dir, shuffling=True, name='medicalimage'):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    # Dataset filenames, and shuffling.
    img_dir = os.path.join(dataset_dir, 'img')
    weakly_label_dir = os.path.join(dataset_dir, 'weakly_label_gt')
    label_dir = os.path.join(dataset_dir, 'mask_gt')
    liver_mask_dir = os.path.join(dataset_dir, 'liver_mask')
    case_names = os.listdir(img_dir)
    case_names = sorted(case_names)
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(case_names)

    # Process dataset files.
    i = 0
    fidx = 0
    while i < len(case_names):
        # Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(case_names) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(case_names)))
                sys.stdout.flush()
                case_name = case_names[i]
                _add_to_tfrecord(img_dir, label_dir, weakly_label_dir, liver_mask_dir, case_name, tfrecord_writer)
                i += 1
                j += 1
            fidx += 1

    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the Pascal VOC dataset!')


if __name__ == '__main__':
    dataset_dir = '/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_2'
    output_dir = '/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_2/tfrecords'
    run(dataset_dir, output_dir, name='ISBI2017', shuffling=True)


