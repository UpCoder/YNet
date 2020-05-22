# -*- coding=utf-8 -*-
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

from datasets.dataset_utils import int64_feature, float_feature, bytes_feature, EncodedFloatFeature, EncodedInt64Feature

# Original dataset organisation.
# TFRecords convertion parameters.
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 200
MIN_AREA = 20

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


def _convert_to_example(image_data_np, tumor_fully_mask_np, mask_np, liver_mask_np, bbox_np):
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
    bbox_np = np.asarray(np.asarray(bbox_np, np.float32) / 512., np.float32)
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    labels = []
    for b in bbox_np:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        labels.append(1)
    image_format = b'PNG'
    example = tf.train.Example(features=tf.train.Features(feature={
        # 'image/height': int64_feature(512),
        # 'image/width': int64_feature(512),
        # 'image/channels': int64_feature(3),
        # 'image/shape': int64_feature([512, 512, 3]),
        'image/format': bytes_feature(image_format),
        # 'image/encoded': bytes_feature(np.asarray(image_data_np, np.float32).tostring()),
        # 'livermask/encoded': bytes_feature(np.asarray(liver_mask_np, np.uint8).tostring()),
        # 'fullAnnTumorMask/encoded': bytes_feature(np.asarray(tumor_fully_mask_np, np.uint8).tostring()),
        # 'maskimage/encoded': bytes_feature(np.asarray(mask_np, np.uint8).tostring())}))
        'image/encoded': EncodedFloatFeature(np.asarray(image_data_np, np.float32)),
        'livermask/encoded': EncodedInt64Feature(np.asarray(liver_mask_np, np.int64)),
        'fullAnnTumorMask/encoded': EncodedInt64Feature(np.asarray(tumor_fully_mask_np, np.int64)),
        'maskimage/encoded': EncodedInt64Feature(np.asarray(mask_np, np.int64)),
        'image/object/bbox/xmin': float_feature(xmin),
        'image/object/bbox/xmax': float_feature(xmax),
        'image/object/bbox/ymin': float_feature(ymin),
        'image/object/bbox/ymax': float_feature(ymax),
        'image/object/bbox/label': int64_feature(labels)}))

    return example


def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def run(dataset_dir, output_dir, shuffling=True, name='medicalimage'):
    """Runs the conversion operation.
    1、有病灶，没病灶的slice都有
    2、处理每个case slice之间的不平衡是通过阶梯采样
    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """

    def _get_boundingboxes_from_binary_slice(slice):
        from skimage.measure import label
        res = label(slice, neighbors=8)
        num_roi = np.max(res)
        # print "num_roi is ", num_roi
        boundingboxes = []
        slices = []
        for i in range(1, num_roi + 1):
            xs, ys = np.where(res == i)

            if len(xs) < MIN_AREA:
                continue
            w = np.max(xs) - np.min(xs)
            h = np.max(ys) - np.min(ys)

            boundingboxes.append(
                [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]
            )

            # print np.min(xs), np.min(ys), np.max(xs), np.max(ys)
        # print np.shape(boundingboxes)
        return boundingboxes
    from medicalImage import read_nii_with_header, read_nii
    from skimage.measure import label
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)
    z_axis = 5.0
    img_nii_paths = glob(os.path.join(dataset_dir, 'volume*.nii'))
    imgs = []
    names = []
    masks = []
    tumor_weakly_masks = []
    bboxes = []
    liver_masks = []

    fidx = 0
    sample_id = 0
    tf_filename = _get_output_filename(output_dir, name, fidx)
    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for img_nii_path in img_nii_paths:
            print(os.path.basename(img_nii_path))
            seg_nii_path = os.path.join(dataset_dir,
                                        'segmentation-' + os.path.basename(img_nii_path).split('.')[0].split('-')[
                                            1] + '.nii')
            volume, header = read_nii_with_header(img_nii_path)
            # volume = np.transpose(volume, [1, 0, 2])
            volume = np.asarray(volume, np.float32)
            max_v = 250.
            min_v = -200.
            volume[volume > max_v] = max_v
            volume[volume < min_v] = min_v
            volume -= np.mean(volume)
            min_v = np.min(volume)
            max_v = np.max(volume)
            interv = max_v - min_v
            volume = (volume - min_v) / interv

            z_axis_case = header['voxel_spacing'][-1]
            slice_num = np.ceil((1.0 * z_axis) / (1.0 * z_axis_case))
            if slice_num == 0:
                slice_num = 1
            slice_num = int(slice_num)

            seg = read_nii(seg_nii_path)

            [_, _, channel] = np.shape(volume)
            i = 0
            pos_slice_num = np.sum(np.sum(np.sum(seg == 2, axis=0), axis=0) >= 20)
            neg_rate = (1.0 * pos_slice_num) / (1.0 * channel)  # 正样本的比例
            if neg_rate > 1.0:
                neg_rate = 1.0
            print('neg_rate is ', neg_rate)
            slice_interv = 1
            if pos_slice_num > 150:
                slice_interv = 4
            elif pos_slice_num > 100:
                slice_interv = 3
            elif pos_slice_num > 50:
                slice_interv = 2
            while i < channel:
                seg_slice = seg[:, :, i]
                if np.sum(seg_slice == 2) >= 20:
                    # 病灶的大小大于20
                    seg_slice = seg[:, :, i]

                    # if slice_num != 1:
                    #     mid_slice = []
                    #     for j in range(int(i - slice_num / 2), int(i + slice_num / 2)):
                    #         if j < 0:
                    #             cur_slice = volume[:, :, i]
                    #         elif j >= channel:
                    #             cur_slice = volume[:, :, i]
                    #         else:
                    #             cur_slice = volume[:, :, j]
                    #         mid_slice.append(cur_slice)
                    #     mid_slice = np.mean(mid_slice, axis=0, keepdims=True)
                    # else:
                    #     mid_slice = np.expand_dims(volume[:, :, i], axis=0)
                    mid_slice = np.expand_dims(volume[:, :, i], axis=0)

                    pre_slice = []
                    # pre_end = i - slice_num / 2
                    pre_end = i
                    for j in range(1, slice_num + 1):
                        z = pre_end - j
                        if z < 0:
                            z = 0
                        pre_slice.append(volume[:, :, z])
                    next_slice = []
                    # next_start = i + slice_num / 2
                    next_start = i
                    for j in range(1, slice_num + 1):
                        z = next_start + j
                        if z >= channel:
                            z = channel - 1
                        next_slice.append(volume[:, :, z])
                    pre_slice = np.mean(pre_slice, axis=0, keepdims=True)
                    next_slice = np.mean(next_slice, axis=0, keepdims=True)
                    imgs.append(
                        np.transpose(np.concatenate([pre_slice, mid_slice, next_slice], axis=0),
                                     axes=[1, 2, 0]))

                    names.append(os.path.basename(img_nii_path).split('.')[0].split('-')[1] + '-' + str(i))
                    binary_seg_slice = np.asarray(seg_slice == 2, np.uint8)
                    # print np.max(binary_seg_slice)
                    masks.append(binary_seg_slice)
                    labeled_mask = label(binary_seg_slice)
                    weakly_label_mask = np.zeros_like(binary_seg_slice, np.uint8)
                    for idx in range(1, np.max(labeled_mask) + 1):
                        xs, ys = np.where(labeled_mask == idx)
                        min_xs = np.min(xs)
                        max_xs = np.max(xs)
                        min_ys = np.min(ys)
                        max_ys = np.max(ys)
                        weakly_label_mask[min_xs: max_xs, min_ys: max_ys] = 1
                    liver_masks.append(np.asarray(seg_slice == 1, np.uint8))
                    tumor_weakly_masks.append(weakly_label_mask)
                    bboxes.append(_get_boundingboxes_from_binary_slice(binary_seg_slice))
                    print('add positive, ', np.shape(imgs), np.shape(masks), np.shape(names), np.shape(liver_masks))
                    example = _convert_to_example(imgs[-1], masks[-1], tumor_weakly_masks[-1], liver_masks[-1],
                                                  bboxes[-1])
                    tfrecord_writer.write(example.SerializeToString())
                    i += slice_interv
                else:
                    if np.random.random() < (neg_rate / 3.):
                        mid_slice = np.expand_dims(volume[:, :, i], axis=0)

                        pre_slice = []
                        # pre_end = i - slice_num / 2
                        pre_end = i
                        for j in range(1, slice_num + 1):
                            z = pre_end - j
                            if z < 0:
                                z = 0
                            pre_slice.append(volume[:, :, z])
                        next_slice = []
                        # next_start = i + slice_num / 2
                        next_start = i
                        for j in range(1, slice_num + 1):
                            z = next_start + j
                            if z >= channel:
                                z = channel - 1
                            next_slice.append(volume[:, :, z])
                        pre_slice = np.mean(pre_slice, axis=0, keepdims=True)
                        next_slice = np.mean(next_slice, axis=0, keepdims=True)
                        imgs.append(
                            np.transpose(np.concatenate([pre_slice, mid_slice, next_slice], axis=0),
                                         axes=[1, 2, 0]))
                        names.append(os.path.basename(img_nii_path).split('.')[0].split('-')[1] + '-' + str(i))
                        binary_seg_slice = np.asarray(seg_slice == 2, np.uint8)
                        masks.append(binary_seg_slice)
                        labeled_mask = label(binary_seg_slice)
                        weakly_label_mask = np.zeros_like(binary_seg_slice, np.uint8)
                        for idx in range(1, np.max(labeled_mask) + 1):
                            xs, ys = np.where(labeled_mask == idx)
                            min_xs = np.min(xs)
                            max_xs = np.max(xs)
                            min_ys = np.min(ys)
                            max_ys = np.max(ys)
                            weakly_label_mask[min_xs: max_xs, min_ys: max_ys] = 1

                        liver_masks.append(np.asarray(seg_slice == 1, np.uint8))
                        tumor_weakly_masks.append(weakly_label_mask)
                        bboxes.append(_get_boundingboxes_from_binary_slice(binary_seg_slice))
                        print('add negative: ', np.shape(imgs), np.shape(masks), np.shape(names), np.shape(liver_masks))
                        example = _convert_to_example(imgs[-1], masks[-1], tumor_weakly_masks[-1], liver_masks[-1],
                                                      bboxes[-1])
                        tfrecord_writer.write(example.SerializeToString())
                        sample_id += 1
                    else:
                        print('ignore ', i)
                    i += slice_interv
    print('\nFinished converting the Pascal VOC dataset!')


def run_V2(dataset_dir, output_dir, shuffling=True, name='medicalimage', npy_flag=False):
    """Runs the conversion operation.
    1、只有病灶的slice才有
    2、处理每个case slice之间的不平衡是通过随机采样，每个case最多50个slice
    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    def _get_boundingboxes_from_binary_slice(slice):
        from skimage.measure import label
        res = label(slice, neighbors=8)
        num_roi = np.max(res)
        # print "num_roi is ", num_roi
        boundingboxes = []
        slices = []
        for i in range(1, num_roi + 1):
            xs, ys = np.where(res == i)

            if len(xs) < MIN_AREA:
                continue
            w = np.max(xs) - np.min(xs)
            h = np.max(ys) - np.min(ys)

            boundingboxes.append(
                [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]
            )

            # print np.min(xs), np.min(ys), np.max(xs), np.max(ys)
        # print np.shape(boundingboxes)
        return boundingboxes
    from medicalImage import read_nii_with_header, read_nii
    from skimage.measure import label
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)
    z_axis = 5.0
    max_slice_num = 30
    img_nii_paths = glob(os.path.join(dataset_dir, 'volume*.nii'))
    imgs = []
    names = []
    masks = []
    tumor_weakly_masks = []
    bboxes = []
    liver_masks = []

    fidx = 3
    sample_id = 0
    tf_filename = _get_output_filename(output_dir, name, fidx)
    if not npy_flag:
        tfrecord_writer = tf.python_io.TFRecordWriter(tf_filename)

    for img_nii_path in img_nii_paths:
        # if os.path.basename(img_nii_path) not in ['volume-5.nii', 'volume-12.nii', 'volume-13.nii', 'volume-15.nii',
        #                                           'volume-19.nii', 'volume-20.nii', 'volume-21.nii', 'volume-25.nii',
        #                                           'volume-26.nii']:
        #     continue
        print(os.path.basename(img_nii_path))
        seg_nii_path = os.path.join(dataset_dir,
                                    'segmentation-' + os.path.basename(img_nii_path).split('.')[0].split('-')[
                                        1] + '.nii')
        volume, header = read_nii_with_header(img_nii_path)
        # volume = np.transpose(volume, [1, 0, 2])
        volume = np.asarray(volume, np.float32)
        max_v = 250.
        min_v = -200.
        volume[volume > max_v] = max_v
        volume[volume < min_v] = min_v
        volume -= np.mean(volume)
        min_v = np.min(volume)
        max_v = np.max(volume)
        interv = max_v - min_v
        volume = (volume - min_v) / interv

        z_axis_case = header['voxel_spacing'][-1]
        slice_num = np.ceil((1.0 * z_axis) / (1.0 * z_axis_case))
        if slice_num == 0:
            slice_num = 1
        slice_num = int(slice_num)

        seg = read_nii(seg_nii_path)

        [_, _, channel] = np.shape(volume)
        i = 0
        indexs = np.where(np.sum(np.sum(seg == 2, axis=0), axis=0) >= 20)[0]
        np.random.shuffle(indexs)
        if len(indexs) > max_slice_num:
            indexs = indexs[:max_slice_num]
        pos_slice_num = np.sum(np.sum(np.sum(seg == 2, axis=0), axis=0) >= 20)
        neg_rate = (1.0 * pos_slice_num) / (1.0 * channel)  # 正样本的比例
        if neg_rate > 1.0:
            neg_rate = 1.0
        print('neg_rate is ', neg_rate)

        for i in indexs:
            seg_slice = seg[:, :, i]
            # if slice_num != 1:
            #     mid_slice = []
            #     for j in range(int(i - slice_num / 2), int(i + slice_num / 2)):
            #         if j < 0:
            #             cur_slice = volume[:, :, i]
            #         elif j >= channel:
            #             cur_slice = volume[:, :, i]
            #         else:
            #             cur_slice = volume[:, :, j]
            #         mid_slice.append(cur_slice)
            #     mid_slice = np.mean(mid_slice, axis=0, keepdims=True)
            # else:
            #     mid_slice = np.expand_dims(volume[:, :, i], axis=0)
            mid_slice = np.expand_dims(volume[:, :, i], axis=0)

            # pre_slice = []
            # # pre_end = i - slice_num / 2
            # pre_end = i
            # for j in range(1, slice_num + 1):
            #     z = pre_end - j
            #     if z < 0:
            #         z = 0
            #     pre_slice.append(volume[:, :, z])
            pre_slice = np.expand_dims(volume[:, :, i-1], axis=0)
            # next_slice = []
            # # next_start = i + slice_num / 2
            # next_start = i
            # for j in range(1, slice_num + 1):
            #     z = next_start + j
            #     if z >= channel:
            #         z = channel - 1
            #     next_slice.append(volume[:, :, z])
            # pre_slice = np.mean(pre_slice, axis=0, keepdims=True)
            # next_slice = np.mean(next_slice, axis=0, keepdims=True)
            next_slice = np.expand_dims(volume[:, :, i+1], axis=0)
            imgs.append(
                np.transpose(np.concatenate([pre_slice, mid_slice, next_slice], axis=0),
                             axes=[1, 2, 0]))

            names.append(os.path.basename(img_nii_path).split('.')[0].split('-')[1] + '-' + str(i))
            binary_seg_slice = np.asarray(seg_slice == 2, np.uint8)
            # print np.max(binary_seg_slice)
            masks.append(binary_seg_slice)
            labeled_mask = label(binary_seg_slice)
            weakly_label_mask = np.zeros_like(binary_seg_slice, np.uint8)
            for idx in range(1, np.max(labeled_mask) + 1):
                xs, ys = np.where(labeled_mask == idx)
                min_xs = np.min(xs)
                max_xs = np.max(xs)
                min_ys = np.min(ys)
                max_ys = np.max(ys)
                weakly_label_mask[min_xs: max_xs, min_ys: max_ys] = 1
            liver_masks.append(np.asarray(seg_slice == 1, np.uint8))
            tumor_weakly_masks.append(weakly_label_mask)
            bboxes.append(_get_boundingboxes_from_binary_slice(binary_seg_slice))
            if not npy_flag:
                print('add positive, ', np.shape(imgs), np.shape(masks), np.shape(names), np.shape(liver_masks))
                example = _convert_to_example(imgs[-1], masks[-1], tumor_weakly_masks[-1], liver_masks[-1],
                                              bboxes[-1])
                tfrecord_writer.write(example.SerializeToString())
    if npy_flag:
        np.save(os.path.join(output_dir, 'imgs_' + str(fidx) + '.npy'), np.asarray(imgs, np.float32))
        np.save(os.path.join(output_dir, 'mask_' + str(fidx) + '.npy'), np.asarray(masks, np.float32))
    print('\nFinished converting the Pascal VOC dataset!')


if __name__ == '__main__':
    dataset_dir = '/media/give/CBMIR/ld/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training_Batch_2'
    output_dir = '/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_2/tfrecords_V2_V2'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    run_V2(dataset_dir, output_dir, name='ISBI2017', shuffling=True, npy_flag=True)


