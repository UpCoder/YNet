# -*- coding=utf-8 -*-
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
import cv2
from tf_extended.instance_balanced_weight import tf_cal_gt_for_single_image

slim = tf.contrib.slim
import util
from glob import glob


def get_split(split_name, dataset_dir, file_pattern, num_samples, reader=None):
    dataset_dir = util.io.get_absolute_path(dataset_dir)

    if util.str.contains(file_pattern, '%'):
        file_pattern = util.io.join_path(dataset_dir, file_pattern % split_name)
    else:
        file_pattern = util.io.join_path(dataset_dir, file_pattern)
    # Allowing None in the signature so that dataset_factory can use the default.
    print(file_pattern)
    print(glob(file_pattern))
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

    labels_to_names = {0: 'background', 1: 'lesion'}
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


def get_samples(sess, N):
    dataset = get_split('train',
                        '/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_2/tfrecords_V2',
                        'ISBI2017*.tfrecord', 1300)
    with tf.name_scope('ISBI_data_provider'):
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=2,
            common_queue_capacity=1000 * 4,
            common_queue_min=700 * 4,
            shuffle=True)
        # Get for SSD network: image, labels, bboxes.

    [image, full_mask, weak_mask, liver_mask] = provider.get(['image', 'full_mask', 'mask_image', 'liver_mask'])
    image = tf.identity(image, 'input_image')
    print 'image, full_mask, weak_mask, liver_mask are'
    print image, full_mask, weak_mask, weak_mask

    image = tf.reshape(image, [512, 512, 3])
    full_mask = tf.reshape(full_mask, [512, 512, 1])
    weak_mask = tf.reshape(weak_mask, [512, 512, 1])
    liver_mask = tf.reshape(liver_mask, [512, 512, 1])

    full_mask = tf.concat([full_mask, full_mask, full_mask], axis=-1)
    weak_mask = tf.concat([weak_mask, weak_mask, weak_mask], axis=-1)
    liver_mask = tf.concat([liver_mask, liver_mask, liver_mask], axis=-1)

    image = tf.identity(image, 'input_image')

    full_mask = tf.identity(full_mask, 'full_mask')
    weak_mask = tf.identity(weak_mask, 'weak_mask')
    liver_mask = tf.identity(liver_mask, 'liver_mask')
    liver_mask = tf.cast(liver_mask, tf.uint8)
    weak_mask = tf.cast(weak_mask, tf.uint8)
    full_mask = tf.cast(full_mask, tf.uint8)

    print('Image: ', image)
    print('MaskImage: ', weak_mask)
    print('fullMask', full_mask)
    print('LiverMask: ', liver_mask)
    from preprocessing import preprocessing_factory
    image, full_mask, liver_mask = preprocessing_factory.get_preprocessing(is_training=True,
                                                                            method='segmentation')(image,
                                                                                                   full_mask,
                                                                                                   liver_mask,
                                                                                                   out_shape=[256, 256])
    image = tf.identity(image, 'processed_image')
    full_mask = tf.identity(full_mask, 'processed_mask_image')
    liver_mask = tf.identity(liver_mask, 'processed_mask_image')

    # batch them
    with tf.name_scope('ISBI_batch'):
        b_image, b_mask_image, b_liver_mask, b_weak_mask = \
            tf.train.batch(
                [image, full_mask, liver_mask, weak_mask],
                batch_size=4,
                num_threads=24,
                capacity=500)
    with tf.name_scope('ISBI_prefetch_queue'):
        batch_queue = slim.prefetch_queue.prefetch_queue(
            [b_image, b_mask_image, b_liver_mask, b_weak_mask],
            capacity=50)

    b_image, b_full_mask_image, b_liver_mask, b_weak_mask_image = batch_queue.dequeue()

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    print('ok')
    # 使用start_queue_runners 启动队列填充
    threads = tf.train.start_queue_runners(sess, coord)
    count = 0

    imgs = []
    full_masks = []
    weak_masks = []
    liver_masks = []

    for idx in range(N):
        b_image_v, b_full_mask_image_v, b_weak_mask_image_v, b_liver_mask_v = sess.run(
            [b_image, b_full_mask_image, b_weak_mask_image, b_liver_mask])
        print np.shape(b_image_v), np.max(b_image_v), np.min(b_image_v), np.max(b_liver_mask_v)
        imgs.extend(b_image_v)
        full_masks.extend(b_full_mask_image_v)
        weak_masks.extend(b_weak_mask_image_v)
        liver_masks.extend(b_liver_mask_v)
    return np.asarray(imgs, np.float32), np.asarray(full_masks, np.uint8), np.asarray(weak_masks,
                                                                                      np.uint8), np.asarray(
        liver_masks, np.uint8)


if __name__ == '__main__':
    dataset = get_split('train',
                        '/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_2',
                        'ISBI2017*.tfrecord', 1300)
    with tf.name_scope('ISBI_data_provider'):
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=2,
            common_queue_capacity=1000 * 4,
            common_queue_min=700 * 4,
            shuffle=True)
        # Get for SSD network: image, labels, bboxes.

    [image, full_mask, weak_mask, liver_mask, glabel, x1, x2, x3, x4, y1, y2, y3, y4] = provider.get(
        ['image', 'full_mask', 'mask_image', 'liver_mask', 'object/label',
         'object/oriented_bbox/x1',
         'object/oriented_bbox/x2',
         'object/oriented_bbox/x3',
         'object/oriented_bbox/x4',
         'object/oriented_bbox/y1',
         'object/oriented_bbox/y2',
         'object/oriented_bbox/y3',
         'object/oriented_bbox/y4'])
    gxs = tf.transpose(tf.stack([x1, x2, x3, x4]))  # shape = (N, 4)
    gys = tf.transpose(tf.stack([y1, y2, y3, y4]))
    image = tf.identity(image, 'input_image')
    print 'image, full_mask, weak_mask, liver_mask are'
    print image, full_mask, weak_mask, weak_mask
    print gxs, gys, glabel
    image = tf.reshape(image, [512, 512, 3])
    full_mask = tf.reshape(full_mask, [512, 512])
    weak_mask = tf.reshape(weak_mask, [512, 512])
    liver_mask = tf.reshape(liver_mask, [512, 512])

    _, pixel_cls_weight = \
        tf_cal_gt_for_single_image(gxs, gys, glabel)
    with tf.name_scope('ISBI_batch'):
        b_image, b_full_mask_image, b_weak_mask_image, b_liver_mask, b_pixel_cls_weight = \
            tf.train.batch(
                [image, full_mask, weak_mask, liver_mask, pixel_cls_weight],
                batch_size=1,
                num_threads=1,
                capacity=500)
    with tf.name_scope('ISBI_prefetch_queue'):
        batch_queue = slim.prefetch_queue.prefetch_queue(
            [b_image, b_full_mask_image, b_weak_mask_image, b_liver_mask, b_pixel_cls_weight],
            capacity=50)
    b_image, b_full_mask_image, b_weak_mask_image, b_liver_mask, b_pixel_cls_weight = batch_queue.dequeue()
    scales = [[256, 256], [272, 272], [288, 288], [304, 304], [320, 320], [352, 352], [336, 336]]
    random_scale_idx = tf.random_uniform([], minval=0, maxval=len(scales), dtype=tf.int32)
    scales_tensor = tf.convert_to_tensor(np.asarray(scales, np.int32), tf.int32)
    random_scale = scales_tensor[random_scale_idx]
    b_image = tf.image.resize_images(b_image, random_scale)
    b_mask_image = tf.image.resize_images(b_full_mask_image, random_scale)
    b_liver_mask = tf.image.resize_images(b_liver_mask, random_scale)
    b_pixel_cls_weight = tf.image.resize_images(b_pixel_cls_weight, random_scale)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        print('ok')
        # 使用start_queue_runners 启动队列填充
        threads = tf.train.start_queue_runners(sess, coord)
        count = 0
        for idx in range(100):
            b_image_v, b_full_mask_image_v, b_weak_mask_image_v, b_liver_mask_v, b_pixel_cls_weight_v = sess.run(
                [b_image, b_full_mask_image, b_weak_mask_image, b_liver_mask, b_pixel_cls_weight])
            print idx, np.shape(b_image_v), np.max(b_image_v), np.min(b_image_v), np.max(b_liver_mask_v), np.shape(
                b_pixel_cls_weight_v), np.max(b_pixel_cls_weight_v), np.min(b_pixel_cls_weight_v), np.unique(
                b_pixel_cls_weight_v), np.sum(b_full_mask_image_v)
            print np.shape(b_image_v), np.shape(b_mask_image)
            max_cls_weight = np.max(b_pixel_cls_weight_v)
            cv2.imwrite('./%d.png' % idx, np.asarray(b_image_v[0] * 255., np.uint8))
            cv2.imwrite('./%d_mask.png' % idx, np.asarray(b_full_mask_image_v[0] * 255., np.uint8))
            cv2.imwrite('./%d_weak_mask.png' % idx, np.asarray(b_weak_mask_image_v[0] * 255., np.uint8))
            cv2.imwrite('./%d_liver_mask.png' % idx, np.asarray(b_liver_mask_v[0] * 255., np.uint8))
            cv2.imwrite('./%d_pixel_weight.png' % idx,
                        np.asarray(b_pixel_cls_weight_v[0] / (1.0 * max_cls_weight) * 255., np.uint8))