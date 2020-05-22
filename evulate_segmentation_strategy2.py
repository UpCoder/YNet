# -*- coding=utf-8 -*-

import numpy as np
import math
import tensorflow as tf
from preprocessing import ssd_vgg_preprocessing, segmentation_preprocessing
import util
import cv2
from nets import UNetWL as UNet
from nets import UNetWLAttention as UNetAttention
from glob import glob


slim = tf.contrib.slim
import config
import os
# =========================================================================== #
# Checkpoint and running Flags
# =========================================================================== #
tf.app.flags.DEFINE_string('checkpoint_path', None, 
   'the path of pretrained model to be used. If there are checkpoints\
    in train_dir, this config will be ignored.')

tf.app.flags.DEFINE_float('gpu_memory_fraction', -1, 
  'the gpu memory fraction to be used. If less than 0, allow_growth = True is used.')


# =========================================================================== #
# I/O and preprocessing Flags.
# =========================================================================== #
tf.app.flags.DEFINE_integer(
    'num_readers', 1,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_bool('preprocessing_use_rotation', False, 
             'Whether to use rotation for data augmentation')

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'dataset_name', 'icdar2015', 'The name of the dataset to load.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')
tf.app.flags.DEFINE_string('dataset_dir', 
           util.io.get_absolute_path('~/dataset/ICDAR2015/Challenge4/ch4_test_images'), 
           'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer('eval_image_width', 256, 'Train image size')
tf.app.flags.DEFINE_integer('eval_image_height', 256, 'Train image size')
tf.app.flags.DEFINE_bool('using_moving_average', True, 
                         'Whether to use ExponentionalMovingAverage')
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999, 
                          'The decay rate of ExponentionalMovingAverage')
tf.app.flags.DEFINE_string('pred_dir', '', '')
tf.app.flags.DEFINE_string('pred_vis_dir', '', '')
tf.app.flags.DEFINE_string('decoder', 'upsampling', '')
tf.app.flags.DEFINE_string('pred_assign_label_path', '', '')
tf.app.flags.DEFINE_string('recovery_img_dir', '', '')
tf.app.flags.DEFINE_string('recovery_feature_map_dir', '', '')
tf.app.flags.DEFINE_bool('update_center', False, '')
tf.app.flags.DEFINE_integer('batch_size', None, 'The number of samples in each batch.')
tf.app.flags.DEFINE_bool('test_flag', False, '')
tf.app.flags.DEFINE_bool('attention_flag', False, '')
tf.app.flags.DEFINE_bool('center_block_flag', True, '')
tf.app.flags.DEFINE_integer('num_centers_k', 2, 'split the image into k^2 block to compute the center')
FLAGS = tf.app.flags.FLAGS


def config_initialization():
    # image shape and feature layers shape inference
    image_shape = (FLAGS.eval_image_height, FLAGS.eval_image_width)
    
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    
    tf.logging.set_verbosity(tf.logging.DEBUG)
    # config.load_config(FLAGS.checkpoint_path)
    # config.load_config(util.io.get_dir(FLAGS.checkpoint_path))
    # config.load_config('/home/give/PycharmProjects/ISBI_Detection')
    config.load_config('./')
    config.init_config(image_shape, 
                       batch_size = FLAGS.batch_size,
                       pixel_conf_threshold = 0.5,
                       link_conf_threshold = 0.1,
                       num_gpus = 1, 
                   )
    util.proc.set_proc_name('test_pixel_link_on'+ '_' + FLAGS.dataset_name)


def get_assign_label(centers, feature_map):
    '''
    获得assign的label，根据欧式距离
    :param centers:
    :param feature_map:
    :return:
    '''
    distance = map(lambda center: np.sum((feature_map-center) ** 2, axis=-1), centers)
    print np.max(feature_map), np.min(feature_map)

    distance = np.asarray(distance, np.float32)
    assign_label = np.argmax(distance, axis=0)
    return assign_label


def evulate_dir():
    from metrics import dice, IoU
    with tf.name_scope('test'):
        image = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])
        image_shape_placeholder = tf.placeholder(tf.int32, shape=[2])
        processed_image = segmentation_preprocessing.segmentation_preprocessing(image, None,
                                                                                out_shape=[FLAGS.eval_image_width,
                                                                                           FLAGS.eval_image_height],
                                                                                is_training=False)
        b_image = tf.expand_dims(processed_image, axis=0)
        print('the decoder is ', FLAGS.decoder)
        net = UNet.UNet(b_image, None, is_training=False, decoder=FLAGS.decoder, update_center_flag=FLAGS.update_center,
                        batch_size=1, init_center_value=None, update_center_strategy=2)
        # print slim.get_variables_to_restore()
        global_step = slim.get_or_create_global_step()

    sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    if FLAGS.gpu_memory_fraction < 0:
        sess_config.gpu_options.allow_growth = True
    elif FLAGS.gpu_memory_fraction > 0:
        sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction

    # Variables to restore: moving avg. or normal weights.
    if FLAGS.using_moving_average:
        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        variables_to_restore[global_step.op.name] = global_step
    else:
        variables_to_restore = slim.get_variables_to_restore()

    saver = tf.train.Saver()

    case_names = util.io.ls(FLAGS.dataset_dir)
    case_names.sort()

    checkpoint = FLAGS.checkpoint_path
    if not os.path.exists(FLAGS.pred_dir):
        os.mkdir(FLAGS.pred_dir)
    if not os.path.exists(FLAGS.pred_vis_dir):
        os.mkdir(FLAGS.pred_vis_dir)
    if not os.path.exists(FLAGS.recovery_img_dir):
        os.mkdir(FLAGS.recovery_img_dir)
    pixel_recovery_features = tf.image.resize_images(net.pixel_recovery_features, image_shape_placeholder)
    with tf.Session(config=sess_config) as sess:
        saver.restore(sess, checkpoint)
        centers = sess.run(net.centers)
        print('sum of centers is ', np.sum(centers))

        IoUs = []
        dices = []
        IoUs_det = []
        dices_det = []
        for iter, case_name in enumerate(case_names):
            image_data = util.img.imread(
                glob(util.io.join_path(FLAGS.dataset_dir, case_name, 'images', '*.png'))[0], rgb=True)
            mask = cv2.imread(glob(util.io.join_path(FLAGS.dataset_dir, case_name, 'weakly_label_whole_mask.png'))[0])[
                   :, :, 0]
            whole_mask = cv2.imread(glob(util.io.join_path(FLAGS.dataset_dir, case_name, 'whole_mask.png'))[0])[:, :, 0]

            pixel_cls_scores, recovery_img, recovery_feature_map, b_image_v, global_step_v = sess.run(
                [net.pixel_cls_scores, net.pixel_recovery_value, pixel_recovery_features, b_image, global_step],
                feed_dict={
                    image: image_data,
                    image_shape_placeholder: np.shape(image_data)[:2]
                })
            print global_step_v
            print '%d / %d: %s' % (iter + 1, len(case_names), case_name), np.shape(pixel_cls_scores), np.max(
                pixel_cls_scores[:, :, :, 1]), np.min(pixel_cls_scores[:, :, :, 1]), np.shape(
                recovery_img), np.max(recovery_img), np.min(recovery_img), np.max(b_image_v), np.min(
                b_image_v), np.shape(b_image_v)
            print np.shape(recovery_feature_map), np.shape(mask)
            pred_vis_path = util.io.join_path(FLAGS.pred_vis_dir, case_name + '.png')
            pred_path = util.io.join_path(FLAGS.pred_dir, case_name + '.png')
            pos_score = np.asarray(pixel_cls_scores > 0.5, np.uint8)[0, :, :, 1]
            pred = cv2.resize(pos_score, tuple(np.shape(image_data)[:2][::-1]), interpolation=cv2.INTER_NEAREST)
            IoUs_det.append(IoU(whole_mask, pred))
            dices_det.append(dice(whole_mask, pred))
            cv2.imwrite(pred_vis_path, np.asarray(pred * 200, np.uint8))
            cv2.imwrite(pred_path, np.asarray(pred, np.uint8))
            recovery_img_path = util.io.join_path(FLAGS.recovery_img_dir, case_name + '.png')
            cv2.imwrite(recovery_img_path, np.asarray(recovery_img[0] * 255, np.uint8))

            xs, ys = np.where(pred == 1)
            features = recovery_feature_map[0, xs, ys, :]
            neg_features = recovery_feature_map[0, pred == 0]
            pos_features = recovery_feature_map[0, pred == 1]
            pos_features = np.mean(pos_features, axis=0, keepdims=True)
            neg_features = np.mean(neg_features, axis=0, keepdims=True)
            centers = np.concatenate([pos_features, neg_features], axis=0)
            distance = map(lambda center: np.sum((features - center) ** 2, axis=-1), centers)
            assign_label = np.argmin(distance, axis=0)
            assign_label += 1
            assign_label[assign_label != 1] = 0
            seg_pred_mask = np.zeros_like(mask)
            seg_pred_mask[xs, ys] = assign_label
            IoUs.append(IoU(whole_mask, seg_pred_mask))
            dices.append(dice(whole_mask, seg_pred_mask))
            if not os.path.exists(os.path.join(FLAGS.pred_dir+'-seg')):
                os.mkdir(os.path.join(FLAGS.pred_dir+'-seg'))
            seg_save_path = os.path.join(FLAGS.pred_dir+'-seg', case_name + '.png')
            cv2.imwrite(seg_save_path, np.asarray(seg_pred_mask, np.uint8))

            if not os.path.exists(os.path.join(FLAGS.pred_vis_dir + '-seg')):
                os.mkdir(os.path.join(FLAGS.pred_vis_dir + '-seg'))
            seg_vis_save_path = os.path.join(FLAGS.pred_vis_dir + '-seg', case_name + '.png')
            cv2.imwrite(seg_vis_save_path, np.asarray(seg_pred_mask * 200, np.uint8))

            print case_name, IoUs[-1], dices[-1], IoUs_det[-1], dices_det[-1]
        print 'mean of Dice is ', np.mean(dices)
        print 'mean of IoU is ', np.mean(IoUs)

        print 'mean of Dice_det is ', np.mean(dices_det)
        print 'mean of IoU_det is ', np.mean(IoUs_det)


def test_dir():
    from metrics import dice, IoU
    with tf.name_scope('test'):
        image = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])
        image_shape_placeholder = tf.placeholder(tf.int32, shape=[2])
        processed_image = segmentation_preprocessing.segmentation_preprocessing(image, None, None,
                                                                                out_shape=[FLAGS.eval_image_width,
                                                                                           FLAGS.eval_image_height],
                                                                                is_training=False)
        b_image = tf.expand_dims(processed_image, axis=0)
        print('the decoder is ', FLAGS.decoder)
        if not FLAGS.attention_flag:
            net = UNet.UNet(b_image, None, None, is_training=False, decoder=FLAGS.decoder,
                            update_center_flag=FLAGS.update_center,
                            batch_size=1, init_center_value=None, pixel_cls_weight=None,
                            update_center_strategy=1, num_centers_k=2, full_annotation_flag=False,
                            output_shape_tensor=np.asarray([FLAGS.eval_image_width, FLAGS.eval_image_height]))
        else:
            net = UNetAttention.UNet(b_image, None, is_training=False, decoder=FLAGS.decoder,
                            update_center_flag=FLAGS.update_center,
                            batch_size=1, init_center_value=None, update_center_strategy=2)
        # print slim.get_variables_to_restore()
        global_step = slim.get_or_create_global_step()

    sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    if FLAGS.gpu_memory_fraction < 0:
        sess_config.gpu_options.allow_growth = True
    elif FLAGS.gpu_memory_fraction > 0:
        sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction

    # Variables to restore: moving avg. or normal weights.
    if FLAGS.using_moving_average:
        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        variables_to_restore[global_step.op.name] = global_step
    else:
        variables_to_restore = slim.get_variables_to_restore()

    saver = tf.train.Saver()

    case_names = util.io.ls(FLAGS.dataset_dir)
    case_names.sort()

    checkpoint = FLAGS.checkpoint_path
    if not os.path.exists(FLAGS.pred_dir):
        os.mkdir(FLAGS.pred_dir)
    if not os.path.exists(FLAGS.pred_vis_dir):
        os.mkdir(FLAGS.pred_vis_dir)
    if not os.path.exists(FLAGS.recovery_img_dir):
        os.mkdir(FLAGS.recovery_img_dir)
    pixel_recovery_features = tf.image.resize_images(net.pixel_recovery_features, image_shape_placeholder)
    with tf.Session(config=sess_config) as sess:
        saver.restore(sess, checkpoint)
        centers = sess.run(net.centers)
        print('sum of centers is ', np.sum(centers))

        IoUs = []
        dices = []
        IoUs_det = []
        dices_det = []
        for iter, case_name in enumerate(case_names):
            image_data = util.img.imread(
                glob(util.io.join_path(FLAGS.dataset_dir, case_name, 'images', '*.png'))[0], rgb=True)
            pixel_cls_scores, recovery_img, recovery_feature_map, b_image_v, global_step_v = sess.run(
                [net.pixel_cls_scores, net.pixel_recovery_value, pixel_recovery_features, b_image, global_step],
                feed_dict={
                    image: image_data,
                    image_shape_placeholder: np.shape(image_data)[:2]
                })
            print global_step_v
            print '%d / %d: %s' % (iter + 1, len(case_names), case_name), np.shape(pixel_cls_scores), np.max(
                pixel_cls_scores[:, :, :, 1]), np.min(pixel_cls_scores[:, :, :, 1]), np.shape(
                recovery_img), np.max(recovery_img), np.min(recovery_img), np.max(b_image_v), np.min(
                b_image_v), np.shape(b_image_v)
            print np.shape(recovery_feature_map)
            pred_vis_path = util.io.join_path(FLAGS.pred_vis_dir, case_name + '.png')
            pred_path = util.io.join_path(FLAGS.pred_dir, case_name + '.png')
            pos_score = np.asarray(pixel_cls_scores > 0.5, np.uint8)[0, :, :, 1]
            pred = cv2.resize(pos_score, tuple(np.shape(image_data)[:2][::-1]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(pred_vis_path, np.asarray(pred * 200, np.uint8))
            cv2.imwrite(pred_path, np.asarray(pred, np.uint8))
            recovery_img_path = util.io.join_path(FLAGS.recovery_img_dir, case_name + '.png')
            cv2.imwrite(recovery_img_path, np.asarray(recovery_img[0] * 255, np.uint8))

            xs, ys = np.where(pred == 1)
            features = recovery_feature_map[0, xs, ys, :]
            neg_features = recovery_feature_map[0, pred == 0]
            pos_features = recovery_feature_map[0, pred == 1]
            pos_features = np.mean(pos_features, axis=0, keepdims=True)
            neg_features = np.mean(neg_features, axis=0, keepdims=True)
            centers = np.concatenate([pos_features, neg_features], axis=0)
            distance = map(lambda center: np.sum((features - center) ** 2, axis=-1), centers)
            assign_label = np.argmin(distance, axis=0)
            assign_label += 1
            assign_label[assign_label != 1] = 0
            seg_pred_mask = np.zeros_like(image_data[:, :, 0])
            seg_pred_mask[xs, ys] = assign_label
            if not os.path.exists(os.path.join(FLAGS.pred_dir+'-seg')):
                os.mkdir(os.path.join(FLAGS.pred_dir+'-seg'))
            seg_save_path = os.path.join(FLAGS.pred_dir+'-seg', case_name + '.png')
            cv2.imwrite(seg_save_path, np.asarray(seg_pred_mask, np.uint8))

            if not os.path.exists(os.path.join(FLAGS.pred_vis_dir + '-seg')):
                os.mkdir(os.path.join(FLAGS.pred_vis_dir + '-seg'))
            seg_vis_save_path = os.path.join(FLAGS.pred_vis_dir + '-seg', case_name + '.png')
            cv2.imwrite(seg_vis_save_path, np.asarray(seg_pred_mask * 200, np.uint8))
            print case_name

def main(_):
    config_initialization()
    # test()
    if FLAGS.test_flag:
        test_dir()
    else:
        evulate_dir()
                
    
if __name__ == '__main__':
    tf.app.run()
