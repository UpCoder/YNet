# -*- coding=utf-8 -*-

import numpy as np
import math
import tensorflow as tf
from preprocessing import ssd_vgg_preprocessing, segmentation_preprocessing
import util
import cv2
from nets import UNetWL as UNet
from nets import UNetWLAttention as UNetAttention
from nets import UNetWLBlocks_ms as UNetBlocksMS
from glob import glob
from datasets.medicalImage import fill_region, close_operation, open_operation, get_kernel_filters, save_mhd_image
from skimage.measure import label
slim = tf.contrib.slim
import config
import os
import pickle
import pydensecrf.densecrf as dcrf
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
import gc
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# scales = [[256, 256], [272, 272], [288, 288], [304, 304], [320, 320], [352, 352], [336, 336]]
scales = [[512, 512]]
# scales = [[480, 480], [496, 496], [512, 512], [528, 528], [544, 544], [560, 560]]
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
tf.app.flags.DEFINE_bool('nii_flag', False, '')
tf.app.flags.DEFINE_integer('num_centers_k', 2, 'split the image into k^2 block to compute the center')
tf.app.flags.DEFINE_bool('nii_case_flag', False, '')
tf.app.flags.DEFINE_bool('full_annotation_flag', False, '')
tf.app.flags.DEFINE_bool('dense_connection_flag', True, '')
tf.app.flags.DEFINE_bool('learnable_connection_flag', True, '')
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


def evulate_dir_nii_weakly_new():
    '''
    新版本的评估代码
    :return:
    '''
    from metrics import dice, IoU
    from datasets.medicalImage import convertCase2PNGs, image_expand
    nii_dir = '/home/give/Documents/dataset/ISBI2017/Training_Batch_1'
    save_dir = '/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_1/DLSC_0/niis'
    # restore_path = '/home/give/PycharmProjects/weakly_label_segmentation/logs/ISBI2017_V2/1s_agumentation_weakly-upsampling-2/model.ckpt-168090'

    restore_path = '/home/give/PycharmProjects/weakly_label_segmentation/logs/ISBI2017_V2/1s_agumentation_weakly_V3-upsampling-1/model.ckpt-167824'

    nii_parent_dir = os.path.dirname(nii_dir)
    with tf.name_scope('test'):
        image = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
        image_shape_placeholder = tf.placeholder(tf.int32, shape=[2])
        input_shape_placeholder = tf.placeholder(tf.int32, shape=[2])
        processed_image = segmentation_preprocessing.segmentation_preprocessing(image, None, None,
                                                                                out_shape=input_shape_placeholder,
                                                                                is_training=False)
        b_image = tf.expand_dims(processed_image, axis=0)

        net = UNetBlocksMS.UNet(b_image, None, None, is_training=False, decoder=FLAGS.decoder,
                                update_center_flag=FLAGS.update_center,
                                batch_size=2, init_center_value=None, update_center_strategy=1,
                                num_centers_k=FLAGS.num_centers_k, full_annotation_flag=False,
                                output_shape_tensor=input_shape_placeholder)

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

    nii_pathes = glob(os.path.join(nii_dir, 'volume-*.nii'))

    checkpoint = restore_path
    # pixel_recovery_features = tf.image.resize_images(net.pixel_recovery_features, image_shape_placeholder)
    with tf.Session(config=sess_config) as sess:
        saver.restore(sess, checkpoint)

        global_gt = []
        global_pred = []
        global_pred_grabcut = []

        case_dices = []
        case_IoUs = []
        case_dices_grabcut = []
        case_IoUs_grabcut = []

        for iter, nii_path in enumerate(nii_pathes):
            # if os.path.basename(nii_path) in ['volume-15.nii', 'volume-25.nii']:
            #     continue
            if os.path.basename(nii_path) != 'volume-0.nii':
                continue
            nii_path_basename = os.path.basename(nii_path)

            pred_dir = os.path.join(save_dir, nii_path_basename.split('.')[0].split('-')[1], 'pred')
            pred_vis_dir = os.path.join(save_dir, nii_path_basename.split('.')[0].split('-')[1], 'pred_vis')
            recovery_img_dir = os.path.join(save_dir, nii_path_basename.split('.')[0].split('-')[1], 'recovery_img')
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            if not os.path.exists(pred_vis_dir):
                os.makedirs(pred_vis_dir)
            if not os.path.exists(recovery_img_dir):
                os.makedirs(recovery_img_dir)

            seg_path = os.path.join(nii_dir, 'segmentation-' + nii_path.split('.')[0].split('-')[1] + '.nii')

            case_preds = []
            case_gts = []
            case_preds_grabcut = []

            # case_recover_features = []
            print(nii_path, seg_path)
            imgs, tumor_masks, liver_masks, tumor_weak_masks = convertCase2PNGs(nii_path, seg_path, save_dir=None)
            print(len(imgs), len(tumor_masks), len(liver_masks), len(tumor_masks))

            for slice_idx, (image_data, liver_mask, whole_mask) in enumerate(zip(imgs, liver_masks, tumor_masks)):
                pixel_cls_scores_ms = []
                pixel_recover_feature_ms = []
                for single_scale in scales:
                    pixel_recover_feature, pixel_cls_scores, b_image_v, global_step_v, net_centers = sess.run(
                        [net.pixel_recovery_features, net.pixel_cls_scores, b_image, global_step, net.centers],
                        feed_dict={
                            image: image_data,
                            image_shape_placeholder: np.shape(image_data)[:2],
                            input_shape_placeholder: single_scale
                        })
                    pixel_cls_scores_ms.append(
                        cv2.resize(pixel_cls_scores[0, :, :, 1], tuple(np.shape(image_data)[:2][::-1])))
                    pixel_recover_feature_ms.append(pixel_recover_feature[0])
                    del pixel_recover_feature
                pixel_cls_scores = np.mean(pixel_cls_scores_ms, axis=0)
                pixel_recover_feature = np.mean(pixel_recover_feature_ms, axis=0)
                # case_recover_features.append(pixel_recover_feature)

                if np.sum(whole_mask) != 0:
                    pred = np.asarray(pixel_cls_scores > 0.6, np.uint8)
                    # 开操作 先腐蚀，后膨胀
                    # 闭操作 先膨胀，后腐蚀
                    # pred = close_operation(pred, kernel_size=3)
                    pred = open_operation(pred, kernel_size=3)
                    pred = fill_region(pred)
                    from grabcut import grabcut
                    pred_grabcut = np.asarray(image_expand(pred, kernel_size=5), np.uint8)
                    xs, ys = np.where(pred_grabcut == 1)
                    print(np.shape(pred_grabcut))
                    if len(xs) == 0:
                        pred_grabcut = np.zeros_like(whole_mask)
                        print(np.min(pred_grabcut), np.max(pred_grabcut), np.sum(pred_grabcut))
                    else:
                        min_xs = np.min(xs)
                        max_xs = np.max(xs)
                        min_ys = np.min(ys)
                        max_ys = np.max(ys)
                        pred_grabcut = grabcut(np.asarray(image_data * 255., np.uint8), [min_xs, min_ys, max_xs, max_ys])
                        pred_grabcut = np.asarray(pred_grabcut == 255, np.uint8)
                        print(np.unique(pred_grabcut))
                        cv2.imwrite('./tmp/%d_gt.png' % slice_idx, np.asarray(whole_mask * 200, np.uint8))
                        cv2.imwrite('./tmp/%d_pred.png' % slice_idx, np.asarray(pred * 200, np.uint8))
                        cv2.imwrite('./tmp/%d_pred_grabcut.png' % slice_idx, np.asarray(pred_grabcut * 200, np.uint8))

                else:
                    pred = np.zeros_like(whole_mask)
                    pred_grabcut = np.zeros_like(whole_mask)

                global_gt.append(whole_mask)
                case_gts.append(whole_mask)
                case_preds.append(pred)
                case_preds_grabcut.append(pred_grabcut)
                global_pred.append(pred)
                global_pred_grabcut.append(pred_grabcut)

                print '%d / %d: %s' % (slice_idx + 1, len(imgs), os.path.basename(nii_path)), np.shape(
                    pixel_cls_scores), np.max(
                    pixel_cls_scores), np.min(pixel_cls_scores), np.shape(pixel_recover_feature)
                del pixel_recover_feature, pixel_recover_feature_ms
                gc.collect()

            case_dice = dice(case_gts, case_preds)
            case_IoU = IoU(case_gts, case_preds)
            case_dice_grabcut = dice(case_gts, case_preds_grabcut)
            case_IoU_grabcut = IoU(case_gts, case_preds_grabcut)
            print('case dice: ', case_dice)
            print('case IoU: ', case_IoU)
            print('case dice grabcut: ', case_dice_grabcut)
            print('case IoU grabcut: ', case_IoU_grabcut)
            case_dices.append(case_dice)
            case_IoUs.append(case_IoU)
            case_dices_grabcut.append(case_dice_grabcut)
            case_IoUs_grabcut.append(case_IoU_grabcut)
        print 'global dice is ', dice(global_gt, global_pred)
        print 'global IoU is ', IoU(global_gt, global_pred)

        print('mean of case dice is ', np.mean(case_dices))
        print('mean of case IoU is ', np.mean(case_IoUs))

        print('mean of case dice (grabcut) is ', np.mean(case_dices_grabcut))
        print('mean of case IoU (grabcut) is ', np.mean(case_IoUs_grabcut))

        print('global dice (grabcut) is ', dice(global_gt, global_pred_grabcut))
        print('global IoU (grabcut) is ', IoU(global_gt, global_pred_grabcut))


def main(_):
    config_initialization()
    evulate_dir_nii_weakly_new()


if __name__ == '__main__':
    tf.app.run()