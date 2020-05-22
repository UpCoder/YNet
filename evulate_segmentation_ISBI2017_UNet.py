# -*- coding=utf-8 -*-

import numpy as np
import math
import tensorflow as tf
from preprocessing import ssd_vgg_preprocessing, segmentation_preprocessing
import util
import cv2
from nets import UNetWL as UNet
from glob import glob
from datasets.medicalImage import fill_region, close_operation, open_operation, save_mhd_image
slim = tf.contrib.slim
import config
import os
import pickle
import pydensecrf.densecrf as dcrf
import gc
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
    from post_processing import cluster_postprocessing, net_center_posprocessing
    nii_dir = '/home/give/Documents/dataset/ISBI2017/Training_Batch_1'
    save_dir = '/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_1/DLSC_0/niis'
    # restore_path = '/home/give/PycharmProjects/weakly_label_segmentation/logs/ISBI2017_V2/1s_agumentation_weakly-upsampling-2/model.ckpt-168090'

    restore_path = '/home/give/PycharmProjects/weakly_label_segmentation/logs/ISBI2017_V2/UNet-upsampling-1/model.ckpt-137572'


    nii_parent_dir = os.path.dirname(nii_dir)
    with tf.name_scope('test'):
        image = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
        image_shape_placeholder = tf.placeholder(tf.int32, shape=[2])
        input_shape_placeholder = tf.placeholder(tf.int32, shape=[2])
        processed_image = segmentation_preprocessing.segmentation_preprocessing(image, None, None,
                                                                                out_shape=input_shape_placeholder,
                                                                                is_training=False)
        b_image = tf.expand_dims(processed_image, axis=0)

        net = UNet.UNet(b_image, None, None, is_training=False, decoder=FLAGS.decoder,
                        update_center_flag=FLAGS.update_center, batch_size=FLAGS.batch_size,
                        update_center_strategy=1,
                        num_centers_k=FLAGS.num_centers_k,
                        full_annotation_flag=False, output_shape_tensor=input_shape_placeholder)

        # net = UNet.UNet(b_image, None, None, is_training=False, decoder=FLAGS.decoder,
        #                         update_center_flag=FLAGS.update_center,
        #                         batch_size=2, init_center_value=None, update_center_strategy=1,
        #                         num_centers_k=FLAGS.num_centers_k, full_annotation_flag=False,
        #                         output_shape_tensor=input_shape_placeholder)

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
        global_pred_kmeans = []
        global_pred_centers = []
        global_pred_crf = []

        case_dices = []
        case_IoUs = []
        case_dices_kmeans = [] # kmeans
        case_IoUs_kmeans = []  # kmeans
        case_dices_centers = [] # net centers
        case_IoUs_centers = [] # net centers
        case_dices_crf = []
        case_IoUs_crf = []
        for iter, nii_path in enumerate(nii_pathes):
            # if os.path.basename(nii_path) not in ['volume-5.nii', 'volume-12.nii', 'volume-13.nii', 'volume-15.nii', 'volume-19.nii', 'volume-20.nii', 'volume-21.nii', 'volume-25.nii', 'volume-26.nii']:
            #     continue
            if os.path.basename(nii_path) in ['volume-15.nii', 'volume-25.nii']:
                continue
            # if os.path.basename(nii_path) != 'volume-0.nii':
            #     continue
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
            case_preds_kmeans = []
            case_preds_centers = []
            case_gts = []

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

                # pred = cv2.resize(pos_score, tuple(np.shape(image_data)[:2][::-1]), interpolation=cv2.INTER_NEAREST)
                if np.sum(whole_mask) != 0:
                    pred = np.asarray(pixel_cls_scores > 0.6, np.uint8)
                    # 开操作 先腐蚀，后膨胀
                    # 闭操作 先膨胀，后腐蚀
                    # pred = close_operation(pred, kernel_size=3)
                    pred = open_operation(pred, kernel_size=3)
                    pred = fill_region(pred)

                    # 再计算kmeans 的结果
                    pred_kmeans = np.asarray(image_expand(pred, kernel_size=5), np.uint8)
                    # pred_seg = image_expand(pred_seg, 5)
                    pred_kmeans = cluster_postprocessing(pred_kmeans, whole_mask, pixel_recover_feature, k=2)
                    pred_kmeans[image_data[:, :, 1] < (10./255.)] = 0
                    pred_kmeans = close_operation(pred_kmeans, kernel_size=5)
                    pred_kmeans = fill_region(pred_kmeans)

                    # 计算根据center得到的结果
                    # pixel_recover_feature, net_centers
                    # pred_centers = np.asarray(image_expand(pred, kernel_size=5), np.uint8)
                    pred_centers = np.asarray(pred, np.uint8)
                    pred_centers = net_center_posprocessing(pred_centers, centers=net_centers,
                                                            pixel_wise_feature=pixel_recover_feature, gt=whole_mask)
                    pred_centers[image_data[:, :, 1] < (10. / 255.)] = 0
                    pred_centers = close_operation(pred_centers, kernel_size=5)
                    pred_centers = fill_region(pred_centers)
                else:
                    pred = np.zeros_like(whole_mask)
                    pred_kmeans = np.zeros_like(whole_mask)
                    pred_centers = np.zeros_like(whole_mask)

                global_gt.append(whole_mask)
                case_gts.append(whole_mask)
                case_preds.append(pred)
                case_preds_kmeans.append(pred_kmeans)
                case_preds_centers.append(pred_centers)
                global_pred.append(pred)
                global_pred_kmeans.append(pred_kmeans)
                global_pred_centers.append(pred_centers)

                print '%d / %d: %s' % (slice_idx + 1, len(imgs), os.path.basename(nii_path)), np.shape(
                    pixel_cls_scores), np.max(
                    pixel_cls_scores), np.min(pixel_cls_scores), np.shape(pixel_recover_feature)
                del pixel_recover_feature, pixel_recover_feature_ms
                gc.collect()

            save_mhd_image(np.transpose(np.asarray(case_preds, np.uint8), axes=[0, 2, 1]),
                           os.path.join(save_dir, nii_path_basename.split('.')[0].split('-')[1], 'pred.mhd'))
            save_mhd_image(np.transpose(np.asarray(case_preds_kmeans, np.uint8), axes=[0, 2, 1]),
                           os.path.join(save_dir, nii_path_basename.split('.')[0].split('-')[1], 'pred_kmeans.mhd'))
            save_mhd_image(np.transpose(np.asarray(case_preds_centers, np.uint8), axes=[0, 2, 1]),
                           os.path.join(save_dir, nii_path_basename.split('.')[0].split('-')[1], 'pred_centers.mhd'))
            case_dice = dice(case_gts, case_preds)
            case_IoU = IoU(case_gts, case_preds)
            case_dice_kmeans = dice(case_gts, case_preds_kmeans)
            case_IoU_kmeans = IoU(case_gts, case_preds_kmeans)
            case_dice_centers = dice(case_gts, case_preds_centers)
            case_IoU_centers = IoU(case_gts, case_preds_centers)
            print('case dice: ', case_dice)
            print('case IoU: ', case_IoU)
            print('case dice kmeans: ', case_dice_kmeans)
            print('case IoU kmeans: ', case_IoU_kmeans)
            print('case dice centers: ', case_dice_centers)
            print('case IoU centers: ', case_IoU_centers)
            case_dices.append(case_dice)
            case_IoUs.append(case_IoU)
            case_dices_kmeans.append(case_dice_kmeans)
            case_IoUs_kmeans.append(case_IoU_kmeans)
            case_dices_centers.append(case_dice_centers)
            case_IoUs_centers.append(case_IoU_centers)

        print 'global dice is ', dice(global_gt, global_pred)
        print 'global IoU is ', IoU(global_gt, global_pred)

        print('mean of case dice is ', np.mean(case_dices))
        print('mean of case IoU is ', np.mean(case_IoUs))

        print 'global dice (kmeans) is ', dice(global_gt, global_pred_kmeans)
        print 'global IoU (kmeans) is ', IoU(global_gt, global_pred_kmeans)

        print 'mean of case dice (kmeans) is ', np.mean(case_dices_kmeans)
        print 'mean of case IoU (kmenas) is ', np.mean(case_IoUs_kmeans)

        print 'global dice (centers) is ', dice(global_gt, global_pred_centers)
        print 'global IoU (centers) is ', IoU(global_gt, global_pred_centers)

        print 'mean of case dice (centers) is ', np.mean(case_dices_centers)
        print 'mean of case IoU (centers) is ', np.mean(case_IoUs_centers)


def evulate_dir_nii():
    threshold = 0.6
    print('threshold = ', threshold)
    from metrics import dice, IoU
    from datasets.medicalImage import convertCase2PNGs
    nii_dir = '/home/give/Documents/dataset/ISBI2017/Training_Batch_1'

    save_dir = '/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_1/UNet/niis'
    restore_path = '/home/give/PycharmProjects/weakly_label_segmentation/logs/ISBI2017_V2/UNet-upsampling/model.ckpt-121234'

    nii_parent_dir = os.path.dirname(nii_dir)
    with tf.name_scope('test'):
        image = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
        image_shape_placeholder = tf.placeholder(tf.int32, shape=[2])
        input_shape_placeholder = tf.placeholder(tf.int32, shape=[2])
        processed_image = segmentation_preprocessing.segmentation_preprocessing(image, None, None,
                                                                                out_shape=input_shape_placeholder,
                                                                                is_training=False)
        b_image = tf.expand_dims(processed_image, axis=0)

        net = UNet.UNet(b_image, None, None, is_training=False, decoder=FLAGS.decoder,
                        update_center_flag=FLAGS.update_center,
                        batch_size=2, init_center_value=None, update_center_strategy=2,
                        num_centers_k=FLAGS.num_centers_k, full_annotation_flag=True,
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

        IoUs = []
        dices = []

        global_gt = []
        global_pred = []

        case_dices = []
        case_IoUs = []
        for iter, nii_path in enumerate(nii_pathes):
            # if os.path.basename(nii_path) in ['volume-5.nii', 'volume-12.nii', 'volume-13.nii', 'volume-15.nii', 'volume-19.nii', 'volume-20.nii', 'volume-21.nii', 'volume-25.nii', 'volume-26.nii']:
            #     continue
            if os.path.basename(nii_path) in ['volume-15.nii', 'volume-25.nii']:
                continue
            # if not nii_path.endswith('14.nii'):
            #     continue
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
            print(nii_path, seg_path)
            imgs, tumor_masks, liver_masks, tumor_weak_masks = convertCase2PNGs(nii_path, seg_path, save_dir=None)
            print(len(imgs), len(tumor_masks), len(liver_masks), len(tumor_masks))
            for slice_idx, (image_data, liver_mask, whole_mask) in enumerate(zip(imgs, liver_masks, tumor_masks)):
                pixel_cls_scores_ms = []
                for single_scale in scales:
                    pixel_cls_scores, b_image_v, global_step_v = sess.run(
                        [net.pixel_cls_scores, b_image, global_step],
                        feed_dict={
                            image: image_data,
                            image_shape_placeholder: np.shape(image_data)[:2],
                            input_shape_placeholder: single_scale
                        })
                    pixel_cls_scores_ms.append(
                        cv2.resize(pixel_cls_scores[0, :, :, 1], tuple(np.shape(image_data)[:2][::-1])))
                pixel_cls_scores = np.mean(pixel_cls_scores_ms, axis=0)
                print '%d / %d: %s' % (slice_idx + 1, len(imgs), os.path.basename(nii_path)), np.shape(
                    pixel_cls_scores), np.max(
                    pixel_cls_scores), np.min(pixel_cls_scores)

                # pred = cv2.resize(pos_score, tuple(np.shape(image_data)[:2][::-1]), interpolation=cv2.INTER_NEAREST)
                if np.sum(whole_mask) != 0:
                    pred = np.asarray(pixel_cls_scores > threshold, np.uint8)
                    # 开操作 先腐蚀，后膨胀
                    # pred = open_operation(pos_score, kernel_size=5)
                    pred = fill_region(pred)
                    IoUs.append(IoU(whole_mask, pred))
                    dices.append(dice(whole_mask, pred))
                else:
                    pred = np.zeros_like(whole_mask)
                global_gt.append(whole_mask)
                case_gts.append(whole_mask)
                case_preds.append(pred)
                global_pred.append(pred)

            save_mhd_image(np.transpose(np.asarray(case_preds, np.uint8), axes=[0, 2, 1]),
                           os.path.join(save_dir, nii_path_basename.split('.')[0].split('-')[1], 'pred.mhd'))
            case_dice = dice(case_gts, case_preds)
            case_IoU = IoU(case_gts, case_preds)
            print('case dice: ', case_dice)
            print('case IoU: ', case_IoU)
            case_dices.append(case_dice)
            case_IoUs.append(case_IoU)
        print 'mean of Dice is ', np.mean(dices)
        print 'mean of IoU is ', np.mean(IoUs)

        print 'global dice (seg) is ', dice(global_gt, global_pred)
        print 'global IoU (seg) is ', IoU(global_gt, global_pred)

        print('mean of case dice is ', np.mean(case_dices))
        print('mean of case IoU is ', np.mean(case_IoUs))


def dense_crf(feature_map, output_probs):
    h = output_probs.shape[0]
    w = output_probs.shape[1]

    output_probs = np.expand_dims(output_probs, 0)
    output_probs = np.append(1 - output_probs, output_probs, axis=0)

    d = dcrf.DenseCRF2D(w, h, 2)
    U = -np.log(output_probs)
    U = U.reshape((2, -1))
    U = np.ascontiguousarray(U)
    img = np.ascontiguousarray(feature_map)

    d.setUnaryEnergy(U)

    from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
    from sklearn.decomposition import PCA
    pca = PCA(n_components=16, whiten=True)
    # print(np.shape(img))
    pca_image = pca.fit_transform(np.reshape(img, [-1, 128]))
    img = np.reshape(pca_image, [512, 512, 16])

    pairwise_energy = create_pairwise_bilateral(sdims=(20, 20), schan=(3,), img=img, chdim=2)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseEnergy(pairwise_energy, compat=10)

    # d.addPairwiseBilateral(sxy=2, srgb=3, rgbim=img, compat=10)

    Q = d.inference(5)

    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

    return Q


def evulate_dir_nii_weakly_feature_crf():
    '''
    新版本的评估代码
    :return:
    '''

    from metrics import dice, IoU
    from datasets.medicalImage import convertCase2PNGs, image_expand
    nii_dir = '/home/give/Documents/dataset/ISBI2017/Training_Batch_1'
    save_dir = '/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_1/DLSC_0/niis'
    # restore_path = '/home/give/PycharmProjects/weakly_label_segmentation/logs/ISBI2017_V2/1s_agumentation_weakly-upsampling-2/model.ckpt-168090'

    restore_path = '/home/give/PycharmProjects/weakly_label_segmentation/logs/ISBI2017_V2/UNet-upsampling-1/model.ckpt-137572'

    nii_parent_dir = os.path.dirname(nii_dir)
    with tf.name_scope('test'):
        image = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
        image_shape_placeholder = tf.placeholder(tf.int32, shape=[2])
        input_shape_placeholder = tf.placeholder(tf.int32, shape=[2])
        processed_image = segmentation_preprocessing.segmentation_preprocessing(image, None, None,
                                                                                out_shape=input_shape_placeholder,
                                                                                is_training=False)
        b_image = tf.expand_dims(processed_image, axis=0)

        net = UNet.UNet(b_image, None, None, is_training=False, decoder=FLAGS.decoder,
                        update_center_flag=FLAGS.update_center,
                        batch_size=2, init_center_value=None, update_center_strategy=2,
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
        global_pred_crf = []

        case_dices = []
        case_IoUs = []
        case_dices_crf = []
        case_IoUs_crf = []
        for iter, nii_path in enumerate(nii_pathes):
            # if os.path.basename(nii_path) not in ['volume-5.nii', 'volume-12.nii', 'volume-13.nii', 'volume-15.nii', 'volume-19.nii', 'volume-20.nii', 'volume-21.nii', 'volume-25.nii', 'volume-26.nii']:
            #     continue
            if os.path.basename(nii_path) in ['volume-15.nii', 'volume-25.nii']:
                continue
            # if os.path.basename(nii_path) != 'volume-0.nii':
            #     continue
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
            case_preds_crf = []
            case_gts = []

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

                # pred = cv2.resize(pos_score, tuple(np.shape(image_data)[:2][::-1]), interpolation=cv2.INTER_NEAREST)
                if np.sum(whole_mask) != 0:
                    pred = np.asarray(pixel_cls_scores > 0.6, np.uint8)
                    # 开操作 先腐蚀，后膨胀
                    # 闭操作 先膨胀，后腐蚀
                    # pred = close_operation(pred, kernel_size=3)
                    pred = open_operation(pred, kernel_size=3)
                    pred = fill_region(pred)

                    # 再计算kmeans 的结果
                    pred_crf = dense_crf(pixel_recover_feature, pixel_cls_scores)
                    pred_crf = np.asarray(pred_crf, np.uint8)
                    if np.sum(pred_crf) != 0:
                        pred_crf = open_operation(pred_crf, kernel_size=3)
                        pred_crf = fill_region(pred_crf)


                else:
                    pred = np.zeros_like(whole_mask)
                    pred_crf = np.zeros_like(whole_mask)

                global_gt.append(whole_mask)
                case_gts.append(whole_mask)
                case_preds.append(pred)
                case_preds_crf.append(pred_crf)
                global_pred.append(pred)
                global_pred_crf.append(pred_crf)

                print '%d / %d: %s' % (slice_idx + 1, len(imgs), os.path.basename(nii_path)), np.shape(
                    pixel_cls_scores), np.max(
                    pixel_cls_scores), np.min(pixel_cls_scores), np.shape(pixel_recover_feature)
                del pixel_recover_feature, pixel_recover_feature_ms
                gc.collect()

            save_mhd_image(np.transpose(np.asarray(case_preds_crf, np.uint8), axes=[0, 2, 1]),
                           os.path.join(save_dir, nii_path_basename.split('.')[0].split('-')[1], 'pred_feature_crf.mhd'))
            case_dice = dice(case_gts, case_preds)
            case_IoU = IoU(case_gts, case_preds)
            case_dice_crf = dice(case_gts, case_preds_crf)
            case_IoU_crf = IoU(case_gts, case_preds_crf)
            print('case dice: ', case_dice)
            print('case IoU: ', case_IoU)
            print('case dice kmeans: ', case_dice_crf)
            print('case IoU kmeans: ', case_IoU_crf)
            case_dices.append(case_dice)
            case_IoUs.append(case_IoU)
            case_dices_crf.append(case_dice_crf)
            case_IoUs_crf.append(case_IoU_crf)

        print 'global dice is ', dice(global_gt, global_pred)
        print 'global IoU is ', IoU(global_gt, global_pred)

        print('mean of case dice is ', np.mean(case_dices))
        print('mean of case IoU is ', np.mean(case_IoUs))

        print 'global dice (crf) is ', dice(global_gt, global_pred_crf)
        print 'global IoU (crf) is ', IoU(global_gt, global_pred_crf)

        print 'mean of case dice (crf) is ', np.mean(case_dices_crf)
        print 'mean of case IoU (crf) is ', np.mean(case_IoUs_crf)


def evulate_dir_nii_weakly_crf():
    from metrics import dice, IoU
    from datasets.medicalImage import convertCase2PNGs, image_expand
    nii_dir = '/home/give/Documents/dataset/ISBI2017/Training_Batch_1'
    save_dir = '/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_1/DLSC_0/niis'
    restore_path = '/home/give/PycharmProjects/weakly_label_segmentation/logs/ISBI2017_V2/UNet-upsampling-1/model.ckpt-137572'

    nii_parent_dir = os.path.dirname(nii_dir)
    with tf.name_scope('test'):
        image = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
        image_shape_placeholder = tf.placeholder(tf.int32, shape=[2])
        input_shape_placeholder = tf.placeholder(tf.int32, shape=[2])
        processed_image = segmentation_preprocessing.segmentation_preprocessing(image, None, None,
                                                                                out_shape=input_shape_placeholder,
                                                                                is_training=False)
        b_image = tf.expand_dims(processed_image, axis=0)

        net = UNet.UNet(b_image, None, None, is_training=False, decoder=FLAGS.decoder,
                        update_center_flag=FLAGS.update_center,
                        batch_size=2, init_center_value=None, update_center_strategy=2,
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

        IoUs = []
        dices = []

        global_gt = []
        global_pred = []
        global_pred_crf = []

        case_dices = []
        case_IoUs = []
        case_dices_crf = []
        case_IoUs_crf = []
        for iter, nii_path in enumerate(nii_pathes):
            # if os.path.basename(nii_path) in ['volume-5.nii', 'volume-12.nii', 'volume-13.nii', 'volume-15.nii', 'volume-19.nii', 'volume-20.nii', 'volume-21.nii', 'volume-25.nii', 'volume-26.nii']:
            #     continue
            if os.path.basename(nii_path) in ['volume-15.nii', 'volume-25.nii']:
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
            case_preds_crf = []
            case_gts = []

            # case_recover_features = []
            print(nii_path, seg_path)
            imgs, tumor_masks, liver_masks, tumor_weak_masks = convertCase2PNGs(nii_path, seg_path, save_dir=None)
            print(len(imgs), len(tumor_masks), len(liver_masks), len(tumor_masks))
            for slice_idx, (image_data, liver_mask, whole_mask) in enumerate(zip(imgs, liver_masks, tumor_masks)):
                pixel_cls_scores_ms = []
                for single_scale in scales:
                    pixel_cls_scores, b_image_v, global_step_v = sess.run(
                        [net.pixel_cls_scores, b_image, global_step],
                        feed_dict={
                            image: image_data,
                            image_shape_placeholder: np.shape(image_data)[:2],
                            input_shape_placeholder: single_scale
                        })
                    pixel_cls_scores_ms.append(
                        cv2.resize(pixel_cls_scores[0, :, :, 1], tuple(np.shape(image_data)[:2][::-1])))
                pixel_cls_scores = np.mean(pixel_cls_scores_ms, axis=0)
                # case_recover_features.append(pixel_recover_feature)

                # pred = cv2.resize(pos_score, tuple(np.shape(image_data)[:2][::-1]), interpolation=cv2.INTER_NEAREST)
                if np.sum(whole_mask) != 0:
                    pred = np.asarray(pixel_cls_scores > 0.6, np.uint8)
                    # 开操作 先腐蚀，后膨胀
                    # 闭操作 先膨胀，后腐蚀
                    # pred = close_operation(pred, kernel_size=3)
                    pred = open_operation(pred, kernel_size=3)
                    pred = fill_region(pred)
                    IoUs.append(IoU(whole_mask, pred))
                    dices.append(dice(whole_mask, pred))

                    pred_crf = dense_crf(np.asarray(image_data * 255., np.uint8), pixel_cls_scores)
                    pred_crf = np.asarray(pred_crf, np.uint8)
                    if np.sum(pred_crf) != 0:
                        pred_crf = open_operation(pred_crf, kernel_size=3)
                        pred_crf = fill_region(pred_crf)


                else:
                    pred = np.zeros_like(whole_mask)
                    pred_crf = np.zeros_like(whole_mask)
                global_gt.append(whole_mask)
                case_gts.append(whole_mask)
                case_preds.append(pred)
                case_preds_crf.append(pred_crf)
                global_pred.append(pred)
                global_pred_crf.append(pred_crf)
                print '%d / %d: %s' % (slice_idx + 1, len(imgs), os.path.basename(nii_path)), np.shape(
                    pixel_cls_scores), np.max(
                    pixel_cls_scores), np.min(pixel_cls_scores)

            save_mhd_image(np.transpose(np.asarray(case_preds, np.uint8), axes=[0, 2, 1]),
                           os.path.join(save_dir, nii_path_basename.split('.')[0].split('-')[1], 'pred.mhd'))
            save_mhd_image(np.transpose(np.asarray(case_preds_crf, np.uint8), axes=[0, 2, 1]),
                           os.path.join(save_dir, nii_path_basename.split('.')[0].split('-')[1], 'pred_crf.mhd'))
            case_dice = dice(case_gts, case_preds)
            case_IoU = IoU(case_gts, case_preds)
            case_dice_crf = dice(case_gts, case_preds_crf)
            case_IoU_crf = IoU(case_gts, case_preds_crf)
            print('case dice: ', case_dice)
            print('case IoU: ', case_IoU)
            print('case dice crf: ', case_dice_crf)
            print('case IoU crf: ', case_IoU_crf)
            case_dices.append(case_dice)
            case_IoUs.append(case_IoU)
            case_dices_crf.append(case_dice_crf)
            case_IoUs_crf.append(case_IoU_crf)

        print 'global dice is ', dice(global_gt, global_pred)
        print 'global IoU is ', IoU(global_gt, global_pred)

        print('mean of case dice is ', np.mean(case_dices))
        print('mean of case IoU is ', np.mean(case_IoUs))

        print 'global dice (crf) is ', dice(global_gt, global_pred_crf)
        print 'global IoU (crf) is ', IoU(global_gt, global_pred_crf)

        print 'mean of case dice is ', np.mean(case_dices_crf)
        print 'mean of case IoU is ', np.mean(case_IoUs_crf)

def evulate_dir_nii_weakly_kmeans_pixel():
    '''
    新版本的评估代码
    :return:
    '''
    from metrics import dice, IoU
    from datasets.medicalImage import convertCase2PNGs, image_expand
    from post_processing import net_center_posprocessing, cluster_postprocessing
    nii_dir = '/home/give/Documents/dataset/ISBI2017/Training_Batch_1'
    save_dir = '/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_1/DLSC_0/niis'
    restore_path = '/home/give/PycharmProjects/weakly_label_segmentation/logs/ISBI2017_V2/UNet-upsampling-1/model.ckpt-137572'
    with tf.name_scope('test'):
        image = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
        image_shape_placeholder = tf.placeholder(tf.int32, shape=[2])
        input_shape_placeholder = tf.placeholder(tf.int32, shape=[2])
        processed_image = segmentation_preprocessing.segmentation_preprocessing(image, None, None,
                                                                                out_shape=input_shape_placeholder,
                                                                                is_training=False)
        b_image = tf.expand_dims(processed_image, axis=0)

        net = UNet.UNet(b_image, None, None, is_training=False, decoder=FLAGS.decoder,
                        update_center_flag=FLAGS.update_center,
                        batch_size=2, init_center_value=None, update_center_strategy=2,
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
        global_pred_kmeans = []
        global_pred_crf = []

        case_dices = []
        case_IoUs = []
        case_dices_kmeans = [] # kmeans
        case_IoUs_kmeans = []  # kmeans

        for iter, nii_path in enumerate(nii_pathes):
            if os.path.basename(nii_path) in ['volume-15.nii', 'volume-25.nii']:
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
            case_preds_kmeans = []
            case_gts = []

            # case_recover_features = []
            print(nii_path, seg_path)
            imgs, tumor_masks, liver_masks, tumor_weak_masks = convertCase2PNGs(nii_path, seg_path, save_dir=None)
            print(len(imgs), len(tumor_masks), len(liver_masks), len(tumor_masks))

            for slice_idx, (image_data, liver_mask, whole_mask) in enumerate(zip(imgs, liver_masks, tumor_masks)):
                pixel_cls_scores_ms = []
                for single_scale in scales:
                    pixel_cls_scores, b_image_v, global_step_v, net_centers = sess.run(
                        [net.pixel_cls_scores, b_image, global_step, net.centers],
                        feed_dict={
                            image: image_data,
                            image_shape_placeholder: np.shape(image_data)[:2],
                            input_shape_placeholder: single_scale
                        })
                    pixel_cls_scores_ms.append(
                        cv2.resize(pixel_cls_scores[0, :, :, 1], tuple(np.shape(image_data)[:2][::-1])))
                pixel_cls_scores = np.mean(pixel_cls_scores_ms, axis=0)
                if np.sum(whole_mask) != 0:
                    pred = np.asarray(pixel_cls_scores > 0.6, np.uint8)
                    # 开操作 先腐蚀，后膨胀
                    # 闭操作 先膨胀，后腐蚀
                    # pred = close_operation(pred, kernel_size=3)
                    pred = open_operation(pred, kernel_size=3)
                    pred = fill_region(pred)

                    # 再计算kmeans 的结果
                    pred_kmeans = np.asarray(image_expand(pred, kernel_size=5), np.uint8)
                    # pred_seg = image_expand(pred_seg, 5)
                    pred_kmeans = cluster_postprocessing(pred_kmeans, whole_mask, image_data, k=2)
                    pred_kmeans[image_data[:, :, 1] < (10./255.)] = 0
                    pred_kmeans = close_operation(pred_kmeans, kernel_size=5)
                    pred_kmeans = fill_region(pred_kmeans)


                else:
                    pred = np.zeros_like(whole_mask)
                    pred_kmeans = np.zeros_like(whole_mask)

                global_gt.append(whole_mask)
                case_gts.append(whole_mask)
                case_preds.append(pred)
                case_preds_kmeans.append(pred_kmeans)
                global_pred.append(pred)
                global_pred_kmeans.append(pred_kmeans)

                print '%d / %d: %s' % (slice_idx + 1, len(imgs), os.path.basename(nii_path)), np.shape(
                    pixel_cls_scores), np.max(
                    pixel_cls_scores), np.min(pixel_cls_scores)
                gc.collect()

            save_mhd_image(np.transpose(np.asarray(case_preds, np.uint8), axes=[0, 2, 1]),
                           os.path.join(save_dir, nii_path_basename.split('.')[0].split('-')[1], 'pred.mhd'))
            save_mhd_image(np.transpose(np.asarray(case_preds_kmeans, np.uint8), axes=[0, 2, 1]),
                           os.path.join(save_dir, nii_path_basename.split('.')[0].split('-')[1], 'pred_kmeans.mhd'))
            case_dice = dice(case_gts, case_preds)
            case_IoU = IoU(case_gts, case_preds)
            case_dice_kmeans = dice(case_gts, case_preds_kmeans)
            case_IoU_kmeans = IoU(case_gts, case_preds_kmeans)
            print('case dice: ', case_dice)
            print('case IoU: ', case_IoU)
            print('case dice kmeans: ', case_dice_kmeans)
            print('case IoU kmeans: ', case_IoU_kmeans)
            case_dices.append(case_dice)
            case_IoUs.append(case_IoU)
            case_dices_kmeans.append(case_dice_kmeans)
            case_IoUs_kmeans.append(case_IoU_kmeans)

        print 'global dice is ', dice(global_gt, global_pred)
        print 'global IoU is ', IoU(global_gt, global_pred)

        print('mean of case dice is ', np.mean(case_dices))
        print('mean of case IoU is ', np.mean(case_IoUs))

        print 'global dice (kmeans) is ', dice(global_gt, global_pred_kmeans)
        print 'global IoU (kmeans) is ', IoU(global_gt, global_pred_kmeans)

        print 'mean of case dice (kmeans) is ', np.mean(case_dices_kmeans)
        print 'mean of case IoU (kmenas) is ', np.mean(case_IoUs_kmeans)


def main(_):
    config_initialization()

    print('full annotation flag')
    # evulate_dir_nii()

    # evulate_dir_nii_weakly_new()
    # evulate_dir_nii_weakly_feature_crf()
    # evulate_dir_nii_weakly_crf()
    evulate_dir_nii_weakly_kmeans_pixel()


if __name__ == '__main__':
    tf.app.run()
# 544 / 828: volume-4.nii (1, 256, 256, 2) 0.0005035501 1.9400452e-09 (1, 256, 256, 1) 0.016835693 6.260295e-05 0.003921569 0.0 (1, 256, 256, 3)