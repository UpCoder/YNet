# -*- coding=utf-8 -*-

import numpy as np
import math
import tensorflow as tf
from preprocessing import ssd_vgg_preprocessing, segmentation_preprocessing
import util
import cv2
from nets import UNetWL as UNet
from nets import UNetWLAttention as UNetAttention
from nets import UNetWLBlocks as UNetBlocks
from glob import glob
from datasets.medicalImage import fill_region, close_operation, open_operation, get_kernel_filters, save_mhd_image
from skimage.measure import label
slim = tf.contrib.slim
import config
import os
import pickle
import pydensecrf.densecrf as dcrf
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
    def dense_crf(img, output_probs):
        h = output_probs.shape[0]
        w = output_probs.shape[1]

        output_probs = np.expand_dims(output_probs, 0)
        output_probs = np.append(1 - output_probs, output_probs, axis=0)

        d = dcrf.DenseCRF2D(w, h, 2)
        U = -np.log(output_probs)
        U = U.reshape((2, -1))
        U = np.ascontiguousarray(U)
        img = np.ascontiguousarray(img)

        d.setUnaryEnergy(U)

        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=3, srgb=13, rgbim=img, compat=10)

        Q = d.inference(5)
        Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

        return Q
    with tf.name_scope('test'):
        image = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])
        image_shape_placeholder = tf.placeholder(tf.int32, shape=[2])
        processed_image = segmentation_preprocessing.segmentation_preprocessing(image, None, None,
                                                                                out_shape=[FLAGS.eval_image_width,
                                                                                           FLAGS.eval_image_height],
                                                                                is_training=False)
        b_image = tf.expand_dims(processed_image, axis=0)
        print('the decoder is ', FLAGS.decoder)
        if FLAGS.attention_flag:
            # 使用Attention册落
            net = UNetAttention.UNet(b_image, None, is_training=False, decoder=FLAGS.decoder,
                                     update_center_flag=FLAGS.update_center,
                                     batch_size=1, init_center_value=None, update_center_strategy=2)
        elif FLAGS.center_block_flag:
            print('using the center block flag')
            # 使用block策略
            net = UNetBlocks.UNet(b_image, None, None, is_training=False, decoder=FLAGS.decoder,
                                  update_center_flag=FLAGS.update_center,
                                  batch_size=3, init_center_value=None, update_center_strategy=2,
                                  num_centers_k=FLAGS.num_centers_k)
        else:
            net = UNet.UNet(b_image, None, is_training=False, decoder=FLAGS.decoder,
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

    img_dir = os.path.join(FLAGS.dataset_dir, 'img')
    weakly_label_dir = os.path.join(FLAGS.dataset_dir, 'weakly_label_gt')
    fully_label_dir = os.path.join(FLAGS.dataset_dir, 'mask_gt')
    liver_mask_dir = os.path.join(FLAGS.dataset_dir, 'liver_mask')
    case_names = os.listdir(img_dir)
    case_names = [case_name.split('.')[0] for case_name in case_names]

    checkpoint = FLAGS.checkpoint_path
    if not os.path.exists(FLAGS.pred_dir):
        os.mkdir(FLAGS.pred_dir)
    if not os.path.exists(FLAGS.pred_vis_dir):
        os.mkdir(FLAGS.pred_vis_dir)
    if not os.path.exists(FLAGS.recovery_img_dir):
        os.mkdir(FLAGS.recovery_img_dir)
    if not os.path.exists(FLAGS.recovery_feature_map_dir):
        os.mkdir(FLAGS.recovery_feature_map_dir)
    pixel_recovery_features = tf.image.resize_images(net.pixel_recovery_features, image_shape_placeholder)
    with tf.Session(config=sess_config) as sess:
        saver.restore(sess, checkpoint)
        centers = sess.run(net.centers)
        print('sum of centers is ', np.sum(centers))

        IoUs = []
        dices = []
        IoUs_det = []
        dices_det = []
        dices_crf = []
        IoUs_crf = []

        global_gt = []
        global_pred_det = []
        global_pred_seg = []
        global_pred_crf = []
        for iter, case_name in enumerate(case_names):
            # if case_name != '61-151':
            #      continue
            image_data = util.img.imread(
                glob(util.io.join_path(img_dir, case_name + '*.png'))[0], rgb=True)
            mask = cv2.imread(glob(util.io.join_path(weakly_label_dir, case_name + '*.png'))[0])[:, :, 0]
            whole_mask = cv2.imread(glob(util.io.join_path(fully_label_dir, case_name + '*.png'))[0])[
                   :, :, 0]
            liver_mask = cv2.imread(os.path.join(liver_mask_dir, case_name + '.png'))

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
            global_gt.append(whole_mask)
            global_pred_det.append(pred)

            cv2.imwrite(pred_vis_path, np.asarray(pred * 200, np.uint8))
            cv2.imwrite(pred_path, np.asarray(pred, np.uint8))
            recovery_img_path = util.io.join_path(FLAGS.recovery_img_dir, case_name + '.png')
            recovery_img = recovery_img[0]
            recovery_img = cv2.resize(recovery_img, tuple(np.shape(image_data)[:2]))
            recovery_img *= 255
            recovery_img = np.asarray(recovery_img, np.uint8)
            cv2.imwrite(recovery_img_path, recovery_img)

            xs, ys = np.where(pred == 1)
            centers_list = []
            pos_features = recovery_feature_map[0, xs, ys]
            pos_features = np.mean(pos_features, axis=0, keepdims=True)
            liver_coords = np.where(liver_mask == 1)
            liver_features = recovery_feature_map[0, liver_coords[0], liver_coords[1]]
            liver_features = np.mean(liver_features, axis=0, keepdims=True)
            blank_coords = np.where(image_data < 10)
            blank_features = recovery_feature_map[0, blank_coords[0], blank_coords[1]]
            blank_features = np.mean(blank_features, axis=0, keepdims=True)
            centers_list.append(pos_features)
            centers_list.append(liver_features)
            centers_list.append(blank_features)
            num_pos = 1

            # 取不同的联通分量作为不同的positive center
            # labeled_pos_mask = label(np.asarray(pred, np.uint8))
            # num_connect = np.max(labeled_pos_mask)
            # for idx in range(1, num_connect + 1):
            #     cur_pred = np.asarray(labeled_pos_mask == 1, np.uint8)
            #     pos_features = recovery_feature_map[0, cur_pred == 1]
            #     pos_features = np.mean(pos_features, axis=0, keepdims=True)
            #     centers_list.append(pos_features)
            # num_pos = len(centers_list)
            # neg_features = recovery_feature_map[0, pred == 0]
            # neg_features = np.mean(neg_features, axis=0, keepdims=True)
            # centers_list.append(neg_features)

            # 按块划分
            # _, h, w, _ = np.shape(recovery_feature_map)
            # cell_h = h / FLAGS.num_centers_k
            # cell_w = w / FLAGS.num_centers_k
            # for i in range(FLAGS.num_centers_k):
            #     for j in range(FLAGS.num_centers_k):
            #         cell_features = recovery_feature_map[0, cell_h * i:cell_h * (i + 1), cell_w * j:cell_w * (j + 1), :]
            #         cell_pred = pred[cell_h * i:cell_h * (i + 1), cell_w * j:cell_w * (j + 1)]
            #         cell_xs, cell_ys = np.where(cell_pred == 0)
            #         cell_features = cell_features[cell_xs, cell_ys, :]
            #         cell_features_center = np.mean(cell_features, axis=0, keepdims=True)
            #         centers_list.append(cell_features_center)


            # dilated 之后取不同的边划分
            # for i in [35]:
            #     kernel_size = i
            #     kernels = get_kernel_filters(kernel_size)
            #     kernal_names = ['0-whole', '1-left', '2-right', '3-top', '4-bottom']
            #
            #     for idx in range(1, num_connect + 1):
            #         whole_ring_mask = None
            #         cur_connect = np.asarray(labeled_pos_mask == idx, np.uint8)
            #         cur_connect = image_expand(cur_connect, kernel_size=7)
            #         cur_connect = np.asarray(cur_connect, np.float32)
            #         for kernel_idx, kernel in enumerate(kernels):
            #             dilated_pred = cv2.filter2D(cur_connect, -1, kernel)
            #             dilated_pred = np.asarray(dilated_pred, np.uint8)
            #             ring_area_mask = np.logical_and(np.asarray(dilated_pred, np.bool),
            #                                             np.logical_not(np.asarray(cur_connect, np.bool)))
            #             if whole_ring_mask is None:
            #                 whole_ring_mask = ring_area_mask
            #             else:
            #                 ring_area_mask = np.logical_and(np.logical_not(ring_area_mask), whole_ring_mask)
            #             # from datasets.medicalImage import show_image
            #             # show_image(np.asarray(ring_area_mask, np.uint8))
            #             cv2.imwrite(
            #                 os.path.join('./tmp', case_name + '-' + str(idx) + '-' + kernal_names[kernel_idx] + '.png'),
            #                 np.asarray(ring_area_mask * 200, np.uint8))
            #             ring_xs, ring_ys = np.where(ring_area_mask == 1)
            #             ring_reatures = recovery_feature_map[0, ring_xs, ring_ys, :]
            #             centers_list.append(np.mean(ring_reatures, axis=0, keepdims=True))

            centers = np.concatenate(centers_list, axis=0)
            features = recovery_feature_map[0, xs, ys, :]
            distance = map(lambda center: np.sum((features - center) ** 2, axis=-1), centers)
            multi_color_seg = np.zeros_like(mask)
            assign_label = np.argmin(distance, axis=0)
            assign_label += 1
            multi_color_seg[xs, ys] = assign_label
            assign_label[assign_label > num_pos] = 0
            seg_pred_mask = np.zeros_like(mask)
            seg_pred_mask[xs, ys] = assign_label

            # post-preprocessing
            # 膨胀+腐蚀 (闭操作)
            seg_pred_mask = close_operation(seg_pred_mask, kernel_size=11)
            # seg_pred_mask = open_operation(seg_pred_mask, kernel_size=11)
            # 填充空洞
            seg_pred_mask = fill_region(seg_pred_mask)
            IoUs.append(IoU(whole_mask, seg_pred_mask))
            dices.append(dice(whole_mask, seg_pred_mask))
            global_pred_seg.append(seg_pred_mask)
            if not os.path.exists(os.path.join(FLAGS.pred_dir+'-seg')):
                os.mkdir(os.path.join(FLAGS.pred_dir+'-seg'))
            seg_save_path = os.path.join(FLAGS.pred_dir+'-seg', case_name + '.png')
            cv2.imwrite(seg_save_path, np.asarray(seg_pred_mask, np.uint8))

            if not os.path.exists(os.path.join(FLAGS.pred_vis_dir + '-seg')):
                os.mkdir(os.path.join(FLAGS.pred_vis_dir + '-seg'))
            seg_vis_save_path = os.path.join(FLAGS.pred_vis_dir + '-seg', case_name + '.png')
            cv2.imwrite(seg_vis_save_path, np.asarray(seg_pred_mask * 200, np.uint8))
            seg_vis_save_path = os.path.join(FLAGS.pred_vis_dir + '-seg', case_name + '_assigned.png')
            cv2.imwrite(seg_vis_save_path, np.asarray(multi_color_seg * 70, np.uint8))


            # 提取每个联通分量的features
            labeled_pos_mask = label(np.asarray(pred, np.uint8))
            num_connect = np.max(labeled_pos_mask)
            features = []   # 存储不同的联通分量对应的特征 N，X，C, X是变量
            for idx in range(1, num_connect + 1):
                cur_pred = np.asarray(labeled_pos_mask == idx, np.uint8)
                cur_xs, cur_ys = np.where(cur_pred == 1)
                pos_features = recovery_feature_map[0, cur_xs, cur_ys]
                connect_obj = {}
                connect_obj['xs'] = cur_xs
                connect_obj['ys'] = cur_ys
                connect_obj['features'] = pos_features
                features.append(connect_obj)
            feature_save_path = os.path.join(FLAGS.recovery_feature_map_dir, case_name + '.pickle')
            with open(feature_save_path, 'wb') as f:
                pickle.dump(features, f)
            print('features is written at ', feature_save_path), np.shape(
                recovery_feature_map), len(features), num_connect

            if not os.path.exists(os.path.join(FLAGS.pred_dir+'-CRF')):
                os.mkdir(os.path.join(FLAGS.pred_dir+'-CRF'))

            seg_save_path = os.path.join(FLAGS.pred_dir+'-CRF', case_name + '.png')
            pixel_cls_scores = pixel_cls_scores[0, :, :, 1]
            pixel_cls_scores = cv2.resize(pixel_cls_scores, tuple(np.shape(image_data)[:2]))
            recovery_img = np.expand_dims(recovery_img, axis=2)
            recovery_img = np.concatenate([recovery_img, recovery_img, recovery_img], axis=-1)
            pred_crf = dense_crf(recovery_img, pixel_cls_scores)
            cv2.imwrite(seg_save_path, np.asarray(pred_crf, np.uint8))

            if not os.path.exists(os.path.join(FLAGS.pred_vis_dir + '-CRF')):
                os.mkdir(os.path.join(FLAGS.pred_vis_dir + '-CRF'))
            seg_vis_save_path = os.path.join(FLAGS.pred_vis_dir + '-CRF', case_name + '.png')
            cv2.imwrite(seg_vis_save_path, np.asarray(pred_crf * 200, np.uint8))
            dices_crf.append(dice(whole_mask, pred_crf))
            IoUs_crf.append(IoU(whole_mask, pred_crf))
            global_pred_crf.append(pred_crf)

            print case_name,  dices[-1], IoUs[-1], dices_crf[-1], IoUs_crf[-1], dices_det[-1], IoUs_det[-1]

        print 'mean of Dice is ', np.mean(dices)
        print 'mean of IoU is ', np.mean(IoUs)

        print 'mean of Dice_det is ', np.mean(dices_det)
        print 'mean of IoU_det is ', np.mean(IoUs_det)

        print 'mean of Dice_crf is ', np.mean(dices_crf)
        print 'mean of IoU_crf is ', np.mean(IoUs_crf)

        print 'global dice (seg) is ', dice(global_gt, global_pred_seg)
        print 'global IoU (seg) is ', IoU(global_gt, global_pred_seg)
        print 'global dice (det) is ', dice(global_gt, global_pred_det)
        print 'global IoU (det) is ', IoU(global_gt, global_pred_det)
        print 'global dice (CRF) is ', dice(global_gt, global_pred_crf)
        print 'global IoU (CRF) is ', IoU(global_gt, global_pred_crf)


def evulate_dir_full_annotation():
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

        net = UNetBlocks.UNet(b_image, None, None, is_training=False, decoder=FLAGS.decoder,
                              update_center_flag=FLAGS.update_center,
                              batch_size=3, init_center_value=None, update_center_strategy=2,
                              num_centers_k=FLAGS.num_centers_k, full_annotation_flag=True)
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

    img_dir = os.path.join(FLAGS.dataset_dir, 'img')
    weakly_label_dir = os.path.join(FLAGS.dataset_dir, 'weakly_label_gt')
    fully_label_dir = os.path.join(FLAGS.dataset_dir, 'mask_gt')
    liver_mask_dir = os.path.join(FLAGS.dataset_dir, 'liver_mask')
    case_names = os.listdir(img_dir)
    case_names = [case_name.split('.')[0] for case_name in case_names]

    checkpoint = FLAGS.checkpoint_path
    if not os.path.exists(FLAGS.pred_dir):
        os.mkdir(FLAGS.pred_dir)
    if not os.path.exists(FLAGS.pred_vis_dir):
        os.mkdir(FLAGS.pred_vis_dir)
    if not os.path.exists(FLAGS.recovery_img_dir):
        os.mkdir(FLAGS.recovery_img_dir)
    with tf.Session(config=sess_config) as sess:
        saver.restore(sess, checkpoint)

        IoUs_det = []
        dices_det = []

        global_gt = []
        global_pred_det = []
        for iter, case_name in enumerate(case_names):
            image_data = util.img.imread(
                glob(util.io.join_path(img_dir, case_name + '*.png'))[0], rgb=True)
            mask = cv2.imread(glob(util.io.join_path(weakly_label_dir, case_name + '*.png'))[0])[:, :, 0]
            whole_mask = cv2.imread(glob(util.io.join_path(fully_label_dir, case_name + '*.png'))[0])[
                   :, :, 0]
            liver_mask = cv2.imread(os.path.join(liver_mask_dir, case_name + '.png'))

            pixel_cls_scores, b_image_v, global_step_v = sess.run(
                [net.pixel_cls_scores, b_image, global_step],
                feed_dict={
                    image: image_data,
                    image_shape_placeholder: np.shape(image_data)[:2]
                })
            print '%d / %d: %s' % (iter + 1, len(case_names), case_name), np.shape(pixel_cls_scores), np.max(
                pixel_cls_scores[:, :, :, 1]), np.min(pixel_cls_scores[:, :, :, 1])

            pred_vis_path = util.io.join_path(FLAGS.pred_vis_dir, case_name + '.png')
            pred_path = util.io.join_path(FLAGS.pred_dir, case_name + '.png')
            pos_score = np.asarray(pixel_cls_scores > 0.5, np.uint8)[0, :, :, 1]
            pred = cv2.resize(pos_score, tuple(np.shape(image_data)[:2][::-1]), interpolation=cv2.INTER_NEAREST)
            # pred = close_operation(pred, kernel_size=11)
            # pred = fill_region(pred)
            IoUs_det.append(IoU(whole_mask, pred))
            dices_det.append(dice(whole_mask, pred))
            global_gt.append(whole_mask)
            global_pred_det.append(pred)

            cv2.imwrite(pred_vis_path, np.asarray(pred * 200, np.uint8))
            cv2.imwrite(pred_path, np.asarray(pred, np.uint8))

            print 'ok', case_name, IoUs_det[-1], dices_det[-1]


        print 'mean of Dice_det is ', np.mean(dices_det)
        print 'mean of IoU_det is ', np.mean(IoUs_det)

        print 'global dice (det) is ', dice(global_gt, global_pred_det)
        print 'global IoU (det) is ', IoU(global_gt, global_pred_det)


def evulate_dir_nii():
    from metrics import dice, IoU
    from datasets.medicalImage import convertCase2PNGs
    nii_dir = '/home/ld/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_1/nii/Training_Batch_1'
    nii_parent_dir = os.path.dirname(nii_dir)
    with tf.name_scope('test'):
        image = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])
        image_shape_placeholder = tf.placeholder(tf.int32, shape=[2])
        processed_image = segmentation_preprocessing.segmentation_preprocessing(image, None,
                                                                                out_shape=[FLAGS.eval_image_width,
                                                                                           FLAGS.eval_image_height],
                                                                                is_training=False)
        b_image = tf.expand_dims(processed_image, axis=0)
        print('the decoder is ', FLAGS.decoder)
        if FLAGS.attention_flag:
            # 使用Attention册落
            net = UNetAttention.UNet(b_image, None, is_training=False, decoder=FLAGS.decoder,
                                     update_center_flag=FLAGS.update_center,
                                     batch_size=1, init_center_value=None, update_center_strategy=2)
        elif FLAGS.center_block_flag:
            print('using the center block flag')
            # 使用block策略
            net = UNetBlocks.UNet(b_image, None, None, is_training=False, decoder=FLAGS.decoder,
                                  update_center_flag=FLAGS.update_center,
                                  batch_size=3, init_center_value=None, update_center_strategy=2,
                                  num_centers_k=FLAGS.num_centers_k)
        else:
            net = UNet.UNet(b_image, None, is_training=False, decoder=FLAGS.decoder,
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

    nii_pathes = glob(os.path.join(nii_dir, 'volume-*.nii'))

    checkpoint = FLAGS.checkpoint_path
    pred_dir = os.path.join(nii_parent_dir, 'pred')
    pred_vis_dir = os.path.join(nii_parent_dir, 'pred_vis')
    recovery_img_dir = os.path.join(nii_parent_dir, 'recovery_img')
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    if not os.path.exists(pred_vis_dir):
        os.mkdir(pred_vis_dir)
    if not os.path.exists(recovery_img_dir):
        os.mkdir(recovery_img_dir)
    pixel_recovery_features = tf.image.resize_images(net.pixel_recovery_features, image_shape_placeholder)
    with tf.Session(config=sess_config) as sess:
        saver.restore(sess, checkpoint)
        centers = sess.run(net.centers)
        print('sum of centers is ', np.sum(centers))

        IoUs = []
        dices = []
        IoUs_det = []
        dices_det = []

        global_gt = []
        global_pred_det = []
        global_pred_seg = []

        case_dices = []
        case_dice_dets = []
        case_IoUs = []
        case_IoU_dets = []
        for iter, nii_path in enumerate(nii_pathes):
            seg_path = os.path.join(nii_dir, 'segmentation-' + nii_path.split('.')[0].split('-')[1] + '.nii')
            print(nii_path, seg_path)
            preds_case = []
            preds_det_case = []
            gts_case = []
            imgs, idxs, tumor_masks, liver_masks = convertCase2PNGs(nii_path, seg_path, save_dir=None)
            for slice_idx, (image_data, liver_mask, whole_mask) in enumerate(zip(imgs, liver_masks, tumor_masks)):
                pixel_cls_scores, recovery_img, recovery_feature_map, b_image_v, global_step_v = sess.run(
                    [net.pixel_cls_scores, net.pixel_recovery_value, pixel_recovery_features, b_image, global_step],
                    feed_dict={
                        image: image_data,
                        image_shape_placeholder: np.shape(image_data)[:2]
                    })
                print global_step_v
                print '%d / %d: %s' % (slice_idx + 1, len(imgs), os.path.basename(nii_path)), np.shape(
                    pixel_cls_scores), np.max(
                    pixel_cls_scores[:, :, :, 1]), np.min(pixel_cls_scores[:, :, :, 1]), np.shape(
                    recovery_img), np.max(recovery_img), np.min(recovery_img), np.max(b_image_v), np.min(
                    b_image_v), np.shape(b_image_v), np.max(image_data), np.min(image_data)
                print np.shape(recovery_feature_map)

                pos_score = np.asarray(pixel_cls_scores > 0.5, np.uint8)[0, :, :, 1]
                pred = cv2.resize(pos_score, tuple(np.shape(image_data)[:2][::-1]), interpolation=cv2.INTER_NEAREST)
                IoUs_det.append(IoU(whole_mask, pred))
                dices_det.append(dice(whole_mask, pred))
                global_gt.append(whole_mask)
                global_pred_det.append(pred)
                preds_det_case.append(pred)
                gts_case.append(whole_mask)

                xs, ys = np.where(pred == 1)
                centers_list = []
                pos_features = recovery_feature_map[0, xs, ys]
                pos_features = np.mean(pos_features, axis=0, keepdims=True)
                liver_coords = np.where(liver_mask == 1)
                liver_features = recovery_feature_map[0, liver_coords[0], liver_coords[1]]
                liver_features = np.mean(liver_features, axis=0, keepdims=True)
                blank_coords = np.where(image_data < 10)
                blank_features = recovery_feature_map[0, blank_coords[0], blank_coords[1]]
                blank_features = np.mean(blank_features, axis=0, keepdims=True)
                centers_list.append(pos_features)
                centers_list.append(liver_features)
                centers_list.append(blank_features)
                num_pos = 1

                centers = np.concatenate(centers_list, axis=0)
                features = recovery_feature_map[0, xs, ys, :]
                distance = map(lambda center: np.sum((features - center) ** 2, axis=-1), centers)
                assign_label = np.argmin(distance, axis=0)
                assign_label += 1
                assign_label[assign_label > num_pos] = 0
                seg_pred_mask = np.zeros_like(whole_mask)
                seg_pred_mask[xs, ys] = assign_label

                # post-preprocessing
                # 膨胀+腐蚀 (闭操作)
                seg_pred_mask = close_operation(seg_pred_mask, kernel_size=11)
                # seg_pred_mask = open_operation(seg_pred_mask, kernel_size=11)
                # 填充空洞
                seg_pred_mask = fill_region(seg_pred_mask)
                preds_case.append(seg_pred_mask)
                IoUs.append(IoU(whole_mask, seg_pred_mask))
                dices.append(dice(whole_mask, seg_pred_mask))
                global_pred_seg.append(seg_pred_mask)
                if not os.path.exists(os.path.join(FLAGS.pred_dir+'-seg')):
                    os.mkdir(os.path.join(FLAGS.pred_dir+'-seg'))

                if not os.path.exists(os.path.join(FLAGS.pred_vis_dir + '-seg')):
                    os.mkdir(os.path.join(FLAGS.pred_vis_dir + '-seg'))
                print os.path.basename(nii_path), slice_idx, IoUs[-1], dices[-1], IoUs_det[-1], dices_det[-1]
            case_dice_det = dice(gts_case, preds_det_case)
            case_IoU_det = IoU(gts_case, preds_det_case)
            case_dice = dice(gts_case, preds_case)
            case_IoU = IoU(gts_case, preds_case)
            print('case dice_det: ', case_dice_det)
            print('case IoU_det: ', case_IoU_det)
            print('case dice_det: ', case_dice)
            print('case IoU_det: ', case_IoU)
            case_dices.append(case_dice)
            case_IoUs.append(case_IoU)
            case_dice_dets.append(case_dice_det)
            case_IoU_dets.append(case_IoU_det)
        print 'mean of Dice is ', np.mean(dices)
        print 'mean of IoU is ', np.mean(IoUs)

        print 'mean of Dice_det is ', np.mean(dices_det)
        print 'mean of IoU_det is ', np.mean(IoUs_det)

        print 'global dice (seg) is ', dice(global_gt, global_pred_seg)
        print 'global IoU (seg) is ', IoU(global_gt, global_pred_det)
        print 'global dice (det) is ', dice(global_gt, global_pred_det)
        print 'global IoU (det) is ', IoU(global_gt, global_pred_det)

        print('mean of case dice is ', np.mean(case_dices))
        print('mean of case IoU is ', np.mean(case_IoUs))
        print('mean of case dice_det is ', np.mean(case_dice_dets))
        print('mean of case IoU_dets is ', np.mean(case_IoU_dets))


def evulate_nii_case(case_name=None, save_dir=None):
    from metrics import dice, IoU
    from datasets.medicalImage import convertCase2PNGs
    if case_name is None:
        case_name = '/media/give/CBMIR/ld/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training_Batch_2/volume-45.nii'
        seg_case_name = '/media/give/CBMIR/ld/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training_Batch_2/segmentation-45.nii'
    else:
        seg_case_name = os.path.join(os.path.dirname(case_name),
                                     'segmentation-' + os.path.basename(case_name).split('-')[1])
    save_dir = os.path.join(save_dir, os.path.basename(case_name).split('.')[0])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    case_base_name = os.path.basename(case_name).split('-')[1].split('.')[0]
    with tf.name_scope('test'):
        image = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])
        image_shape_placeholder = tf.placeholder(tf.int32, shape=[2])
        processed_image = segmentation_preprocessing.segmentation_preprocessing(image, None, None,
                                                                                out_shape=[FLAGS.eval_image_width,
                                                                                           FLAGS.eval_image_height],
                                                                                is_training=False)
        b_image = tf.expand_dims(processed_image, axis=0)
        print('the decoder is ', FLAGS.decoder)
        if FLAGS.attention_flag:
            # 使用Attention册落
            net = UNetAttention.UNet(b_image, None, is_training=False, decoder=FLAGS.decoder,
                                     update_center_flag=FLAGS.update_center,
                                     batch_size=1, init_center_value=None, update_center_strategy=2)
        elif FLAGS.center_block_flag:
            print('using the center block flag')
            # 使用block策略
            net = UNetBlocks.UNet(b_image, None, None, is_training=False, decoder=FLAGS.decoder,
                                  update_center_flag=FLAGS.update_center,
                                  batch_size=3, init_center_value=None, update_center_strategy=2,
                                  num_centers_k=FLAGS.num_centers_k)
        else:
            net = UNet.UNet(b_image, None, is_training=False, decoder=FLAGS.decoder,
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

    checkpoint = FLAGS.checkpoint_path
    # pred_dir = os.path.join(save_dir, 'pred')
    # pred_vis_dir = os.path.join(save_dir, 'pred_vis')
    # recovery_img_dir = os.path.join(save_dir, 'recovery_img')
    # if not os.path.exists(pred_dir):
    #     os.mkdir(pred_dir)
    # if not os.path.exists(pred_vis_dir):
    #     os.mkdir(pred_vis_dir)
    # if not os.path.exists(recovery_img_dir):
    #     os.mkdir(recovery_img_dir)
    pixel_recovery_features = tf.image.resize_images(net.pixel_recovery_features, image_shape_placeholder)
    with tf.Session(config=sess_config) as sess:
        saver.restore(sess, checkpoint)
        centers = sess.run(net.centers)
        print('sum of centers is ', np.sum(centers))

        IoUs = []
        dices = []
        IoUs_det = []
        dices_det = []

        global_gt = []
        global_pred_det = []
        global_pred_seg = []

        print(case_name, seg_case_name)
        preds_case = []
        preds_det_case = []
        gts_case = []
        imgs, idxs, tumor_masks, liver_masks = convertCase2PNGs(case_name, seg_case_name, save_dir=None)
        for slice_idx, (image_data, liver_mask, whole_mask) in enumerate(zip(imgs, liver_masks, tumor_masks)):
            pixel_cls_scores, recovery_img, recovery_feature_map, b_image_v, global_step_v = sess.run(
                [net.pixel_cls_scores, net.pixel_recovery_value, pixel_recovery_features, b_image, global_step],
                feed_dict={
                    image: image_data,
                    image_shape_placeholder: np.shape(image_data)[:2]
                })
            print global_step_v
            print '%d / %d: %s' % (slice_idx + 1, len(imgs), os.path.basename(case_name)), np.shape(
                pixel_cls_scores), np.max(
                pixel_cls_scores[:, :, :, 1]), np.min(pixel_cls_scores[:, :, :, 1]), np.shape(
                recovery_img), np.max(recovery_img), np.min(recovery_img), np.max(b_image_v), np.min(
                b_image_v), np.shape(b_image_v), np.max(image_data), np.min(image_data)
            print np.shape(recovery_feature_map)

            pos_score = np.asarray(pixel_cls_scores > 0.5, np.uint8)[0, :, :, 1]
            pred = cv2.resize(pos_score, tuple(np.shape(image_data)[:2][::-1]), interpolation=cv2.INTER_NEAREST)
            IoUs_det.append(IoU(whole_mask, pred))
            dices_det.append(dice(whole_mask, pred))
            global_gt.append(whole_mask)
            global_pred_det.append(pred)
            preds_det_case.append(pred)
            gts_case.append(whole_mask)

            xs, ys = np.where(pred == 1)
            centers_list = []
            pos_features = recovery_feature_map[0, xs, ys]
            pos_features = np.mean(pos_features, axis=0, keepdims=True)
            liver_coords = np.where(liver_mask == 1)
            liver_features = recovery_feature_map[0, liver_coords[0], liver_coords[1]]
            liver_features = np.mean(liver_features, axis=0, keepdims=True)
            blank_coords = np.where(image_data < 10)
            blank_features = recovery_feature_map[0, blank_coords[0], blank_coords[1]]
            blank_features = np.mean(blank_features, axis=0, keepdims=True)
            centers_list.append(pos_features)
            centers_list.append(liver_features)
            centers_list.append(blank_features)
            num_pos = 1

            centers = np.concatenate(centers_list, axis=0)
            features = recovery_feature_map[0, xs, ys, :]
            distance = map(lambda center: np.sum((features - center) ** 2, axis=-1), centers)
            assign_label = np.argmin(distance, axis=0)
            assign_label += 1
            assign_label[assign_label > num_pos] = 0
            seg_pred_mask = np.zeros_like(whole_mask)
            seg_pred_mask[xs, ys] = assign_label

            # post-preprocessing
            # seg_pred_mask[image_data[:, :, 1] < 110] = 0
            # 膨胀+腐蚀 (闭操作)
            seg_pred_mask = close_operation(seg_pred_mask, kernel_size=11)
            # seg_pred_mask = open_operation(seg_pred_mask, kernel_size=11)
            # 填充空洞
            seg_pred_mask = fill_region(seg_pred_mask)
            preds_case.append(seg_pred_mask)
            IoUs.append(IoU(whole_mask, seg_pred_mask))
            dices.append(dice(whole_mask, seg_pred_mask))
            global_pred_seg.append(seg_pred_mask)
            if not os.path.exists(os.path.join(FLAGS.pred_dir + '-seg')):
                os.mkdir(os.path.join(FLAGS.pred_dir + '-seg'))

            if not os.path.exists(os.path.join(FLAGS.pred_vis_dir + '-seg')):
                os.mkdir(os.path.join(FLAGS.pred_vis_dir + '-seg'))
            print os.path.basename(case_name), slice_idx, IoUs[-1], dices[-1], IoUs_det[-1], dices_det[-1]
        case_dice_det = dice(gts_case, preds_det_case)
        case_IoU_det = IoU(gts_case, preds_det_case)
        case_dice = dice(gts_case, preds_case)
        case_IoU = IoU(gts_case, preds_case)
        print('case dice_det: ', case_dice_det)
        print('case IoU_det: ', case_IoU_det)
        print('case dice_det: ', case_dice)
        print('case IoU_det: ', case_IoU)

        print 'mean of Dice is ', np.mean(dices)
        print 'mean of IoU is ', np.mean(IoUs)

        print 'mean of Dice_det is ', np.mean(dices_det)
        print 'mean of IoU_det is ', np.mean(IoUs_det)

        print 'global dice (seg) is ', dice(global_gt, global_pred_seg)
        print 'global IoU (seg) is ', IoU(global_gt, global_pred_det)
        print 'global dice (det) is ', dice(global_gt, global_pred_det)
        print 'global IoU (det) is ', IoU(global_gt, global_pred_det)
        save_mhd_image(np.transpose(np.asarray(preds_det_case, np.uint8), axes=[0, 2, 1]),
                       os.path.join(save_dir, case_base_name + '_det.mhd'))
        save_mhd_image(np.transpose(np.asarray(preds_case, np.uint8), axes=[0, 2, 1]),
                       os.path.join(save_dir, case_base_name + '_seg.mhd'))



def test_dir():
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
        if not FLAGS.attention_flag:
            net = UNet.UNet(b_image, None, is_training=False, decoder=FLAGS.decoder,
                            update_center_flag=FLAGS.update_center,
                            batch_size=1, init_center_value=None, update_center_strategy=2)
        elif FLAGS.center_block_flag:
            # 使用block策略
            net = UNetBlocks.UNet(b_image, None, is_training=False, decoder=FLAGS.decoder,
                                  update_center_flag=FLAGS.update_center,
                                  batch_size=1, init_center_value=None, update_center_strategy=2,
                                  num_centers_k=FLAGS.num_centers_k)
        else:
            # 使用Attention册落
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


def evulate_from_restore(pred_seg_dir, pred_det_dir, gt_dir):
    def _compute_mean_metrics(gts, masks, threshold=0.5):
        save_dir = os.path.join(os.getcwd(), 'tmp')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        IoUs = []
        dices = []
        idx = 0
        for gt, mask in zip(gts, masks):
            intersection = np.sum(np.logical_and(gt != 0, mask != 0))
            union = np.sum(gt != 0) + np.sum(mask != 0)
            if union == 0:
                IoUs.append(100.0)
                dices.append(100.0)
                continue
            IoU = (1.0 * intersection) / (1.0 * union - 1.0 * intersection) * 100.
            dice = (2.0 * intersection) / (1.0 * union) * 100
            IoUs.append(IoU)
            dices.append(dice)
            idx += 1
        return np.asarray(dices, np.float32), np.asarray(IoUs, np.float32)

    def _compute_global_metrics(gts, masks):
        intersection = np.sum(np.logical_and(gts != 0, masks != 0))
        union = np.sum(gts != 0) + np.sum(masks != 0)
        global_dice = (2.0 * intersection) / (1.0 * union) * 100
        global_IoU = (1.0 * intersection) / (1.0 * union - 1.0 * intersection) * 100
        return global_dice, global_IoU

    case_names = os.listdir(gt_dir)
    gt_masks = []
    pred_seg_masks = []
    pred_det_masks = []
    from tqdm import tqdm
    for case_name in tqdm(case_names):
        gt_mask = os.path.join(gt_dir, case_name)
        gt_mask = cv2.imread(gt_mask)[:, :, 0]
        pred_seg_mask = os.path.join(pred_seg_dir, case_name)
        pred_seg_mask = cv2.imread(pred_seg_mask)[:, :, 0]
        pred_det_mask = os.path.join(pred_det_dir, case_name)
        pred_det_mask = cv2.imread(pred_det_mask)[:, :, 0]

        pred_seg_masks.append(pred_seg_mask)
        pred_det_masks.append(pred_det_mask)
        gt_masks.append(gt_mask)
    gt_masks = np.asarray(gt_masks, np.uint8)
    pred_det_masks = np.asarray(pred_det_masks, np.uint8)
    pred_seg_masks = np.asarray(pred_seg_masks, np.uint8)
    mDice, mIoU = _compute_mean_metrics(gt_masks, pred_seg_masks)
    mDice_det, mIoU_det = _compute_mean_metrics(gt_masks, pred_det_masks)
    print('Mean of Metrics')
    print('dice: %.4f, IoU: %.4f' % (np.mean(mDice), np.mean(mIoU)))
    print('dice_det: %.4f, IoU_det: %.4f' % (np.mean(mDice_det), np.mean(mIoU_det)))
    gDice, gIoU = _compute_global_metrics(gt_masks, pred_seg_masks)
    gDice_det, gIoU_det = _compute_global_metrics(gt_masks, pred_det_masks)
    print('Global of Metrics')
    print('dice: %.4f, IoU: %.4f' % (gDice, gIoU))
    print('dice_det: %.4f, IoU_det: %.4f' % (gDice_det, gIoU_det))


def main(_):
    config_initialization()
    evulate_restore_flag = False
    nii_case_flag = False
    # test()
    if FLAGS.test_flag:
        test_dir()
    elif FLAGS.nii_flag:
        evulate_dir_nii()
    elif evulate_restore_flag:
        gt_dir = '/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_1/mask_gt'
        pred_seg_dir = '/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_1/DLSC_0/pred-seg'
        pred_det_dir = '/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_1/DLSC_0/pred'
        evulate_from_restore(pred_seg_dir, pred_det_dir, gt_dir)
    elif nii_case_flag:
        save_dir = '/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_2/DLSC_0/cases'
        evulate_nii_case(None, save_dir)
    elif FLAGS.full_annotation_flag:
        evulate_dir_full_annotation()
    else:
        evulate_dir()
                
    
if __name__ == '__main__':
    tf.app.run()
# 544 / 828: volume-4.nii (1, 256, 256, 2) 0.0005035501 1.9400452e-09 (1, 256, 256, 1) 0.016835693 6.260295e-05 0.003921569 0.0 (1, 256, 256, 3)