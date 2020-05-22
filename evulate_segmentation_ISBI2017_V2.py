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

import gc
from post_processing import net_center_posprocessing, cluster_postprocessing
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
        image = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
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


def evulate_dir_nii():
    threshold = 0.5
    print('threshold = ', threshold)
    from metrics import dice, IoU
    from datasets.medicalImage import convertCase2PNGs
    nii_dir = '/home/give/Documents/dataset/ISBI2017/Training_Batch_1'

    save_dir = '/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_1/DLSC_0_full_annotation/niis'
    restore_path = '/home/give/PycharmProjects/weakly_label_segmentation/logs/ISBI2017_V2/1s_agumentation_weakly_V3-upsampling-fully/model.ckpt-235870'

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
            imgs = np.expand_dims(imgs[:, :, :, 1], axis=3)
            imgs = np.concatenate([imgs, imgs, imgs], axis=3)
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
                    pred = close_operation(pred, kernel_size=5)
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

        print 'global dice is ', dice(global_gt, global_pred)
        print 'global IoU is ', IoU(global_gt, global_pred)

        print('mean of case dice is ', np.mean(case_dices))
        print('mean of case IoU is ', np.mean(case_IoUs))


def evulate_dir_nii_weakly_crf():
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
        d.addPairwiseBilateral(sxy=2, srgb=3, rgbim=img, compat=10)

        Q = d.inference(5)

        Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

        return Q
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


def evulate_dir_nii_weakly_wo_update():
    from metrics import dice, IoU
    from datasets.medicalImage import convertCase2PNGs, image_expand
    nii_dir = '/media/give/CBMIR/ld/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training_Batch_1'
    save_dir = '/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_1/DLSC_0_wo_update/niis'
    restore_path = '/home/give/PycharmProjects/weakly_label_segmentation/logs/ISBI2017_V2/1s_agumentation_weakly-upsampling-2-wo_update/model.ckpt-170026'

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
        global_pred_seg = []

        case_dices = []
        case_IoUs = []
        case_dices_seg = []
        case_IoUs_seg = []
        for iter, nii_path in enumerate(nii_pathes):
            if os.path.basename(nii_path) in ['volume-5.nii', 'volume-12.nii', 'volume-13.nii', 'volume-15.nii', 'volume-19.nii', 'volume-20.nii', 'volume-21.nii', 'volume-25.nii', 'volume-26.nii']:
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
            case_preds_seg = []
            case_gts = []

            # case_recover_features = []
            print(nii_path, seg_path)
            imgs, tumor_masks, liver_masks, tumor_weak_masks = convertCase2PNGs(nii_path, seg_path, save_dir=None)
            print(len(imgs), len(tumor_masks), len(liver_masks), len(tumor_masks))
            for slice_idx, (image_data, liver_mask, whole_mask) in enumerate(zip(imgs, liver_masks, tumor_masks)):
                pixel_cls_scores_ms = []
                pixel_recover_feature_ms = []
                for single_scale in scales:
                    pixel_recover_feature, pixel_cls_scores, b_image_v, global_step_v = sess.run(
                        [net.pixel_recovery_features, net.pixel_cls_scores, b_image, global_step],
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
                    IoUs.append(IoU(whole_mask, pred))
                    dices.append(dice(whole_mask, pred))

                    # 再计算kmeans 的结果
                    pred_seg = np.asarray(image_expand(pred, kernel_size=5), np.uint8)
                    # pred_seg = image_expand(pred_seg, 5)
                    pred_seg = cluster_postprocessing(pred_seg, whole_mask, pixel_recover_feature, k=2)
                    pred_seg[image_data[:, :, 1] < (10./255.)] = 0
                    pred_seg = close_operation(pred_seg, kernel_size=5)
                    pred_seg = fill_region(pred_seg)

                else:
                    pred = np.zeros_like(whole_mask)
                    pred_seg = np.zeros_like(whole_mask)
                global_gt.append(whole_mask)
                case_gts.append(whole_mask)
                case_preds.append(pred)
                case_preds_seg.append(pred_seg)
                global_pred.append(pred)
                global_pred_seg.append(pred_seg)
                print '%d / %d: %s' % (slice_idx + 1, len(imgs), os.path.basename(nii_path)), np.shape(
                    pixel_cls_scores), np.max(
                    pixel_cls_scores), np.min(pixel_cls_scores), np.shape(pixel_recover_feature)
                del pixel_recover_feature, pixel_recover_feature_ms
                gc.collect()

            save_mhd_image(np.transpose(np.asarray(case_preds, np.uint8), axes=[0, 2, 1]),
                           os.path.join(save_dir, nii_path_basename.split('.')[0].split('-')[1], 'pred.mhd'))
            save_mhd_image(np.transpose(np.asarray(case_preds_seg, np.uint8), axes=[0, 2, 1]),
                           os.path.join(save_dir, nii_path_basename.split('.')[0].split('-')[1], 'pred_seg.mhd'))
            case_dice = dice(case_gts, case_preds)
            case_IoU = IoU(case_gts, case_preds)
            case_dice_seg = dice(case_gts, case_preds_seg)
            case_IoU_seg = IoU(case_gts, case_preds_seg)
            print('case dice: ', case_dice)
            print('case IoU: ', case_IoU)
            print('case dice seg: ', case_dice_seg)
            print('case IoU seg: ', case_IoU_seg)
            case_dices.append(case_dice)
            case_IoUs.append(case_IoU)
            case_dices_seg.append(case_dice_seg)
            case_IoUs_seg.append(case_IoU_seg)

        print 'global dice is ', dice(global_gt, global_pred)
        print 'global IoU is ', IoU(global_gt, global_pred)

        print('mean of case dice is ', np.mean(case_dices))
        print('mean of case IoU is ', np.mean(case_IoUs))


        print 'global dice (seg) is ', dice(global_gt, global_pred_seg)
        print 'global IoU (seg) is ', IoU(global_gt, global_pred_seg)

        print 'mean of case dice is ', np.mean(case_dices_seg)
        print 'mean of case IoU is ', np.mean(case_IoUs_seg)


def evulate_dir_nii_weakly_wo_learnable_connection():
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
    from metrics import dice, IoU
    from datasets.medicalImage import convertCase2PNGs, image_expand
    nii_dir = '/media/give/CBMIR/ld/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training_Batch_1'
    save_dir = '/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_1/DLSC_0_wo_learnable_connection/niis'
    restore_path = '/home/give/PycharmProjects/weakly_label_segmentation/logs/ISBI2017_V2/1s_agumentation_weakly_wo_learnable_connection-upsampling-2/model.ckpt-145121'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
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
                                batch_size=2, init_center_value=None, update_center_strategy=2,
                                num_centers_k=FLAGS.num_centers_k, full_annotation_flag=False,
                                output_shape_tensor=input_shape_placeholder, dense_connection_flag=True,
                                learnable_merged_flag=False)

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
        global_pred_seg = []

        case_dices = []
        case_IoUs = []
        case_dices_seg = []
        case_IoUs_seg = []
        for iter, nii_path in enumerate(nii_pathes):
            if os.path.basename(nii_path) in ['volume-5.nii', 'volume-12.nii', 'volume-13.nii', 'volume-15.nii', 'volume-19.nii', 'volume-20.nii', 'volume-21.nii', 'volume-25.nii', 'volume-26.nii']:
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
            case_preds_seg = []
            case_gts = []

            # case_recover_features = []
            print(nii_path, seg_path)
            imgs, tumor_masks, liver_masks, tumor_weak_masks = convertCase2PNGs(nii_path, seg_path, save_dir=None)
            print(len(imgs), len(tumor_masks), len(liver_masks), len(tumor_masks))
            for slice_idx, (image_data, liver_mask, whole_mask) in enumerate(zip(imgs, liver_masks, tumor_masks)):
                pixel_cls_scores_ms = []
                pixel_recover_feature_ms = []
                for single_scale in scales:
                    pixel_recover_feature, pixel_cls_scores, b_image_v, global_step_v = sess.run(
                        [net.pixel_recovery_features, net.pixel_cls_scores, b_image, global_step],
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
                    IoUs.append(IoU(whole_mask, pred))
                    dices.append(dice(whole_mask, pred))

                    # 再计算kmeans 的结果
                    pred_seg = np.asarray(image_expand(pred, kernel_size=5), np.uint8)
                    # pred_seg = image_expand(pred_seg, 5)
                    pred_seg = cluster_postprocessing(pred_seg, whole_mask, pixel_recover_feature, k=2)
                    pred_seg[image_data[:, :, 1] < (10./255.)] = 0
                    pred_seg = close_operation(pred_seg, kernel_size=5)
                    pred_seg = fill_region(pred_seg)

                else:
                    pred = np.zeros_like(whole_mask)
                    pred_seg = np.zeros_like(whole_mask)
                global_gt.append(whole_mask)
                case_gts.append(whole_mask)
                case_preds.append(pred)
                case_preds_seg.append(pred_seg)
                global_pred.append(pred)
                global_pred_seg.append(pred_seg)
                print '%d / %d: %s' % (slice_idx + 1, len(imgs), os.path.basename(nii_path)), np.shape(
                    pixel_cls_scores), np.max(
                    pixel_cls_scores), np.min(pixel_cls_scores), np.shape(pixel_recover_feature)
                del pixel_recover_feature, pixel_recover_feature_ms
                gc.collect()

            save_mhd_image(np.transpose(np.asarray(case_preds, np.uint8), axes=[0, 2, 1]),
                           os.path.join(save_dir, nii_path_basename.split('.')[0].split('-')[1], 'pred.mhd'))
            save_mhd_image(np.transpose(np.asarray(case_preds_seg, np.uint8), axes=[0, 2, 1]),
                           os.path.join(save_dir, nii_path_basename.split('.')[0].split('-')[1], 'pred_seg.mhd'))
            case_dice = dice(case_gts, case_preds)
            case_IoU = IoU(case_gts, case_preds)
            case_dice_seg = dice(case_gts, case_preds_seg)
            case_IoU_seg = IoU(case_gts, case_preds_seg)
            print('case dice: ', case_dice)
            print('case IoU: ', case_IoU)
            print('case dice seg: ', case_dice_seg)
            print('case IoU seg: ', case_IoU_seg)
            case_dices.append(case_dice)
            case_IoUs.append(case_IoU)
            case_dices_seg.append(case_dice_seg)
            case_IoUs_seg.append(case_IoU_seg)

        print 'global dice is ', dice(global_gt, global_pred)
        print 'global IoU is ', IoU(global_gt, global_pred)

        print('mean of case dice is ', np.mean(case_dices))
        print('mean of case IoU is ', np.mean(case_IoUs))


        print 'global dice (seg) is ', dice(global_gt, global_pred_seg)
        print 'global IoU (seg) is ', IoU(global_gt, global_pred_seg)

        print 'mean of case dice is ', np.mean(case_dices_seg)
        print 'mean of case IoU is ', np.mean(case_IoUs_seg)


def evulate_dir_nii_weakly():
    from metrics import dice, IoU
    from datasets.medicalImage import convertCase2PNGs, image_expand
    nii_dir = '/home/give/Documents/dataset/ISBI2017/Training_Batch_1'
    save_dir = '/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_1/DLSC_0/niis'
    # restore_path = '/home/give/PycharmProjects/weakly_label_segmentation/logs/ISBI2017_V2/1s_agumentation_weakly-upsampling-2/model.ckpt-168090'

    restore_path = '/home/give/PycharmProjects/weakly_label_segmentation/logs/ISBI2017_V2/1s_agumentation_weakly_V3-upsampling-2/model.ckpt-211635'

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
        global_pred_seg = []
        global_pred_crf = []

        case_dices = []
        case_IoUs = []
        case_dices_seg = []
        case_IoUs_seg = []
        case_dices_crf = []
        case_IoUs_crf = []
        for iter, nii_path in enumerate(nii_pathes):
            if os.path.basename(nii_path) in ['volume-5.nii', 'volume-12.nii', 'volume-13.nii', 'volume-15.nii', 'volume-19.nii', 'volume-20.nii', 'volume-21.nii', 'volume-25.nii', 'volume-26.nii']:
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
            case_preds_seg = []
            case_gts = []

            # case_recover_features = []
            print(nii_path, seg_path)
            imgs, tumor_masks, liver_masks, tumor_weak_masks = convertCase2PNGs(nii_path, seg_path, save_dir=None)
            print(len(imgs), len(tumor_masks), len(liver_masks), len(tumor_masks))
            for slice_idx, (image_data, liver_mask, whole_mask) in enumerate(zip(imgs, liver_masks, tumor_masks)):
                pixel_cls_scores_ms = []
                pixel_recover_feature_ms = []
                for single_scale in scales:
                    pixel_recover_feature, pixel_cls_scores, b_image_v, global_step_v = sess.run(
                        [net.pixel_recovery_features, net.pixel_cls_scores, b_image, global_step],
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
                    IoUs.append(IoU(whole_mask, pred))
                    dices.append(dice(whole_mask, pred))

                    # 再计算kmeans 的结果
                    pred_seg = np.asarray(image_expand(pred, kernel_size=5), np.uint8)
                    # pred_seg = image_expand(pred_seg, 5)
                    pred_seg = cluster_postprocessing(pred_seg, whole_mask, pixel_recover_feature, k=2)
                    pred_seg[image_data[:, :, 1] < (10./255.)] = 0
                    pred_seg = close_operation(pred_seg, kernel_size=5)
                    pred_seg = fill_region(pred_seg)

                else:
                    pred = np.zeros_like(whole_mask)
                    pred_seg = np.zeros_like(whole_mask)
                global_gt.append(whole_mask)
                case_gts.append(whole_mask)
                case_preds.append(pred)
                case_preds_seg.append(pred_seg)
                global_pred.append(pred)
                global_pred_seg.append(pred_seg)
                print '%d / %d: %s' % (slice_idx + 1, len(imgs), os.path.basename(nii_path)), np.shape(
                    pixel_cls_scores), np.max(
                    pixel_cls_scores), np.min(pixel_cls_scores), np.shape(pixel_recover_feature)
                del pixel_recover_feature, pixel_recover_feature_ms
                gc.collect()

            save_mhd_image(np.transpose(np.asarray(case_preds, np.uint8), axes=[0, 2, 1]),
                           os.path.join(save_dir, nii_path_basename.split('.')[0].split('-')[1], 'pred.mhd'))
            save_mhd_image(np.transpose(np.asarray(case_preds_seg, np.uint8), axes=[0, 2, 1]),
                           os.path.join(save_dir, nii_path_basename.split('.')[0].split('-')[1], 'pred_seg.mhd'))
            case_dice = dice(case_gts, case_preds)
            case_IoU = IoU(case_gts, case_preds)
            case_dice_seg = dice(case_gts, case_preds_seg)
            case_IoU_seg = IoU(case_gts, case_preds_seg)
            print('case dice: ', case_dice)
            print('case IoU: ', case_IoU)
            print('case dice seg: ', case_dice_seg)
            print('case IoU seg: ', case_IoU_seg)
            case_dices.append(case_dice)
            case_IoUs.append(case_IoU)
            case_dices_seg.append(case_dice_seg)
            case_IoUs_seg.append(case_IoU_seg)

        print 'global dice is ', dice(global_gt, global_pred)
        print 'global IoU is ', IoU(global_gt, global_pred)

        print('mean of case dice is ', np.mean(case_dices))
        print('mean of case IoU is ', np.mean(case_IoUs))


        print 'global dice (seg) is ', dice(global_gt, global_pred_seg)
        print 'global IoU (seg) is ', IoU(global_gt, global_pred_seg)

        print 'mean of case dice is ', np.mean(case_dices_seg)
        print 'mean of case IoU is ', np.mean(case_IoUs_seg)


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

    # restore_path = '/home/give/PycharmProjects/weakly_label_segmentation/logs/ISBI2017_V2/1s_agumentation_weakly_V3-upsampling-1/model.ckpt-167824'
    restore_path = '/home/give/PycharmProjects/weakly_label_segmentation/logs/ISBI2017_V2/1s_agumentation_weakly_V3/-upsampling-True-1-False-True-True/model.ckpt-170919'
    nii_parent_dir = os.path.dirname(nii_dir)
    with tf.name_scope('test'):
        image = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
        image_shape_placeholder = tf.placeholder(tf.int32, shape=[2])
        input_shape_placeholder = tf.placeholder(tf.int32, shape=[2])
        processed_image = segmentation_preprocessing.segmentation_preprocessing(image, None, None,
                                                                                out_shape=input_shape_placeholder,
                                                                                is_training=False)
        b_image = tf.expand_dims(processed_image, axis=0)
        # batch_size = 2
        # image = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, None, 3])
        # image_shape_placeholder = tf.placeholder(tf.int32, shape=[2])
        # input_shape_placeholder = tf.placeholder(tf.int32, shape=[2])
        # with tf.name_scope('preprocessing_val'):
        #     print('image is ', image)
        #     b_image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        #     b_image = tf.image.resize_images(b_image, input_shape_placeholder,
        #                                   method=tf.image.ResizeMethod.BILINEAR,
        #                                   align_corners=False)

        net = UNetBlocksMS.UNet(b_image, None, None, is_training=False, decoder=FLAGS.decoder,
                                update_center_flag=FLAGS.update_center,
                                batch_size=2, init_center_value=None, update_center_strategy=1,
                                num_centers_k=FLAGS.num_centers_k, full_annotation_flag=False,
                                output_shape_tensor=input_shape_placeholder, dense_connection_flag=True,
                                learnable_merged_flag=True)

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


            # start_idx = 0
            # pixel_recover_feature = []
            # pixel_cls_scores = []
            # while start_idx < len(imgs):
            #     end_idx = start_idx + batch_size
            #     if end_idx > len(imgs):
            #         end_idx = len(imgs)
            #
            #     batch_images = imgs[start_idx: end_idx]
            #     pixel_recover_feature_batch, pixel_cls_scores_batch, b_image_v, global_step_v, net_centers = sess.run(
            #         [net.pixel_recovery_features, net.pixel_cls_scores, b_image, global_step, net.centers],
            #         feed_dict={
            #             image: batch_images,
            #             image_shape_placeholder: np.shape(batch_images)[1:3],
            #             input_shape_placeholder: [512, 512]
            #         })
            #     pixel_recover_feature.extend(pixel_recover_feature_batch)
            #     pixel_cls_scores.extend(pixel_cls_scores_batch)
            #     start_idx = end_idx
            # print(len(pixel_recover_feature), len(pixel_cls_scores))
            # for slice_idx, (image_data, liver_mask, whole_mask) in enumerate(zip(imgs, liver_masks, tumor_masks)):
            #     if np.sum(whole_mask) != 0:
            #         pred = np.asarray(pixel_cls_scores[slice_idx] > 0.6, np.uint8)
            #         # 开操作 先腐蚀，后膨胀
            #         # 闭操作 先膨胀，后腐蚀
            #         # pred = close_operation(pred, kernel_size=3)
            #         pred = open_operation(pred, kernel_size=3)
            #         pred = fill_region(pred)
            #
            #         # 再计算kmeans 的结果
            #         pred_kmeans = np.asarray(image_expand(pred, kernel_size=5), np.uint8)
            #         # pred_seg = image_expand(pred_seg, 5)
            #         pred_kmeans = cluster_postprocessing(pred_kmeans, whole_mask, pixel_recover_feature[slice_idx], k=2)
            #         pred_kmeans[image_data[:, :, 1] < (10./255.)] = 0
            #         pred_kmeans = close_operation(pred_kmeans, kernel_size=5)
            #         pred_kmeans = fill_region(pred_kmeans)
            #
            #         # 计算根据center得到的结果
            #         # pixel_recover_feature, net_centers
            #         pred_centers = np.asarray(image_expand(pred, kernel_size=5), np.uint8)
            #         pred_centers = net_center_posprocessing(pred_centers, centers=net_centers,
            #                                                 pixel_wise_feature=pixel_recover_feature[slice_idx],
            #                                                 gt=whole_mask)
            #         pred_centers[image_data[:, :, 1] < (10. / 255.)] = 0
            #         pred_centers = close_operation(pred_centers, kernel_size=5)
            #         pred_centers = fill_region(pred_centers)
            #     else:
            #         pred = np.zeros_like(whole_mask)
            #         pred_kmeans = np.zeros_like(whole_mask)
            #         pred_centers = np.zeros_like(whole_mask)
            #
            #     global_gt.append(whole_mask)
            #     case_gts.append(whole_mask)
            #     case_preds.append(pred)
            #     case_preds_kmeans.append(pred_kmeans)
            #     case_preds_centers.append(pred_centers)
            #     global_pred.append(pred)
            #     global_pred_kmeans.append(pred_kmeans)
            #     global_pred_centers.append(pred_centers)

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


def evulate_dir_nii_weakly_kmeans_pixel():
    '''
    新版本的评估代码
    :return:
    '''
    from metrics import dice, IoU
    from datasets.medicalImage import convertCase2PNGs, image_expand
    nii_dir = '/home/give/Documents/dataset/ISBI2017/Training_Batch_1'
    save_dir = '/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_1/DLSC_0/niis'
    # restore_path = '/home/give/PycharmProjects/weakly_label_segmentation/logs/ISBI2017_V2/1s_agumentation_weakly-upsampling-2/model.ckpt-168090'

    # restore_path = '/home/give/PycharmProjects/weakly_label_segmentation/logs/ISBI2017_V2/1s_agumentation_weakly_V3-upsampling-1/model.ckpt-167824'
    restore_path = '/home/give/PycharmProjects/weakly_label_segmentation/logs/ISBI2017_V2/1s_agumentation_weakly_V3/-upsampling-True-1-False-True-True/model.ckpt-170919'
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
                                output_shape_tensor=input_shape_placeholder, dense_connection_flag=True,
                                learnable_merged_flag=True)

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


def evulate_dir_nii_weakly_wo_dense_learnable_new():
    '''
    新版本的评估代码
    :return:
    '''
    from metrics import dice, IoU
    from datasets.medicalImage import convertCase2PNGs, image_expand
    nii_dir = '/home/give/Documents/dataset/ISBI2017/Training_Batch_1'
    save_dir = '/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_1/DLSC_0/niis'
    # restore_path = '/home/give/PycharmProjects/weakly_label_segmentation/logs/ISBI2017_V2/1s_agumentation_weakly-upsampling-2/model.ckpt-168090'

    restore_path = '/home/give/PycharmProjects/weakly_label_segmentation/logs/ISBI2017_V2/1s_agumentation_weakly_V3/-upsampling-True-1-False-False-False/model.ckpt-98237'

    nii_parent_dir = os.path.dirname(nii_dir)
    with tf.name_scope('test'):
        image = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
        image_shape_placeholder = tf.placeholder(tf.int32, shape=[2])
        input_shape_placeholder = tf.placeholder(tf.int32, shape=[2])
        processed_image = segmentation_preprocessing.segmentation_preprocessing(image, None, None,
                                                                                out_shape=input_shape_placeholder,
                                                                                is_training=False)
        b_image = tf.expand_dims(processed_image, axis=0)
        # batch_size = 2
        # image = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, None, 3])
        # image_shape_placeholder = tf.placeholder(tf.int32, shape=[2])
        # input_shape_placeholder = tf.placeholder(tf.int32, shape=[2])
        # with tf.name_scope('preprocessing_val'):
        #     print('image is ', image)
        #     b_image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        #     b_image = tf.image.resize_images(b_image, input_shape_placeholder,
        #                                   method=tf.image.ResizeMethod.BILINEAR,
        #                                   align_corners=False)

        net = UNetBlocksMS.UNet(b_image, None, None, is_training=False, decoder=FLAGS.decoder,
                                update_center_flag=FLAGS.update_center,
                                batch_size=2, init_center_value=None, update_center_strategy=1,
                                num_centers_k=FLAGS.num_centers_k, full_annotation_flag=False,
                                dense_connection_flag=False, learnable_merged_flag=False,
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


            # start_idx = 0
            # pixel_recover_feature = []
            # pixel_cls_scores = []
            # while start_idx < len(imgs):
            #     end_idx = start_idx + batch_size
            #     if end_idx > len(imgs):
            #         end_idx = len(imgs)
            #
            #     batch_images = imgs[start_idx: end_idx]
            #     pixel_recover_feature_batch, pixel_cls_scores_batch, b_image_v, global_step_v, net_centers = sess.run(
            #         [net.pixel_recovery_features, net.pixel_cls_scores, b_image, global_step, net.centers],
            #         feed_dict={
            #             image: batch_images,
            #             image_shape_placeholder: np.shape(batch_images)[1:3],
            #             input_shape_placeholder: [512, 512]
            #         })
            #     pixel_recover_feature.extend(pixel_recover_feature_batch)
            #     pixel_cls_scores.extend(pixel_cls_scores_batch)
            #     start_idx = end_idx
            # print(len(pixel_recover_feature), len(pixel_cls_scores))
            # for slice_idx, (image_data, liver_mask, whole_mask) in enumerate(zip(imgs, liver_masks, tumor_masks)):
            #     if np.sum(whole_mask) != 0:
            #         pred = np.asarray(pixel_cls_scores[slice_idx] > 0.6, np.uint8)
            #         # 开操作 先腐蚀，后膨胀
            #         # 闭操作 先膨胀，后腐蚀
            #         # pred = close_operation(pred, kernel_size=3)
            #         pred = open_operation(pred, kernel_size=3)
            #         pred = fill_region(pred)
            #
            #         # 再计算kmeans 的结果
            #         pred_kmeans = np.asarray(image_expand(pred, kernel_size=5), np.uint8)
            #         # pred_seg = image_expand(pred_seg, 5)
            #         pred_kmeans = cluster_postprocessing(pred_kmeans, whole_mask, pixel_recover_feature[slice_idx], k=2)
            #         pred_kmeans[image_data[:, :, 1] < (10./255.)] = 0
            #         pred_kmeans = close_operation(pred_kmeans, kernel_size=5)
            #         pred_kmeans = fill_region(pred_kmeans)
            #
            #         # 计算根据center得到的结果
            #         # pixel_recover_feature, net_centers
            #         pred_centers = np.asarray(image_expand(pred, kernel_size=5), np.uint8)
            #         pred_centers = net_center_posprocessing(pred_centers, centers=net_centers,
            #                                                 pixel_wise_feature=pixel_recover_feature[slice_idx],
            #                                                 gt=whole_mask)
            #         pred_centers[image_data[:, :, 1] < (10. / 255.)] = 0
            #         pred_centers = close_operation(pred_centers, kernel_size=5)
            #         pred_centers = fill_region(pred_centers)
            #     else:
            #         pred = np.zeros_like(whole_mask)
            #         pred_kmeans = np.zeros_like(whole_mask)
            #         pred_centers = np.zeros_like(whole_mask)
            #
            #     global_gt.append(whole_mask)
            #     case_gts.append(whole_mask)
            #     case_preds.append(pred)
            #     case_preds_kmeans.append(pred_kmeans)
            #     case_preds_centers.append(pred_centers)
            #     global_pred.append(pred)
            #     global_pred_kmeans.append(pred_kmeans)
            #     global_pred_centers.append(pred_centers)

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

def evulate_dir_nii_weakly_feature_crf():
    '''
    新版本的评估代码
    :return:
    '''
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

        pca_image = pca.fit_transform(np.reshape(img, [-1, 256]))
        img = np.reshape(pca_image, [512, 512, 16])

        pairwise_energy = create_pairwise_bilateral(sdims=(20, 20), schan=(3,), img=img, chdim=2)
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseEnergy(pairwise_energy, compat=10)

        # d.addPairwiseBilateral(sxy=2, srgb=3, rgbim=img, compat=10)

        Q = d.inference(5)

        Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

        return Q
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

    # sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    # if FLAGS.gpu_memory_fraction < 0:
    #     sess_config.gpu_options.allow_growth = True
    # elif FLAGS.gpu_memory_fraction > 0:
    #     sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction

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
    with tf.Session() as sess:
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
    if FLAGS.full_annotation_flag:
        print('full annotation flag')
        evulate_dir_nii()
    else:
        print('weakly annotation flag')
        # evulate_dir_nii_weakly()
        # evulate_dir_nii_weakly_wo_dense_learnable_new()
        # evulate_dir_nii_weakly_new()
        evulate_dir_nii_weakly_kmeans_pixel()
        # evulate_dir_nii_weakly_feature_crf()
        # evulate_dir_nii_weakly_crf()
        # evulate_dir_nii_weakly_wo_update()
        # evulate_dir_nii_weakly_wo_learnable_connection()


if __name__ == '__main__':
    tf.app.run()
# 544 / 828: volume-4.nii (1, 256, 256, 2) 0.0005035501 1.9400452e-09 (1, 256, 256, 1) 0.016835693 6.260295e-05 0.003921569 0.0 (1, 256, 256, 3)


# overview 描述不清
# 强调我们得到多器官分割的model，基于多个单器官的model，without annotation
# related work 里面介绍有许多有监督的方法做多器官分割，但是需要大量标记。本文提供一个新思路，就是利用多个基于多个单器官的model，生成多器官分割。这种思路全新的思路。
# 直接应用存在的问题，1，耗费时间大。2，不同数据集之间可能存在gap
#