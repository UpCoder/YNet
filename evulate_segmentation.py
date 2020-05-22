# -*- coding=utf-8 -*-

import numpy as np
import math
import tensorflow as tf
from preprocessing import ssd_vgg_preprocessing, segmentation_preprocessing
import util
import cv2
from nets import UNetWL as UNet
from glob import glob
from nets import UNetWLBlocks_ms_DSW as UNetBlocksMS
from datasets.medicalImage import fill_region, close_operation, open_operation, get_kernel_filters, save_mhd_image
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from skimage.measure import label
slim = tf.contrib.slim
import config
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
scales = [[256, 256]]
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


def cluster_postprocessing(pred, gt, features, k=2, method='kmeans'):
    '''
    利用聚类的方法对pred进行后处理
    :param pred: pred的模版，我们对pred的前景进行处理 [512, 512]
    :param gt: gt，用于计算cluster后的匹配关系 [512, 512]
    :param features: 特征 [512, 512, C]
    :param k: 2
    :param method: str, default: kmeans
    :return: processed_pred
    '''
    from metrics import dice
    def _compute_cluster(data, k, method='kmeans'):
        '''
        :param data:
        :param k:
        :return:
        '''
        if method == 'kmeans':
            k_mean_obj = KMeans(n_clusters=k, n_jobs=6, verbose=0, max_iter=500, tol=1e-6)
        elif method == 'spectral':
            k_mean_obj = SpectralClustering(n_clusters=k, n_jobs=6)
        elif method == 'gmm':
            k_mean_obj = GaussianMixture(n_components=k, tol=1e-6, max_iter=500, verbose=0)
        else:
            print('the method do not support!', method)
            assert False
        k_mean_obj.fit(data)
        return k_mean_obj
    new_mask = np.zeros_like(pred, np.uint8)
    labeled_pred = label(pred)
    for label_idx in range(1, np.max(labeled_pred) + 1):
        cur_connect_pred = np.asarray(labeled_pred == label_idx, np.uint8)
        cur_xs, cur_ys = np.where(cur_connect_pred == 1)
        cur_features = features[cur_xs, cur_ys, : ]
        k_mean_obj = _compute_cluster(cur_features, k, method)
        pred_label = k_mean_obj.predict(cur_features)
        cur_mask = np.zeros_like(gt, np.uint8)
        cur_mask[cur_xs, cur_ys] = (pred_label + 1)
        max_dice = -0.5
        target = -1
        for target_k in range(1, k + 1):
            cur_dice = dice(gt, cur_mask == target_k)
            if cur_dice > max_dice:
                max_dice = cur_dice
                target = target_k
        if max_dice == 0.0:
            min_num = 512 * 512
            for target_k in range(1, k + 1):
                num = np.sum(cur_mask == target_k)
                if num < min_num:
                    target = target_k
                    min_num = num
        new_mask[cur_xs, cur_ys] = ((pred_label + 1) == target)
    return new_mask


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


def generate_recovery_image_feature_map():
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
                        batch_size=1)
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
    checkpoint_name = util.io.get_filename(str(checkpoint))
    IoUs = []
    dices = []

    pixel_recovery_features = tf.image.resize_images(net.pixel_recovery_features, image_shape_placeholder)
    with tf.Session(config=sess_config) as sess:
        saver.restore(sess, checkpoint)
        centers = sess.run(net.centers)
        print('sum of centers is ', np.sum(centers))

        for iter, case_name in enumerate(case_names):
            image_data = util.img.imread(
                glob(util.io.join_path(FLAGS.dataset_dir, case_name, 'images', '*.png'))[0], rgb=True)
            mask = cv2.imread(glob(util.io.join_path(FLAGS.dataset_dir, case_name, 'whole_mask.png'))[0])[
                   :, :, 0]

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
            cv2.imwrite(pred_vis_path, np.asarray(pred * 200, np.uint8))
            cv2.imwrite(pred_path, np.asarray(pred, np.uint8))
            recovery_img_path = util.io.join_path(FLAGS.recovery_img_dir, case_name + '.png')
            cv2.imwrite(recovery_img_path, np.asarray(recovery_img[0] * 255, np.uint8))
            recovery_feature_map_path = util.io.join_path(FLAGS.recovery_feature_map_dir, case_name + '.npy')

            xs, ys = np.where(mask == 1)
            features = recovery_feature_map[0][xs, ys, :]
            print 'the size of feature map is ', np.shape(np.asarray(features, np.float32))

            np.save(recovery_feature_map_path, np.asarray(features, np.float32))


def test():
    with tf.name_scope('test'):
        image = tf.placeholder(dtype=tf.int32, shape = [None, None, 3])
        image_shape = tf.placeholder(dtype = tf.int32, shape = [3, ])
        processed_image = segmentation_preprocessing.segmentation_preprocessing(image, None, out_shape=[256, 256],
                                                                                is_training=False)
        b_image = tf.expand_dims(processed_image, axis = 0)
        net = UNet.UNet(b_image, None, is_training=False, decoder=FLAGS.decoder)
        global_step = slim.get_or_create_global_step()

    sess_config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True)
    if FLAGS.gpu_memory_fraction < 0:
        sess_config.gpu_options.allow_growth = True
    elif FLAGS.gpu_memory_fraction > 0:
        sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction;
    
    checkpoint_dir = util.io.get_dir(FLAGS.checkpoint_path)
    logdir = util.io.join_path(checkpoint_dir, 'test', FLAGS.dataset_name + '_' +FLAGS.dataset_split_name)

    # Variables to restore: moving avg. or normal weights.
    if FLAGS.using_moving_average:
        variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        variables_to_restore[global_step.op.name] = global_step
    else:
        variables_to_restore = slim.get_variables_to_restore()
    
    saver = tf.train.Saver(var_list = variables_to_restore)

    case_names = util.io.ls(FLAGS.dataset_dir)
    case_names.sort()
    
    checkpoint = FLAGS.checkpoint_path
    checkpoint_name = util.io.get_filename(str(checkpoint))
    IoUs = []
    dices = []
    with tf.Session(config = sess_config) as sess:
        saver.restore(sess, checkpoint)
        centers = sess.run(net.centers)

        for iter, case_name in enumerate(case_names):
            image_data = util.img.imread(
                glob(util.io.join_path(FLAGS.dataset_dir, case_name, 'images', '*.png'))[0], rgb=True)

            pixel_cls_scores, pixel_recovery_feature_map = sess.run(
                [net.pixel_cls_scores, net.pixel_recovery_features[-1]],
                feed_dict = {
                    image: image_data
            })
            print '%d/%d: %s'%(iter + 1, len(case_names), case_name), np.shape(pixel_cls_scores)
            pos_score = np.asarray(pixel_cls_scores > 0.5, np.uint8)[0, :, :, 1]
            pred = cv2.resize(pos_score, tuple(np.shape(image_data)[:2][::-1]), interpolation=cv2.INTER_NEAREST)
            gt = util.img.imread(util.io.join_path(FLAGS.dataset_dir, case_name, 'weakly_label_whole_mask.png'))[:, :, 1]
            intersection = np.sum(np.logical_and(gt != 0, pred != 0))
            union = np.sum(gt != 0) + np.sum(pred != 0)
            IoU = (1.0 * intersection) / (1.0 * union - 1.0 * intersection) * 100
            dice = (2.0 * intersection) / (1.0 * union) * 100
            IoUs.append(IoU)
            dices.append(dice)
            cv2.imwrite(util.io.join_path(FLAGS.pred_path, case_name + '.png'), pred)
            cv2.imwrite(util.io.join_path(FLAGS.pred_vis_path, case_name + '.png'),
                        np.asarray(pred * 200))
            assign_label = get_assign_label(centers, pixel_recovery_feature_map)[0]
            assign_label = assign_label * pos_score
            assign_label += 1
            cv2.imwrite(util.io.join_path(FLAGS.pred_assign_label_path, case_name + '.png'),
                        np.asarray(assign_label * 100, np.uint8))
    print('total mean of IoU is ', np.mean(IoUs))
    print('total mean of dice is ', np.mean(dices))


def net_center_posprocessing(pred, centers, pixel_wise_feature, gt):
    '''
    根据net维护的center，得到seg的结果
    :param pred: [512, 512]
    :param centers: [2, 256]
    :param pixel_wise_feature: [512, 512, 256]
    :param gt: [512, 512]
    :return:
    '''
    # print(np.shape(pred), np.shape(centers), np.shape(pixel_wise_feature), np.shape(gt))

    from metrics import dice
    distances1 = np.sum((pixel_wise_feature - centers[0]) ** 2, axis=2, keepdims=True)
    distances2 = np.sum((pixel_wise_feature - centers[1]) ** 2, axis=2, keepdims=True)
    distances = np.concatenate([distances1, distances2], axis=2)
    optimized_pred = np.argmin(distances, axis=2)
    # print(np.max(optimized_pred), np.min(optimized_pred))
    res_pred = np.zeros_like(pred, np.uint8)
    res_dice = 0.0
    for i in range(2):
        final_optimized_pred = np.logical_and(
            np.asarray(pred == 1, np.bool),
            np.asarray(optimized_pred == i, np.bool)
        )
        cur_dice = dice(gt, np.asarray(final_optimized_pred, np.uint8))
        if res_dice <= cur_dice:
            res_dice = cur_dice
            res_pred = np.asarray(final_optimized_pred, np.uint8)
        else:
            continue
    return np.asarray(res_pred, np.uint8)


def evulate_dir_nii_weakly():
    '''
    新版本的评估代码
    :return:
    '''
    import os
    from metrics import dice, IoU
    from datasets.medicalImage import convertCase2PNGs, image_expand
    case_names = util.io.ls(FLAGS.dataset_dir)
    case_names.sort()

    restore_path = '/home/give/PycharmProjects/weakly_label_segmentation/logs/DSW/DSW-upsampling-True-1-False-True-True/model.ckpt-10450'

    with tf.name_scope('test'):
        image = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])
        input_shape_placeholder = tf.placeholder(tf.int32, shape=[2])
        processed_image = segmentation_preprocessing.segmentation_preprocessing(image, None, None,
                                                                                out_shape=[256, 256],
                                                                                is_training=False)

        b_image = tf.expand_dims(processed_image, axis=0)

        net = UNetBlocksMS.UNet(b_image, None, None, is_training=False, decoder=FLAGS.decoder,
                                update_center_flag=FLAGS.update_center,
                                batch_size=2, init_center_value=None, update_center_strategy=1,
                                full_annotation_flag=False,
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

    checkpoint = restore_path
    # pixel_recovery_features = tf.image.resize_images(net.pixel_recovery_features, image_shape_placeholder)
    with tf.Session(config=sess_config) as sess:
        saver.restore(sess, checkpoint)

        global_gt = []
        global_pred = []
        global_pred_kmeans = []
        global_pred_centers = []

        case_dices = []
        case_IoUs = []
        case_dices_kmeans = [] # kmeans
        case_IoUs_kmeans = []  # kmeans
        case_dices_centers = [] # net centers
        case_IoUs_centers = [] # net centers
        for iter, case_name in enumerate(case_names):
            # image_data = util.img.imread(
            #     glob(util.io.join_path(FLAGS.dataset_dir, case_name, 'images', '*.png'))[0])
            # gt = util.img.imread(util.io.join_path(FLAGS.dataset_dir, case_name, 'weakly_label_whole_mask.png'))[:, :, 1]

            image_data = cv2.imread(
                glob(util.io.join_path(FLAGS.dataset_dir, case_name, 'images', '*.png'))[0])
            gt = cv2.imread(glob(util.io.join_path(FLAGS.dataset_dir, case_name, 'whole_mask.png'))[0])[
                   :, :, 1]
            gt = cv2.resize(gt, (256, 256), interpolation=cv2.INTER_NEAREST)
            # if case_name == '0402a81e75262469925ea893b6706183832e85324f7b1e08e634129f5d522cdd':
            #     gt = np.transpose(gt, axes=[1, 0])
            print(np.shape(image_data), np.shape(gt))
            recover_image, pixel_recover_feature, pixel_cls_scores, b_image_v, global_step_v, net_centers = sess.run(
                [net.pixel_recovery_logits, net.pixel_recovery_features, net.pixel_cls_scores, b_image, global_step, net.centers],
                feed_dict={
                    image: image_data,
                    input_shape_placeholder: [256, 256],
                })

            pred = np.asarray(pixel_cls_scores[0, :, :, 1] > 0.5, np.uint8)
            pixel_recover_feature = pixel_recover_feature[0]
            # pred = cv2.resize(pred, tuple(np.shape(image_data)[:2][::-1]))
            # pixel_recover_feature = cv2.resize(pixel_recover_feature, tuple(np.shape(image_data)[:2][::-1]))
            print(np.shape(pred), case_name)
            cv2.imwrite('./tmp/%s_gt.png' % (case_name), np.asarray(gt * 200, np.uint8))
            cv2.imwrite('./tmp/%s_img.png' % (case_name), np.asarray(cv2.resize(image_data, (256, 256)), np.uint8))
            cv2.imwrite('./tmp/%s_recover_img.png' % (case_name), np.asarray(recover_image[0] * 255., np.uint8))
            cv2.imwrite('./tmp/%s_pred.png' % (case_name), np.asarray(pred * 200, np.uint8))
            # 开操作 先腐蚀，后膨胀
            # 闭操作 先膨胀，后腐蚀
            # pred = close_operation(pred, kernel_size=3)
            # pred = open_operation(pred, kernel_size=3)
            # pred = fill_region(pred)

            # 再计算kmeans 的结果
            pred_kmeans = np.asarray(image_expand(pred, kernel_size=5), np.uint8)
            # pred_seg = image_expand(pred_seg, 5)
            print(np.shape(pred_kmeans), np.shape(gt), np.shape(pixel_recover_feature))
            pred_kmeans = cluster_postprocessing(pred_kmeans, gt, pixel_recover_feature, k=2)
            # pred_kmeans[image_data[:, :, 1] < (10./255.)] = 0
            # pred_kmeans = close_operation(pred_kmeans, kernel_size=5)
            pred_kmeans = fill_region(pred_kmeans)

            # 计算根据center得到的结果
            # pixel_recover_feature, net_centers
            # pred_centers = np.asarray(image_expand(pred, kernel_size=5), np.uint8)
            pred_centers = np.asarray(pred, np.uint8)
            pred_centers = net_center_posprocessing(pred_centers, centers=net_centers,
                                                    pixel_wise_feature=pixel_recover_feature, gt=gt)
            # pred_centers[image_data[:, :, 1] < (10. / 255.)] = 0
            # pred_centers = close_operation(pred_centers, kernel_size=5)
            pred_centers = fill_region(pred_centers)
            cv2.imwrite('./tmp/%s_pred_center.png' % (case_name), np.asarray(pred_centers * 200, np.uint8))
            cv2.imwrite('./tmp/%s_pred_kmeans.png' % (case_name), np.asarray(pred_kmeans * 200, np.uint8))

            global_gt.append(gt)
            global_pred.append(pred)
            global_pred_kmeans.append(pred_kmeans)
            global_pred_centers.append(pred_centers)

            case_dice = dice(gt, pred)
            case_IoU = IoU(gt, pred)
            case_dice_kmeans = dice(gt, pred_kmeans)
            case_IoU_kmeans = IoU(gt, pred_kmeans)
            case_dice_centers = dice(gt, pred_centers)
            case_IoU_centers = IoU(gt, pred_centers)
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


def main(_):
    config_initialization()
    # test()
    evulate_dir_nii_weakly()
    # generate_recovery_image_feature_map()
                
    
if __name__ == '__main__':
    tf.app.run()
