#encoding = utf-8

import numpy as np
import math
import tensorflow as tf
from preprocessing import ssd_vgg_preprocessing, segmentation_preprocessing
import util
import cv2
from nets import pixel_link_symbol, UNet
from glob import glob


slim = tf.contrib.slim
import config
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

tf.app.flags.DEFINE_integer('eval_image_width', 1280, 'Train image size')
tf.app.flags.DEFINE_integer('eval_image_height', 768, 'Train image size')
tf.app.flags.DEFINE_bool('using_moving_average', True, 
                         'Whether to use ExponentionalMovingAverage')
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999, 
                          'The decay rate of ExponentionalMovingAverage')
tf.app.flags.DEFINE_string('pred_path', '', '')
tf.app.flags.DEFINE_string('pred_vis_path', '', '')

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
                       batch_size = 1, 
                       pixel_conf_threshold = 0.5,
                       link_conf_threshold = 0.1,
                       num_gpus = 1, 
                   )
    util.proc.set_proc_name('test_pixel_link_on'+ '_' + FLAGS.dataset_name)


def test():
    with tf.name_scope('test'):
        image = tf.placeholder(dtype=tf.int32, shape = [None, None, 3])
        image_shape = tf.placeholder(dtype = tf.int32, shape = [3, ])
        processed_image = segmentation_preprocessing.segmentation_preprocessing(image, None, out_shape=[256, 256],
                                                                                is_training=False)
        b_image = tf.expand_dims(processed_image, axis = 0)
        net = UNet.UNet(b_image, None, is_training=False)
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
    
    with tf.Session(config = sess_config) as sess:
        saver.restore(sess, checkpoint)

        for iter, case_name in enumerate(case_names):
            image_data = util.img.imread(
                glob(util.io.join_path(FLAGS.dataset_dir, case_name, 'images', '*.png'))[0], rgb=True)
            pixel_cls_scores, = sess.run(
                [net.pixel_cls_scores],
                feed_dict = {
                    image: image_data
            })
            print '%d/%d: %s'%(iter + 1, len(case_names), case_name), np.shape(pixel_cls_scores)
            pos_score = np.asarray(pixel_cls_scores > 0.5, np.uint8)[0, :, :, 1]
            pos_score = cv2.resize(pos_score, tuple(np.shape(image_data)[:2]), interpolation=cv2.INTER_NEAREST)

            cv2.imwrite(util.io.join_path(FLAGS.pred_path, case_name + '.png'), pos_score)
            cv2.imwrite(util.io.join_path(FLAGS.pred_vis_path, case_name + '.png'),
                        np.asarray(pos_score * 200))
         

def main(_):
    config_initialization()
    test()
                
    
if __name__ == '__main__':
    tf.app.run()
