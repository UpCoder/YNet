# -*- coding=utf-8 -*-
# 加入超参数，K，决定把整个图像划分成几块，分别计算center
# test code to make sure the ground truth calculation and data batch works well.

import numpy as np
import tensorflow as tf # test
from tensorflow.python.ops import control_flow_ops

from datasets import dataset_factory

from nets import UNetWL as UNet
from nets import UNetWLAttention as UNetAttention
from nets import UNetWLBlocks as UNetBlocks
from nets import UNetWLBlocks_ms as UNetBlocksMS
from tf_extended.instance_balanced_weight import tf_cal_gt_for_single_image
from preprocessing.preprocessing_factory import scales
import util
slim = tf.contrib.slim
import config
# =========================================================================== #
# Checkpoint and running Flags
# =========================================================================== #
tf.app.flags.DEFINE_string('train_dir', None, 
                           'the path to store checkpoints and eventfiles for summaries')

tf.app.flags.DEFINE_string('checkpoint_path', None, 
   'the path of pretrained model to be used. If there are checkpoints in train_dir, this config will be ignored.')

tf.app.flags.DEFINE_float('gpu_memory_fraction', -1, 
                          'the gpu memory fraction to be used. If less than 0, allow_growth = True is used.')

tf.app.flags.DEFINE_integer('batch_size', None, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'The number of gpus can be used.')
tf.app.flags.DEFINE_integer('max_number_of_steps', 1000000, 'The maximum number of training steps.')
tf.app.flags.DEFINE_integer('log_every_n_steps', 1, 'log frequency')
tf.app.flags.DEFINE_bool("ignore_missing_vars", False, '')
# tf.app.flags.DEFINE_string('checkpoint_exclude_scopes', ['pixel_link_modify', 'pixel_cls_modify'], 'checkpoint_exclude_scopes')
checkpoint_exclude_scopes = ['pixel_recovery', 'pixel_cls']
# =========================================================================== #
# Optimizer configs.
# =========================================================================== #
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'learning rate.')
tf.app.flags.DEFINE_float('momentum', 0.9, 'The momentum for the MomentumOptimizer')
tf.app.flags.DEFINE_float('weight_decay', 0.00001, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_bool('using_moving_average', True, 'Whether to use ExponentionalMovingAverage')
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999, 'The decay rate of ExponentionalMovingAverage')

# =========================================================================== #
# I/O and preprocessing Flags.
# =========================================================================== #
tf.app.flags.DEFINE_integer(
    'num_readers', 1,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 24,
    'The number of threads used to create the batches.')

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'dataset_name', None, 'The name of the dataset to load.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string(
    'decoder', 'upsampling', 'the method of decoder, such as upsampling, transpose'
)
tf.app.flags.DEFINE_integer('train_image_width', 256, 'Train image size')
tf.app.flags.DEFINE_integer('train_image_height', 256, 'Train image size')
tf.app.flags.DEFINE_bool('update_center', False, '')
tf.app.flags.DEFINE_bool('attention_flag', False, '')
tf.app.flags.DEFINE_integer('update_center_strategy', 1, '1 is similar to center loss, 2 is our proposed')
tf.app.flags.DEFINE_bool('center_block_flag', True, '')
tf.app.flags.DEFINE_integer('num_centers_k', 2, 'split the image into k^2 block to compute the center')
tf.app.flags.DEFINE_bool('full_annotation_flag', False, '')
FLAGS = tf.app.flags.FLAGS


def config_initialization():
    # image shape and feature layers shape inference
    image_shape = (FLAGS.train_image_height, FLAGS.train_image_width)
    
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    
    tf.logging.set_verbosity(tf.logging.DEBUG)
    util.init_logger(
        log_file = 'log_train_pixel_link_%d_%d.log'%image_shape, 
                    log_path = FLAGS.train_dir + '-' + FLAGS.decoder, stdout = False, mode = 'a')

    config.load_config('./')
    config.init_config(image_shape, 
                       batch_size = FLAGS.batch_size, 
                       weight_decay = FLAGS.weight_decay, 
                       num_gpus = FLAGS.num_gpus
                   )

    batch_size = config.batch_size
    batch_size_per_gpu = config.batch_size_per_gpu
        
    tf.summary.scalar('batch_size', batch_size)
    tf.summary.scalar('batch_size_per_gpu', batch_size_per_gpu)

    util.proc.set_proc_name('train_pixel_link_on'+ '_' + FLAGS.dataset_name)
    
    dataset = dataset_factory.get_dataset(FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)
    # config.print_config(FLAGS, dataset)
    return dataset


def create_dataset_batch_queue(dataset, preprocessing_flag=True):
    from preprocessing import preprocessing_factory

    with tf.device('/cpu:0'):
        with tf.name_scope(FLAGS.dataset_name + '_data_provider'):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=FLAGS.num_readers,
                common_queue_capacity=1000 * config.batch_size,
                common_queue_min=700 * config.batch_size,
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

        liver_mask = tf.cast(tf.logical_or(tf.equal(liver_mask, 1), tf.equal(full_mask, 1)), tf.uint8)
        if FLAGS.full_annotation_flag:
            mask_image = full_mask
        else:
            mask_image = weak_mask

        image = tf.reshape(image, [512, 512, 3])
        mask_image = tf.reshape(mask_image, [512, 512, 1])
        liver_mask = tf.reshape(liver_mask, [512, 512, 1])
        mask_image = tf.concat([mask_image, mask_image, mask_image], axis=-1)
        liver_mask = tf.concat([liver_mask, liver_mask, liver_mask], axis=-1)
        image = tf.identity(image, 'input_image')

        mask_image = tf.identity(mask_image, 'mask_image')
        liver_mask = tf.identity(liver_mask, 'liver_mask')
        liver_mask = tf.cast(liver_mask, tf.uint8)
        mask_image = tf.cast(mask_image, tf.uint8)
        _, pixel_cls_weight = \
            tf_cal_gt_for_single_image(gxs, gys, glabel)

        print('Image: ', image)
        print('MaskImage: ', mask_image)
        print('LiverMask: ', liver_mask)
        print('pixel_cls_weights is ', pixel_cls_weight)
        pixel_cls_weight = tf.expand_dims(pixel_cls_weight, axis=2)
        image, mask_image, liver_mask, pixel_cls_weight = preprocessing_factory.get_preprocessing(is_training=True,
                                                                                                  method='segmentation_with_weight')(
            image,
            mask_image,
            liver_mask,
            pixel_cls_weight)
        from preprocessing import tf_image
        # liver_mask = tf_image.resize_image(liver_mask, config.train_image_shape,
        #                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        #                                    align_corners=False)
        image = tf.identity(image, 'processed_image')
        mask_image = tf.identity(mask_image, 'processed_mask_image')
        liver_mask = tf.identity(liver_mask, 'processed_mask_image')
        pixel_cls_weight = tf.identity(pixel_cls_weight, 'processed_pixel_weight')
        
        # batch them
        with tf.name_scope(FLAGS.dataset_name + '_batch'):
            b_image, b_mask_image, b_liver_mask, b_pixel_cls_weight = \
                tf.train.batch(
                    [image, mask_image, liver_mask, pixel_cls_weight],
                    batch_size=config.batch_size_per_gpu,
                    num_threads=FLAGS.num_preprocessing_threads,
                    capacity=500)
        with tf.name_scope(FLAGS.dataset_name + '_prefetch_queue'):
            batch_queue = slim.prefetch_queue.prefetch_queue(
                [b_image, b_mask_image, b_liver_mask, b_pixel_cls_weight],
                capacity=50)
    return batch_queue    


def sum_gradients(clone_grads):                        
    averaged_grads = []
    for grad_and_vars in zip(*clone_grads):
        grads = []
        var = grad_and_vars[0][1]
        try:
            for g, v in grad_and_vars:
                assert v == var
                grads.append(g)
            grad = tf.add_n(grads, name = v.op.name + '_summed_gradients')
        except:
            import pdb
            pdb.set_trace()
        
        averaged_grads.append((grad, v))
    return averaged_grads


def create_clones(batch_queue):        
    with tf.device('/cpu:0'):
        global_step = slim.create_global_step()
        learning_rate = tf.constant(FLAGS.learning_rate, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # optimizer = tf.train.MomentumOptimizer(learning_rate,
        #                                        momentum=FLAGS.momentum, name='Momentum')

        tf.summary.scalar('learning_rate', learning_rate)
    # place clones
    pixel_link_loss = 0 # for summary only
    gradients = []
    for clone_idx, gpu in enumerate(config.gpus):
        do_summary = clone_idx == 0 # only summary on the first clone
        reuse = clone_idx > 0
        with tf.variable_scope(tf.get_variable_scope(), reuse = reuse):
            with tf.name_scope(config.clone_scopes[clone_idx]) as clone_scope:
                with tf.device(gpu) as clone_device:
                    b_image, b_mask_image, b_liver_mask, b_pixel_cls_weight = batch_queue.dequeue()
                    # random resize
                    random_scale_idx = tf.random_uniform([], minval=0, maxval=len(scales), dtype=tf.int32)
                    scales_tensor = tf.convert_to_tensor(np.asarray(scales, np.int32), tf.int32)
                    random_scale = scales_tensor[random_scale_idx]
                    tf.summary.scalar('random_scale', random_scale[0])
                    b_image = tf.image.resize_images(b_image, random_scale)
                    b_mask_image = tf.image.resize_images(b_mask_image, random_scale,
                                                          tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    b_liver_mask = tf.image.resize_images(b_liver_mask, random_scale,
                                                          tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    b_pixel_cls_weight = tf.image.resize_images(b_pixel_cls_weight, random_scale)

                    # net = pixel_link_symbol.PixelLinkNetModify(b_image, b_mask_image, is_training=True)
                    # net = UNet.UNet(b_image, None, is_training=False, decoder=FLAGS.decoder)

                    net = UNet.UNet(b_image, b_mask_image, b_liver_mask, is_training=True, decoder=FLAGS.decoder,
                                    update_center_flag=FLAGS.update_center, batch_size=FLAGS.batch_size,
                                    update_center_strategy=FLAGS.update_center_strategy,
                                    num_centers_k=FLAGS.num_centers_k,
                                    full_annotation_flag=FLAGS.full_annotation_flag,
                                    output_shape_tensor=random_scale, pixel_cls_weight=b_pixel_cls_weight)
                    net.build_loss(do_summary=do_summary)
                    if FLAGS.update_center:
                        if FLAGS.update_center_strategy == 1:
                            update_center_op = net.update_centers(alpha=0.5)
                        elif FLAGS.update_center_strategy == 2:
                            print 'update_center_strategy is 2'
                        else:
                            print('the update_center_strategy do not support!')
                            assert False
                    
                    # gather losses
                    losses = tf.get_collection(tf.GraphKeys.LOSSES, clone_scope)
                    print('losses are ', losses)
                    # binary cross entropy, dice, mse, center loss
                    if FLAGS.full_annotation_flag:
                        assert len(losses) == 2
                    elif FLAGS.update_center:
                        assert len(losses) == 4
                    else:
                        assert len(losses) == 3

                    total_clone_loss = tf.add_n(losses) / config.num_clones
                    pixel_link_loss += total_clone_loss

                    # gather regularization loss and add to clone_0 only
                    if clone_idx == 0:
                        regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                        total_clone_loss = total_clone_loss + regularization_loss
                    
                    # compute clone gradients
                    clone_gradients = optimizer.compute_gradients(total_clone_loss)
                    gradients.append(clone_gradients)
                    
    tf.summary.scalar('pixel_link_loss', pixel_link_loss)
    tf.summary.scalar('regularization_loss', regularization_loss)

    # add all gradients together
    # note that the gradients do not need to be averaged, because the average operation has been done on loss.
    averaged_gradients = sum_gradients(gradients)
    if FLAGS.update_center:
        with tf.control_dependencies([update_center_op]):
            apply_grad_op = optimizer.apply_gradients(averaged_gradients, global_step=global_step)
    else:
        with tf.control_dependencies([]):
            apply_grad_op = optimizer.apply_gradients(averaged_gradients, global_step=global_step)
    
    train_ops = [apply_grad_op]
    
    bn_update_op = util.tf.get_update_op()
    if bn_update_op is not None:
        train_ops.append(bn_update_op)
    
    # moving average
    if FLAGS.using_moving_average:
        tf.logging.info('using moving average in training, \
        with decay = %f'%(FLAGS.moving_average_decay))
        ema = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([apply_grad_op]):# ema after updating
            train_ops.append(tf.group(ema_op))
         
    train_op = control_flow_ops.with_dependencies(train_ops, pixel_link_loss, name='train_op')
    return train_op

    
def train(train_op):
    summary_op = tf.summary.merge_all()
    sess_config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True)
    if FLAGS.gpu_memory_fraction < 0:
        sess_config.gpu_options.allow_growth = True
    elif FLAGS.gpu_memory_fraction > 0:
        sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction
    if not FLAGS.full_annotation_flag:
        real_train_dir = FLAGS.train_dir + '-' + FLAGS.decoder + '-' + str(FLAGS.update_center_strategy)
    else:
        real_train_dir = FLAGS.train_dir + '-' + FLAGS.decoder
    if FLAGS.attention_flag:
        real_train_dir += '-attention'
    print('real train dir is ', real_train_dir)
    init_fn = util.tf.get_init_fn(checkpoint_path=FLAGS.checkpoint_path,
                                  train_dir=real_train_dir,
                                  ignore_missing_vars=FLAGS.ignore_missing_vars,
                                  checkpoint_exclude_scopes=checkpoint_exclude_scopes)

    saver = tf.train.Saver(max_to_keep = 500, write_version = 2)
    # print slim.get_variables_to_restore()
    slim.learning.train(
            train_op,
            logdir = real_train_dir,
            init_fn = init_fn,
            summary_op = summary_op,
            number_of_steps = FLAGS.max_number_of_steps,
            log_every_n_steps = FLAGS.log_every_n_steps,
            save_summaries_secs = 30,
            saver = saver,
            save_interval_secs = 2400,
            session_config = sess_config
    )


def main(_):
    dataset = config_initialization()   
    batch_queue = create_dataset_batch_queue(dataset)
    train_op = create_clones(batch_queue)
    train(train_op)


if __name__ == '__main__':
    tf.app.run()
