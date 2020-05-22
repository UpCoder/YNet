# -*- coding=utf-8 -*-
# weakly label version
# 在计算center的时候，不仅仅有两个center，而是根据block确定
import tensorflow as tf
from OHEM import OHNM_batch
import numpy as np
slim = tf.contrib.slim

MODEL_TYPE_vgg16 = 'vgg16'
MODEL_TYPE_vgg16_no_dilation = 'vgg16_no_dilation'
dice_coff = 1.0
ms_flag = False
FUSE_TYPE_cascade_conv1x1_upsample_sum = 'cascade_conv1x1_upsample_sum'
FUSE_TYPE_cascade_conv1x1_128_upsamle_sum_conv1x1_2 = \
                            'cascade_conv1x1_128_upsamle_sum_conv1x1_2'
FUSE_TYPE_cascade_conv1x1_128_upsamle_concat_conv1x1_2 = \
                            'cascade_conv1x1_128_upsamle_concat_conv1x1_2'
skip_connection_setting = {
    # 'vgg16': ['conv1_2','conv2_2', 'conv3_3', 'conv4_3', 'fc7'],
    'vgg16': ['conv2_2', 'conv3_3', 'conv4_3', 'fc7'],
    'res50': [],
}



def transpose2D(input_tensor, upsample_rate, transpose_kernel_size, use_batchnorm, is_training):
    print('decoder is tranpose2D')
    input_shape = input_tensor.get_shape().as_list()
    output = slim.conv2d_transpose(input_tensor, num_outputs=input_shape[-1], kernel_size=transpose_kernel_size,
                                   stride=upsample_rate, biases_initializer=None)
    if use_batchnorm:
        output = slim.batch_norm(output, is_training=is_training)
    output = tf.nn.relu(output)
    return output


def upsampling2D(input_tensor, upsample_rate):
    print('decoder is upsampling2D', input_tensor)
    sh = tf.shape(input_tensor)
    newShape = upsample_rate * sh[1:3]
    return tf.image.resize_images(input_tensor, newShape)


class TaskNetworkModuleV2(object):
    '''
    对比上一个版本，本版本有multi scale 的功能
    '''
    def __init__(self, input_tensors, output_dim, output_shape, arg_sc, name, decoder='upsampling', is_training=True,
                 hidden_dim=128, skip_connection=True, last_layer_activation=tf.nn.relu, dense_connection_flag=True,
                 learnable_merged_flag=True):
        last_output = None
        # regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
        # 从深层到浅层
        final_output = []
        different_scale_outputs = []
        with tf.variable_scope(name):
            with slim.arg_scope(arg_sc):
                for idx, input_tensor in enumerate(input_tensors):
                    if skip_connection is True:
                        if last_output is not None:
                            if learnable_merged_flag:
                                alpha = tf.get_variable('alpha_' + str(idx), shape=[], dtype=tf.float32,
                                                        initializer=tf.ones_initializer(),
                                                        regularizer=None)
                                tf.summary.scalar('alpha_' + str(idx), alpha)
                            if decoder == 'upsampling':
                                last_output = upsampling2D(last_output, 2)
                            elif decoder == 'transpose':
                                last_output = transpose2D(last_output, 2, (4, 4), True, is_training=is_training)
                            print('the last output is ', last_output)
                            print('the output is ', input_tensor)
                            if learnable_merged_flag:
                                output = tf.concat([input_tensor, alpha * last_output], axis=-1)
                            else:
                                output = tf.concat([input_tensor, last_output], axis=-1)
                        else:
                            output = input_tensor
                    elif last_output is not None:
                        # 没有skip connection
                        output = upsampling2D(last_output, 2)
                    else:
                        output = input_tensor

                    output = slim.conv2d(output, hidden_dim, kernel_size=1, stride=1,
                                         scope='level_' + str(idx) + '_1x1')
                    output = slim.conv2d(output, hidden_dim, kernel_size=3, stride=1,
                                         scope='level_' + str(idx) + '_3x3')
                    last_output = output
                    different_scale_outputs.append(last_output)
                    final_output.append(tf.image.resize_images(output, output_shape))
                if dense_connection_flag:
                    final_output = slim.conv2d(tf.concat(final_output, -1), hidden_dim * len(input_tensors) / 2,
                                               kernel_size=1, stride=1, scope='merged_1x1')
                else:
                    final_output = slim.conv2d(final_output[-1], hidden_dim, kernel_size=3, stride=1,
                                               scope='wo_dense_conn')
                final_output = slim.conv2d(final_output, hidden_dim * len(input_tensors) / 2,
                                           kernel_size=3, stride=1, scope='merged_3x3',
                                           activation_fn=last_layer_activation)
                self.final_feature_map = final_output
                final_output = slim.conv2d(final_output, output_dim,
                                           kernel_size=1, stride=1, scope='logits', activation_fn=None,
                                           normalizer_fn=None)
                self.output = final_output

class TaskNetworkModuleV3(object):
    '''
    对比上一个版本，更改了alpha的位置
    '''
    def __init__(self, input_tensors, output_dim, output_shape, arg_sc, name, decoder='upsampling', is_training=True,
                 hidden_dim=128, skip_connection=True, last_layer_activation=tf.nn.relu, dense_connection_flag=True,
                 learnable_merged_flag=True):
        print('execute TaskNetworkMudoleV3')
        last_output = None
        # regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
        # 从深层到浅层
        final_output = []
        different_scale_outputs = []
        print('input_tensors are ', input_tensors)
        with tf.variable_scope(name):
            with slim.arg_scope(arg_sc):
                for idx, input_tensor in enumerate(input_tensors):
                    if skip_connection is True:
                        if last_output is not None:
                            if decoder == 'upsampling':
                                last_output = upsampling2D(last_output, 2)
                            elif decoder == 'transpose':
                                last_output = transpose2D(last_output, 2, (4, 4), True, is_training=is_training)
                            print('the last output is ', last_output)
                            print('the output is ', input_tensor)
                            output = tf.concat([input_tensor, last_output], axis=-1)
                        else:
                            output = input_tensor
                    elif last_output is not None:
                        # 没有skip connection
                        output = upsampling2D(last_output, 2)
                    else:
                        output = input_tensor

                    output = slim.conv2d(output, hidden_dim, kernel_size=1, stride=1,
                                         scope='level_' + str(idx) + '_1x1')
                    output = slim.conv2d(output, hidden_dim, kernel_size=3, stride=1,
                                         scope='level_' + str(idx) + '_3x3')
                    last_output = output
                    different_scale_outputs.append(last_output)
                    if learnable_merged_flag:
                        alpha = tf.get_variable('alpha_' + str(idx), shape=[], dtype=tf.float32,
                                                initializer=tf.ones_initializer(),
                                                regularizer=None)
                        alpha = tf.nn.sigmoid(alpha)
                        tf.summary.scalar('alpha_' + str(idx), alpha)
                        final_output.append(tf.image.resize_images(output * (1 + alpha), output_shape))
                    else:
                        final_output.append(tf.image.resize_images(output, output_shape))
                if dense_connection_flag:
                    final_output = slim.conv2d(tf.concat(final_output, -1), hidden_dim * len(input_tensors) / 2,
                                               kernel_size=1, stride=1, scope='merged_1x1')
                else:
                    final_output = slim.conv2d(final_output[-1], hidden_dim, kernel_size=3, stride=1,
                                               scope='wo_dense_conn')
                final_output = slim.conv2d(final_output, hidden_dim * len(input_tensors) / 2,
                                           kernel_size=3, stride=1, scope='merged_3x3',
                                           activation_fn=last_layer_activation)
                self.final_feature_map = final_output
                final_output = slim.conv2d(final_output, output_dim,
                                           kernel_size=1, stride=1, scope='logits', activation_fn=None,
                                           normalizer_fn=None)
                self.output = final_output


class UNet(object):
    def __init__(self, inputs, mask_input, liver_mask_input, is_training, base_model='vgg16', decoder='upsampling',
                 update_center_flag=False, batch_size=4, init_center_value=None, similarity_alpha=1.0,
                 pixel_cls_weight=None, update_center_strategy=1, num_centers_k=2, full_annotation_flag=False,
                 output_shape_tensor=None, dense_connection_flag=True, learnable_merged_flag=True):
        print('inputs are ', inputs)
        print('mask_input', mask_input)
        print('liver_mask_input', liver_mask_input)
        print('is_training: ', is_training)
        print('dense connection flag is ', dense_connection_flag)
        print('learnable merged flag is ', learnable_merged_flag)
        self.inputs = inputs
        self.is_training = is_training
        if liver_mask_input is not None:
            self.liver_mask = tf.cast(liver_mask_input[:, :, :, 1], tf.bool)
            tf.summary.image('liver_mask', tf.cast(tf.expand_dims(self.liver_mask, axis=3), tf.float32), max_outputs=1)
        self.mask_input = mask_input
        self.base_model = base_model
        self.decoder = decoder
        self.batch_size = batch_size
        self.num_classes = 2
        self.hidden_dim = 128
        self.dense_connection_flag = dense_connection_flag
        self.learnable_merged_flag = learnable_merged_flag

        self.img_shape = output_shape_tensor
        self.pixel_cls_weight = pixel_cls_weight
        if self.pixel_cls_weight is not None:
            print('pixel_cls_weight is ', self.pixel_cls_weight), tf.cast(
                tf.squeeze(self.pixel_cls_weight, axis=-1) / tf.reduce_max(self.pixel_cls_weight,
                                                                           axis=[1, 2, 3]) * 255.,
                tf.uint8)
            vis_pixel_cls_weight = tf.map_fn(lambda (pixel_cls_weight, max_value): pixel_cls_weight / max_value,
                                             (self.pixel_cls_weight,
                                              tf.reduce_max(self.pixel_cls_weight, axis=[1, 2, 3])), dtype=tf.float32)
            vis_pixel_cls_weight = tf.cast(vis_pixel_cls_weight * 255.0, tf.uint8)
            tf.summary.image('pixel_weight', vis_pixel_cls_weight, max_outputs=1)
        self.stride = 2
        self.recovery_channal = 1
        self.similarity_alpha = similarity_alpha
        self.update_center_flag = update_center_flag
        self.init_center_value = init_center_value
        self.update_center_strategy = update_center_strategy
        self.num_centers_k = num_centers_k
        self.inputs_mask = tf.greater_equal(self.inputs[:, :, :, 1], 10.0 / 255.0)    # 生成input的mask
        tf.summary.image('inputs_mask', tf.expand_dims(tf.cast(self.inputs_mask, tf.float32), axis=3), max_outputs=1)
        self.kernel_sizes = [11, 15, 21]
        self.full_annotation_flag = full_annotation_flag

        self._build_network()
        self._up_down_layers()
        self._logits_to_scores()
        # if self.update_center_flag:
        #     self._compute_qij()
        #     self._compute_pij()

    def _build_network(self):
        import config
        if config.model_type == MODEL_TYPE_vgg16:
            from nets import vgg
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                weights_regularizer=slim.l2_regularizer(config.weight_decay),
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                biases_initializer=tf.zeros_initializer()):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    padding='SAME') as sc:
                    self.arg_scope = sc
                    self.net, self.end_points = vgg.basenet(
                        inputs=self.inputs, pooling='MAX')

        elif config.model_type == MODEL_TYPE_vgg16_no_dilation:
            from nets import vgg
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                weights_regularizer=slim.l2_regularizer(config.weight_decay),
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                biases_initializer=tf.zeros_initializer()):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    padding='SAME') as sc:
                    self.arg_scope = sc
                    self.net, self.end_points = vgg.basenet(
                        inputs=self.inputs, dilation=False, pooling='MAX')
        else:
            raise ValueError('model_type not supported:%s' % (config.model_type))

    def _up_down_layers(self):
        import config
        input_tensors = []
        for idx in range(0, len(skip_connection_setting[self.base_model]))[::-1]:  # [4, 3, 2, 1, 0]
            print('basemode: ', self.base_model)
            current_layer_name = skip_connection_setting[self.base_model][idx]
            current_layer = self.end_points[current_layer_name]
            input_tensors.append(current_layer)
        # 针对每个pixel进行分类，区分前景背景
        pixel_cls_module = TaskNetworkModuleV3(input_tensors, config.num_classes,
                                               self.img_shape / self.stride,
                                               self.arg_scope, name='pixel_cls', is_training=self.is_training,
                                               decoder=self.decoder, hidden_dim=self.hidden_dim,
                                               dense_connection_flag=self.dense_connection_flag,
                                               learnable_merged_flag=self.learnable_merged_flag)

        # 试图恢复pixel的值，提取real feature map，用于计算相似度
        if not self.full_annotation_flag:
            pixel_recovery_module = TaskNetworkModuleV3(input_tensors, self.recovery_channal,
                                                        self.img_shape / self.stride,
                                                        self.arg_scope, name='pixel_recovery', is_training=self.is_training,
                                                        decoder=self.decoder, hidden_dim=self.hidden_dim,
                                                        last_layer_activation=tf.nn.leaky_relu,
                                                        dense_connection_flag=self.dense_connection_flag,
                                                        learnable_merged_flag=self.learnable_merged_flag)

            # self.pixel_recovery_logits = tf.nn.sigmoid(pixel_recovery_module.output)
            self.pixel_recovery_logits = pixel_recovery_module.output
            self.pixel_recovery_features = pixel_recovery_module.final_feature_map
            self.pixel_recovery_features_num_channal = self.pixel_recovery_features.get_shape().as_list()[-1]
            # build center
            if self.update_center_flag and self.init_center_value is not None:
                self.centers = tf.get_variable('centers', [self.num_classes, self.pixel_recovery_features_num_channal],
                                               dtype=tf.float32,
                                               initializer=lambda shape, dtype, partition_info: np.asarray(
                                                   self.init_center_value,
                                                   np.float32),
                                               trainable=False)
            elif self.update_center_flag:
                self.centers = tf.get_variable('centers', [self.num_classes, self.pixel_recovery_features_num_channal],
                                               dtype=tf.float32,
                                               initializer=tf.zeros_initializer(),
                                               trainable=False)
            else:
                self.centers = tf.get_variable('centers', [self.batch_size, self.num_classes + 1,
                                                           self.pixel_recovery_features_num_channal],
                                               dtype=tf.float32, initializer=tf.truncated_normal_initializer(),
                                               trainable=False)
            tf.summary.scalar('sum_centers', tf.reduce_sum(self.centers))
        self.pixel_cls_logits = pixel_cls_module.output

        # tf.summary.histogram('centers', self.centers)
        print('pixel_cls_logits is ', self.pixel_cls_logits)
        if self.stride != 1:
            self.pixel_cls_logits = tf.image.resize_images(self.pixel_cls_logits, self.img_shape)
            if not self.full_annotation_flag:
                self.pixel_recovery_logits = tf.image.resize_images(self.pixel_recovery_logits,
                                                                    self.img_shape)
                self.pixel_recovery_features = tf.image.resize_images(self.pixel_recovery_features, self.img_shape)

    def _flat_pixel_values(self, values):
        shape = values.shape.as_list()
        values = tf.reshape(values, shape=[shape[0], -1, shape[-1]])
        return values

    def _logits_to_scores(self):
        self.pixel_cls_scores = tf.nn.softmax(self.pixel_cls_logits)
        tf.summary.scalar('mean_pixel_cls_score', tf.reduce_mean(self.pixel_cls_scores[:, :, :, 1]))
        tf.summary.image('pred_mask_float',
                         tf.cast(tf.expand_dims(self.pixel_cls_scores[:, :, :, 1], axis=3), tf.float32),
                         max_outputs=1)
        tf.summary.image('pred_mask',
                         tf.cast(tf.expand_dims(self.pixel_cls_scores[:, :, :, 1], axis=3) * 250.0, tf.uint8),
                         max_outputs=1)

        tf.summary.image('image', self.inputs, max_outputs=1)
        tf.summary.scalar('max_image', tf.reduce_max(self.inputs))
        if not self.full_annotation_flag:
            self.pixel_recovery_value = self.pixel_recovery_logits
            self.pixel_recovery_logits_flatten = self._flat_pixel_values(self.pixel_recovery_logits)
            self.pixel_recovery_value_flatten = self._flat_pixel_values(self.pixel_recovery_value)
            tf.summary.image('recovery_image', tf.cast(self.pixel_recovery_value * 255, tf.uint8), max_outputs=1)
            tf.summary.scalar('max_recovery_image', tf.reduce_max(self.pixel_recovery_value))
            tf.summary.scalar('min_recovery_image', tf.reduce_min(self.pixel_recovery_value))
            self.assign_label = tf.argmin(tf.concat(
                [tf.expand_dims(tf.reduce_sum(tf.square(self.pixel_recovery_features - self.centers[0]), axis=3), axis=3),
                 tf.expand_dims(tf.reduce_sum(
                     tf.square(self.pixel_recovery_features - self.centers[1]), axis=3), axis=3)], axis=3), axis=3)
            if self.mask_input is not None:
                tf.summary.image('assign_seg_label',
                                 tf.expand_dims(
                                     tf.cast((self.assign_label + 1) * 200 * tf.cast(self.mask_input[:, :, :, 1], tf.int64),
                                             tf.uint8), axis=3))

        if self.mask_input is not None:
            tf.summary.image('mask', self.mask_input * 200, max_outputs=1)
        self.pixel_cls_logits_flatten = \
            self._flat_pixel_values(self.pixel_cls_logits)

        self.pixel_cls_scores_flatten = \
            self._flat_pixel_values(self.pixel_cls_scores)

        tf.summary.scalar('max_pixel_cls_scores', tf.reduce_max(self.pixel_cls_scores_flatten[:, :, 1]))
        tf.summary.scalar('min_pixel_cls_scores', tf.reduce_min(self.pixel_cls_scores_flatten[:, :, 1]))
        tf.summary.scalar('pixel_cls_score_variance', tf.reduce_mean(
            (self.pixel_cls_scores_flatten - tf.reduce_mean(self.pixel_cls_scores_flatten[:, :, 1])) ** 2))

    def _get_weakly_label_loss(self):
        '''
        总体上来说还是l2 loss，是为了让类内更加聚集
        逐步的更新center
        :return:
        '''
        # TODO: 考虑是否是要选择hard example，因为现在在计算recovery loss的时候我们并没有选择hard example
        # TODO: 考虑给每个sample都建立对应的center，而不是基于整个batch计算center
        # 限制feature的大小，避免因为太大，而导致计算l2loss过大
        # pixel_recovery_features = tf.nn.sigmoid(self.pixel_recovery_features)
        pixel_recovery_features = self.pixel_recovery_features

        # 计算膨胀后的差值，组成neg masks
        depth = pixel_recovery_features.get_shape().as_list()[-1]
        kernel_size = 11
        kernel = tf.convert_to_tensor(np.zeros([kernel_size, kernel_size, 1], dtype=np.float32))
        print('pos_mask', self.pos_mask)
        dilation = tf.clip_by_value(
            tf.squeeze(tf.nn.dilation2d(tf.cast(tf.expand_dims(self.pos_mask, axis=3), tf.float32), filter=kernel,
                                        strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME'), axis=3), 0.0, 1.0)
        print('dilation is ', dilation)
        neg_masks = tf.logical_and(self.inputs_mask,
                                   tf.cast(tf.cast(dilation, tf.int32) - tf.cast(self.pos_mask, tf.int32),
                                           tf.bool))

        def _get_wealy_label_loss_sample(pixel_recovery_features_sample, centers_sample, pos_mask_sample,
                                         input_mask_sample, neg_mask_sample):
            # 计算负样本

            # neg_mask_sample = tf.logical_and(input_mask_sample, tf.logical_not(pos_mask_sample))

            # assign_label = tf.stop_gradient(assign_label)
            print('pos_mask_sample is ', pos_mask_sample, pixel_recovery_features_sample)
            assign_features = tf.gather(centers_sample, tf.cast(pos_mask_sample, tf.int32))
            l2loss = tf.reduce_sum((assign_features - pixel_recovery_features_sample) ** 2, axis=-1)
            l2loss_pos = l2loss * tf.cast(pos_mask_sample, tf.float32)
            # l2loss_neg = l2loss * tf.cast(self.selected_neg_mask, tf.float32)
            l2loss_neg = l2loss * tf.cast(neg_mask_sample, tf.float32)  # 使用全部的negative points
            l2loss = l2loss_pos + l2loss_neg
            n_points = tf.reduce_sum(tf.cast(pos_mask_sample, tf.float32)) + tf.reduce_sum(
                tf.cast(neg_mask_sample, tf.float32)) # 使用全部的negative points
            print('l2loss is ', l2loss)
            l2loss = tf.cond(n_points > 0, lambda: tf.reduce_mean(l2loss), lambda : 0.0)
            # l2loss = tf.reduce_sum(l2loss) / (
            #         tf.reduce_sum(tf.cast(self.pos_mask, tf.float32)) + tf.reduce_sum(
            #     tf.cast(self.selected_neg_mask, tf.float32)) + 1.0)
            print('l2loss is ', l2loss)
            return l2loss
        l2losses = tf.map_fn(lambda
                                 (pixel_recovery_features_sample, pos_mask_sample, input_mask_sample, neg_mask_sample):
                             _get_wealy_label_loss_sample(pixel_recovery_features_sample, self.centers,
                                                          pos_mask_sample, input_mask_sample, neg_mask_sample),
                             (pixel_recovery_features, self.pos_mask, self.inputs_mask, neg_masks),
                             dtype=tf.float32)
        tf.summary.image('dilation', tf.expand_dims(tf.cast(tf.cast(dilation, tf.int8) * 100, tf.uint8), axis=3),
                         max_outputs=1)
        tf.summary.image('neg_masks', tf.expand_dims(tf.cast(tf.cast(neg_masks, tf.int8) * 100, tf.uint8), axis=3),
                         max_outputs=1)
        return tf.reduce_mean(l2losses)

    def _get_weakly_label_loss_with_center(self):
        '''
        总体上来说还是l2 loss，是为了让类内更加聚集
        不停的更换center
        :return:
        '''
        # TODO: 考虑是否是要选择hard example，因为现在在计算recovery loss的时候我们并没有选择hard example
        # TODO: 考虑给每个sample都建立对应的center，而不是基于整个batch计算center

        # 先计算center
        def _get_dilated_kernel(kernel_size):
            '''
            返回进行kernel操作的5个模版 （1个是正常的dilated操作，还有四个是分别对四个方向进行单独进行dilated的操作）
            :param kernel_size:
            :return:  [5, kernel_size, kernel_size]
            '''
            kernel_whole = np.ones([kernel_size, kernel_size], np.uint8)
            half_size = kernel_size // 2
            kernel_left = np.copy(kernel_whole, np.uint8)
            kernel_left[:, half_size + 1:] = 0
            kernel_right = np.copy(kernel_whole, np.uint8)
            kernel_right[:, :half_size] = 0

            kernel_top = np.copy(kernel_whole, np.uint8)
            kernel_top[half_size + 1:, :] = 0

            kernel_bottom = np.copy(kernel_whole, np.uint8)
            kernel_bottom[:half_size, :] = 0

            return np.concatenate([
                np.expand_dims(kernel_whole, axis=0),
                np.expand_dims(kernel_left, axis=0),
                np.expand_dims(kernel_right, axis=0),
                np.expand_dims(kernel_top, axis=0),
                np.expand_dims(kernel_bottom, axis=0),
            ], axis=0)
        # TODO: 考虑给每个sample都建立对应的center，而不是基于整个batch计算center
        print('pos_mask', self.pos_mask)
        print('neg_mask', self.selected_neg_mask)
        print('pixel_recovery_feaures', self.pixel_recovery_features)
        # 限制feature的大小，避免因为太大，而导致计算l2loss过大
        pixel_recovery_features = tf.nn.sigmoid(self.pixel_recovery_features)
        # pixel_recovery_features = self.pixel_recovery_features

        def _update_single_sample(pixel_recovery_feature_sample, pos_mask_sample, input_mask_sample, liver_mask_sample,
                                  centers):
            '''
            针对每一个sample计算其类中心
            :param pixel_recovery_feature_sample: [W, H, C]
            :param pos_mask_sample: [W, H]
            :param input_mask_sample: [W, H]
            :param centers: [17, C]
            :return:
            '''
            # 一个三个sample，一个是肝脏区域，一个是pos区域，一个是黑色背景区域
            neg_mask_sample = liver_mask_sample
            # 所有正样本的特征 (bounding box内部)
            pos_features = tf.gather_nd(pixel_recovery_feature_sample, tf.where(pos_mask_sample))
            # 所有负样本的特征 (bounding box外部)
            neg_features = tf.gather_nd(pixel_recovery_feature_sample, tf.where(neg_mask_sample))
            # 计算均值，如果当前没有正样本，则返回保存的上一个batch内的正样本的均值
            pos_features = tf.cond(tf.equal(tf.shape(pos_features)[0], 0), lambda: tf.expand_dims(centers[0], axis=0),
                                   lambda: tf.reduce_mean(pos_features, axis=0, keepdims=True))
            # 计算均值，如果当前没有负样本，则返回保存的上一个batch内的负样本的均值
            neg_features = tf.cond(tf.equal(tf.shape(neg_features)[0], 0), lambda: tf.expand_dims(centers[1], axis=0),
                                   lambda: tf.reduce_mean(neg_features, axis=0, keepdims=True))

            blank_mask_sample = tf.logical_not(input_mask_sample)
            blank_features = tf.gather_nd(pixel_recovery_feature_sample, tf.where(blank_mask_sample))
            blank_features = tf.cond(tf.equal(tf.shape(blank_features)[0], 0),
                                     lambda: tf.expand_dims(centers[2], axis=0),
                                     lambda: tf.reduce_mean(blank_features, axis=0, keepdims=True))
            center_lists = []
            center_lists.append(pos_features)
            center_lists.append(neg_features)
            center_lists.append(blank_features)
            # idx = 3
            # kernels_ring_masks = []
            # for kernel_size in self.kernel_sizes:
            #     # 针对当前的kernel 大小，生成对应的kernel 操作模版
            #     kernels = _get_dilated_kernel(kernel_size)
            #     ring_masks = []
            #     whole_ring_mask = None
            #     for kernel in kernels:
            #         kernel = tf.convert_to_tensor(kernel, tf.float32)
            #         kernel = tf.expand_dims(kernel, 2)
            #         kernel = tf.expand_dims(kernel, 3)
            #         # 进行膨胀操作
            #         dilated_pos_mask = tf.nn.conv2d(
            #             tf.expand_dims(tf.expand_dims(tf.cast(pos_mask_sample, tf.float32), axis=0), axis=3),
            #             kernel, [1, 1, 1, 1], 'SAME')
            #         # 转化为二值
            #         dilated_pos_mask = tf.greater_equal(dilated_pos_mask, 1)
            #         # 删除多余的纬度，还是将其变成W*H的形式
            #         dilated_pos_mask = tf.squeeze(dilated_pos_mask, axis=3)
            #         dilated_pos_mask = tf.squeeze(dilated_pos_mask, axis=0)
            #         # 取ring 和 negative mask 都是True的地方
            #         ring_area_mask = tf.logical_and(dilated_pos_mask, neg_mask_sample)
            #         if whole_ring_mask is None:
            #             whole_ring_mask = ring_area_mask
            #         else:
            #             # 变成一边，而不是三边
            #             ring_area_mask = tf.logical_and(tf.logical_not(ring_area_mask), whole_ring_mask)
            #         ring_masks.append(tf.expand_dims(ring_area_mask, axis=0))
            #         ring_feature = tf.gather_nd(pixel_recovery_feature_sample, tf.where(ring_area_mask))
            #         ring_feature = tf.cond(tf.equal(tf.reduce_sum(tf.cast(ring_area_mask, tf.float32)), 0.0),
            #                                lambda: tf.expand_dims(centers[idx], axis=0),
            #                                lambda: tf.reduce_mean(ring_feature, axis=0, keepdims=True))
            #         idx += 1
            #         center_lists.append(ring_feature)
            #     kernels_ring_masks.append(tf.expand_dims(tf.concat(ring_masks, axis=0), axis=0))
            # print('center_lists is ', center_lists)
            updated_feature = tf.concat(center_lists, axis=0)
            # return updated_feature, tf.concat(kernels_ring_masks, axis=0)
            return updated_feature

        updated_feature = tf.map_fn(
            lambda (pixel_recovery_feature_sample, pos_mask_sample, input_mask_sample, liver_mask_sample, centers): _update_single_sample(
                pixel_recovery_feature_sample, pos_mask_sample, input_mask_sample, liver_mask_sample, centers),
            (pixel_recovery_features, self.pos_mask, self.inputs_mask, self.liver_mask, self.centers), dtype=tf.float32)


        def _get_wealy_label_loss_sample(pixel_recovery_features_sample, centers_sample, pos_mask_sample, input_mask_sample):
            # 计算负样本
            neg_mask_sample = tf.logical_and(input_mask_sample, tf.logical_not(pos_mask_sample))
            distance = tf.map_fn(
                lambda center_feature: tf.reduce_sum((pixel_recovery_features_sample - center_feature) ** 2, axis=-1),
                centers_sample, dtype=tf.float32)
            print('distance is ', distance)
            assign_label = tf.argmin(distance, axis=0)
            vis_tensor = tf.cast(255.0 - tf.cast(assign_label, tf.float32) * 70, tf.uint8)
            vis_tensor = tf.expand_dims(vis_tensor, axis=2)

            assign_label = tf.stop_gradient(assign_label)
            assign_features = tf.gather(centers_sample, assign_label)
            l2loss = tf.reduce_sum((assign_features - pixel_recovery_features_sample) ** 2, axis=-1)
            l2loss_pos = l2loss * tf.cast(pos_mask_sample, tf.float32)
            # l2loss_neg = l2loss * tf.cast(self.selected_neg_mask, tf.float32)
            l2loss_neg = l2loss * tf.cast(neg_mask_sample, tf.float32)  # 使用全部的negative points
            l2loss = l2loss_pos + l2loss_neg
            # n_points = tf.reduce_sum(tf.cast(self.pos_mask, tf.float32)) + tf.reduce_sum(
            #     tf.cast(self.selected_neg_mask, tf.float32))
            n_points = tf.reduce_sum(tf.cast(self.pos_mask, tf.float32)) + tf.reduce_sum(
                tf.cast(neg_mask_sample, tf.float32)) # 使用全部的negative points
            l2loss = tf.cond(n_points > 0, lambda: tf.reduce_sum(l2loss) / n_points, lambda : 0.0)
            # l2loss = tf.reduce_sum(l2loss) / (
            #         tf.reduce_sum(tf.cast(self.pos_mask, tf.float32)) + tf.reduce_sum(
            #     tf.cast(self.selected_neg_mask, tf.float32)) + 1.0)
            return l2loss, vis_tensor
        l2losses, vis_tensors = tf.map_fn(lambda
                                 (pixel_recovery_features_sample, centers_sample, pos_mask_sample, input_mask_sample):
                             _get_wealy_label_loss_sample(pixel_recovery_features_sample, centers_sample,
                                                          pos_mask_sample, input_mask_sample),
                             (pixel_recovery_features, updated_feature, self.pos_mask, self.inputs_mask),
                             dtype=(tf.float32, tf.uint8))
        tf.summary.image('assign_label', vis_tensors, max_outputs=1)
        return tf.reduce_mean(l2losses)


    def update_centers_V2(self):
        '''
        计算pos area和neg area 的feature 均值，根据此更新center
        相当于针对每个batch，我们都有一个center，避免不同sample之间，病灶有所差异
        :return:
        '''

        def _get_dilated_kernel(kernel_size):
            '''
            返回进行kernel操作的5个模版 （1个是正常的dilated操作，还有四个是分别对四个方向进行单独进行dilated的操作）
            :param kernel_size:
            :return:  [5, kernel_size, kernel_size]
            '''
            kernel_whole = np.ones([kernel_size, kernel_size], np.uint8)
            half_size = kernel_size // 2
            kernel_left = np.copy(kernel_whole, np.uint8)
            kernel_left[:, half_size + 1:] = 0
            kernel_right = np.copy(kernel_whole, np.uint8)
            kernel_right[:, :half_size] = 0

            kernel_top = np.copy(kernel_whole, np.uint8)
            kernel_top[half_size + 1:, :] = 0

            kernel_bottom = np.copy(kernel_whole, np.uint8)
            kernel_bottom[:half_size, :] = 0

            return np.concatenate([
                np.expand_dims(kernel_whole, axis=0),
                np.expand_dims(kernel_left, axis=0),
                np.expand_dims(kernel_right, axis=0),
                np.expand_dims(kernel_top, axis=0),
                np.expand_dims(kernel_bottom, axis=0),
            ], axis=0)
        # TODO: 考虑给每个sample都建立对应的center，而不是基于整个batch计算center
        print('pos_mask', self.pos_mask)
        print('neg_mask', self.selected_neg_mask)
        print('pixel_recovery_feaures', self.pixel_recovery_features)
        pixel_recovery_features = tf.nn.sigmoid(self.pixel_recovery_features)
        # pixel_recovery_features = self.pixel_recovery_features

        def _update_single_sample(pixel_recovery_feature_sample, pos_mask_sample, input_mask_sample, centers):
            '''
            针对每一个sample计算其类中心
            :param pixel_recovery_feature_sample: [W, H, C]
            :param pos_mask_sample: [W, H]
            :param input_mask_sample: [W, H]
            :param centers: [17, C]
            :return:
            '''
            neg_mask_sample = tf.logical_and(input_mask_sample, tf.logical_not(pos_mask_sample))
            # 所有正样本的特征 (bounding box内部)
            pos_features = tf.gather_nd(pixel_recovery_feature_sample, tf.where(pos_mask_sample))
            # 所有负样本的特征 (bounding box外部)
            neg_features = tf.gather_nd(pixel_recovery_feature_sample, tf.where(neg_mask_sample))
            # 计算均值，如果当前没有正样本，则返回保存的上一个batch内的正样本的均值
            pos_features = tf.cond(tf.equal(tf.shape(pos_features)[0], 0), lambda: tf.expand_dims(centers[0], axis=0),
                                   lambda: tf.reduce_mean(pos_features, axis=0, keepdims=True))
            # 计算均值，如果当前没有负样本，则返回保存的上一个batch内的负样本的均值
            neg_features = tf.cond(tf.equal(tf.shape(neg_features)[0], 0), lambda: tf.expand_dims(centers[1], axis=0),
                                   lambda: tf.reduce_mean(neg_features, axis=0, keepdims=True))

            blank_mask_sample = tf.logical_not(input_mask_sample)
            blank_features = tf.gather_nd(pixel_recovery_feature_sample, tf.where(blank_mask_sample))
            blank_features = tf.cond(tf.equal(tf.shape(blank_features)[0], 0),
                                     lambda: tf.expand_dims(centers[2], axis=0),
                                     lambda: tf.reduce_mean(blank_features, axis=0, keepdims=True))
            center_lists = []
            center_lists.append(pos_features)
            center_lists.append(neg_features)
            center_lists.append(blank_features)
            idx = 3
            kernels_ring_masks = []
            for kernel_size in self.kernel_sizes:
                # 针对当前的kernel 大小，生成对应的kernel 操作模版
                kernels = _get_dilated_kernel(kernel_size)
                ring_masks = []
                whole_ring_mask = None
                for kernel in kernels:
                    kernel = tf.convert_to_tensor(kernel, tf.float32)
                    kernel = tf.expand_dims(kernel, 2)
                    kernel = tf.expand_dims(kernel, 3)
                    # 进行膨胀操作
                    dilated_pos_mask = tf.nn.conv2d(
                        tf.expand_dims(tf.expand_dims(tf.cast(pos_mask_sample, tf.float32), axis=0), axis=3),
                        kernel, [1, 1, 1, 1], 'SAME')
                    # 转化为二值
                    dilated_pos_mask = tf.greater_equal(dilated_pos_mask, 1)
                    # 删除多余的纬度，还是将其变成W*H的形式
                    dilated_pos_mask = tf.squeeze(dilated_pos_mask, axis=3)
                    dilated_pos_mask = tf.squeeze(dilated_pos_mask, axis=0)
                    # 取ring 和 negative mask 都是True的地方
                    ring_area_mask = tf.logical_and(dilated_pos_mask, neg_mask_sample)
                    if whole_ring_mask is None:
                        whole_ring_mask = ring_area_mask
                    else:
                        # 变成一边，而不是三边
                        ring_area_mask = tf.logical_and(tf.logical_not(ring_area_mask), whole_ring_mask)
                    ring_masks.append(tf.expand_dims(ring_area_mask, axis=0))
                    ring_feature = tf.gather_nd(pixel_recovery_feature_sample, tf.where(ring_area_mask))
                    ring_feature = tf.cond(tf.equal(tf.reduce_sum(tf.cast(ring_area_mask, tf.float32)), 0.0),
                                           lambda: tf.expand_dims(centers[idx], axis=0),
                                           lambda: tf.reduce_mean(ring_feature, axis=0, keepdims=True))
                    idx += 1
                    center_lists.append(ring_feature)
                kernels_ring_masks.append(tf.expand_dims(tf.concat(ring_masks, axis=0), axis=0))
            print('center_lists is ', center_lists)
            updated_feature = tf.concat(center_lists, axis=0)
            return updated_feature, tf.concat(kernels_ring_masks, axis=0)

        updated_feature, kernels_ring_masks = tf.map_fn(
            lambda (pixel_recovery_feature_sample, pos_mask_sample, input_mask_sample, centers): _update_single_sample(
                pixel_recovery_feature_sample, pos_mask_sample, input_mask_sample, centers),
            (pixel_recovery_features, self.pos_mask, self.inputs_mask, self.centers), dtype=(tf.float32, tf.bool))
        center_update_op = tf.assign(self.centers, updated_feature)

        print kernels_ring_masks
        kernels_ring_masks = kernels_ring_masks[0]
        return center_update_op, kernels_ring_masks

    def update_centers(self, alpha):
        '''
        采用center loss的更新策略
        :param alpha:
        :return:
        '''
        # pixel_recovery_features = tf.nn.sigmoid(self.pixel_recovery_features)
        pixel_recovery_features = self.pixel_recovery_features
        print('centers are ', self.centers)
        # distance = tf.map_fn(
        #     lambda center_feature: tf.reduce_sum((pixel_recovery_features - center_feature) ** 2, axis=-1),
        #     self.centers, dtype=tf.float32)

        # assign_label = tf.argmin(distance, axis=0)
        # prior_cond = tf.logical_or(self.pos_mask, self.selected_neg_mask)
        # assign_label = tf.gather_nd(assign_label, tf.where(prior_cond))
        assign_label = tf.cast(self.pos_mask, tf.int32)

        # assign_features = tf.gather(self.centers, assign_label)
        # # pred_features = tf.gather_nd(pixel_recovery_features, tf.where(prior_cond))
        # pred_features = pixel_recovery_features
        # diff = tf.reshape(assign_features - pred_features, [-1, 256])
        # unique_label, unique_idx, unique_count = tf.unique_with_counts(tf.reshape(assign_label, [-1]))
        # appear_times = tf.gather(unique_count, unique_idx)
        # diff = diff / tf.expand_dims(tf.cast(1 + appear_times, tf.float32), axis=1)
        # diff = alpha * diff
        # centers_update_op = tf.scatter_sub(self.centers, tf.reshape(assign_label, [-1]), diff)
        # return centers_update_op

        assign_features = tf.gather(self.centers, assign_label)
        # pred_features = tf.gather_nd(pixel_recovery_features, tf.where(prior_cond))
        pred_features = pixel_recovery_features
        diff = assign_features - pred_features
        print('diff is ', diff)
        kernel_size = 11
        kernel = tf.convert_to_tensor(np.zeros([kernel_size, kernel_size, 1]), tf.float32)
        erosion = tf.clip_by_value(
            tf.squeeze(tf.nn.dilation2d(tf.cast(tf.expand_dims(self.pos_mask, axis=3), tf.float32), filter=kernel,
                                        strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME'), axis=3), 0.0, 1.0)
        neg_masks = tf.logical_and(self.inputs_mask,
                                   tf.cast(tf.cast(erosion, tf.int32) - tf.cast(self.pos_mask, tf.int32),
                                           tf.bool))
        selected_masks = tf.logical_or(tf.cast(self.pos_mask, tf.bool), tf.cast(neg_masks, tf.bool))
        selected_assign_label = tf.gather(tf.reshape(assign_label, [-1]),
                                          tf.where(tf.reshape(selected_masks, [-1]))[:, 0])
        selected_diff = tf.gather(tf.reshape(diff, [-1, 256]), tf.where(tf.reshape(selected_masks, [-1]))[:, 0])
        unique_label, unique_idx, unique_count = tf.unique_with_counts(tf.reshape(selected_assign_label, [-1]))
        appear_times = tf.gather(unique_count, unique_idx)
        selected_diff = selected_diff / tf.expand_dims(tf.cast(1 + appear_times, tf.float32), axis=1)
        selected_diff = alpha * selected_diff
        centers_update_op = tf.scatter_sub(self.centers, selected_assign_label, selected_diff)
        return centers_update_op

        # 计算选择的负样本
        # depth = pixel_recovery_features.get_shape().as_list()[-1]
        # kernel_size = 11
        # kernel = tf.convert_to_tensor(np.zeros([kernel_size, kernel_size, 1]), tf.float32)
        # erosion = tf.clip_by_value(
        #     tf.squeeze(tf.nn.dilation2d(tf.cast(tf.expand_dims(self.pos_mask, axis=3), tf.float32), filter=kernel,
        #                                 strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME'), axis=3), 0.0, 1.0)
        # neg_masks = tf.logical_and(self.inputs_mask,
        #                            tf.cast(tf.cast(erosion, tf.int32) - tf.cast(self.pos_mask, tf.int32),
        #                                    tf.bool))
        # selected_masks = tf.logical_or(tf.cast(self.pos_mask, tf.bool), tf.cast(neg_masks, tf.bool))
        # selected_masks_flatten = tf.reshape(selected_masks, [-1])
        # selected_label_flatten = tf.gather(tf.reshape(assign_label, [-1]), tf.where(selected_masks_flatten))
        # selected_center_features_flatten = tf.gather(tf.gather(self.centers, tf.reshape(assign_label, [-1])),
        #                                              tf.where(selected_masks_flatten)[:, 0])
        # selected_pred_features_flatten = tf.gather(tf.reshape(pixel_recovery_features, shape=[-1, depth]),
        #                                            tf.where(selected_masks_flatten)[:, 0])
        # print('selected_pred_features_flatten ', tf.reshape(pixel_recovery_features, shape=[-1, depth]),
        #       tf.where(selected_masks_flatten), selected_pred_features_flatten)
        # print('selected_center_features_flatten ', tf.gather(self.centers, assign_label),
        #       selected_center_features_flatten)
        # print('selected_label_flatten ', selected_label_flatten)
        # print('selected_masks_flatten ', selected_masks_flatten)
        # diff = tf.reshape(tf.square(selected_center_features_flatten - selected_pred_features_flatten), [-1, depth])
        # unique_label, unique_idx, unique_count = tf.unique_with_counts(tf.squeeze(selected_label_flatten, axis=1))
        # appear_times = tf.gather(unique_count, unique_idx)
        # print('diff is ', diff)
        # appear_times = tf.where(tf.equal(appear_times, 0), tf.ones_like(appear_times), appear_times)
        # diff = diff / tf.expand_dims(tf.cast(1 + appear_times, tf.float32), axis=1)
        # diff = alpha * diff
        # print('diff is ', diff)
        # centers_update_op = tf.scatter_sub(self.centers, tf.squeeze(selected_label_flatten, axis=1), diff)
        # return centers_update_op


    def build_cls_loss(self, batch_size, pos_mask, neg_mask, pos_mask_flatten, neg_mask_flatten, n_pos, do_summary, pixel_cls_loss_weight_lambda):
        from losses import loss_with_binary_dice
        pixel_cls_weight_flatten = tf.reshape(self.pixel_cls_weight, [batch_size, -1])
        with tf.name_scope('pixel_cls_loss'):
            def no_pos():
                return tf.constant(.0)

            def has_pos():
                print('the pixel_cls_logits_flatten is ', self.pixel_cls_logits_flatten)
                print('the pos_mask is ', pos_mask_flatten)
                pixel_cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.pixel_cls_logits_flatten,
                    labels=tf.cast(pos_mask_flatten, dtype=tf.int32))

                # return tf.reduce_mean(pixel_cls_loss)
                pixel_neg_scores = self.pixel_cls_scores_flatten[:, :, 0]
                shape = pos_mask.get_shape().as_list()
                selected_neg_pixel_mask = OHNM_batch(batch_size, pixel_neg_scores, pos_mask_flatten, neg_mask_flatten)

                tf.summary.image('selected_neg_mask_cls',
                                 tf.expand_dims(tf.cast(tf.reshape(selected_neg_pixel_mask, tf.shape(pos_mask)),
                                         tf.float32), axis=3), max_outputs=1)
                n_neg = tf.cast(tf.reduce_sum(selected_neg_pixel_mask), tf.float32)
                selected_neg_pixel_mask = tf.cond(tf.equal(n_neg, 0), lambda: neg_mask_flatten,
                                                  lambda: tf.cast(selected_neg_pixel_mask, tf.bool))
                n_neg = tf.cast(tf.reduce_sum(tf.cast(selected_neg_pixel_mask, tf.float32)), tf.float32)

                cur_pixel_cls_weights = tf.cast(pos_mask_flatten, tf.float32) + \
                                        tf.cast(selected_neg_pixel_mask, tf.float32)
                # instance balanced weight
                # cur_pixel_cls_weights = tf.cast(pixel_cls_weight_flatten, tf.float32) + \
                #                         tf.cast(selected_neg_pixel_mask, tf.float32)
                loss = tf.reduce_sum(pixel_cls_loss * cur_pixel_cls_weights) / (n_neg + n_pos)

                # shape = pos_mask.get_shape().as_list()
                # pixel_neg_scores = tf.abs(self.pixel_cls_scores_flatten[:, :, 0] - 0.5)
                # pixel_neg_loss = tf.cast(pixel_cls_loss * tf.cast(neg_mask_flatten, tf.float32), tf.float32)
                # selected_neg_pixel_mask = OHNM_batch(batch_size, pixel_neg_loss, pos_mask_flatten, neg_mask_flatten)
                # tf.summary.image('selected_neg_mask_cls',
                #                  tf.cast(tf.reshape(selected_neg_pixel_mask, [shape[0], shape[1], shape[2], 1]),
                #                          tf.float32), max_outputs=1)
                # cur_pixel_cls_weights = tf.cast(selected_neg_pixel_mask, tf.float32) + tf.cast(pos_mask_flatten,
                #                                                                                tf.float32)
                # # 排除可能有零的情况
                # loss = tf.cond(tf.equal(tf.reduce_sum(cur_pixel_cls_weights), 0), lambda: 0.0,
                #                lambda: tf.reduce_sum(pixel_cls_loss * cur_pixel_cls_weights) / tf.reduce_sum(
                #                    cur_pixel_cls_weights))

                return loss

            pixel_cls_loss = has_pos()
            # pixel_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            #     logits=self.pixel_cls_logits_flatten,
            #     labels=tf.cast(pos_mask_flatten, dtype=tf.int32)))
            pixel_cls_dice_loss, pixel_cls_dice = loss_with_binary_dice(self.pixel_cls_scores, pos_mask, axis=[1, 2])
            tf.add_to_collection(tf.GraphKeys.LOSSES, pixel_cls_loss * pixel_cls_loss_weight_lambda)
            tf.add_to_collection(tf.GraphKeys.LOSSES, pixel_cls_dice_loss * pixel_cls_loss_weight_lambda * dice_coff)
        return pixel_cls_loss, pixel_cls_dice, pixel_cls_dice_loss

    def build_loss(self, do_summary=True):
        """
        The loss consists of two parts: pixel_cls_loss + link_cls_loss,
            and link_cls_loss is calculated only on positive pixels
        """
        import config

        count_warning = tf.get_local_variable(
            name='count_warning', initializer=tf.constant(0.0))
        batch_size = config.batch_size_per_gpu
        background_label = config.background_label
        text_label = config.text_label
        pixel_link_neg_loss_weight_lambda = config.pixel_link_neg_loss_weight_lambda
        pixel_cls_loss_weight_lambda = config.pixel_cls_loss_weight_lambda


        # build the cls loss
        mask_input = tf.split(self.mask_input, num_or_size_splits=3, axis=-1)
        self.pos_mask = tf.squeeze(tf.equal(mask_input[1], text_label), axis=3)
        pos_mask_flatten = tf.reshape(self.pos_mask, [batch_size, -1])
        self.neg_mask = tf.squeeze(tf.equal(mask_input[1], background_label), axis=3)
        print('the pos_mask=', self.pos_mask)
        print('the neg_mask=', self.neg_mask)
        neg_mask_flatten = tf.reshape(self.neg_mask, [batch_size, -1])

        tf.summary.image('pos_mask', tf.cast(tf.expand_dims(self.pos_mask, axis=3), tf.float32), max_outputs=1)
        tf.summary.image('neg_mask', tf.cast(tf.expand_dims(self.neg_mask, axis=3), tf.float32), max_outputs=1)

        n_pos = tf.reduce_sum(tf.cast(self.pos_mask, dtype=tf.float32))
        pixel_cls_loss, pixel_cls_dice, pixel_cls_dice_loss = self.build_cls_loss(batch_size, self.pos_mask,
                                                                                  self.neg_mask,
                                                                                  pos_mask_flatten, neg_mask_flatten,
                                                                                  n_pos, do_summary,
                                                                                  pixel_cls_loss_weight_lambda)

        # build the recovery loss
        if not self.full_annotation_flag:
            # recovery_loss = tf.square((self.inputs[:, :, :, 1] - tf.squeeze(self.pixel_recovery_value, axis=3)))
            # recovery_loss_positive = tf.cond(tf.reduce_sum(tf.cast(self.pos_mask, tf.float32)) > 0, lambda: tf.reduce_sum(
            #     recovery_loss * tf.cast(self.pos_mask, tf.float32)) / tf.reduce_sum(
            #     tf.cast(self.pos_mask, tf.float32)), lambda: 0.0)
            # selected_neg_mask = OHNM_batch(self.batch_size, recovery_loss, self.pos_mask, self.neg_mask)
            # self.selected_neg_mask = tf.cast(selected_neg_mask, tf.bool)
            # tf.summary.image('selected_neg_mask_recovery', tf.cast(tf.expand_dims(selected_neg_mask, axis=3), tf.float32),
            #                  max_outputs=1)
            # recovery_loss_negative = tf.cond(tf.reduce_sum(tf.cast(self.selected_neg_mask, tf.float32)) > 0,
            #                                  lambda: tf.reduce_sum(
            #                                      recovery_loss * tf.cast(self.selected_neg_mask,
            #                                                              tf.float32)) / tf.reduce_sum(
            #                                      tf.cast(self.selected_neg_mask, tf.float32)), lambda: 0.0)
            # balanced_recovery_loss = 2.0 * recovery_loss_positive + recovery_loss_negative
            # final_recovery_loss = balanced_recovery_loss
            final_recovery_loss = 1.0 * tf.reduce_mean(
                tf.square((self.inputs[:, :, :, 1] - tf.squeeze(self.pixel_recovery_value, axis=3))))
            tf.add_to_collection(tf.GraphKeys.LOSSES, final_recovery_loss)


        # build the center loss
        if self.update_center_flag and not self.full_annotation_flag:
            print('has the weakly label loss with center')
            lambda_center_loss = 2.0
            if self.update_center_strategy == 2:
                center_loss = self._get_weakly_label_loss_with_center() * lambda_center_loss
            elif self.update_center_strategy == 1:
                center_loss = self._get_weakly_label_loss() * lambda_center_loss

            tf.add_to_collection(tf.GraphKeys.LOSSES, center_loss)
        else:
            print('has not the weakly label loss with center')

        if do_summary:
            tf.summary.scalar('pixel_cls_loss', pixel_cls_loss)
            tf.summary.scalar('pixel_cls_dice', pixel_cls_dice)
            tf.summary.scalar('pixel_cls_dice_loss', pixel_cls_dice_loss)
            if not self.full_annotation_flag:
                tf.summary.scalar('pixel_recovery_loss', final_recovery_loss)
            # tf.summary.scalar('threshold_tensor', threshold_tensor)
            if self.update_center_flag and not self.full_annotation_flag:
                tf.summary.scalar('center_loss', center_loss)
