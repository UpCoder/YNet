# -*- coding=utf-8 -*-
# weakly label version
import tensorflow as tf
from OHEM import OHNM_batch
import numpy as np
slim = tf.contrib.slim

MODEL_TYPE_vgg16 = 'vgg16'
MODEL_TYPE_vgg16_no_dilation = 'vgg16_no_dilation'
dice_coff = 0.5
ms_flag = False
FUSE_TYPE_cascade_conv1x1_upsample_sum = 'cascade_conv1x1_upsample_sum'
FUSE_TYPE_cascade_conv1x1_128_upsamle_sum_conv1x1_2 = \
                            'cascade_conv1x1_128_upsamle_sum_conv1x1_2'
FUSE_TYPE_cascade_conv1x1_128_upsamle_concat_conv1x1_2 = \
                            'cascade_conv1x1_128_upsamle_concat_conv1x1_2'
skip_connection_setting = {
    'vgg16': ['conv1_2','conv2_2', 'conv3_3', 'conv4_3', 'fc7'],
    'res50': [],
}
def compute_EDM_tf(tensor):
    '''
    计算tensor的欧式距离
    :param tensor: N*C
    :return: N*N
    '''
    shape = tensor.get_shape().as_list()
    G = tf.matmul(tensor, tf.transpose(tensor, perm=[1, 0]))
    diag_tensor = tf.expand_dims(tf.diag_part(G), axis=0)
    H = tf.tile(diag_tensor, [shape[0], 1])
    D = tf.sqrt(H + tf.transpose(H, perm=[1, 0]) - 2. * G)
    return D


class GlobalSimilarityAttention:
    def __init__(self, feature_map, name, arg_sc):
        '''

        :param feature_map: N * H * W * C
        :param name:
        :param arg_sc:
        '''
        with tf.variable_scope(name):
            with slim.arg_scope(arg_sc):
                shape = feature_map.get_shape().as_list()
                feature_map = tf.reshape(feature_map, [shape[0], -1, shape[-1]])
                distance = tf.map_fn(lambda feature: compute_EDM_tf(feature), feature_map)
                print('distance is ', distance)
                alpha = tf.sigmoid(distance)
                print('the similarity matrix is ', alpha)
                print('the feature_map is ', feature_map)
                o = tf.matmul(alpha, feature_map)
                print('delta feature is ', o)
                beta = tf.get_variable('beta', [], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
                feature_map = beta * o + feature_map
                feature_map = tf.reshape(feature_map, shape)
                self.output_feature_map = feature_map


class LocalSimilarityAttention:
    def __init__(self, feature_map, k, name, arg_sc):
        with tf.variable_scope(name):
            with slim.arg_scope(arg_sc):
                shape = feature_map.get_shape().as_list()
                cropped_feature_map = feature_map[:, k//2:shape[1]-k//2, k//2:shape[2]-k//2, :]
                cropped_feature_map = tf.reshape(cropped_feature_map, [shape[0], -1, shape[-1]])
                patches = tf.extract_image_patches(feature_map, ksizes=[1, k, k, 1], strides=[1, 1, 1, 1],
                                                   rates=[1, 1, 1, 1], padding='VALID')
                patches_shape = patches.get_shape().as_list()
                patches = tf.reshape(patches, [patches_shape[0], -1, k * k, shape[-1]])
                patches = tf.transpose(patches, perm=[0, 2, 1, 3])
                print('the patches is ', patches)
                print('the cropped_feature_map is ', cropped_feature_map)
                # 计算每个pixel和相邻的pixel 特征之间的距离
                distance = tf.map_fn(
                    lambda (feature_example, patch_example): tf.map_fn(
                        lambda patch_neighbor: tf.reduce_sum((feature_example - patch_neighbor) ** 2, axis=-1),
                        patch_example),
                    [cropped_feature_map, patches], dtype=tf.float32)

                print('distance is ', distance)
                alpha = tf.sigmoid(distance)
                o = tf.reduce_sum(patches * tf.expand_dims(alpha, axis=3), axis=1)
                beta = tf.get_variable('beta', [], tf.float32, initializer=tf.constant_initializer(0.0))
                cropped_feature_map = o * beta + cropped_feature_map
                print 'cropped_feature_map is ', cropped_feature_map
                output_feature_map = tf.reshape(cropped_feature_map, [shape[0], shape[1]-k+1, shape[2]-k+1, shape[3]])
                output_feature_map = tf.image.resize_images(output_feature_map, [shape[1], shape[2]])
                print('output_feature_map is ', output_feature_map)
                self.output_feature_map = output_feature_map


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
    print('decoder is upsampling2D')
    input_shape = input_tensor.get_shape().as_list()
    return tf.image.resize_images(input_tensor, [input_shape[1] * upsample_rate, input_shape[2] * upsample_rate])


class TaskNetworkModuleV2(object):
    '''
    对比上一个版本，本版本有multi scale 的功能
    '''
    def __init__(self, input_tensors, output_dim, output_shape, arg_sc, name, decoder='upsampling', is_training=True, hidden_dim=128):
        last_output = None
        # regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
        # 从深层到浅层
        final_output = []
        learnable_merged_flag = True
        different_scale_outputs = []
        with tf.variable_scope(name):
            with slim.arg_scope(arg_sc):
                for idx, input_tensor in enumerate(input_tensors):
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
                    output = slim.conv2d(output, hidden_dim, kernel_size=1, stride=1,
                                         scope='level_' + str(idx) + '_1x1')
                    output = slim.conv2d(output, hidden_dim, kernel_size=3, stride=1,
                                         scope='level_' + str(idx) + '_3x3')
                    last_output = output
                    different_scale_outputs.append(last_output)
                    final_output.append(tf.image.resize_images(output, output_shape))
                final_output = slim.conv2d(tf.concat(final_output, -1), hidden_dim * len(input_tensors) / 2,
                                           kernel_size=1, stride=1, scope='merged_1x1')
                final_output = slim.conv2d(final_output, hidden_dim * len(input_tensors) / 2,
                                           kernel_size=3, stride=1, scope='merged_3x3')
                local_similarity_attention = LocalSimilarityAttention(final_output, k=3, name='final_out',
                                                                      arg_sc=arg_sc)
                final_output = local_similarity_attention.output_feature_map
                self.final_feature_map = final_output
                final_output = slim.conv2d(final_output, output_dim,
                                           kernel_size=1, stride=1, scope='logits', activation_fn=None,
                                           normalizer_fn=None)
                self.output = final_output


class UNet(object):
    def __init__(self, inputs, mask_input, is_training, base_model='vgg16', decoder='upsampling',
                 update_center_flag=False, batch_size=4, init_center_value=None, similarity_alpha=1.0,
                 update_center_strategy=1):
        self.inputs = inputs
        self.is_training = is_training
        self.mask_input = mask_input
        self.base_model = base_model
        self.decoder = decoder
        self.batch_size = batch_size
        self.num_classes = 2
        self.hidden_dim = 128
        self.img_size = 256
        self.stride = 1
        self.recovery_channal = 1
        self.similarity_alpha = similarity_alpha
        self.update_center_flag = update_center_flag
        self.init_center_value = init_center_value
        self.update_center_strategy = update_center_strategy

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
        pixel_cls_module = TaskNetworkModuleV2(input_tensors, config.num_classes,
                                               [self.img_size / self.stride, self.img_size / self.stride],
                                               self.arg_scope, name='pixel_cls', is_training=self.is_training,
                                               decoder=self.decoder, hidden_dim=self.hidden_dim)

        # 试图恢复pixel的值，提取real feature map，用于计算相似度
        pixel_recovery_module = TaskNetworkModuleV2(input_tensors, self.recovery_channal,
                                                    [self.img_size / self.stride, self.img_size / self.stride],
                                                    self.arg_scope, name='pixel_recovery', is_training=self.is_training,
                                                    decoder=self.decoder, hidden_dim=self.hidden_dim)

        self.pixel_cls_logits = pixel_cls_module.output
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
        else:
            self.centers = tf.get_variable('centers', [self.num_classes, self.pixel_recovery_features_num_channal],
                                           dtype=tf.float32, initializer=tf.truncated_normal_initializer(),
                                           trainable=False)
        tf.summary.scalar('sum_centers', tf.reduce_sum(self.centers))
        # tf.summary.histogram('centers', self.centers)
        if self.stride != 1:
            self.pixel_cls_logits = tf.image.resize_images(self.pixel_cls_logits, [self.img_size, self.img_size])
            self.pixel_recovery_logits = tf.image.resize_images(self.pixel_recovery_logits,
                                                                [self.img_size, self.img_size])

    def _flat_pixel_values(self, values):
        shape = values.shape.as_list()
        values = tf.reshape(values, shape=[shape[0], -1, shape[-1]])
        return values

    def _logits_to_scores(self):
        self.pixel_cls_scores = tf.nn.softmax(self.pixel_cls_logits)
        # self.pixel_recovery_value = tf.nn.sigmoid(self.pixel_recovery_logits)
        self.pixel_recovery_value = self.pixel_recovery_logits

        tf.summary.image('pred_mask', tf.expand_dims(self.pixel_cls_scores[:, :, :, 1], axis=3) * 200.0, max_outputs=1)
        tf.summary.image('image', self.inputs, max_outputs=1)
        tf.summary.scalar('max_image', tf.reduce_max(self.inputs))
        tf.summary.image('recovery_image', self.pixel_recovery_value, max_outputs=1)
        tf.summary.scalar('max_recovery_image', tf.reduce_max(self.pixel_recovery_value))
        tf.summary.scalar('min_recovery_image', tf.reduce_min(self.pixel_recovery_value))
        if self.mask_input is not None:
            tf.summary.image('mask', self.mask_input * 200, max_outputs=1)
        self.pixel_cls_logits_flatten = \
            self._flat_pixel_values(self.pixel_cls_logits)
        self.pixel_recovery_logits_flatten = self._flat_pixel_values(self.pixel_recovery_logits)

        self.pixel_cls_scores_flatten = \
            self._flat_pixel_values(self.pixel_cls_scores)
        self.pixel_recovery_value_flatten = self._flat_pixel_values(self.pixel_recovery_value)

    def _get_weakly_label_loss(self):
        '''
        总体上来说还是l2 loss，是为了让类内更加聚集
        :return:
        '''
        # pixel_recovery_features = tf.nn.sigmoid(self.pixel_recovery_features)
        pixel_recovery_features = self.pixel_recovery_features
        distance = tf.map_fn(
            lambda center_feature: tf.reduce_sum((pixel_recovery_features - center_feature) ** 2, axis=-1),
            self.centers, dtype=tf.float32)
        assign_label = tf.argmin(distance, axis=0)
        vis_tensor = assign_label + 1
        vis_tensor = tf.cast(vis_tensor * tf.cast(tf.logical_or(self.pos_mask, self.selected_neg_mask), tf.int64) * 100,
                             tf.uint8)
        tf.summary.image('assign_label', tf.expand_dims(vis_tensor, axis=3), max_outputs=1)
        assign_label = tf.stop_gradient(assign_label)
        assign_features = tf.gather(self.centers, assign_label)
        l2loss = tf.reduce_sum((assign_features - pixel_recovery_features) ** 2, axis=-1)
        l2loss_pos = l2loss * tf.cast(self.pos_mask, tf.float32)
        l2loss_neg = l2loss * tf.cast(self.selected_neg_mask, tf.float32)
        l2loss = l2loss_pos + l2loss_neg
        l2loss = tf.reduce_sum(l2loss) / (
                tf.reduce_sum(tf.cast(self.pos_mask, tf.float32)) + tf.reduce_sum(
            tf.cast(self.selected_neg_mask, tf.float32)))
        return l2loss

    def update_centers_V2(self):
        '''
        计算pos area和neg area 的feature 均值，根据此更新center
        相当于针对每个batch，我们都有一个center，避免不同sample之间，病灶有所差异
        :return:
        '''
        # TODO: 扩展到每个simple都更新一个center，然后再去计算loss
        pos_features = tf.gather_nd(self.pixel_recovery_features, tf.where(self.pos_mask))
        neg_features = tf.gather_nd(self.pixel_recovery_features, tf.where(self.selected_neg_mask))
        pos_features = tf.reduce_mean(pos_features, axis=0, keepdims=True)
        neg_features = tf.reduce_mean(neg_features, axis=0, keepdims=True)
        updated_feature = tf.concat([pos_features, neg_features], axis=0)
        center_update_op = tf.assign(self.centers, updated_feature)
        return center_update_op

    def update_centers(self, alpha):
        '''
        采用center loss的更新策略
        :param alpha:
        :return:
        '''
        # pixel_recovery_features = tf.nn.sigmoid(self.pixel_recovery_features)
        pixel_recovery_features = self.pixel_recovery_features
        distance = tf.map_fn(
            lambda center_feature: tf.reduce_sum((pixel_recovery_features - center_feature) ** 2, axis=-1),
            self.centers, dtype=tf.float32)
        assign_label = tf.argmin(distance, axis=0)
        prior_cond = tf.logical_or(self.pos_mask, self.selected_neg_mask)
        # select_cond0 = tf.logical_and(prior_cond, tf.equal(assign_label, 0))
        # select_cond1 = tf.logical_and(prior_cond, tf.equal(assign_label, 1))
        # zero_mean_feature = tf.gather_nd(pixel_recovery_features, tf.where(select_cond0))
        # one_mean_feature = tf.gather_nd(pixel_recovery_features, tf.where(select_cond1))
        # zero_mean_feature = tf.reduce_mean(zero_mean_feature, keepdims=True, axis=0)
        # zero_mean_feature = tf.where(tf.equal(tf.shape(zero_mean_feature)[0], 0),
        #                              tf.expand_dims(self.centers[0], axis=0), zero_mean_feature)
        #
        # one_mean_feature = tf.reduce_mean(one_mean_feature, keepdims=True, axis=0)
        # one_mean_feature = tf.where(tf.equal(tf.shape(one_mean_feature)[0], 0), tf.expand_dims(self.centers[1], axis=0),
        #                             one_mean_feature)
        # print zero_mean_feature, one_mean_feature
        # updated_feature = tf.concat([zero_mean_feature, one_mean_feature], axis=0)
        # centers_update_op = tf.assign(self.centers, updated_feature)

        assign_label = tf.gather_nd(assign_label, tf.where(prior_cond))
        assign_features = tf.gather(self.centers, assign_label)
        pred_features = tf.gather_nd(pixel_recovery_features, tf.where(prior_cond))
        diff = assign_features - pred_features
        unique_label, unique_idx, unique_count = tf.unique_with_counts(assign_label)
        appear_times = tf.gather(unique_count, unique_idx)
        diff = diff / tf.expand_dims(tf.cast(1 + appear_times, tf.float32), axis=1)
        diff = alpha * diff
        centers_update_op = tf.scatter_sub(self.centers, assign_label, diff)
        return centers_update_op

    def _compute_qij(self):
        similarity_abs = tf.map_fn(
            lambda center: tf.pow(
                (1 + tf.reduce_sum(tf.square(self.pixel_recovery_features - center), axis=3) / self.similarity_alpha),
                (self.similarity_alpha + 1.) / 2. * -1.0), self.centers)
        print 'similarity_abs is ', similarity_abs
        print 'centers is ', self.centers
        similarity_abs = tf.transpose(similarity_abs, [1, 2, 3, 0])
        similarity_rel = similarity_abs / tf.reduce_sum(similarity_abs, axis=3, keepdims=True) # 相对相似距离
        self.qij = similarity_rel

    def _compute_pij(self):
        # fj = tf.transpose(tf.reduce_sum(self.qij, axis=[0, 1, 2], keepdims=True), [3, 0, 1, 2])
        fj = tf.reduce_sum(self.qij, axis=[0, 1, 2])
        print 'fj is ', fj
        print 'self.qij transpose is ', tf.transpose(self.qij, [3, 0, 1, 2])
        pij_abs = tf.map_fn(lambda (single_fj, single_qij): tf.div(tf.square(single_qij), single_fj),
                            (fj, tf.transpose(self.qij, [3, 0, 1, 2])), dtype=tf.float32)
        pij_abs = tf.transpose(pij_abs, [1, 2, 3, 0])
        pij_rel = pij_abs / tf.reduce_sum(pij_abs, axis=3, keepdims=True)
        self.pij = pij_rel

    def build_cls_loss(self, batch_size, pos_mask, neg_mask, pos_mask_flatten, neg_mask_flatten, n_pos, do_summary, pixel_cls_loss_weight_lambda):
        from losses import loss_with_binary_dice
        with tf.name_scope('pixel_cls_loss'):
            def no_pos():
                return tf.constant(.0)

            def has_pos():
                print('the pixel_cls_logits_flatten is ', self.pixel_cls_logits_flatten)
                print('the pos_mask is ', pos_mask_flatten)
                pixel_cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.pixel_cls_logits_flatten,
                    labels=tf.cast(pos_mask_flatten, dtype=tf.int32))

                # pixel_neg_scores = self.pixel_cls_scores_flatten[:, :, 0]
                # selected_neg_pixel_mask = OHNM_batch(batch_size, pixel_neg_scores, pos_mask_flatten, neg_mask_flatten)
                #
                # cur_pixel_cls_weights = tf.cast(selected_neg_pixel_mask, tf.float32)
                # n_neg = tf.cast(tf.reduce_sum(selected_neg_pixel_mask), tf.float32)
                # loss = tf.reduce_sum(pixel_cls_loss * cur_pixel_cls_weights) / (n_neg + n_pos)

                # return loss
                return tf.reduce_mean(pixel_cls_loss)

            pixel_cls_loss = has_pos()
            pixel_cls_dice_loss, pixel_cls_dice = loss_with_binary_dice(self.pixel_cls_scores, pos_mask, axis=[1, 2])
            tf.add_to_collection(tf.GraphKeys.LOSSES, pixel_cls_loss * pixel_cls_loss_weight_lambda)
            tf.add_to_collection(tf.GraphKeys.LOSSES, pixel_cls_dice_loss * pixel_cls_loss_weight_lambda * dice_coff)
        return pixel_cls_loss, pixel_cls_dice, pixel_cls_dice_loss

    def compute_KL(self):
        X = tf.distributions.Categorical(probs=self.pij)
        Y = tf.distributions.Categorical(probs=self.qij)
        return tf.distributions.kl_divergence(X, Y)

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
        neg_mask_flatten = tf.reshape(self.neg_mask, [batch_size, -1])
        print('the pos_mask=', self.pos_mask)
        print('the neg_mask=', self.neg_mask)
        n_pos = tf.reduce_sum(tf.cast(self.pos_mask, dtype=tf.float32))
        pixel_cls_loss, pixel_cls_dice, pixel_cls_dice_loss = self.build_cls_loss(batch_size, self.pos_mask,
                                                                                  self.neg_mask,
                                                                                  pos_mask_flatten, neg_mask_flatten,
                                                                                  n_pos, do_summary,
                                                                                  pixel_cls_loss_weight_lambda)

        # build the selected mask
        # shape = self.pos_mask.get_shape().as_list()
        # random_tensor = tf.random_normal(shape)
        # threshold_tensor = tf.reduce_sum(tf.cast(self.pos_mask, tf.float32)) / (shape[0] * shape[1] * shape[2])
        # self.selected_neg_mask = tf.logical_and(self.neg_mask, tf.less(random_tensor, threshold_tensor))

        # build the recovery loss
        recovery_loss = tf.square((self.inputs[:, :, :, 1] - tf.squeeze(self.pixel_recovery_value, axis=3)))
        recovery_loss_positive = tf.reduce_sum(recovery_loss * tf.cast(self.pos_mask, tf.float32))/ tf.reduce_sum(
            tf.cast(self.pos_mask, tf.float32))
        selected_neg_mask = OHNM_batch(self.batch_size, recovery_loss, self.pos_mask, self.neg_mask)
        self.selected_neg_mask = tf.cast(selected_neg_mask, tf.bool)
        tf.summary.image('selected_neg_mask', tf.cast(tf.expand_dims(selected_neg_mask, axis=3), tf.float32),
                         max_outputs=1)
        recovery_loss_negative = tf.reduce_sum(recovery_loss * tf.cast(selected_neg_mask, tf.float32)) / tf.reduce_sum(
            tf.cast(selected_neg_mask, tf.float32))
        balanced_recovery_loss = 2.0 * recovery_loss_positive + recovery_loss_negative
        tf.add_to_collection(tf.GraphKeys.LOSSES, balanced_recovery_loss)


        # build the center loss
        if self.update_center_flag:
            lambda_center_loss = 2.0
            center_loss = self._get_weakly_label_loss() * lambda_center_loss
            tf.add_to_collection(tf.GraphKeys.LOSSES, center_loss)

            # kl_divergence = self.compute_KL()
            # selected_neg_mask = OHNM_batch(self.batch_size, kl_divergence, self.pos_mask, self.neg_mask)
            # prior_mask = tf.logical_or(self.pos_mask, tf.cast(selected_neg_mask, bool))
            # kl_divergence = tf.reduce_sum(self.compute_KL() * tf.cast(prior_mask, tf.float32))
            # tf.add_to_collection(tf.GraphKeys.LOSSES, kl_divergence)
            # tf.summary.image('kl_selected_neg_mask', tf.cast(tf.expand_dims(selected_neg_mask, axis=3), tf.float32),
            #                  max_outputs=1)
            # tf.summary.image('assign_label_0',
            #                  tf.expand_dims(self.qij[:, :, :, 0] * tf.cast(prior_mask, tf.float32), axis=3),
            #                  max_outputs=1)
            # tf.summary.image('assign_label_1',
            #                  tf.expand_dims(self.qij[:, :, :, 1] * tf.cast(prior_mask, tf.float32), axis=3),
            #                  max_outputs=1)
        if do_summary:
            tf.summary.scalar('pixel_cls_loss', pixel_cls_loss)
            tf.summary.scalar('pixel_cls_dice', pixel_cls_dice)
            tf.summary.scalar('pixel_cls_dice_loss', pixel_cls_dice_loss)
            tf.summary.scalar('pixel_recovery_loss', balanced_recovery_loss)
            # tf.summary.scalar('threshold_tensor', threshold_tensor)
            if self.update_center_flag:
                tf.summary.scalar('center_loss', center_loss)


if __name__ == '__main__':
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME') as sc:
            input_tensor = tf.random_normal([4, 128, 128, 320])
            global_attention_module = GlobalSimilarityAttention(input_tensor, 'gam', sc)
            local_attention_module = LocalSimilarityAttention(input_tensor, 5, 'lam', sc)

