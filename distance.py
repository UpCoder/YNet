# -*- coding=utf-8 -*-
import numpy as np
import tensorflow as tf


def test_tensorflow_func():
    def test_gather():
        with tf.Session() as sess:
            x_tensor = tf.concat([tf.ones([1, 10]), tf.zeros([1, 10])], axis=0)

            labels = tf.convert_to_tensor(np.random.random_integers(0, 1, size=[2, 5, 5]))
            features = tf.gather(x_tensor, labels)
            print features
            print sess.run(features)

    def test_mapfn():
        centers = tf.ones([2, 128], tf.float32)
        features = tf.random_normal([4, 256, 256, 128], dtype=tf.float32)
        distance = tf.map_fn(
            lambda center_feature: tf.reduce_sum((features - center_feature) ** 2, axis=-1),
            centers, dtype=tf.float32)
        print distance

    def test_unique_with_counts():
        assign_label = tf.convert_to_tensor(np.asarray([1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], np.uint8))
        unique_label, unique_idx, unique_count = tf.unique_with_counts(assign_label)
        appear_times = tf.gather(unique_count, unique_idx)
        with tf.Session() as sess:
            print sess.run(appear_times)
            print sess.run(unique_label)
            print sess.run(unique_idx)
            print sess.run(unique_count)

    def test_gatherV2():
        features = tf.random_normal([4, 256, 256, 128])
        assign_label = tf.convert_to_tensor(np.random.random_integers(0, 1, [4, 256, 256]))
        zero_mean_feature = tf.gather_nd(features, tf.where(tf.equal(assign_label, 0)))
        print zero_mean_feature

    def test_topk():
        features = tf.convert_to_tensor(np.asarray([0.1 * i for i in range(10)]))

        topk, _ = tf.nn.top_k(-features, k=tf.convert_to_tensor(5))
        threshold = topk[-1]
        print(threshold)
        with tf.Session() as sess:
            features_v, topk_v = sess.run([features, topk])
            print topk_v[-1]
            print features_v

    def test_dilation2d():
        gray_image = tf.convert_to_tensor(np.asarray(
            [[0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 1, 0, 0],
             [0, 0, 1, 1, 1, 0, 0],
             [0, 0, 1, 1, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0]]
            , np.uint8), tf.float32)
        gray_image = tf.expand_dims(tf.expand_dims(gray_image, axis=0), axis=3)
        print gray_image
        kernel = tf.convert_to_tensor(np.asarray([
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
        ], np.float32), tf.float32)
        kernel = tf.expand_dims(kernel, axis=2)
        kernel = tf.expand_dims(kernel, axis=3)
        dilated_image = tf.nn.conv2d(gray_image, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
        dilated_image = tf.cast(tf.greater_equal(dilated_image, 1), tf.uint8)
        print dilated_image
        with tf.Session() as sess:
            gray_image_v, dilated_image_v = sess.run([gray_image, dilated_image])
            gray_image_v = np.squeeze(gray_image_v)
            dilated_image_v = np.squeeze(dilated_image_v)
            print gray_image_v
            print dilated_image_v

    def test_get_kernel():
        def _get_dilated_kernel(kernel_size):
            '''
            返回进行kernel操作的5个模版 （1个是正常的dilated操作，还有四个是分别对四个方向进行单独进行dilated的操作）
            :param kernel_size:
            :return:  [5, kernel_size, kernel_size]
            '''
            kernel_whole = np.ones([kernel_size, kernel_size], np.uint8)
            half_size = kernel_size // 2
            kernel_left = np.copy(kernel_whole, np.uint8)
            kernel_left[:, half_size+1:] = 0
            kernel_right = np.copy(kernel_whole, np.uint8)
            kernel_right[:, :half_size] = 0

            kernel_top = np.copy(kernel_whole, np.uint8)
            kernel_top[half_size+1:, :] = 0

            kernel_bottom = np.copy(kernel_whole, np.uint8)
            kernel_bottom[:half_size, :] = 0

            return np.concatenate([
                np.expand_dims(kernel_whole, axis=0),
                np.expand_dims(kernel_left, axis=0),
                np.expand_dims(kernel_right, axis=0),
                np.expand_dims(kernel_top, axis=0),
                np.expand_dims(kernel_bottom, axis=0),
            ], axis=0)
        print _get_dilated_kernel(3)
    # test_gather()
    # test_mapfn()
    # test_unique_with_counts()
    # test_gatherV2()
    test_topk()
    # test_dilation2d()
    # test_get_kernel()


def compute_alpha(tensor, sess):
    D = compute_EDM_tf(tensor, sess)
    alpha = 1-tf.sigmoid(D)
    print sess.run(alpha)
    return alpha


def compute_EDM_tf(tensor, sess):
    '''
    计算tensor的欧式距离
    :param tensor: N*C
    :return: N*N
    '''
    shape = tensor.get_shape().as_list()
    G = tf.matmul(tensor, tf.transpose(tensor, perm=[1, 0]))
    print 'G is ', G, sess.run(G)
    diag_tensor = tf.expand_dims(tf.diag_part(G), axis=0)
    H = tf.tile(diag_tensor, [shape[0], 1])
    print 'diag_tensor is ', diag_tensor, sess.run(diag_tensor)
    print 'H tensor is ', H, sess.run(H)
    D = tf.sqrt(H + tf.transpose(H, perm=[1, 0]) - 2. * G)
    return D


def compute_squared_EDM_method4(X):
    # 获得矩阵都行和列，因为是行向量，因此一共有n个向量
    n,m = X.shape
    # 计算Gram 矩阵
    G = np.dot(X, X.T)
    print('G is ', G)
    # 因为是行向量，n是向量个数,沿y轴复制n倍，x轴复制一倍
    H = np.tile(np.diagonal(G), (n, 1))
    print('H is ', H)
    return np.sqrt(H + H.T - 2*G)


if __name__ == '__main__':
    # X = np.asarray([
    #     [1, 2, 3],
    #     [3, 4, 4]
    # ], np.float32)
    # print compute_squared_EDM_method4(X)
    #
    # with tf.Session() as sess:
    #     D_tensor = compute_EDM_tf(tf.convert_to_tensor(X), sess)
    #     print 'D is ', D_tensor, sess.run(D_tensor)
    #     compute_alpha(tf.convert_to_tensor(X), sess)

    test_tensorflow_func()