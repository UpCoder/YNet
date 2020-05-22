import os
import tensorflow as tf
from preprocessing import tf_image
from tensorflow.python.ops import random_ops


def preprocessing_training(image, mask, liver_mask, pixel_weight, prob=0.5):
    with tf.name_scope('preprocessing_training'):
        print('image is ', image)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # with tf.name_scope('rotate'):
        #     rnd = tf.random_uniform((), minval=0, maxval=1, name='rotate')
        #     def rotate():
        #         k = random_ops.random_uniform([], 0, 10000)
        #         k = tf.cast(k, tf.int32)
        #         if liver_mask is not None:
        #             return tf.image.rot90(image, k=k), tf.image.rot90(mask, k=k), tf.image.rot90(liver_mask,
        #                                                                                          k=k), tf.image.rot90(
        #                 pixel_weight, k=k)
        #         else:
        #             return tf.image.rot90(image, k=k), tf.image.rot90(mask, k=k), None, tf.image.rot90(pixel_weight,
        #                                                                                                k=k)
        #
        #     def no_rotate():
        #         return image, mask, liver_mask, pixel_weight
        #     image, mask, liver_mask, pixel_weight = tf.cond(tf.less(rnd, prob), rotate, no_rotate)

        with tf.name_scope('flip_left_right'):
            def flip_left_right():
                if liver_mask is not None:
                    return tf.image.flip_left_right(image), tf.image.flip_left_right(mask), tf.image.flip_left_right(
                        liver_mask), tf.image.flip_left_right(pixel_weight)
                else:
                    return tf.image.flip_left_right(image), tf.image.flip_left_right(
                        mask), None, tf.image.flip_left_right(pixel_weight)

            def no_flip_left_right():
                return image, mask, liver_mask, pixel_weight
            rnd = tf.random_uniform((), minval=0, maxval=1, name='flip_left_right')
            image, mask, liver_mask, pixel_weight = tf.cond(tf.less(rnd, prob), flip_left_right, no_flip_left_right)

        with tf.name_scope('flip_up_down'):
            def flip_up_down():
                if liver_mask is not None:
                    return tf.image.flip_up_down(image), tf.image.flip_up_down(mask), tf.image.flip_up_down(
                        liver_mask), tf.image.flip_up_down(pixel_weight)
                else:
                    return tf.image.flip_up_down(image), tf.image.flip_up_down(mask), None, tf.image.flip_up_down(
                        pixel_weight)

            def no_flip_up_down():
                return image, mask, liver_mask, pixel_weight

            rnd = tf.random_uniform((), minval=0, maxval=1, name='flip_up_down')
            image, mask, liver_mask, pixel_weight = tf.cond(tf.less(rnd, prob), flip_up_down, no_flip_up_down)

        # image = tf_image.resize_image(image, out_shape,
        #                               method=tf.image.ResizeMethod.BILINEAR,
        #                               align_corners=False)
        # mask = tf_image.resize_image(mask, out_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        #                              align_corners=False)
        # if liver_mask is not None:
        #     liver_mask = tf_image.resize_image(liver_mask, out_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        #                                        align_corners=False)
    return image, mask, liver_mask, pixel_weight


def preprocessing_val(image):
    with tf.name_scope('preprocessing_val'):
        print('image is ', image)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image


def segmentation_preprocessing(image, mask, liver_mask, pixel_weight, is_training):
    if is_training:
        return preprocessing_training(image, mask, liver_mask, pixel_weight)
    else:
        return preprocessing_val(image)