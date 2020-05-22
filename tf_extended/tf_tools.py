# -*- coding=utf-8 -*-
import tensorflow as tf


def showParametersInCkpt(ckpt_dir):
    '''
    显示出ckpt文件中的参数名称
    :param ckpt_dir:
    :return:
    '''
    from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
    latest_ckp = tf.train.latest_checkpoint(ckpt_dir)
    # print_tensors_in_checkpoint_file(latest_ckp, all_tensors=True, tensor_name='', all_tensor_names=True)
    from tensorflow.contrib.framework.python.framework import checkpoint_utils
    var_list = checkpoint_utils.list_variables(latest_ckp)
    for v in var_list:
        print(v)


if __name__ == '__main__':
    # ckpt_dir = '/home/give/PycharmProjects/weakly_label_segmentation/logs/1s_weakly_label-transpose'
    # showParametersInCkpt(ckpt_dir)

    tensor = tf.random_normal([4, 256, 256, 128])
    res = tf.extract_image_patches(tensor, [1, 5, 5, 1], [1, 1, 1, 1], [1, 1, 1, 1], 'VALID')
    print res
