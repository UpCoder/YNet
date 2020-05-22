# -*- coding=utf-8 -*-
import os
import cv2
import numpy as np
from glob import glob
import shutil


def look_color_img():
    color_img_path = '/home/give/Documents/dataset/data-science-bowl-2018/stage1_train/be771d6831e3f8f1af4696bc08a582f163735db5baf9906e4729acc6a05e1187/images/be771d6831e3f8f1af4696bc08a582f163735db5baf9906e4729acc6a05e1187.png'
    color_img = cv2.imread(color_img_path)
    print np.shape(color_img)


def countNumofBHImgs(img_dir='/home/give/Documents/dataset/data-science-bowl-2018/stage1_train'):
    '''
    统计黑白图像的个数
    :param img_dir:
    :return:
    '''
    case_names = os.listdir(img_dir)
    count = 0
    for case_name in case_names:
        img_path = glob(os.path.join(img_dir, case_name, 'images', '*.png'))[0]
        img = cv2.imread(img_path)
        if np.sum(img[:, :, 0] != img[:, :, 1]) == 0:
            count += 1
    print 'the number of black-white images is ', count


def copy2BHImages(original_dir, copy2dir):
    '''
    将黑白图像copy到另一个文件夹里面
    :param original_dir:
    :param copy2dir:
    :return:
    '''
    case_names = os.listdir(original_dir)
    count = 0
    for case_name in case_names:
        img_path = glob(os.path.join(original_dir, case_name, 'images', '*.png'))[0]
        img = cv2.imread(img_path)
        if np.sum(img[:, :, 0] != img[:, :, 1]) == 0:
            count += 1
            shutil.copytree(os.path.join(original_dir, case_name), os.path.join(copy2dir, case_name))


def execute_copy2BHImages():
    original_dir = '/home/give/Documents/dataset/data-science-bowl-2018/stage1_test'
    copy2dir = '/home/give/Documents/dataset/data-science-bowl-2018/stage1_test_BW'
    copy2BHImages(original_dir, copy2dir)


def delete_BHImages():
    '''
    删除某些BWImage，因为根据聚类效果发现，这两者不同
    :return:
    '''
    BWDir = '/home/give/Documents/dataset/data-science-bowl-2018/stage1_train_BW'
    kmeans_dir = '/home/give/Documents/dataset/data-science-bowl-2018/stage1_train_BW_kmeans'
    case_names = os.listdir(kmeans_dir)
    case_names = [case_name.split('.')[0] for case_name in case_names]
    for BWCaseName in os.listdir(BWDir):
        if BWCaseName not in case_names:
            print 'delete: ', BWCaseName
            shutil.rmtree(os.path.join(BWDir, BWCaseName))


if __name__ == '__main__':
    # look_color_img()
    # countNumofBHImgs()
    execute_copy2BHImages()
    # delete_BHImages()
