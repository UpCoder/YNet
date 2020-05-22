# -*- coding=utf-8 -*-
import cv2
import os
from glob import glob
import numpy as np
from tqdm import tqdm
import nipy
MIN_AREA = 10


def image_expand(img, kernel_size=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    image = cv2.dilate(img, kernel)
    return image


def image_erode(img, kernel_size=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    image = cv2.erode(img, kernel)
    return image


def read_nii(file_path):
    return nipy.load_image(file_path).get_data()


def read_nii_with_header(file_path):
    img_obj = nipy.load_image(file_path)
    header_obj = img_obj.header
    res_dict = {}
    res_dict['voxel_spacing'] = [header_obj['srow_x'][0], header_obj['srow_y'][1], header_obj['srow_z'][2]]
    img_arr = img_obj.get_data()
    return img_arr, res_dict


def notExistsMake(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def convertCase2PNGV4(volume_path, seg_path, z_axis=5.0):
    volume, header = read_nii_with_header(volume_path)
    volume = np.asarray(volume, np.float32)
    max_v = 180
    min_v = -70
    volume[volume > max_v] = max_v
    volume[volume < min_v] = min_v
    volume -= np.mean(volume)
    min_v = np.min(volume)
    max_v = np.max(volume)
    interv = max_v - min_v
    volume = (volume - min_v) / interv

    z_axis_case = header['voxel_spacing'][-1]
    slice_num = int(z_axis / z_axis_case)
    if slice_num == 0:
        slice_num = 1

    seg = read_nii(seg_path)

    volume = volume * np.asarray(np.asarray(seg!=0, np.uint8))
    # print np.shape(volume), np.shape(seg)
    [_, _, channel] = np.shape(volume)
    imgs = []
    idxs = []
    slices = []
    i = 0

    pos_slice_num = np.sum(np.sum(np.sum(seg == 2, axis=0), axis=0) != 0)
    total_slice_num = np.shape(seg)[-1]
    print('pos_slice_num is ', pos_slice_num, total_slice_num)
    neg_rate = (3.0 * pos_slice_num) / total_slice_num # 正样本是负样本的
    liver_masks = []
    if neg_rate > 1.0:
        neg_rate = 1.0
    while i < channel:
        seg_slice = seg[:, :, i]
        if np.sum(seg_slice == 1) != 0:
            # 有tumor
            pre_slice = volume[:, :, i-slice_num:i]
            pre_slice = np.mean(pre_slice, axis=-1, keepdims=True)
            next_slice = volume[:, :, i+1: i+slice_num+1]
            next_slice = np.mean(next_slice, axis=-1, keepdims=True)

            mid_slice = volume[:, :, i]
            imgs.append(np.concatenate([pre_slice, np.expand_dims(mid_slice, 2), next_slice],axis=-1))

            idxs.append(i)
            binary_seg_slice = np.asarray(seg_slice == 2, np.uint8)
            binary_seg_slice = image_expand(image_erode(binary_seg_slice, kernel_size=5), kernel_size=5)
            slices.append(binary_seg_slice)
            liver_masks.append(np.asarray(seg[:, :, i] == 1, np.uint8))
            i += slice_num
        else:
            i += 1
    del volume, seg
    return imgs, idxs, slices, liver_masks


def convertDir2PNGV4(nii_dir, save_dir, z_axis=5.0):
    '''
    考虑z-axix的平衡
    在将short 边大于64的缩小0.1倍
    并且保存了mask
    :param nii_dir:
    :param save_dir:
    :param z_axis: Z轴解像度默认是5.0
    :return:
    '''
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    img_dir = os.path.join(save_dir, 'img')
    vis_gt_dir = os.path.join(save_dir, 'vis_gt')
    mask_gt_dir = os.path.join(save_dir, 'mask_gt')
    liver_mask_dir = os.path.join(save_dir, 'liver_mask')
    notExistsMake(img_dir)
    notExistsMake(vis_gt_dir)
    notExistsMake(mask_gt_dir)
    notExistsMake(liver_mask_dir)

    segmentation_paths = glob(os.path.join(nii_dir, 'segmentation-*.nii'))
    volume_paths = glob(os.path.join(nii_dir, 'volume-*.nii'))
    segmentation_paths.sort()
    volume_paths.sort()
    for (volume_path, seg_path) in tqdm(zip(volume_paths, segmentation_paths)):
        file_id = int(os.path.basename(seg_path).split('.')[0].split('-')[1])
        imgs, idxs, slices, liver_masks = convertCase2PNGV4(volume_path, seg_path, z_axis=z_axis)
        for img, idx, slice, liver_mask in zip(imgs, idxs, slices, liver_masks):
            print(os.path.join(img_dir, '%d-%d.png' % (file_id, idx)))
            cv2.imwrite(os.path.join(img_dir, '%d-%d.png' % (file_id, idx)), np.asarray(img * 255.0, np.uint8))
            cv2.imwrite(os.path.join(mask_gt_dir, '%d-%d.png' % (file_id, idx)),
                        np.asarray(slice * 1.0, np.uint8))
            cv2.imwrite(os.path.join(liver_mask_dir, '%d-%d.png' % (file_id, idx)),
                        np.asarray(liver_mask * 1.0, np.uint8))
            cv2.imwrite(os.path.join(vis_gt_dir, '%d-%d.png' % (file_id, idx)),
                        np.asarray(slice * 200.0, np.uint8))


if __name__ == '__main__':
    batch_name = 'Batch_2'
    convertDir2PNGV4(
        '/media/give/Seagate Expansion Drive/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training_' + batch_name,
        '/media/give/Seagate Expansion Drive/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/PNG/' + batch_name,
        z_axis=1.0)

