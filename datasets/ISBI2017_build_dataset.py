# -*- coding=utf-8 -*-
import cv2
import os
from glob import glob
import numpy as np
from tqdm import tqdm
from datasets.medicalImage import read_nii, image_erode, image_expand, read_nii_with_header
MIN_AREA = 10


def notExistsMake(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def convertCase2PNGV4(volume_path, seg_path, z_axis=5.0, short_edge=64):
    volume, header = read_nii_with_header(volume_path)
    # volume = np.transpose(volume, [1, 0, 2])
    volume = np.asarray(volume, np.float32)
    max_v = 300.
    min_v = -350.
    # max_v = 180
    # min_v = -70
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
    # print np.shape(volume), np.shape(seg)
    [_, _, channel] = np.shape(volume)
    imgs = []
    bboxes = []
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
        if np.sum(seg_slice == 2) != 0:
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
            # print np.max(binary_seg_slice)
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
    def _write_txt_file(boxes, file_path):
        with open(file_path, 'w') as gt_file:
            lines = []
            for box in boxes:
                line = 'Tumor %d %d %d %d\n' % (box[0], box[1], box[2], box[3])
                lines.append(line)
            gt_file.writelines(lines)
            gt_file.close()
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
        # imgs, bboxes, idxs = convertCase2PNG(volume_path, seg_path)
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


def convertMask2WeaklyLabel(gt_dir, weakly_label_dir, vis_weakly_label_dir):
    notExistsMake(weakly_label_dir)
    notExistsMake(vis_weakly_label_dir)
    from skimage.measure import label
    names = os.listdir(gt_dir)
    names = [name.split('.')[0] for name in names]
    for name in names:
        mask_path = os.path.join(gt_dir, name + '.png')
        mask = cv2.imread(mask_path)[:, :, 1]
        labeled_img = label(mask)
        weakly_label_mask = np.zeros_like(mask, np.uint8)
        for idx in range(1, np.max(labeled_img) + 1):
            xs, ys = np.where(labeled_img == idx)
            min_xs = np.min(xs)
            max_xs = np.max(xs)
            min_ys = np.min(ys)
            max_ys = np.max(ys)
            weakly_label_mask[min_xs: max_xs, min_ys: max_ys] = 1
        cv2.imwrite(os.path.join(weakly_label_dir, name + '.png'), np.asarray(weakly_label_mask, np.uint8))
        cv2.imwrite(os.path.join(vis_weakly_label_dir, name + '.png'), np.asarray(weakly_label_mask * 200, np.uint8))


def convertPNG2NPY(dataset_dir, save_dir):
    '''
    将PNG转成npy格式的数据，用于训练UNet和UNet++
    :param dataset_dir:
    :param save_dir:
    :return:
    '''
    img_dir = os.path.join(dataset_dir, 'img')
    mask_dir = os.path.join(dataset_dir, 'mask_gt')
    case_names = os.listdir(img_dir)
    imgs = []
    masks = []
    for case_name in tqdm(case_names):
        img_path = os.path.join(img_dir, case_name)
        mask_path = os.path.join(mask_dir, case_name)
        imgs.append(cv2.imread(img_path))
        masks.append(cv2.imread(mask_path))
    np.save(os.path.join(save_dir, 'imgs.npy'), np.asarray(imgs, np.uint8))
    np.save(os.path.join(save_dir, 'TumorMasks.npy'), np.asarray(masks, np.uint8))
    np.save(os.path.join(save_dir, 'casenames.npy'), np.asarray(case_names))


def convertNII2NPY(nii_dir, save_dir, suffix='nii', z_axis=5.0, target_name='liver'):
    img_nii_paths = glob(os.path.join(nii_dir, 'volume*.nii'))
    if target_name == 'liver':
        target_id = 1
    elif target_name == 'tumor':
        target_id = 2
    imgs = []
    names = []
    masks = []
    for img_nii_path in img_nii_paths:
        print(os.path.basename(img_nii_path))
        seg_nii_path = os.path.join(nii_dir, 'segmentation-' + os.path.basename(img_nii_path).split('.')[0].split('-')[
            1] + '.nii')
        volume, header = read_nii_with_header(img_nii_path)
        # volume = np.transpose(volume, [1, 0, 2])
        volume = np.asarray(volume, np.float32)
        max_v = 300.
        min_v = -350.
        # max_v = 180
        # min_v = -70
        volume[volume > max_v] = max_v
        volume[volume < min_v] = min_v
        volume -= np.mean(volume)
        min_v = np.min(volume)
        max_v = np.max(volume)
        interv = max_v - min_v
        volume = (volume - min_v) / interv

        z_axis_case = header['voxel_spacing'][-1]
        slice_num = np.ceil((1.0 * z_axis) / (1.0 * z_axis_case))
        if slice_num == 0:
            slice_num = 1
        slice_num = int(slice_num)

        seg = read_nii(seg_nii_path)

        [_, _, channel] = np.shape(volume)
        i = 0
        pos_slice_num = np.sum(np.sum(np.sum(seg == target_id, axis=0), axis=0) >= 20)
        neg_rate = pos_slice_num / channel  # 正样本的比例
        liver_masks = []
        if neg_rate > 1.0:
            neg_rate = 1.0

        while i < channel:
            seg_slice = seg[:, :, i]
            if np.sum(seg_slice == target_id) >= 20:
                # 病灶的大小大于20
                seg_slice = seg[:, :, i]

                if slice_num != 1:
                    mid_slice = []
                    for j in range(int(i - slice_num / 2), int(i + slice_num / 2)):
                        if j < 0:
                            cur_slice = volume[:, :, i]
                        elif j >= channel:
                            cur_slice = volume[:, :, i]
                        else:
                            cur_slice = volume[:, :, j]
                        mid_slice.append(cur_slice)
                    mid_slice = np.mean(mid_slice, axis=0, keepdims=True)
                else:
                    mid_slice = np.expand_dims(volume[:, :, i], axis=0)

                pre_slice = []
                pre_end = i - slice_num / 2
                for j in range(1, slice_num + 1):
                    z = pre_end - j
                    if z < 0:
                        z = 0
                    pre_slice.append(volume[:, :, z])
                next_slice = []
                next_start = i + slice_num / 2
                for j in range(1, slice_num + 1):
                    z = next_start + j
                    if z >= channel:
                        z = channel - 1
                    next_slice.append(volume[:, :, z])
                pre_slice = np.mean(pre_slice, axis=0, keepdims=True)
                next_slice = np.mean(next_slice, axis=0, keepdims=True)
                imgs.append(
                    np.transpose(np.concatenate([pre_slice, mid_slice, next_slice], axis=0),
                                 axes=[1, 2, 0]))

                names.append(os.path.basename(img_nii_path).split('.')[0].split('-')[1] + '-' + str(i))
                binary_seg_slice = np.asarray(seg_slice == target_id, np.uint8)
                binary_seg_slice = image_expand(image_erode(binary_seg_slice, kernel_size=5), kernel_size=5)
                # print np.max(binary_seg_slice)
                masks.append(binary_seg_slice)
                print('add positive, ', np.shape(imgs), np.shape(masks), np.shape(names))
                i += 2
            else:
                # if np.random.random() < (neg_rate / 2.):
                #     if slice_num != 1:
                #         mid_slice = []
                #         for j in range(int(i - slice_num / 2), int(i + slice_num / 2)):
                #             if j < 0:
                #                 cur_slice = volume[:, :, i]
                #             elif j >= channel:
                #                 cur_slice = volume[:, :, i]
                #             else:
                #                 cur_slice = volume[:, :, j]
                #             mid_slice.append(cur_slice)
                #         mid_slice = np.mean(mid_slice, axis=0, keepdims=True)
                #     else:
                #         mid_slice = np.expand_dims(volume[:, :, i], axis=0)
                #
                #     pre_slice = []
                #     pre_end = i - slice_num / 2
                #     for j in range(1, slice_num + 1):
                #         z = pre_end - j
                #         if z < 0:
                #             z = 0
                #         pre_slice.append(volume[:, :, z])
                #     next_slice = []
                #     next_start = i + slice_num / 2
                #     for j in range(1, slice_num + 1):
                #         z = next_start + j
                #         if z >= channel:
                #             z = channel - 1
                #         next_slice.append(volume[:, :, z])
                #     pre_slice = np.mean(pre_slice, axis=0, keepdims=True)
                #     next_slice = np.mean(next_slice, axis=0, keepdims=True)
                #     imgs.append(
                #         np.transpose(np.concatenate([pre_slice, mid_slice, next_slice], axis=0),
                #                      axes=[1, 2, 0]))
                #     names.append(os.path.basename(img_nii_path).split('.')[0].split('-')[1] + '-' + str(i))
                #     binary_seg_slice = np.asarray(seg_slice == 2, np.uint8)
                #     binary_seg_slice = image_expand(image_erode(binary_seg_slice, kernel_size=5), kernel_size=5)
                #     # print np.max(binary_seg_slice)
                #     masks.append(binary_seg_slice)
                #     print('add negative: ', np.shape(imgs), np.shape(masks), np.shape(names))
                # else:
                #     print('ignore ', i)
                i += 1
    state = np.random.get_state()
    np.random.shuffle(imgs)
    np.random.set_state(state)
    np.random.shuffle(masks)
    np.random.set_state(state)
    np.random.shuffle(names)
    print(np.shape(imgs), np.shape(masks), np.shape(names))
    np.save(os.path.join(save_dir, target_name + '_imgs_' + suffix + '.npy'), imgs)
    np.save(os.path.join(save_dir, target_name + '_masks_' + suffix + '.npy'), masks)
    np.save(os.path.join(save_dir, target_name + '_names_' + suffix + '.npy'), names)


if __name__ == '__main__':
    batch_name = 'Batch_1'
    # convertDir2PNGV4('/media/give/CBMIR/ld/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training_' + batch_name,
    #                  '/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/' + batch_name)
    # convertMask2WeaklyLabel(
    #     '/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/' + batch_name + '/mask_gt',
    #     '/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/' + batch_name + '/weakly_label_gt',
    #     '/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/' + batch_name + '/vis_weakly_label_gt'
    # )

    # convertPNG2NPY('/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/' + batch_name,
    #                '/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/' + batch_name)
    convertNII2NPY('/media/give/CBMIR/ld/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training_' + batch_name,
                   save_dir='/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/' + batch_name,
                   target_name='tumor')

