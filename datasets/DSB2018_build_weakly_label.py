import os
import cv2
import numpy as np
from skimage.measure import label
from tqdm import tqdm


def convert2WeaklyLabelMask(dataset_dir):
    casenames = os.listdir(dataset_dir)
    for casename in tqdm(casenames):
        case_dir = os.path.join(dataset_dir, casename)
        whole_mask_path = os.path.join(case_dir, 'whole_mask.png')
        mask = cv2.imread(whole_mask_path)[:, :, 1]
        weakly_labeled_mask = np.zeros_like(mask, np.uint8)
        labeled_mask = label(mask, neighbors=8)
        for labeled_id in range(1, np.max(labeled_mask) + 1):
            cur_mask = np.asarray(labeled_mask == labeled_id, np.uint8)
            xs, ys = np.where(cur_mask != 0)
            min_xs = np.min(xs)
            max_xs = np.max(xs)
            min_ys = np.min(ys)
            max_ys = np.max(ys)
            cur_mask[min_xs: max_xs, min_ys: max_ys] = 1
            weakly_labeled_mask += cur_mask
        weakly_labeled_mask = np.asarray(weakly_labeled_mask != 0, np.uint8)
        cv2.imwrite(os.path.join(case_dir, 'weakly_label_whole_mask.png'), weakly_labeled_mask)
        cv2.imwrite(os.path.join(case_dir, 'weakly_label_whole_mask_show.png'), weakly_labeled_mask * 250)


if __name__ == '__main__':
    dataset_dir = '/home/give/Documents/dataset/data-science-bowl-2018/stage1_train'
    convert2WeaklyLabelMask(dataset_dir)