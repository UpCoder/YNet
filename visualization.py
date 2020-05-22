# -*- coding=utf-8 -*-
import numpy as np
from datasets.medicalImage import read_nii, read_mhd_image, save_mhd_image


def generate_visualization(gt_path, pred_path, save_path):
    gt = read_nii(gt_path)
    gt = np.transpose(gt, axes=[2, 1, 0])
    pred = read_mhd_image(pred_path)

    tp = np.asarray(np.logical_and(gt == 2, pred == 1), np.uint8) * 2
    fp = np.asarray(np.logical_and(gt != 2, pred == 1), np.uint8) * 6
    fn = np.asarray(np.logical_and(gt == 2, pred == 0), np.uint8) * 4
    vis = tp + fp + fn
    print(np.shape(vis))
    save_mhd_image(vis, save_path)


if __name__ == '__main__':

    for case_id in [27]:
        gt_path = '/Users/Shared/Previously Relocated Items/Security/工作/paper/2020/MICCAI/visualization/%d/segmentation-%d.nii' % (case_id, case_id)
        pred_path = '/Users/Shared/Previously Relocated Items/Security/工作/paper/2020/MICCAI/visualization/%d/pred_centers.mhd' % case_id
        vis_path = '/Users/Shared/Previously Relocated Items/Security/工作/paper/2020/MICCAI/visualization/%d/pred_centers_vis.mhd' % case_id
        generate_visualization(gt_path, pred_path, vis_path)

        pred_path = '/Users/Shared/Previously Relocated Items/Security/工作/paper/2020/MICCAI/visualization/%d/pred_kmeans.mhd' % case_id
        vis_path = '/Users/Shared/Previously Relocated Items/Security/工作/paper/2020/MICCAI/visualization/%d/pred_kmeans_vis.mhd' % case_id
        generate_visualization(gt_path, pred_path, vis_path)

        pred_path = '/Users/Shared/Previously Relocated Items/Security/工作/paper/2020/MICCAI/visualization/%d/pred_crf.mhd' % case_id
        vis_path = '/Users/Shared/Previously Relocated Items/Security/工作/paper/2020/MICCAI/visualization/%d/pred_crf_vis.mhd' % case_id
        generate_visualization(gt_path, pred_path, vis_path)

        pred_path = '/Users/Shared/Previously Relocated Items/Security/工作/paper/2020/MICCAI/visualization/%d/pred.mhd' % case_id
        vis_path = '/Users/Shared/Previously Relocated Items/Security/工作/paper/2020/MICCAI/visualization/%d/pred_vis.mhd' % case_id
        generate_visualization(gt_path, pred_path, vis_path)