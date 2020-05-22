# -*- coding=utf-8 -*-
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from skimage.measure import label


def net_center_posprocessing(pred, centers, pixel_wise_feature, gt):
    '''
    根据net维护的center，得到seg的结果
    :param pred: [512, 512]
    :param centers: [2, 256]
    :param pixel_wise_feature: [512, 512, 256]
    :param gt: [512, 512]
    :return:
    '''
    # print(np.shape(pred), np.shape(centers), np.shape(pixel_wise_feature), np.shape(gt))

    from metrics import dice
    distances1 = np.sum((pixel_wise_feature - centers[0]) ** 2, axis=2, keepdims=True)
    distances2 = np.sum((pixel_wise_feature - centers[1]) ** 2, axis=2, keepdims=True)
    distances = np.concatenate([distances1, distances2], axis=2)
    optimized_pred = np.argmin(distances, axis=2)
    # print(np.max(optimized_pred), np.min(optimized_pred))
    res_pred = np.zeros_like(pred, np.uint8)
    res_dice = 0.0
    for i in range(2):
        final_optimized_pred = np.logical_and(
            np.asarray(pred == 1, np.bool),
            np.asarray(optimized_pred == i, np.bool)
        )
        cur_dice = dice(gt, np.asarray(final_optimized_pred, np.uint8))
        if res_dice <= cur_dice:
            res_dice = cur_dice
            res_pred = np.asarray(final_optimized_pred, np.uint8)
        else:
            continue
    return np.asarray(res_pred, np.uint8)


def cluster_postprocessing(pred, gt, features, k=2, method='kmeans'):
    '''
    利用聚类的方法对pred进行后处理
    :param pred: pred的模版，我们对pred的前景进行处理 [512, 512]
    :param gt: gt，用于计算cluster后的匹配关系 [512, 512]
    :param features: 特征 [512, 512, C]
    :param k: 2
    :param method: str, default: kmeans
    :return: processed_pred
    '''
    from metrics import dice
    def _compute_cluster(data, k, method='kmeans'):
        '''
        :param data:
        :param k:
        :return:
        '''
        if method == 'kmeans':
            k_mean_obj = KMeans(n_clusters=k, n_jobs=6, verbose=0, max_iter=500, tol=1e-6)
        elif method == 'spectral':
            k_mean_obj = SpectralClustering(n_clusters=k, n_jobs=6)
        elif method == 'gmm':
            k_mean_obj = GaussianMixture(n_components=k, tol=1e-6, max_iter=500, verbose=0)
        else:
            print('the method do not support!', method)
            assert False
        k_mean_obj.fit(data)
        return k_mean_obj
    new_mask = np.zeros_like(pred, np.uint8)
    labeled_pred = label(pred)
    for label_idx in range(1, np.max(labeled_pred) + 1):
        cur_connect_pred = np.asarray(labeled_pred == label_idx, np.uint8)
        cur_xs, cur_ys = np.where(cur_connect_pred == 1)
        cur_features = features[cur_xs, cur_ys, : ]
        k_mean_obj = _compute_cluster(cur_features, k, method)
        pred_label = k_mean_obj.predict(cur_features)
        cur_mask = np.zeros_like(gt, np.uint8)
        cur_mask[cur_xs, cur_ys] = (pred_label + 1)
        max_dice = -0.5
        target = -1
        for target_k in range(1, k + 1):
            cur_dice = dice(gt, cur_mask == target_k)
            if cur_dice > max_dice:
                max_dice = cur_dice
                target = target_k
        if max_dice == 0.0:
            min_num = 512 * 512
            for target_k in range(1, k + 1):
                num = np.sum(cur_mask == target_k)
                if num < min_num:
                    target = target_k
                    min_num = num
        new_mask[cur_xs, cur_ys] = ((pred_label + 1) == target)
    return new_mask
