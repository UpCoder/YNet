# -*- coding=utf-8 -*-
import numpy as np
import cv2
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
import os
from glob import glob
from tqdm import tqdm
import pickle


def evulate_acc(img_dir, save_dir, mapping_dict):
    from metrics import acc_supervised, dice, IoU
    case_names = os.listdir(save_dir)
    case_names = [case_name.split('.')[0] for case_name in case_names]
    pred = []
    gt = []
    dices = []
    IoUs = []
    for case_name in case_names:
        pred_img = cv2.imread(os.path.join(save_dir, case_name + '.png'))[:, :, 0]
        # print np.unique(pred_img)
        for key in mapping_dict.keys():
            pred_img[pred_img == key] = mapping_dict[key]
        mask_img = cv2.imread(glob(os.path.join(img_dir, case_name, 'whole_mask.png'))[0])[:, :, 0]
        dices.append(dice(mask_img, pred_img))
        IoUs.append(IoU(mask_img, pred_img))
        pred.extend(np.reshape(pred_img, [-1]))
        gt.extend(np.reshape(mask_img, [-1]))
        print case_name, dices[-1], IoUs[-1]
    gt = np.asarray(gt, np.uint8)
    pred = np.asarray(pred, np.uint8)
    print 'gt shape: ', np.shape(gt)
    print 'pred shape: ', np.shape(pred)
    print 'ACC is ',
    print acc_supervised(gt, pred)
    print 'mean of dice is ',
    print np.mean(dices)
    print 'mean of IoU is ',
    print np.mean(IoUs)


def unlabeled_kmeans():
    '''
    完全是无label的k-means
    :return:
    '''

    def load_data(img_dir):
        case_names = os.listdir(img_dir)
        sorted(case_names)
        features = []
        for case_name in case_names[:50]:
            img_path = glob(os.path.join(img_dir, case_name, 'images', '*.png'))[0]
            img = cv2.imread(img_path)
            features.extend(np.reshape(img, [-1, 3]))
        print np.shape(features)
        return np.asarray(features, np.float32)

    def compute_cluster(data):
        '''
        :param data:
        :return:
        '''
        k_mean_obj = KMeans(n_clusters=2, n_jobs=6, verbose=0)
        k_mean_obj.fit(data)
        return k_mean_obj

    def generate_generate_assign_label_map(img_dir, k_mean_obj, save_dir):
        case_names = os.listdir(img_dir)
        sorted(case_names)
        for case_name in tqdm(case_names):
            img_path = glob(os.path.join(img_dir, case_name, 'images', '*.png'))[0]
            img = cv2.imread(img_path)
            save_img_path = os.path.join(save_dir, case_name + '.png')
            features = np.reshape(img, [-1, 3])
            assign_label = k_mean_obj.predict(features)
            assign_label = np.reshape(assign_label, np.shape(img)[:2])
            cv2.imwrite(save_img_path, np.asarray((assign_label + 1) * 100, np.uint8))
    img_dir = '/home/give/Documents/dataset/data-science-bowl-2018/stage1_train_BW'
    save_dir = '/home/give/Documents/dataset/data-science-bowl-2018/stage1_train_BW_kmeans'
    features = load_data(img_dir)
    k_mean_obj = compute_cluster(features)
    generate_generate_assign_label_map(img_dir, k_mean_obj, save_dir)
    evulate_acc(img_dir, save_dir)


def weakly_label_kmeansV1():
    '''
    只聚类bounding box内的点，原始像素值
    mean of dice is  0.433258328603829
    mean of IoU is  0.34943736
    :return:
    '''
    def load_data(img_dir):
        case_names = os.listdir(img_dir)
        sorted(case_names)
        features = []
        for case_name in case_names[:50]:
            img_path = glob(os.path.join(img_dir, case_name, 'images', '*.png'))[0]
            img = cv2.imread(img_path)
            mask = cv2.imread(glob(os.path.join(img_dir, case_name, 'weakly_label_whole_mask.png'))[0])[:, :, 0]
            xs, ys = np.where(mask == 1)
            if len(xs) == 0:
                print case_name
                continue
            # print len(xs), len(ys), np.shape(mask), np.max(xs), np.min(xs), np.max(ys), np.min(ys)
            features.extend(np.reshape(img[xs, ys, :], [-1, 3]))
        print np.shape(features)
        return np.asarray(features, np.float32)

    def compute_cluster(data):
        '''
        :param data:
        :return:
        '''
        k_mean_obj = KMeans(n_clusters=2, n_jobs=6, verbose=1)
        k_mean_obj.fit(data)
        return k_mean_obj

    def generate_generate_assign_label_map(img_dir, k_mean_obj, save_dir):
        case_names = os.listdir(img_dir)
        sorted(case_names)
        for case_name in tqdm(case_names):
            img_path = glob(os.path.join(img_dir, case_name, 'images', '*.png'))[0]
            img = cv2.imread(img_path)
            mask = cv2.imread(glob(os.path.join(img_dir, case_name, 'weakly_label_whole_mask.png'))[0])[:, :, 0]
            save_pred = np.zeros_like(mask, np.uint8)
            save_img_path = os.path.join(save_dir, case_name + '.png')
            xs, ys = np.where(mask == 1)
            features = img[xs, ys, :]
            assign_label = k_mean_obj.predict(features)
            assign_label += 1
            save_pred[xs, ys] = assign_label
            # shape = np.shape(mask)
            # for i in range(shape[0]):
            #     for j in range(shape[1]):
            #         if mask[i,j] == 1:
            #             features.append(img[i, j])
            # assign_label = k_mean_obj.predict(features)
            # for i in range(shape[0]):
            #     for j in range(shape[1]):
            #         if mask[i, j] == 1:
            #             features.append(img[i, j])
            cv2.imwrite(save_img_path, np.asarray((save_pred) * 100, np.uint8))
    img_dir = '/home/give/Documents/dataset/data-science-bowl-2018/stage1_train_BW'
    save_dir = '/home/give/Documents/dataset/data-science-bowl-2018/stage1_train_BW_kmeans_wl'
    features = load_data(img_dir)
    kmeans_obj = compute_cluster(features)
    generate_generate_assign_label_map(img_dir, kmeans_obj, save_dir)
    evulate_acc(img_dir, save_dir, {100: 0, 200: 1})


def weakly_label_kmeansV2():
    '''
    只聚类bounding box内的点，利用auto encoder提取的特征
    mean of dice is  0.618238877320971
    mean of IoU is  0.5258171
    :return:
    '''
    def load_data(img_dir, feature_map_dir):
        case_names = os.listdir(img_dir)
        sorted(case_names)
        features = []
        for case_name in case_names[:50]:
            feature_map_path = glob(os.path.join(feature_map_dir, case_name + '.npy'))[0]
            feature_map = np.load(feature_map_path)
            features.extend(feature_map)
            print np.shape(features)
        print np.shape(features)
        return np.asarray(features, np.float32)

    def compute_cluster(data):
        '''
        :param data:
        :return:
        '''
        k_mean_obj = KMeans(n_clusters=2, n_jobs=6, verbose=1)
        k_mean_obj.fit(data)
        return k_mean_obj

    def generate_generate_assign_label_map(img_dir, feature_map_dir, k_mean_obj, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        case_names = os.listdir(img_dir)
        sorted(case_names)
        for case_name in tqdm(case_names):
            feature_map_path = os.path.join(feature_map_dir, case_name + '.npy')
            feature_map = np.load(feature_map_path)
            mask = cv2.imread(glob(os.path.join(img_dir, case_name, 'weakly_label_whole_mask.png'))[0])[:, :, 0]
            save_pred = np.zeros_like(mask, np.uint8)
            save_img_path = os.path.join(save_dir, case_name + '.png')
            xs, ys = np.where(mask == 1)
            features = feature_map
            assign_label = k_mean_obj.predict(features)
            assign_label += 1
            save_pred[xs, ys] = assign_label
            # shape = np.shape(mask)
            # for i in range(shape[0]):
            #     for j in range(shape[1]):
            #         if mask[i,j] == 1:
            #             features.append(img[i, j])
            # assign_label = k_mean_obj.predict(features)
            # for i in range(shape[0]):
            #     for j in range(shape[1]):
            #         if mask[i, j] == 1:
            #             features.append(img[i, j])
            cv2.imwrite(save_img_path, np.asarray((save_pred) * 100, np.uint8))
    img_dir = '/home/give/Documents/dataset/data-science-bowl-2018/stage1_train_BW'
    feature_map_dir = '/home/give/Documents/dataset/data-science-bowl-2018/weakly_label_result/DLSC_Net/recovery_feature_map'
    save_dir = '/home/give/Documents/dataset/data-science-bowl-2018/weakly_label_result/DLSC_Net/stage1_train_BW_kmeans_wl'
    kmeans_save_path = os.path.join(os.path.dirname(save_dir), 'kmeans.pkl')
    if os.path.exists(kmeans_save_path):
        with open(kmeans_save_path, 'rb') as f:
            kmeans_obj = pickle.load(f)
    else:
        features = load_data(img_dir, feature_map_dir)
        kmeans_obj = compute_cluster(features)
        with open(kmeans_save_path, 'wb') as f:
            pickle.dump(kmeans_obj, f)

    generate_generate_assign_label_map(img_dir, feature_map_dir, kmeans_obj, save_dir)
    evulate_acc(img_dir, save_dir, {100: 1, 200: 0})


def weakly_label_kmeansV3(pickle_path='/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_2/DLSC_0/recovery_feature_map/128-383.pickle', k=2):
    '''
    针对每个bounding box我们聚类
    :return:
    '''
    from metrics import dice, IoU
    def compute_cluster(data, k):
        '''
        :param data:
        :return:
        '''
        k_mean_obj = KMeans(n_clusters=k, n_jobs=6, verbose=0, max_iter=500, tol=1e-6)
        # k_mean_obj = SpectralClustering(n_clusters=k, n_jobs=6)
        # k_mean_obj = GaussianMixture(n_components=k, tol=1e-6, max_iter=500, verbose=0)
        k_mean_obj.fit(data)
        return k_mean_obj
    real_mask_dir = '/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_2/mask_gt'
    real_mask_path = os.path.join(real_mask_dir, os.path.basename(pickle_path).split('.')[0] + '.png')
    real_mask = cv2.imread(real_mask_path)[:, :, 1]
    with open(pickle_path, 'rb') as f:
        connect_objs = pickle.load(f)
    mask = np.zeros([512, 512], np.uint8)
    vis_mask = np.zeros([512, 512], np.uint8)
    # cv2.imwrite(os.path.basename(pickle_path).split('.')[0] + '_real_mask.png',
    #             np.asarray(real_mask * 200, np.uint8))
    for obj_idx, connect_obj in enumerate(connect_objs):
        cur_xs = connect_obj['xs']
        cur_ys = connect_obj['ys']
        cur_features = connect_obj['features']
        k_mean_obj = compute_cluster(cur_features, k)
        pred_label = k_mean_obj.predict(cur_features)
        # pred_label = k_mean_obj.fit_predict(cur_features)
        cur_mask = np.zeros([512, 512], np.uint8)
        cur_mask[cur_xs, cur_ys] = (pred_label + 1)
        # cv2.imwrite(os.path.basename(pickle_path).split('.')[0] + '_' + str(obj_idx) + '.png',
        #             np.asarray(cur_mask * 100, np.uint8))

        max_dice = -0.5
        target = -1
        for target_k in range(1, k + 1):
            cur_dice = dice(real_mask, cur_mask == target_k)
            if cur_dice > max_dice:
                target = target_k
                max_dice = cur_dice
        if max_dice == 0.0:
            # 找到个数最小的一类
            min_num = 512*512
            for target_k in range(1, k + 1):
                num = np.sum(cur_mask == target_k)
                if num < min_num:
                    target = target_k
                    min_num = num
        # print max_dice, target
        mask[cur_xs, cur_ys] = ((pred_label + 1) == target)
        vis_cur_mask = np.zeros([512, 512], np.uint8)
        for target_k in range(1, k+1):
            if target_k == target:
                vis_cur_mask[cur_mask == target] = 2
            else:
                vis_cur_mask[cur_mask == target_k] = 1
        vis_mask += vis_cur_mask
    from datasets.medicalImage import close_operation, fill_region, image_erode
    mask = close_operation(mask, kernel_size=11)
    # seg_pred_mask = open_operation(seg_pred_mask, kernel_size=11)
    # 填充空洞
    mask = fill_region(mask)
    dice = dice(real_mask, mask)
    IoU = IoU(real_mask, mask)

    cv2.imwrite(os.path.basename(pickle_path).split('.')[0] + '_cluster.png', np.asarray(vis_mask * 100, np.uint8))
    cv2.imwrite(os.path.basename(pickle_path).split('.')[0] + '.png', np.asarray(mask * 200, np.uint8))
    print os.path.basename(pickle_path), dice, IoU
    return dice, IoU, real_mask, mask


if __name__ == '__main__':
    # unlabeled_kmeans()
    # weakly_label_kmeansV1()
    weakly_label_kmeansV2()
    # paths = glob(os.path.join(
    #     '/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_2/DLSC_0/recovery_feature_map',
    #     '*.pickle'))
    # dices = []
    # IoUs = []
    # gts = []
    # preds = []
    # for path in paths:
    #     # if not path.endswith('28-48.pickle'):
    #     #     continue
    #     dice, IoU, gt, pred = weakly_label_kmeansV3(path, k=2)
    #     dices.append(dice)
    #     IoUs.append(IoU)
    #     gts.append(gt)
    #     preds.append(pred)
    # print np.mean(dices), np.mean(IoUs)
    # from metrics import dice, IoU
    # print dice(np.asarray(gts, np.uint8), np.asarray(preds, np.uint8)), IoU(np.asarray(gts, np.uint8),
    #                                                                         np.asarray(preds, np.uint8))