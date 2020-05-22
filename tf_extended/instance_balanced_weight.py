# -*- coding=utf-8 -*-
import tensorflow as tf
import numpy as np
import util
PIXEL_CLS_WEIGHT_all_ones = 'PIXEL_CLS_WEIGHT_all_ones'
PIXEL_CLS_WEIGHT_bbox_balanced = 'PIXEL_CLS_WEIGHT_bbox_balanced'
PIXEL_NEIGHBOUR_TYPE_4 = 'PIXEL_NEIGHBOUR_TYPE_4'
PIXEL_NEIGHBOUR_TYPE_8 = 'PIXEL_NEIGHBOUR_TYPE_8'

DECODE_METHOD_join = 'DECODE_METHOD_join'

def get_neighbours_8(x, y):
    """
    Get 8 neighbours of point(x, y)
    """
    return [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1), \
        (x - 1, y),                 (x + 1, y),  \
        (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]


def get_neighbours_4(x, y):
    return [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)]



def is_valid_cord(x, y, w, h):
    """
    Tell whether the 2D coordinate (x, y) is valid or not.
    If valid, it should be on an h x w image
    """
    return x >=0 and x < w and y >= 0 and y < h


def get_neighbours(x, y):
    neighbour_type = PIXEL_NEIGHBOUR_TYPE_8
    if neighbour_type == PIXEL_NEIGHBOUR_TYPE_4:
        return get_neighbours_4(x, y)
    else:
        return get_neighbours_8(x, y)


def tf_cal_gt_for_single_image(xs, ys, labels):
    pixel_cls_label, pixel_cls_weight = \
        tf.py_func(
                    cal_gt_for_single_image,
                    [xs, ys, labels],
                    [tf.int32, tf.float32]
                   )
    score_map_shape = [512, 512]
    pixel_cls_label.set_shape(score_map_shape)
    pixel_cls_weight.set_shape(score_map_shape)
    return pixel_cls_label, pixel_cls_weight


def cal_gt_for_single_image(normed_xs, normed_ys, labels):
    """
    Args:
        xs, ys: both in shape of (N, 4),
            and N is the number of bboxes,
            their values are normalized to [0,1]
        labels: shape = (N,), only two values are allowed:
                                                        -1: ignored
                                                        1: text
    Return:
        pixel_cls_label
        pixel_cls_weight
        pixel_link_label
        pixel_link_weight
    """
    import config
    score_map_shape = [512, 512]
    h, w = score_map_shape
    text_label = 1
    ignore_label = config.ignore_label
    background_label = 0
    bbox_border_width = config.bbox_border_width
    pixel_cls_border_weight_lambda = config.pixel_cls_border_weight_lambda

    # validate the args
    assert np.ndim(normed_xs) == 2
    assert np.shape(normed_xs)[-1] == 4
    assert np.shape(normed_xs) == np.shape(normed_ys)
    assert len(normed_xs) == len(labels)

    #     assert set(labels).issubset(set([text_label, ignore_label, background_label]))

    num_positive_bboxes = np.sum(np.asarray(labels) == text_label)
    # rescale normalized xys to absolute values
    xs = normed_xs * w
    ys = normed_ys * h

    # initialize ground truth values
    mask = np.zeros(score_map_shape, dtype=np.int32)
    pixel_cls_label = np.ones(score_map_shape, dtype=np.int32) * background_label
    pixel_cls_weight = np.zeros(score_map_shape, dtype=np.float32)

    ## get the masks of all bboxes
    bbox_masks = []
    pos_mask = mask.copy()
    for bbox_idx, (bbox_xs, bbox_ys) in enumerate(zip(xs, ys)):
        if labels[bbox_idx] == background_label:
            continue

        bbox_mask = mask.copy()

        bbox_points = zip(bbox_xs, bbox_ys)
        bbox_contours = util.img.points_to_contours(bbox_points)
        util.img.draw_contours(bbox_mask, bbox_contours, idx=-1,
                               color=1, border_width=-1)

        bbox_masks.append(bbox_mask)

        if labels[bbox_idx] == text_label:
            pos_mask += bbox_mask

    # treat overlapped in-bbox pixels as negative,
    # and non-overlapped  ones as positive
    pos_mask = np.asarray(pos_mask == 1, dtype=np.int32)
    # 总的正样本数
    num_positive_pixels = np.sum(pos_mask)

    ## add all bbox_maskes, find non-overlapping pixels
    sum_mask = np.sum(bbox_masks, axis=0)
    not_overlapped_mask = sum_mask == 1

    ## gt and weight calculation
    for bbox_idx, bbox_mask in enumerate(bbox_masks):
        bbox_label = labels[bbox_idx]
        if bbox_label == ignore_label:
            # for ignored bboxes, only non-overlapped pixels are encoded as ignored
            bbox_ignore_pixel_mask = bbox_mask * not_overlapped_mask
            pixel_cls_label += bbox_ignore_pixel_mask * ignore_label
            continue

        if labels[bbox_idx] == background_label:
            continue
        # from here on, only text boxes left.

        # for positive bboxes, all pixels within it and pos_mask are positive
        bbox_positive_pixel_mask = bbox_mask * pos_mask
        # background or text is encoded into cls gt
        pixel_cls_label += bbox_positive_pixel_mask * bbox_label

        # for the pixel cls weights, only positive pixels are set to ones

        num_bbox_pixels = np.sum(bbox_positive_pixel_mask)
        if num_bbox_pixels > 0:
            per_bbox_weight = num_positive_pixels * 1.0 / num_positive_bboxes   # 平均每个box的正像素点个数
            per_pixel_weight = per_bbox_weight / num_bbox_pixels    # num_bbox_pixels 是当前box的正样本的数
            pixel_cls_weight += bbox_positive_pixel_mask * per_pixel_weight

        ## calculate the labels and weights of links
        ### for all pixels in  bboxes, all links are positive at first

        ## the border of bboxes might be distored because of overlapping
        ## so recalculate it, and find the border mask
        new_bbox_contours = util.img.find_contours(bbox_positive_pixel_mask)
        bbox_border_mask = mask.copy()
        util.img.draw_contours(bbox_border_mask, new_bbox_contours, -1,
                               color=1, border_width=bbox_border_width * 2 + 1)
        bbox_border_mask *= bbox_positive_pixel_mask
        bbox_border_cords = np.where(bbox_border_mask)

        ## give more weight to the border pixels if configured
        pixel_cls_weight[bbox_border_cords] *= pixel_cls_border_weight_lambda

    pixel_cls_weight = np.asarray(pixel_cls_weight, dtype=np.float32)
    return pixel_cls_label, pixel_cls_weight