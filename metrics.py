import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

nmi = normalized_mutual_info_score
ari = adjusted_rand_score


def acc_supervised(y_true, y_pred):
    return np.sum(np.asarray(y_true == y_pred, np.float32)) / len(y_true)


def dice(y_true, y_pred):
    intersection = np.asarray(np.logical_and(np.asarray(y_true, np.bool), np.asarray(y_pred, np.bool)), np.float32)
    intersection = np.sum(np.asarray(intersection, np.float32))
    union = np.sum(np.asarray(y_true, np.float32)) + np.sum(np.asarray(y_pred, np.float32))
    if union == 0:
        return 1.0
    return 2. * intersection / union


def IoU(y_true, y_pred):
    intersection = np.asarray(np.logical_and(np.asarray(y_true, np.bool), np.asarray(y_pred, np.bool)), np.float32)
    intersection = np.sum(np.asarray(intersection, np.float32))
    union = np.sum(np.asarray(y_true, np.float32)) + np.sum(np.asarray(y_pred, np.float32))
    if union == 0:
        return 1.0
    return intersection / (union - intersection)


def acc_unsupervised(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size