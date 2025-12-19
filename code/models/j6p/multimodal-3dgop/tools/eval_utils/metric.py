import torch
import numpy as np
from numba import jit


"""
confusionMetric
GT\P    P    N
P      TP    FN
N      FP    TN
"""

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = np.bincount(n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)


def per_class_iu(hist):
    # iou = TP / (TP + FP + FN)
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    return iu


def per_class_biou(hist):
    """only 2 class, background and foreground"""
    # iou = TP / (TP + FP + FN)
    hist = np.array([[hist[0][0], hist[0, 1:].sum()], [hist[1:, 0].sum(), hist[1:, 1:].sum()]])
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    return iu


def per_class_fwiou(hist):
    # FWIOU = [(TP+FN)/(TP+FP+TN+FN)] * [TP / (TP + FP + FN)]
    freq = hist.sum(axis=1) / hist.sum()
    iu = per_class_iu(hist)
    fwiou = freq[freq > 0] * iu[freq > 0]
    fwiou = np.nansum(fwiou)
    return fwiou


def per_class_fwiou_fg(hist):
    """only count foreground"""
    # FWIOU = [(TP+FN)/(TP+FP+TN+FN)] * [TP / (TP + FP + FN)]
    pixel_sum = hist.sum(axis=1)
    freq = pixel_sum[1:] / (hist.sum() - pixel_sum[0])
    iu = per_class_iu(hist)[1:]
    fwiou = freq[freq > 0] * iu[freq > 0]
    return fwiou


def per_class_precision(hist):
    # return each category pixel accuracy(A more accurate way to call it precision)
    # acc = (TP) / TP + FP
    precision = np.diag(hist) / hist.sum(axis=0)
    return precision


def per_class_recall(hist):
    # Recall = (TP) / (TP + FN)
    recall = np.diag(hist) / hist.sum(axis=1)
    return recall


def per_class_f1(hist):
    # 2*precision*recall / (precision + recall)
    precision =  per_class_precision(hist)
    recall = per_class_recall(hist)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def Accuracy(hist):
    # return all class overall pixel accuracy
    # acc = (TP + TN) / (TP + TN + FP + TN)
    acc = np.diag(hist).sum() / hist.sum()
    return acc


def fast_hist_crop(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label)+1)
    hist = hist[unique_label, :]
    hist = hist[:, unique_label]
    return hist


@jit(nopython=True)
def fast_hist_crop_numba(output, target, unique_label):
    pred, label = output.flatten(), target.flatten()
    n = np.max(unique_label)+1
    k = (0 <= label) & (label < n)
    bin_count = np.bincount(n * label[k] + pred[k], minlength=n ** 2)
    hist = bin_count[:n ** 2].reshape(n, n)
    hist = hist[unique_label, :]
    hist = hist[:, unique_label]
    return hist


def gpu_hist_crop_numba(output, target, unique_label, fg_mask=True):
    # unique_label:cuda tensor, fg_mask: only count foreground
    if fg_mask:
        idx = output | target
        output, target = output[idx], target[idx]
    pred, label = output.flatten().long(), target.flatten().long()
    n = torch.max(unique_label)+1
    k = (0 <= label) & (label < n)
    bin_count = torch.bincount(n * label[k] + pred[k], minlength=n ** 2)
    hist = bin_count[:n ** 2].reshape(n, n)
    hist = hist[unique_label, :]
    hist = hist[:, unique_label]
    hist = np.array(hist.cpu())
    return hist