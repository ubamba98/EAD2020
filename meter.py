import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

def predict(X, threshold):
    '''X is sigmoid output of the model'''
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds


def single_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (height, width)
    intersection = (y_true * y_pred_bin).sum()
    if (y_true.sum()==0 and y_pred_bin.sum()==0):
        return 1
    return (2*intersection) / (y_true.sum() + y_pred_bin.sum())

def metric(y_pred_bin, y_true, threshold = 0.5):
    y_pred_bin = (y_pred_bin>threshold).float()
    y_true = y_true.float()
    batch_size = y_true.shape[0]
    channel_num = y_true.shape[1]
    mean_dice_channel = 0.
    for i in range(batch_size):
        for j in range(channel_num):
            channel_dice = single_dice_coef(y_true[i, j, ...],y_pred_bin[i, j, ...])
            mean_dice_channel += channel_dice/(channel_num*batch_size)
    return mean_dice_channel

def metric_pos_neg(probability, truth, threshold=0.5, reduction='none'):
    '''Calculates dice of positive and negative images seperately'''
    '''probability and truth must be torch tensors'''
    batch_size = len(truth)
    channels = truth.shape[1]
    with torch.no_grad():
        probability = probability.view(batch_size,channels,-1)
        truth = truth.view(batch_size,5,-1)
        assert(probability.shape == truth.shape)
        dice_pos_ = np.zeros(channels)
        dice_neg_ = np.zeros(channels)
        for i in range(channels):
            p = (probability[:,i,:] > threshold).float()
            t = (truth[:,i,:] > 0.5).float()
            t_sum = t.sum(-1)
            p_sum = p.sum(-1)
            neg_index = torch.nonzero(t_sum == 0)
            pos_index = torch.nonzero(t_sum >= 1)
            dice_neg = (p_sum == 0).float()
            dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))
            dice_neg = dice_neg[neg_index]
            dice_pos = dice_pos[pos_index]
            dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
            dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
            dice_neg_[i]=dice_neg
            dice_pos_[i]=dice_pos
        dice_neg = dice_neg_.mean()
        dice_pos = dice_pos_.mean()
    return dice_neg, dice_pos

class Meter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''
    def __init__(self, phase, epoch):
        self.base_threshold = 0.5 # <<<<<<<<<<< here's the threshold
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        dice_neg, dice_pos = metric_pos_neg(probs, targets, self.base_threshold)
        dice = metric(probs, targets)
        self.base_dice_scores.append(dice)
        self.dice_neg_scores.append(dice_neg)
        self.dice_pos_scores.append(dice_pos)
        preds = predict(probs, self.base_threshold)
        iou = compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.mean(self.base_dice_scores)
        dice_neg = np.mean(self.dice_neg_scores)
        dice_pos = np.mean(self.dice_pos_scores)
        iou = np.nanmean(self.iou_scores)
        
        return dice, dice_pos, dice_neg, iou

def epoch_log(phase, epoch, epoch_loss, meter, start):
    '''logging the metrics at the end of an epoch'''
    dice, dice_pos, dice_neg, iou = meter.get_metrics()
    print("Loss: %0.4f | IoU: %0.4f | dice: %0.4f | dice_pos: %0.4f | dice_neg: %0.4f" % (epoch_loss, iou, dice, dice_pos, dice_neg))
    return dice, iou

def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    '''computes iou for one ground truth mask and predicted mask'''
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]

def compute_iou_batch(outputs, labels, classes=None):
    '''computes mean iou for a batch of ground truth masks and predicted masks'''
    ious = []
    preds = np.copy(outputs) # copy is imp
    labels = np.array(labels) # tensor to np
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou