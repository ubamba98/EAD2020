import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.metrics import fbeta_score


def single_dice_coef(y_true, y_pred_bin):
    # shape of y_true and y_pred_bin: (height, width)
    intersection = (y_true * y_pred_bin).sum()
    if (y_true.sum()==0 and y_pred_bin.sum()==0):
        return 1
    return (2*intersection) / (y_true.sum() + y_pred_bin.sum())

def single_f2_coef(y_true, y_pred_bin):
    y_true = y_true.cpu().numpy()
    y_pred_bin = y_pred_bin.cpu().numpy()
    f2_score = fbeta_score(y_true, y_pred_bin, beta=2)
    return f2_score

def f2_pytorch(y_true, y_pred_bin):

    tp = (y_true * y_pred_bin).sum()#.to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred_bin)).sum()#.to(torch.float32)
    fp = ((1 - y_true) * y_pred_bin).sum()#.to(torch.float32)
    fn = (y_true * (1 - y_pred_bin)).sum()#.to(torch.float32)

    epsilon = 1e-10

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f2 = 5* (precision*recall) / (4*precision + recall + epsilon)
    return f2

def f2_metric(y_pred_bin, y_true, threshold = 0.5):
    y_pred_bin = (y_pred_bin>threshold).float()
    y_true = y_true.float()
    batch_size = y_true.shape[0]
    channel_num = y_true.shape[1]
    mean_f2_channel = 0.
    for i in range(batch_size):
        for j in range(channel_num):
            channel_f2 = single_f2_coef(y_true[i, j, ...].view(-1),y_pred_bin[i, j, ...].view(-1))
            mean_f2_channel += channel_f2/(channel_num*batch_size)
    return mean_f2_channel

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
        self.f2_scores = []

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        dice_neg, dice_pos = metric_pos_neg(probs, targets, self.base_threshold)
        dice = metric(probs, targets)
        f2 = f2_metric(probs, targets)
        self.base_dice_scores.append(dice)
        self.dice_neg_scores.append(dice_neg)
        self.dice_pos_scores.append(dice_pos)
        self.f2_scores.append(f2)
        # preds = predict(probs, self.base_threshold)
        iou = soft_jaccard_score(outputs, targets)
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.mean(self.base_dice_scores)
        dice_neg = np.mean(self.dice_neg_scores)
        dice_pos = np.mean(self.dice_pos_scores)
        f2 = np.mean(self.f2_scores)
        iou = np.nanmean(self.iou_scores)
        
        return dice, dice_pos, dice_neg, iou, f2

def epoch_log(phase, epoch, epoch_loss, meter, start):
    '''logging the metrics at the end of an epoch'''
    dice, dice_pos, dice_neg, iou, f2 = meter.get_metrics()
    print("Loss: %0.4f | IoU: %0.4f | dice: %0.4f | dice_pos: %0.4f | dice_neg: %0.4f | f2_score: %0.4f" % (epoch_loss, iou, dice, dice_pos, dice_neg, f2))
    return dice, iou

def soft_jaccard_score(y_pred: torch.Tensor, y_true: torch.Tensor, smooth=0.0, eps=1e-7, threshold=0.5) -> torch.Tensor:
    """
    :param y_pred:
    :param y_true:
    :param smooth:
    :param eps:
    :return:
    Shape:
        - Input: :math:`(N, NC, *)` where :math:`*` means
            any number of additional dimensions
        - Target: :math:`(N, NC, *)`, same shape as the input
        - Output: scalar.
    """
    assert y_pred.size() == y_true.size()
    bs = y_true.size(0)
    num_classes = y_pred.size(1)
    dims = (0, 2)
    y_pred = (y_pred>threshold).float()
    y_true = y_true.view(bs, num_classes, -1)
    y_pred = y_pred.view(bs, num_classes, -1)
    
    if dims is not None:
        intersection = torch.sum(y_pred * y_true, dim=dims)
        cardinality = torch.sum(y_pred + y_true, dim=dims)
    else:
        intersection = torch.sum(y_pred * y_true)
        cardinality = torch.sum(y_pred + y_true)

    union = cardinality - intersection
    jaccard_score = (intersection + smooth) / (union.clamp_min(eps) + smooth)
    return jaccard_score.mean().item()