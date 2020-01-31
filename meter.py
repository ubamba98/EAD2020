import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.metrics import fbeta_score, precision_recall_fscore_support
from sklearn.metrics import jaccard_similarity_score as jaccard_score
def f2_metric(y_pred_bin, y_true, threshold = 0.5):
    y_pred_bin = (y_pred_bin>threshold).float()
    y_true = y_true.float()
    batch_size = y_true.shape[0]
    Apred = ((y_pred_bin > 0).astype(np.uint8)).flatten()
    Btrue = ((y_true > 0).astype(np.uint8)).flatten()
    f2_score = []
    jc_score = []
    for i in range(batch_size):
        f2_score.append(fbeta_score(Btrue, Apred, beta=2, average='binary'))
        jc_score.append(jaccard_score(Btrue, Apred)) 
    return np.mean(f2_score), np.mean(jc_score)

def dice(y_pred_bin, y_true, threshold = 0.5):
    y_pred_bin = (y_pred_bin>threshold).float()
    y_true = y_true.float()
    batch_size = y_true.shape[0]
    dice = []
    precision = []
    recall = []
    for i in range(batch_size):
        p, r, fb_score, support = precision_recall_fscore_support( ((y_true[i]> 0).astype(np.uint8)).flatten(), ((y_pred[i]> 0).astype(np.uint8)).flatten(), average='binary')
        dice.append(fb_score)
        precision.append(p)
        recall.append(r)
    return np.mean(dice), np.mean(precision), np.mean(recall)

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
        self.recall = []
        self.precision = []


    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        dice_neg, dice_pos = metric_pos_neg(probs, targets, self.base_threshold)
        dice, p, r = dice(probs, targets)
        f2, iou = f2_metric(probs, targets)
        self.base_dice_scores.append(dice)
        self.precision.append(p)
        self.recall.append(r)
        self.dice_neg_scores.append(dice_neg)
        self.dice_pos_scores.append(dice_pos)
        self.f2_scores.append(f2)
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.mean(self.base_dice_scores)
        precision = np.mean(self.precision)
        recall = np.mean(self.recall)
        dice_neg = np.mean(self.dice_neg_scores)
        dice_pos = np.mean(self.dice_pos_scores)
        f2 = np.mean(self.f2_scores)
        iou = np.nanmean(self.iou_scores)
        
        return dice, dice_pos, dice_neg, iou, f2, precision, recall

def epoch_log(phase, epoch, epoch_loss, meter, start):
    '''logging the metrics at the end of an epoch'''
    dice, dice_pos, dice_neg, iou, f2, precision, recall = meter.get_metrics()
    print("Loss: %0.4f | IoU: %0.4f | dice: %0.4f | dice_pos: %0.4f | dice_neg: %0.4f | f2_score: %0.4f | precision: %0.4f | recall: %0.4f" % (epoch_loss, iou, dice, dice_pos, dice_neg, f2, precision, recall))
    return dice, iou, f2, dice_pos, dice_neg, precision, recall