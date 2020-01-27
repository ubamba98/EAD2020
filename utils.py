import torch
import torch.nn as nn
from torch.nn import functional as F
import segmentation_models_pytorch as smp

class Tversky(nn.Module):
    """ Tversky loss function.

    Parameters
    ----------
    y_true : tensor containing target mask.
    y_pred : tensor containing predicted mask.
    alpha : float
        real value, weight of 'y_pred' class.
    beta : float
        real value, weight of 'y_true' class.
    smooth : float
        small real value used for avoiding division by zero error.

    Returns
    -------
        tensor containing tversky loss.
    """
    
    def __init__(self, alpha = 0.5, beta = 0.5, smooth = 1e-10):
        super(Tversky, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = 1e-10

    def forward(self, y_pred, y_true):
        y_pred = y_pred.type(torch.FloatTensor)
        y_true = y_true.type(torch.FloatTensor)
        y_pred = F.sigmoid(y_pred)
        num_classes = y_true.size(1)
        bs = y_true.size(0)
        tversky = 0
        for i in range(num_classes):
            y_t_i = y_true[:,i,...]
            y_p_i = y_pred[:,i,...]
            y_t_i = y_t_i.reshape(bs,-1)
            y_p_i = y_p_i.reshape(bs,-1)
            truepos = torch.sum(y_t_i*y_p_i, dim=1)
            fp_fn = self.alpha*torch.sum(y_p_i * (1 - y_t_i), dim=1) + self.beta * torch.sum((1 - y_p_i) * y_t_i, dim=1)
            tversky_i = (truepos + self.smooth) / ((truepos + self.smooth) + fp_fn)
            tversky+=torch.mean(tversky_i)
        return 1-tversky/num_classes

class BCEDiceLoss(nn.Module):
    __name__ = 'bce_dice_loss'

    def __init__(self, eps=1e-7, beta=1., activation='sigmoid', ignore_channels=None, threshold=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.beta = beta
        self.eps = eps
        self.threshold = threshold
        self.ignore_channels = ignore_channels
        self.activation = smp.utils.base.Activation(activation)

    def forward(self, y_pr, y_gt):
        bce = self.bce(y_pr, y_gt)
        y_pr = self.activation(y_pr)
        dice = 1 - smp.utils.functional.f_score(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )
        return dice + bce

class DiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, eps=1e-7, beta=1., activation='sigmoid', ignore_channels=None, threshold=None):
        super().__init__()
        self.beta = beta
        self.eps = eps
        self.threshold = threshold
        self.ignore_channels = ignore_channels
        self.activation = smp.utils.base.Activation(activation)

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        dice = 1 - smp.utils.functional.f_score(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )
        return dice


class BCEDiceJaccardLoss(nn.Module):
    __name__ = 'bce_dice_loss'

    def __init__(self, eps=1e-7, beta=1., activation='sigmoid', ignore_channels=None, threshold=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.beta = beta
        self.eps = eps
        self.threshold = threshold
        self.ignore_channels = ignore_channels
        self.activation = smp.utils.base.Activation(activation)

    def forward(self, y_pr, y_gt):
        bce = self.bce(y_pr, y_gt)
        y_pr = self.activation(y_pr)
        dice = 1 - smp.utils.functional.f_score(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )
        jaccard = 1 - smp.utils.functional.jaccard(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )
        return dice + bce + jaccard