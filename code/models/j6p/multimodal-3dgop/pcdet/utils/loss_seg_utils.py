import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


# TODO
class NoiseLoss(nn.Module):
    def __init__(self, weight, ignore_index=255) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, preds, targets):
        return self.loss(preds, targets)


def single_offset_loss_norm_dir(pt_offsets, gt_offsets, valid):
    pt_diff = pt_offsets - gt_offsets   # (b, 2, h, w)
    pt_dist = torch.sum(torch.abs(pt_diff), dim=1)   # (2)
    offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)

    gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)
    gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(0) + 1e-8)
    pt_offsets_norm = torch.norm(pt_offsets, p=2, dim=1)
    pt_offsets_ = pt_offsets / (pt_offsets_norm.unsqueeze(0) + 1e-8)
    dir_diff = 1 - torch.sum(gt_offsets_ * pt_offsets_, dim=1)
    offset_dir_loss = torch.sum(dir_diff * valid) / (torch.sum(valid) + 1e-6)
    return offset_norm_loss + offset_dir_loss


class OHEM(nn.Module):
    """Online Hard Example Mining Sampler for segmentation.

    Args:
        thresh (float, optional): The threshold for hard example selection.
            Below which, are prediction with low confidence. If not
            specified, the hard examples will be pixels of top ``min_kept``
            loss. Default: None.
        min_kept (int, optional): The minimum number of predictions to keep.
            Default: 100000.
        min_kept_neg: keep the negtive points and Default: None
        mask: work for binary_cross_entropy, shape is seg_label.shape
        class_weight: work for cross_entropy, weight is Tensor of size `C`, Shape like Tensor float:[1.0, 2.0, 3.0]
    """

    def __init__(self, thresh=None, min_kept=100000, min_kept_neg=None, ignore_index=255, loss_func='bce', 
                        mask=None, class_weight=None):
        super().__init__()
        self.mask = mask  # work for binary_cross_entropy
        self.class_weight = class_weight  # work for cross_entropy
        self.loss_func = loss_func
        self.ignore_index = ignore_index
        assert min_kept > 1
        self.thresh = thresh
        self.min_kept = min_kept
        self.min_kept_neg = min_kept_neg

    def forward(self, seg_logit, seg_label):
        assert seg_logit.shape[2:] == seg_label.shape[2:]
        assert seg_label.shape[1] == 1
        if self.loss_func == 'bce':
            assert seg_logit.shape[1] == 1
        if self.loss_func == 'ce':
            assert seg_logit.shape[1] > 1
        seg_label = seg_label.squeeze(1).long()
        batch_kept = self.min_kept * seg_label.size(0)
        if self.min_kept_neg:
            batch_kept_neg = self.min_kept_neg * seg_label.size(0)
        valid_mask = seg_label != self.ignore_index
        seg_weight = seg_logit.new_zeros(size=seg_label.size())
        valid_seg_weight = seg_weight[valid_mask]
        if self.thresh is not None:
            seg_prob = F.softmax(seg_logit, dim=1)
            tmp_seg_label = seg_label.clone().unsqueeze(1)
            tmp_seg_label[tmp_seg_label == self.ignore_index] = 0
            seg_prob = seg_prob.gather(1, tmp_seg_label).squeeze(1)
            sort_prob, sort_indices = seg_prob[valid_mask].sort()

            if sort_prob.numel() > 0:
                min_threshold = sort_prob[min(batch_kept,
                                                sort_prob.numel() - 1)]
            else:
                min_threshold = 0.0
            threshold = max(min_threshold, self.thresh)
            valid_seg_weight[seg_prob[valid_mask] < threshold] = 1.
        else:
            if self.loss_func == 'bce':
                losses = F.binary_cross_entropy_with_logits(seg_logit, seg_label.unsqueeze(1).float(), weight=self.mask, reduction='none')
                losses = losses.squeeze(1)
            if self.loss_func == 'ce':
                losses = F.cross_entropy(seg_logit, seg_label, weight=self.class_weight, reduction='none')
            if not self.min_kept_neg:
                _, sort_indices = losses[valid_mask].sort(descending=True)
                valid_seg_weight[sort_indices[:batch_kept]] = 1.
            else:
                losses = losses * (1 - seg_label).float()  # set positive pixel loss to zero
                _, sort_indices = losses[valid_mask].sort(descending=True)
                valid_seg_weight[seg_label[valid_mask] != 0] = 1.
                valid_seg_weight[sort_indices[:batch_kept_neg]] = 1.

        if self.loss_func == 'bce':
            loss = F.binary_cross_entropy_with_logits(seg_logit, seg_label.unsqueeze(1).float(), weight=self.mask, reduction='none')
        if self.loss_func == 'ce':
            loss = F.cross_entropy(seg_logit, seg_label, weight=self.class_weight, reduction='none')
            loss = loss.unsqueeze(1)
        seg_weight[valid_mask] = valid_seg_weight
        
        loss = loss * seg_weight.unsqueeze(1).expand(seg_logit.shape)
        if self.thresh is not None:
            loss = loss.mean()
        else:
            loss = loss.sum() / (torch.sum(seg_weight == 1.) + 1)
        return loss


class SigmoidFocalClassificationLoss(nn.Module):
    """Sigmoid focal cross entropy loss.
      Focal loss down-weights well classified examples and focusses on the hard
      examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
    """

    def __init__(self, alpha=0.25, gamma=2, size_average=True, ignore_index=None):
        """Constructor.
        Args:
            gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
            alpha: optional alpha weighting factor to balance positives vs negatives.
            all_zero_negative: bool. if True, will treat all zero as background.
            else, will treat first label as background. only affect alpha.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, prediction_tensor, target_tensor, mask=None):
        """Compute loss function.
        Args:
            prediction_tensor: A float tensor of shape [b, c, h, w]
            target_tensor: A float tensor of shape [b, h, w]
        Returns:
          loss: a float tensor of shape [b, c, h, w]
        """
        num_classes = prediction_tensor.shape[1]
        # [b, h, w] to [b, c, h, w]
        target_tensor = F.one_hot(torch.clamp(target_tensor.squeeze(1).long(), 0, num_classes - 1), num_classes=num_classes)
        target_tensor = target_tensor.permute(0, 3, 1, 2).float()
        # per_entry_cross_ent = (_sigmoid_cross_entropy_with_logits(labels=target_tensor, logits=prediction_tensor))
        per_entry_cross_ent = F.binary_cross_entropy_with_logits(prediction_tensor, target_tensor, reduction='none')
        modulating_factor = 1.0
        if self.gamma:
            prediction_probabilities = torch.sigmoid(prediction_tensor)
            p_t = ((target_tensor * prediction_probabilities) + ((1 - target_tensor) * (1 - prediction_probabilities)))
            modulating_factor = torch.pow(1.0 - p_t, self.gamma)
        alpha_weight_factor = 1.0
        if self.alpha is not None:
            alpha_weight_factor = (target_tensor * self.alpha + (1 - target_tensor) * (1 - self.alpha))
        loss = (modulating_factor * alpha_weight_factor * per_entry_cross_ent)
        
        if mask is not None:
            loss = loss * mask

        if self.ignore_index is not None:
            ignore_mask = target_tensor == self.ignore_index
            loss = loss * (~ignore_mask).float()
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def binary_dice_loss(pred, target, valid_mask, smooth=1, exponent=2, reduction='mean'):
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

    num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum(pred.pow(exponent) + target.pow(exponent), dim=1) + smooth
    loss = 1 - num / den

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise Exception('Unexpected reduction {}'.format(reduction))
    

class MultiClassDiceLoss(nn.Module):
    """
    pred: N C H W
    target: N H W
    """
    def __init__(self, smooth=1, exponent=2, class_weight=None, ignore_index=255, reduction='mean'):
        super(MultiClassDiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        valid_mask = (target != self.ignore_index).float()
        target = F.one_hot(torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes).float()
        total_loss = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                dice_loss = binary_dice_loss(pred[:, i], target[..., i], valid_mask, self.smooth, self.exponent, self.reduction)
                if self.class_weight is not None:
                    dice_loss *= self.class_weight[i]
                total_loss += dice_loss
        return total_loss / num_classes
    

class FocalLoss(nn.Module):
   """
   copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
   This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
   'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
       Focal_Loss= -1*alpha*(1-pt)*log(pt)
   :param num_class:
   :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
   :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                   focus on hard misclassified example
   :param smooth: (float,double) smooth value when cross entropy
   :param balance_index: (int) balance class index, should be specific when alpha is float
   :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
   """

   def __init__(self, act_fn='softmax', alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True, ignore_index=None):
       super(FocalLoss, self).__init__()
       self.alpha = alpha
       self.gamma = gamma
       self.balance_index = balance_index
       self.smooth = smooth
       self.size_average = size_average
       self.ignore_index = ignore_index
       self.act_fn = act_fn
       assert act_fn in ['softmax', 'sigmoid'], f'Not support act_fn:{act_fn}'

       if self.smooth is not None:
           if self.smooth < 0 or self.smooth > 1.0:
               raise ValueError('smooth value should be in [0,1]')

   def forward(self, logit, target, mask=None):
        if self.act_fn == 'softmax':
            logit = torch.softmax(logit, dim=1)
        elif self.act_fn == 'sigmoid':
            logit = torch.sigmoid(logit)

        num_class = logit.shape[1]
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        
        if self.ignore_index is not None:
            ignore_mask = target == self.ignore_index
            target[ignore_mask] = 0
        else:
            ignore_mask = None
        
        alpha = self.alpha
        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha
        else:
            raise TypeError('Not support alpha type')
        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth/(num_class-1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma
        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        if isinstance(gamma, (list, np.ndarray)):
            assert len(gamma) == num_class
            loss = 0
            for i in range(len(gamma)):
                gamma_mask = target == i
                loss += (-1 * alpha * torch.pow((1 - pt), gamma[i]) * logpt) * gamma_mask.squeeze().float()
        else:
            loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if mask is not None:
            loss = loss * mask.view(loss.size())
        if ignore_mask is not None:
            loss = loss * (~ignore_mask).float().view(loss.size())

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def _sigmoid_cross_entropy_with_logits(logits, labels):
    # to be compatible with tensorflow, we don't use ignore_idx
    loss = torch.clamp(logits, min=0) - logits * labels.type_as(logits)
    loss += torch.log1p(torch.exp(-torch.abs(logits)))
    return loss


def get_focal_loss(preds, labels, mask, gamma=2):
    labels = torch.tensor(labels).clone().detach()
    mask = torch.tensor(mask).clone().detach()
    p = torch.sigmoid(preds)
    p = torch.clamp(p, min=0.01, max=0.99)
    loss1 = -labels * torch.pow((1.0 - p), gamma) * torch.log(p)
    loss2 = -(1.0 - labels) * torch.pow(p, gamma) * torch.log(1.0 - p)
    loss = (loss1 + loss2) * mask
    return  loss.sum() / (mask.sum())


if __name__ == '__main__':
    # test with thresh in ce losses case
    sampler = OHEM(thresh=0.7, min_kept=200, loss_func='ce')
    seg_logit = torch.randn(2, 19, 45, 45)
    seg_label = torch.randint(0, 19, size=(2, 1, 45, 45))
    seg_weight = sampler(seg_logit, seg_label)
    print(seg_weight)

    # test w.o thresh in bce losses case
    sampler = OHEM(min_kept=200, loss_func='bce')
    seg_logit = torch.sigmoid(torch.randn(2, 1, 45, 45))
    seg_label = torch.randint(0, 1, size=(2, 1, 45, 45))
    seg_weight = sampler(seg_logit, seg_label)
    print(seg_weight)

    # test w min_kept_neg in bce losses case
    sampler = OHEM(min_kept=200, min_kept_neg=200, loss_func='bce')
    seg_logit = torch.sigmoid(torch.randn(2, 1, 45, 45))
    seg_label = torch.randint(0, 2, size=(2, 1, 45, 45))
    seg_weight = sampler(seg_logit, seg_label)
    print(seg_weight)