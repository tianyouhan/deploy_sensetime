import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time
from torch.autograd import Function
import sys

class clip_relu(Function):
    def __init__(self, weight_decay, bw_layer=-1):
        self.weight_decay = weight_decay
        self.bw_layer = bw_layer

    def forward(self, input, alpha):
        self.save_for_backward(input, alpha)
        max_alpha = alpha[0]
        # print('forward alpha', max_alpha)
        if not isinstance(max_alpha, float):
            max_alpha = max_alpha.item()
        #print('0 max alpha', max_alpha)
        #print('bw layer', self.bw_layer, (2 ** self.bw_layer - 1) - 1, np.log2((2 ** self.bw_layer - 1) - 1) )
        fl_alpha = np.floor(np.log2((2 ** (self.bw_layer - 1) - 1) / max_alpha));
        max_alpha = np.round(max_alpha * (2 ** fl_alpha)) / (2 ** fl_alpha);
        #print('1 max_alpha', max_alpha)
        input = input.clamp(min=0, max=max_alpha)
        return input

    def backward(self, grad_output):
        input, alpha, = self.saved_tensors
        grad_alpha = (torch.sum(grad_output[input >= float(alpha[0])])).float() / grad_output.size(0)#/grad_output.size(1)
        grad_alpha = grad_alpha.float() + self.weight_decay * alpha
        ## 2019.2.14
        # from lib.config import cfg
        # if cfg.CLIPRELU_GRAD_CLAMP:
        #     grad_alpha.clamp(min=-1e-3, max=1e-3)
        #print('backward alpha', grad_alpha)
        grad_ = grad_output.clone()
        grad_[input <= 0] = 0
        grad_[input >= float(alpha[0])] = 0
        return grad_, grad_alpha

class ClipReLU(nn.Module):
    def __init__(self, inplace=True, alpha=10, weight_decay=0.):
        super(ClipReLU, self).__init__()
        self.clip_relu_max_alpha = torch.nn.Parameter(torch.Tensor([alpha]).float())
        self.weight_decay = weight_decay

    def forward(self, input):
        cliprelu = clip_relu(self.weight_decay, bw_layer=8)
        input = cliprelu(input, self.clip_relu_max_alpha)
        return input

