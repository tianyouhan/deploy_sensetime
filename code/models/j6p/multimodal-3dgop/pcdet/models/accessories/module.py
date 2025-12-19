import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import sys
from .Ristretto import SparseConvRistretto2d
from torch.autograd import Function
from .Clip_ReLU import ClipReLU

class ConvBNRelu(nn.Module):
    def __init__(self, cfg, input_channel, output_channel, kernel_size=3, stride=1, padding=1, BN=nn.BatchNorm2d, groups=1, bias=True, merge_bn=False, winograd4x4=True):
        super(ConvBNRelu, self).__init__()
        self.merge_bn = merge_bn
        self.conv = SparseConvRistretto2d(input_channel, output_channel, kernel_size, stride=stride, groups=groups, padding=padding, winograd4x4=winograd4x4)
        if not merge_bn:
            self.bn = BN(output_channel)
        if cfg.SET_CLIPRELU:
            self.relu = ClipReLU(inplace=True)
        elif cfg.USE_RELU6:
            self.relu = nn.ReLU6(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, sparse_masks = None):
        if not self.merge_bn:
            out = self.relu(self.bn(self.conv(x, sparse_masks = sparse_masks)))
        else:
            out = self.relu(self.conv(x, sparse_masks = sparse_masks))
        return out
'''
def Get_BN(bn_param):
    BN = None
    if bn_param:
        if bn_param["bn_group_size"] == 1:
            bn_param['bn_group'] = None
        else:
            bn_param['bn_group'] = simple_group_split(bn_param['bn_group_size'])

        def BNFunc(*args, **kwargs):
            return SyncBatchNorm2d(*args, **kwargs, group=bn_param['bn_group'], sync_stats=bn_param['bn_sync_stats'], var_mode=syncbnVarMode_t.L2)
        BN = BNFunc
    else:
        BN = nn.BatchNorm2d
    return BN
'''

class AsymmResBNRELU(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1, BN=nn.BatchNorm2d, groups=1):
        super(AsymmResBNRELU, self).__init__()
        assert input_channel == output_channel
        assert stride == 1
        squeeze_ratio = 2
        first_groups = 1
        groups = 4
        middle_channel = input_channel // squeeze_ratio
        aggr_ratio = 1.0 + 1.0 / squeeze_ratio * 2.0
        self.conv1 = nn.Conv2d(input_channel, middle_channel, kernel_size, stride=stride, groups=first_groups, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(int(aggr_ratio * input_channel), int(aggr_ratio * input_channel), kernel_size, stride=stride, groups=4, padding=padding, bias=False)
        self.conv3 = nn.Conv2d(int(aggr_ratio * input_channel), output_channel, kernel_size, stride=stride, groups=groups, padding=padding, bias=False)
        self.bn1 = BN(middle_channel)
        self.bn2 = BN(int(aggr_ratio * input_channel))
        self.bn3 = BN(output_channel)
        self.bn4 = BN(input_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, sparse_masks = None):
        out = self.relu(self.bn1(self.conv1(x)))
        out = torch.cat((out, x, out), dim=1)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)) + self.bn4(x))
        # out = self.relu(self.bn3(self.conv3(out)) + x)
        return out


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        # BN_mom = float(cfg.RPN_STAGE.BN_MOM) if cfg.RPN_STAGE.BN_MOM != -1.0 else 0.01
        self.bn = nn.BatchNorm2d(out_planes) if bn else None #eps=1e-5
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):
    def __init__(self, cfg, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(BasicRFB, self).__init__()
        self.scale = cfg.RPN_STAGE.BACKBONE.RFBBLOCK_SCALE
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=True)
        )
        # self.branch1 = nn.Sequential(
        #        BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
        #        BasicConv(inter_planes, 2*inter_planes, kernel_size=(3,3), stride=stride, padding=(1,1)),
        #        BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1, relu=False)
        #        )

        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2*visual+1,
                      dilation=2*visual+1, relu=True) #2*visual+1
        )

        self.ConvLinear = BasicConv(4 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        # self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):

        x0 = self.branch0(x)
        # x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x2), 1)
        out = self.ConvLinear(out)
        # short = self.shortcut(x)
        out = out * self.scale + x
        out = self.relu(out)

        return out
