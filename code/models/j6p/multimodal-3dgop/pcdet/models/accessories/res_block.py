from pcdet.models.accessories.accessory import SEModule, GloReModule, BNET2d
import torch.nn.functional as F
import torch.nn as nn
import torch
from functools import partial


RELU = nn.ReLU

class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, num_groups, eps=1e-5, affine=True):
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine)

class ResBasicBlock(nn.Module):
    """
    ResNet basic block
    """
    def __init__(self, inplanes, planes, stride=1, dilation=1, padding=1, downsample=None, use_groupnorm=False, num_groups=32):
        super(ResBasicBlock, self).__init__()
        if use_groupnorm:
            BatchNorm2d = partial(GroupNorm, num_groups=num_groups, eps=1e-3)
        else:
            BatchNorm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        Conv2d = partial(nn.Conv2d, bias=False)
        self.conv1 = Conv2d(inplanes, int(planes/4), kernel_size=1, stride=stride, padding=0)
        self.bn1 = BatchNorm2d(int(planes/4))
        self.relu = RELU(inplace=True)
        self.conv2 = Conv2d(int(planes/4), int(planes/4), kernel_size=3, stride=1, padding=padding, dilation=dilation)
        self.bn2 = BatchNorm2d(int(planes/4))
        self.conv3 = Conv2d(int(planes/4), planes, kernel_size=1, stride=1, padding=0)
        self.bn3 = BatchNorm2d(planes)
        self.downsample = downsample


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class IncDepConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, bias=False):
        super(IncDepConv, self).__init__()
        self.bran_in = in_ch // 3
        self.bran_out = out_ch // 3
        self.dps_conv1 = nn.Conv2d(self.bran_in, self.bran_out,
                                   kernel_size=3, padding=1, stride=stride, groups=self.bran_in, bias=bias)
        self.dps_conv2 = nn.Conv2d(self.bran_in, self.bran_out,
                                   kernel_size=5, padding=2, stride=stride, groups=self.bran_in, bias=bias)
        self.dps_conv3 = nn.Conv2d(in_ch - 2*self.bran_in, out_ch - 2*self.bran_out,
                                   kernel_size=7, padding=3, stride=stride, groups=in_ch - 2*self.bran_in, bias=bias)

    def forward(self, x):
        x1 = x[:, :self.bran_in, ...]
        x2 = x[:, self.bran_in:2*self.bran_in, ...]
        x3 = x[:, 2*self.bran_in:, ...]

        x1 = self.dps_conv1(x1)
        x2 = self.dps_conv2(x2)
        x3 = self.dps_conv3(x3)
        return torch.cat([x1, x2, x3], dim=1)


class EfficientConv(nn.Module):
    def __init__(self, cfg, in_ch, out_ch, kernel_size=1, padding=0, stride=1, groups=1, dilation=1, keep_ratio=1., bias=False):
        super(EfficientConv, self).__init__()
        high_ch = int(keep_ratio * out_ch)
        low_ch = out_ch - high_ch
        self.conv_high = nn.Conv2d(in_ch, high_ch, kernel_size=kernel_size, stride=stride,
                                   padding=padding, groups=groups, dilation=dilation, bias=bias)
        self.conv_low = nn.Conv2d(in_ch, low_ch, kernel_size=kernel_size, stride=stride,
                                  padding=padding, groups=groups, dilation=dilation, bias=bias)

        self.conv_high2low = nn.Conv2d(high_ch, low_ch, kernel_size=kernel_size,
                                       padding=padding, groups=groups, dilation=dilation, bias=bias)
        self.conv_low2high = nn.Conv2d(low_ch, high_ch, kernel_size=kernel_size,
                                       padding=padding, groups=groups, dilation=dilation, bias=bias)
        # self.conv_low2high = nn.ConvTranspose2d(low_ch, high_ch, kernel_size=kernel_size, padding=padding, stride=2,
        #                                         groups=groups, dilation=dilation, bias=bias)

        if cfg.RPN_STAGE.BACKBONE.USE_BNET:
            BatchNorm2d = BNET2d
        else:
            # BN_mom = float(cfg.RPN_STAGE.BN_MOM) if cfg.RPN_STAGE.BN_MOM != -1.0 else 0.1
            BatchNorm2d = nn.BatchNorm2d
        self.bn_low = BatchNorm2d(low_ch) #eps=1e-5
        self.bn_high = BatchNorm2d(high_ch)
        self.bn_low_out = BatchNorm2d(low_ch)
        self.bn_high_out = BatchNorm2d(high_ch)

        self.relu = RELU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x_h = self.conv_high(x)
        x_h = self.bn_high(x_h)
        x_l = self.pool(x)
        x_l = self.conv_low(x_l)
        x_l = self.bn_low(x_l)
        x_l2h = self.conv_low2high(self.upsample(x_l))
        x_l2h = self.bn_high_out(x_l2h)
        x_h2l = self.conv_high2low(self.pool(x_h))
        x_h2l = self.bn_low_out(x_h2l)

        # x_h += self.conv_low2high(F.upsample(self.bn_low(x_l), scale_factor=2, mode='bilinear', align_corners=True))
        # or
        x_h = x_h + x_l2h
        x_h = self.relu(x_h)
        # x_h = self.bn_high_out(self.conv_low2high(x_l))
        x_l = x_l + x_h2l
        x_l = self.relu(x_l)

        return torch.cat([x_h, self.upsample(x_l)], dim=1)


class PreActBottleneck(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 stride=1,
                 ratio=4,
                 has_se=False,
                 groups=1,
                 rm_se_bias=False,
                 has_glore=None,
                 dilation=1,
                 mode="NORM",
                 k=1,
                 **topk_args):
        super(PreActBottleneck, self).__init__()
        self.rm_se_bias = rm_se_bias
        self.mode, self.k = mode, k
        self.conv1 = self._bn_relu_conv(in_ch, out_ch // ratio, has_se=has_se)
        self.conv2 = self._bn_relu_conv(out_ch // ratio, out_ch // ratio, k=3, p=dilation, s=stride, g=groups,
                                        d=dilation, **topk_args)
        self.conv3 = self._bn_relu_conv(out_ch // ratio, out_ch)

        if has_glore:
            self.glore = GloReModule(out_ch, pre_act=True)
        else:
            self.glore = None

        if mode == 'UP':
            self.downsample = None
        elif in_ch != out_ch or stride > 1:
            self.downsample = self._bn_relu_conv(in_ch, out_ch, s=stride)
        else:
            self.downsample = None

    def _bn_relu_conv(self, in_ch, out_ch, k=1, p=0, s=1, g=1, d=1, has_se=False, topk=False, **topk_args):
        bn = nn.BatchNorm2d(in_ch)
        relu = RELU(inplace=True)
        conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, stride=s, groups=g, dilation=d)

        if has_se:
            return nn.Sequential(bn, SEModule(in_ch, rm_bias=self.rm_se_bias), relu, conv)

        return nn.Sequential(bn, relu, conv)

    def squeeze_idt(self, idt):
        n, c, h, w = idt.size()
        return idt.view(n, c // self.k, self.k, h, w).sum(2)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.mode == 'UP':
            residual = self.squeeze_idt(x)
        elif self.downsample is not None:
            residual = self.downsample(residual)

        out += residual

        if self.glore is not None:
            out = self.glore(out)

        return out


class PostActBottleneck(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 stride=1,
                 ratio=4,
                 has_se=False,
                 groups=1,
                 dilation=1,
                 se_ratio=16,
                 rm_se_bias=False,
                 has_glore=False,
                 ds_keep_ratio=1.):
        super(PostActBottleneck, self).__init__()
        self.relu = RELU(inplace=True)

        self.conv1 = self._conv_bn_relu(in_ch, out_ch // ratio, ds_keep_ratio=ds_keep_ratio)
        self.conv2 = self._conv_bn_relu(out_ch // ratio, out_ch // ratio,
                                        k=3, p=dilation, s=stride, g=groups, d=dilation, ds_keep_ratio=ds_keep_ratio)
        self.conv3 = self._conv_bn_relu(out_ch // ratio, out_ch, has_relu=False, ds_keep_ratio=ds_keep_ratio)

        if in_ch != out_ch:
            self.downsample = self._conv_bn_relu(in_ch, out_ch, s=stride, has_relu=False, ds_keep_ratio=ds_keep_ratio)
        else:
            self.downsample = None

        if has_se:
            self.se_blk = SEModule(out_ch, rm_bias=rm_se_bias, sqz_ratio=se_ratio)
        else:
            self.se_blk = None

        if has_glore:
            self.glore = GloReModule(out_ch, pre_act=False)
        else:
            self.glore = None

    def _conv_bn_relu(self, in_ch, out_ch, k=1, p=0, s=1, g=1, d=1, has_relu=True, ds_keep_ratio=1.):
        if ds_keep_ratio < 1.:
            return EfficientConv(in_ch, out_ch, kernel_size=k, padding=p, stride=s, groups=g, dilation=d,
                                 keep_ratio=ds_keep_ratio, bias=False)
        else:
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, stride=s, groups=g, dilation=d, bias=False)

        if has_relu:
            return nn.Sequential(conv, nn.BatchNorm2d(out_ch), self.relu)
        else:
            return nn.Sequential(conv, nn.BatchNorm2d(out_ch))

    def T(self, x, dim1=1, dim2=2):
        return x.transpose(dim1, dim2).contiguous()

    def forward(self, x, prev_k=None):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.se_blk is not None:
            out = self.se_blk(out)

        out += residual
        out = self.relu(out)

        if self.glore is not None:
            out = self.glore(out)

        return out


class ShuffleBlockV2(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 stride=1,
                 ratio=0.5,
                 dilation=1,
                 ds_keep_ratio=1.):
        super(ShuffleBlockV2, self).__init__()
        self.relu = RELU(inplace=True)
        self.in_ch_idt = int(in_ch * ratio)
        self.out_ch_idt = int(out_ch * ratio)

        self.in_ch = in_ch - self.in_ch_idt
        self.ch = out_ch - self.out_ch_idt

        self.conv1 = self._conv_bn_relu(self.in_ch, self.ch, ds_keep_ratio=ds_keep_ratio)
        #self.conv1 = self._conv_bn_relu(self.in_ch, self.ch, has_relu=False, ds_keep_ratio=ds_keep_ratio)
        self.conv2 = self._conv_bn_relu(self.ch, self.ch, k=3, p=dilation, s=stride, g=self.out_ch_idt, d=dilation,
                                        has_relu=False, inc_dep=False, ds_keep_ratio=ds_keep_ratio)
        self.conv3 = self._conv_bn_relu(self.ch, self.ch, ds_keep_ratio=ds_keep_ratio)
        #self.conv3 = self._conv_bn_relu(self.ch, self.ch, has_relu=False, ds_keep_ratio=ds_keep_ratio)

        if in_ch != out_ch or stride > 1:
            if stride > 1:
                self.conv_idt = nn.Sequential(
                    self._conv_bn_relu(self.in_ch_idt, self.out_ch_idt, k=3, p=1, g=self.in_ch_idt,
                                       s=stride, has_relu=True, ds_keep_ratio=ds_keep_ratio),
                    self._conv_bn_relu(self.out_ch_idt, self.out_ch_idt, ds_keep_ratio=ds_keep_ratio))
            else:
                self.conv_idt = self._conv_bn_relu(self.in_ch_idt, self.out_ch_idt, ds_keep_ratio=ds_keep_ratio)
        else:
            self.conv_idt = None

    def _conv_bn_relu(self, in_ch, out_ch, k=1, p=0, s=1, g=1, d=1, has_relu=True, inc_dep=False, ds_keep_ratio=1.):
        if ds_keep_ratio < 1.:
            return EfficientConv(in_ch, out_ch, kernel_size=k, padding=p, stride=s, groups=g, dilation=d,
                                 keep_ratio=ds_keep_ratio, bias=False)
        elif inc_dep:
            return IncDepConv(in_ch, out_ch, stride=s, bias=False)
        else:
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, stride=s, groups=g, dilation=d, bias=False)

        if has_relu:
            return nn.Sequential(conv, nn.BatchNorm2d(out_ch), self.relu)
        else:
            return nn.Sequential(conv, nn.BatchNorm2d(out_ch))

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups

        # reshape
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, height, width)

        return x

    def forward(self, x, prev_k=None):
        x_idt = x[:, :self.in_ch_idt, ...]
        x_in = x[:, self.in_ch_idt:, ...]
        out = self.conv1(x_in)
        out = self.conv2(out)
        out = self.conv3(out)

        out = self.channel_shuffle(out, 2)

        if self.conv_idt is not None:
            x_idt = self.conv_idt(x_idt)

        return torch.cat([out, x_idt], dim=1)


class ResBottleneck_Single(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, padding=1):
        super(ResBottleneck_Single, self).__init__()
        BatchNorm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        Conv2d = partial(nn.Conv2d, bias=False)
        self.conv1 = Conv2d(inplanes, int(planes/self.expansion), kernel_size=1, stride=stride, padding=0)
        self.bn1 = BatchNorm2d(int(planes/self.expansion))
        self.relu = RELU(inplace=True)
        self.conv2 = Conv2d(int(planes/self.expansion), int(planes/self.expansion), kernel_size=3, stride=1, padding=padding, dilation=dilation)
        self.bn2 = BatchNorm2d(int(planes/self.expansion))
        self.conv3 = Conv2d(int(planes/self.expansion), planes, kernel_size=1, stride=1, padding=0)
        self.bn3 = BatchNorm2d(planes)
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0),
                BatchNorm2d(planes)
            )
        else:
            downsample = None
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResBasicBlock_Single(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(ResBasicBlock_Single, self).__init__()
        BatchNorm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        Conv2d = partial(nn.Conv2d, bias=False)
        self.conv1 = Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = BatchNorm2d(planes)
        self.relu = RELU(inplace=True)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2d(planes)
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0),
                BatchNorm2d(planes)
            )
        else:
            downsample = None
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out