import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from functools import partial
from pcdet.models.accessories.pytorch_utils import change_default_args
import math
from pcdet.models.backbones_2d import fuser as fuser
from pcdet.models.accessories.res_block import ShuffleBlockV2, EfficientConv, PostActBottleneck, SEModule, ResBasicBlock, ResBasicBlock_Single, ResBottleneck_Single
from pcdet.models.accessories.hourglass import HourglassNet
from pcdet.models.accessories.resnet_new import RPNBase_ResNet
from pcdet.models.accessories.densenet import RPNBase_DenseNet
from pcdet.models.accessories.attention import SELayer, eSEModule
from pcdet.models.accessories.accessory import BNET2d, GhostModule
from pcdet.models.accessories.repvgg import RepVGGBlock
class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, num_groups, eps=1e-5, affine=True):
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine)

RELU = nn.ReLU

class Empty(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Empty, self).__init__()

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            return args[0]
        elif len(args) == 0:
            return None
        return args

# Modified from mmdet to initialize heads
def bias_init_with_prob(prior_prob):
    import numpy as np
    """ initialize conv/fc bias value according to giving probablity"""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init

def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class Sequential(torch.nn.Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, given is a small example::

        # Example of using Sequential
        model = Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))

        # Example of using Sequential with kwargs(python 3.6+)
        model = Sequential(
                  conv1=nn.Conv2d(1,20,5),
                  relu1=nn.ReLU(),
                  conv2=nn.Conv2d(20,64,5),
                  relu2=nn.ReLU()
                )
    """

    def __init__(self, *args, **kwargs):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input, sparse_masks=None):
        for module in self._modules.values():
            from pcdet.models.accessories.Ristretto import SparseConvRistretto2d
            from pcdet.models.accessories.module import ConvBNRelu
            # if ((not cfg.TO_CAFFE) and cfg.ANCHOR_AREA_THRESHOLD > 0) and (isinstance(module, SparseConvRistretto2d) or isinstance(module, ConvBNRelu)):
            #     input = module(input, sparse_masks=sparse_masks)
            # else:
            input = module(input)
        return input


class ConvHead(nn.Module):
    def __init__(self,
                 cfg,
                 in_filters,
                 num_filters,
                 num_class,
                 head_cls_num,
                 num_anchor_per_cls,
                 conv_nums,
                 box_code_size,
                 num_direction_bins,
                 use_direction_classifier,
                 encode_background_as_zeros,
                 dilations,
                 stride,
                 use_se=False,
                 use_res=False,
                 use_glore=False,
                 num_res=1,
                 ratio=4,
                 se_ratio=8,
                 ds_keep_ratio=1.,
                 use_shuffle=False,
                 use_norm=True,
                 split_class=False,
                 split_class_branch=False,
                 split_class_branch_convnum=1,
                 split_class_branch_convfeats=32,
                 num_anchor_per_cls_list=[],
                 ks1=1,
                 firstBN=False,
                 DH_scale=False,
                 ssratio=None,
                 decoupled_head=False,
                 decoupled_ch=None):
        super().__init__()
        ## to caffe
        if cfg.TO_CAFFE:
            num_direction_bins = 1
            num_class = 1
        self.cfg = cfg
        self._num_direction_bins = num_direction_bins
        self._num_class = num_class
        self._num_class_cls = num_class if encode_background_as_zeros else (num_class + 1)
        self._use_direction_classifier = use_direction_classifier
        self._box_code_size = box_code_size
        assert len(dilations) == conv_nums
        assert len(num_anchor_per_cls_list) == head_cls_num
        # num_anchor_per_loc = num_anchor_per_cls * head_cls_num
        self.num_anchor_per_cls_list = num_anchor_per_cls_list
        num_anchor_per_loc = sum(self.num_anchor_per_cls_list)
        self._num_anchor_per_loc = num_anchor_per_loc
        self.split_class = split_class
        self.split_class_branch = split_class_branch
        assert not (split_class and split_class_branch), "split_class and split_class_branch shouldn't be true at the same time"
        self.DH_scale = DH_scale
        self.ssratio = ssratio
        self.decoupled_head = decoupled_head
        self.decoupled_ch = decoupled_ch

        num_cls = num_anchor_per_loc * self._num_class_cls

        # BN_mom = float(cfg.RPN_STAGE.BN_MOM) if cfg.RPN_STAGE.BN_MOM != -1.0 else 0.01
        #BatchNorm2d = change_default_args(eps=1e-3, momentum=BN_mom)(nn.BatchNorm2d)
        # BatchNorm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01) #eps=1e-3
        BatchNorm2d = nn.BatchNorm2d

        ## for quantization
        if cfg.SET_QUANTIZATION:
            from pcdet.models.accessories.Ristretto import SparseConvRistretto2d
            from pcdet.models.accessories.module import ConvBNRelu
            net_module = Sequential(ConvBNRelu(in_filters, num_filters, kernel_size=3,
                                        padding=1, BN=BatchNorm2d,
                                        groups=cfg.RPN_HEAD_GROUP_NUMS[0], stride=stride, merge_bn=not use_norm))
            for i in range(conv_nums):
                net_module.add(ConvBNRelu(num_filters, num_filters, kernel_size=3,
                                            padding=1, BN=BatchNorm2d,
                                            groups=cfg.RPN_HEAD_GROUP_NUMS[i+1], merge_bn=not use_norm))
            self.net = net_module
            kernel_size = 1
            dilation = 1 if head_cls_num == 1 else 1
            padding = 0
            self.conv_cls = SparseConvRistretto2d(cfg, num_filters, num_cls, kernel_size=kernel_size, dilation=dilation, padding=padding)
            self.conv_box = SparseConvRistretto2d(cfg, num_filters, num_anchor_per_loc * box_code_size, kernel_size=kernel_size, dilation=dilation, padding=padding)
            if use_direction_classifier:
                self.conv_dir_cls = SparseConvRistretto2d(cfg, num_filters, num_anchor_per_loc * num_direction_bins, kernel_size=kernel_size, dilation=dilation, padding=padding)
        else:
            if ds_keep_ratio < 1.:
                Conv2d = change_default_args(keep_ratio=ds_keep_ratio)(EfficientConv)
                pad = int((ks1-1)/2)
                net_module = Sequential(Conv2d(cfg, in_filters, num_filters, kernel_size=ks1, padding=pad, stride=stride, bias=False))
            elif use_shuffle:
                Conv2d = change_default_args(ds_keep_ratio=ds_keep_ratio)(ShuffleBlockV2)
                net_module = Sequential(Conv2d(in_filters, num_filters, stride=stride))
            else:
                Conv2d = nn.Conv2d
                pad = int((ks1-1)/2)
                net_module = Sequential(Conv2d(in_filters, num_filters, kernel_size=ks1, padding=pad, stride=stride, bias=False))

            # if firstBN:
            #     net_module.add(BatchNorm2d(num_filters))
            #     net_module.add(RELU(inplace=True))

            if use_se:
                net_module.add(SEModule(num_filters, sqz_ratio=se_ratio))

            for i in range(conv_nums):
                if use_res:
                    for _ in range(num_res):
                        net_module.add(
                            PostActBottleneck(num_filters,
                                              num_filters,
                                              dilation=dilations[i],
                                              has_se=use_se,
                                              has_glore=use_glore,
                                              ratio=ratio,
                                              ds_keep_ratio=ds_keep_ratio)
                        )
                elif use_shuffle:
                    net_module.add(ShuffleBlockV2(num_filters, num_filters))
                else:
                    #net_module.add(Conv2d(num_filters, num_filters, 1, bias=False))  # , padding=dilations[i]
                    #net_module.add(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False))
                    if i == 0:
                        net_module.add(BatchNorm2d(num_filters))
                        net_module.add(RELU(inplace=True))
                        # continue
                    net_module.add(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False))
                    net_module.add(BatchNorm2d(num_filters))
                    net_module.add(RELU(inplace=True))
                    if i == conv_nums - 1:
                        net_module.add(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False))
            self.net = net_module

            if self.DH_scale:
                self.DH_scale_param = nn.parameter.Parameter(torch.ones(1), requires_grad=True)

            kernel_size = 1
            dilation = 1 if head_cls_num == 1 else 1
            padding = 0
            if (not self.split_class) and (not self.split_class_branch):
                if not self.decoupled_head:
                    head_input_ch = num_filters
                else:
                    self.decoupled_conv1 = Sequential(nn.Conv2d(num_filters, self.decoupled_ch, kernel_size=3, padding=1, bias=False))
                    self.decoupled_conv1.add(BatchNorm2d(self.decoupled_ch))
                    self.decoupled_conv1.add(RELU(inplace=True))
                    self.decoupled_conv2 = Sequential(nn.Conv2d(num_filters, self.decoupled_ch, kernel_size=3, padding=1, bias=False))
                    self.decoupled_conv2.add(BatchNorm2d(self.decoupled_ch))
                    self.decoupled_conv2.add(RELU(inplace=True))
                    head_input_ch = self.decoupled_ch
                self.conv_cls = nn.Conv2d(head_input_ch, num_cls, kernel_size=kernel_size, dilation=dilation, padding=padding)
                self.conv_box = nn.Conv2d(head_input_ch, num_anchor_per_loc * box_code_size, kernel_size=kernel_size, dilation=dilation, padding=padding)
                if use_direction_classifier:
                    self.conv_dir_cls = nn.Conv2d(head_input_ch, num_anchor_per_loc * num_direction_bins, kernel_size=kernel_size, dilation=dilation, padding=padding)
                if cfg.RPN_STAGE.IOU_HEAD.USE:
                    self.conv_iou_head = nn.Conv2d(head_input_ch, num_anchor_per_loc * 1, kernel_size=kernel_size, dilation=dilation, padding=padding)
                if cfg.RPN_STAGE.RPN_HEAD.REG_VARIANCE.OUTPUT:
                    self.conv_var_head = nn.Conv2d(head_input_ch, num_anchor_per_loc * box_code_size, kernel_size=kernel_size, dilation=dilation, padding=padding, bias=False)
            else:
                num_filters_final = num_filters
                if self.split_class_branch:
                    assert split_class_branch_convnum > 0, "split_class_branch_convnum should > 0 in split_class_branch model"
                    split_class_branch_group = []
                    for _ in range(head_cls_num):
                        branch_module = Sequential(Conv2d(num_filters, split_class_branch_convfeats, 1, bias=False))
                        branch_module.add(BatchNorm2d(split_class_branch_convfeats))
                        branch_module.add(RELU(inplace=True))
                        for _ in range(split_class_branch_convnum-1):
                            branch_module.add(Conv2d(split_class_branch_convfeats, split_class_branch_convfeats, 1, bias=False))  # , padding=dilations[i]
                            branch_module.add(BatchNorm2d(split_class_branch_convfeats))
                            branch_module.add(RELU(inplace=True))
                        split_class_branch_group.append(branch_module)
                    self.split_class_branch_group = nn.ModuleList(split_class_branch_group)
                    num_filters_final = split_class_branch_convfeats
                conv_cls = []
                for k in range(head_cls_num): 
                    conv_cls.append(nn.Conv2d(num_filters_final, self.num_anchor_per_cls_list[k] * self._num_class_cls, kernel_size=kernel_size, dilation=dilation, padding=padding))
                self.conv_cls = nn.ModuleList(conv_cls)
                conv_box = []
                for k in range(head_cls_num):
                    conv_box.append(nn.Conv2d(num_filters_final, self.num_anchor_per_cls_list[k] * box_code_size, kernel_size=kernel_size, dilation=dilation, padding=padding))
                self.conv_box = nn.ModuleList(conv_box)
                if use_direction_classifier:
                    conv_dir_cls = []
                    for k in range(head_cls_num):
                        conv_dir_cls.append(nn.Conv2d(num_filters_final, self.num_anchor_per_cls_list[k] * num_direction_bins, kernel_size=kernel_size, dilation=dilation, padding=padding))
                    self.conv_dir_cls = nn.ModuleList(conv_dir_cls)

        if cfg.RPN_STAGE.RPN_HEAD.HEAD_INIT_WEIGHT:
            self.init_weight()
        
    def init_weight(self):
        cfg = self.cfg
        # print('Init head with prob 0.01')
        bias_cls = bias_init_with_prob(0.01)
        from ...accessories.Ristretto import SparseConvRistretto2d
        if isinstance(self.conv_cls, SparseConvRistretto2d):
            normal_init(self.conv_cls.conv, std=0.01, bias=bias_cls)
            normal_init(self.conv_box.conv, std=0.01)
        else:
            if (not self.split_class) and (not self.split_class_branch):
                normal_init(self.conv_cls, std=0.01, bias=bias_cls)
                normal_init(self.conv_box, std=0.01)
                if cfg.RPN_STAGE.IOU_HEAD.USE:
                    normal_init(self.conv_iou_head, std=0.01, bias=bias_cls)
                if cfg.RPN_STAGE.RPN_HEAD.REG_VARIANCE.OUTPUT:
                    normal_init(self.conv_var_head, mean=0.0, std=0.0001)
                    #nn.init.constant_(self.conv_var_head.weight, 0.0)
                    #normal_init(self.conv_var_head, mean=0.0, std=0.0001, bias=-6.0)
            else:
                for i, op in enumerate(self.conv_cls):
                    normal_init(op, std=0.01, bias=bias_cls)
                for i, op in enumerate(self.conv_box):
                    normal_init(op, std=0.01)


    def forward(self, x, sparse_masks=None, head_idx=-1, image_idx=-1):
        cfg = self.cfg
        x = self.net(x, sparse_masks=sparse_masks)
        if self.DH_scale:
            scale_attention = self.DH_scale_param * torch.mean(x,dim=[1,2,3])
            scale_attention = torch.clamp((scale_attention+1)*0.5,0,1)
            x = x * scale_attention
        if self.ssratio is not None:
            x = x * self.ssratio
        batch_size = x.shape[0]
        if sparse_masks is None:
            if (not self.split_class) and (not self.split_class_branch):
                if not self.decoupled_head:
                    box_preds = self.conv_box(x)
                    cls_preds = self.conv_cls(x)
                    if self._use_direction_classifier:
                        dir_cls_preds = self.conv_dir_cls(x)
                    if cfg.RPN_STAGE.IOU_HEAD.USE:
                        iou_preds = self.conv_iou_head(x)
                    if cfg.RPN_STAGE.RPN_HEAD.REG_VARIANCE.OUTPUT:
                        var_preds = self.conv_var_head(x)
                else:
                    decoupled_cls = self.decoupled_conv1(x)
                    decoupled_reg = self.decoupled_conv2(x)
                    box_preds = self.conv_box(decoupled_reg)
                    cls_preds = self.conv_cls(decoupled_cls)
                    if self._use_direction_classifier:
                        dir_cls_preds = self.conv_dir_cls(decoupled_cls)
                    if cfg.RPN_STAGE.IOU_HEAD.USE:
                        iou_preds = self.conv_iou_head(decoupled_reg)
                    if cfg.RPN_STAGE.RPN_HEAD.REG_VARIANCE.OUTPUT:
                        var_preds = self.conv_var_head(decoupled_reg)
            elif self.split_class:
                box_preds_list = []
                for i, op in enumerate(self.conv_box):
                    box_preds_list.append(op(x))
                box_preds = torch.cat(box_preds_list, dim=1)
                cls_preds_list = []
                for i, op in enumerate(self.conv_cls):
                    cls_preds_list.append(op(x))
                cls_preds = torch.cat(cls_preds_list, dim=1)
                if self._use_direction_classifier:
                    dir_cls_preds_list = []
                    for i, op in enumerate(self.conv_dir_cls):
                        dir_cls_preds_list.append(op(x))
                    dir_cls_preds = torch.cat(dir_cls_preds_list, dim=1)
            else:
                box_preds_list = []
                cls_preds_list = []
                if self._use_direction_classifier:
                    dir_cls_preds_list = []
                for i, op in enumerate(self.split_class_branch_group):
                    x_branch = op(x)
                    box_preds_list.append(self.conv_box[i](x_branch))
                    cls_preds_list.append(self.conv_cls[i](x_branch))
                    if self._use_direction_classifier:
                        dir_cls_preds_list.append(self.conv_dir_cls[i](x_branch))
                box_preds = torch.cat(box_preds_list, dim=1)
                cls_preds = torch.cat(cls_preds_list, dim=1)
                if self._use_direction_classifier:
                    dir_cls_preds = torch.cat(dir_cls_preds_list, dim=1)

        else:
            box_preds = self.conv_box(x, sparse_masks=sparse_masks)
            cls_preds = self.conv_cls(x, sparse_masks=sparse_masks)
        
        # [N, C, y(H), x(W)]
        C, H, W = box_preds.shape[1:]

        if not cfg.TO_CAFFE:
            box_preds = box_preds.view(-1, self._num_anchor_per_loc,
                                       self._box_code_size, H, W).permute(0, 1, 3, 4, 2).contiguous()
            cls_preds = cls_preds.view(-1, self._num_anchor_per_loc,
                                       self._num_class_cls, H, W).permute(0, 1, 3, 4, 2).contiguous()
            cls_preds = cls_preds.view(batch_size, -1, self._num_class_cls).unsqueeze(-1)

            ret_dict = {
                "box_preds": box_preds.view(batch_size, -1, self._box_code_size),
                "cls_preds": cls_preds.view(batch_size, -1, self._num_class_cls),
            }
            if self._use_direction_classifier:
                dir_cls_preds = dir_cls_preds.view(
                    -1, self._num_anchor_per_loc, self._num_direction_bins, H,
                    W).permute(0, 1, 3, 4, 2).contiguous()
                ret_dict["dir_cls_preds"] = dir_cls_preds.view(batch_size, -1, self._num_direction_bins)
            if cfg.RPN_STAGE.IOU_HEAD.USE:
                iou_preds = iou_preds.view(
                    -1, self._num_anchor_per_loc, 1, H,
                    W).permute(0, 1, 3, 4, 2).contiguous()
                ret_dict["iou_preds"] = iou_preds.view(batch_size, -1, 1)
            if cfg.RPN_STAGE.RPN_HEAD.REG_VARIANCE.OUTPUT:
                var_preds = var_preds.view(-1, self._num_anchor_per_loc, 
                    self._box_code_size, H, W).permute(0, 1, 3, 4, 2).contiguous()
                ret_dict["var_preds"] = var_preds.view(batch_size, -1, self._box_code_size)
        else:
            ret_dict = {
                "box_preds": box_preds,
                "cls_preds": cls_preds
            }
            if self._use_direction_classifier:
                ret_dict["dir_cls_preds"] = self.conv_dir_cls(x)
        return ret_dict, x


class RPNBase(nn.Module):
    """
    RPN components without head.
    """
    def __init__(self,
                 cfg,
                 use_norm=True,
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2),
                 num_filters=(128, 128, 256),
                 upsample_strides=(1, 2, 4),
                 num_upsample_filters=(256, 256, 256),
                 num_input_features=128,
                 use_groupnorm=False,
                 num_groups=32,
                 use_deconv=False,
                 pre_conv=True,
                 split_fpn=False,
                 **kwargs):
        super(RPNBase, self).__init__()
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        self.split_fpn = split_fpn
        self.cfg = cfg
        """
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        """
        if use_norm:
            if use_groupnorm:
                BatchNorm2d = partial(GroupNorm, num_groups=num_groups, eps=1e-3)
            else:
                if cfg.RPN_STAGE.BACKBONE.USE_BNET:
                    BatchNorm2d = BNET2d
                else:
                    # BN_mom = float(cfg.RPN_STAGE.BN_MOM) if cfg.RPN_STAGE.BN_MOM != -1.0 else 0.01
                    BatchNorm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01) #eps=1e-3
                    # BatchNorm2d = nn.BatchNorm2d
            Conv2d = partial(nn.Conv2d, bias=False)
            ConvTranspose2d = partial(nn.ConvTranspose2d, bias=False)
        else:
            BatchNorm2d = Empty
            Conv2d = partial(nn.Conv2d, bias=True)
            ConvTranspose2d = partial(nn.ConvTranspose2d, bias=True)

        in_filters = [num_input_features, *num_filters[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        deblocks = []
        upsample_blocks = []
        fblocks = []
        fpn_blocks = []

        ## for quantization
        if cfg.SET_QUANTIZATION:
            from pcdet.models.accessories.Ristretto import SparseConvRistretto2d
            from pcdet.models.accessories.module import ConvBNRelu
            in_group_nums = cfg.RPN_BASE_IN_GROUP_NUMS
            odd_mid_group_nums = cfg.RPN_BASE_ODD_MID_GROUP_NUMS
            even_mid_group_nums = cfg.RPN_BASE_EVEN_MID_GROUP_NUMS
            out_group_nums = cfg.RPN_BASE_OUT_GROUP_NUMS
            for i, layer_num in enumerate(layer_nums):
                block = Sequential(
                    ConvBNRelu(cfg, in_filters[i], num_filters[i], 3, stride=layer_strides[i], padding=1,
                        BN=BatchNorm2d, groups=in_group_nums[i], merge_bn=not use_norm),
                )
                for j in range(layer_num):
                    if j % 2 == 0:
                        block.add(ConvBNRelu(cfg, num_filters[i], num_filters[i], 3, padding=1,
                            BN=BatchNorm2d, groups=even_mid_group_nums[i], merge_bn=not use_norm))
                    else:
                        block.add(ConvBNRelu(cfg, num_filters[i], num_filters[i], 3, padding=1,
                            BN=BatchNorm2d, groups=odd_mid_group_nums[i], merge_bn=not use_norm))
                blocks.append(block)

                if upsample_strides[i] == 1:
                    upsample = Empty()
                elif upsample_strides[i] == 2:
                    upsample = nn.Upsample(scale_factor=2, mode='nearest')
                elif upsample_strides[i] == 4:
                    upsample = Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.Upsample(scale_factor=2, mode='nearest'))
                deconvBNRelu = ConvBNRelu(cfg, num_filters[i], num_upsample_filters[i], kernel_size=3,
                                        padding=1, BN=BatchNorm2d, groups=out_group_nums[i], merge_bn=not use_norm)
                if pre_conv or use_deconv:
                    deblock = Sequential(deconvBNRelu)
                else:
                    deblock = Sequential(
                            deconvBNRelu)
                upsample_blocks.append(upsample)
                deblocks.append(deblock)
        else:
            for i, layer_num in enumerate(layer_nums):
                if not cfg.RPN_STAGE.BACKBONE.REPVGG:
                    #BatchNorm2d = partial(GroupNorm, num_groups=int(num_filters[i]/16), eps=1e-3)
                    if not cfg.RPN_STAGE.BACKBONE.KEEP_RES:
                        groups = 1
                        block = Sequential(
                            #nn.ZeroPad2d(1),
                            Conv2d(in_filters[i], num_filters[i], 3, stride=layer_strides[i], padding=1, groups=groups),
                            BatchNorm2d(num_filters[i]),
                            RELU(inplace=True)
                        )
                    else:
                        pool_layer = nn.MaxPool2d(kernel_size=layer_strides[i], stride=layer_strides[i], padding=0)
                        #pool_layer = nn.AvgPool2d(kernel_size=layer_strides[i], stride=layer_strides[i], padding=0)
                        #pool_layer = nn.LPPool2d(2, layer_strides[i], stride=layer_strides[i])
                        block = Sequential(
                            Conv2d(in_filters[i], num_filters[i], 3, stride=1, padding=1),
                            pool_layer,
                            BatchNorm2d(num_filters[i]),
                            RELU(inplace=True)
                        )

                    if cfg.RPN_STAGE.BACKBONE.GHOST_MODULE_USE and i in cfg.RPN_STAGE.BACKBONE.GHOST_MODULE_LAYER_RANGE:
                        block = Sequential(GhostModule(in_filters[i], num_filters[i], kernel_size=3, stride=layer_strides[i]))

                    if cfg.RPN_STAGE.BACKBONE.ASYMMRES_MODULE_USE:
                        from pcdet.models.accessories.module import AsymmResBNRELU
                        if i == 0 or i == 1:
                            for j in range(layer_num):
                                block.add(Conv2d(num_filters[i], num_filters[i], 3, padding=1))
                                block.add(BatchNorm2d(num_filters[i]))
                                block.add(RELU(inplace=True))
                        else:
                            if i == 2:
                                asymm_num = 2
                            for j in range(asymm_num):
                                block.add(AsymmResBNRELU(num_filters[i], num_filters[i], 3, padding=1, BN=BatchNorm2d))
                    elif cfg.RPN_STAGE.BACKBONE.RFBBLOCK_USE:
                        from pcdet.models.accessories.module import BasicRFB
                        if i == 0:
                            for j in range(layer_num):
                                groups = cfg.RPN_STAGE.BACKBONE.CONVGROUPS[i] if j != layer_num-1 else 1
                                block.add(Conv2d(num_filters[i], num_filters[i], 3, padding=1, groups=groups))
                                block.add(BatchNorm2d(num_filters[i]))
                                block.add(RELU(inplace=True))
                        else:
                            for j in range(layer_num):
                                groups = cfg.RPN_STAGE.BACKBONE.CONVGROUPS[i] if j != layer_num-1 else 1
                                if j != layer_num - 1:
                                    block.add(Conv2d(num_filters[i], num_filters[i], 3, padding=1, groups=groups))
                                    block.add(BatchNorm2d(num_filters[i]))
                                    block.add(RELU(inplace=True))
                                else:
                                    block.add(BasicRFB(num_filters[i], num_filters[i], 1))
                    else:
                        for j in range(layer_num):
                            if cfg.RPN_STAGE.BACKBONE.GHOST_MODULE_USE and i in cfg.RPN_STAGE.BACKBONE.GHOST_MODULE_LAYER_RANGE:
                                block.add(GhostModule(num_filters[i], num_filters[i], kernel_size=3, stride=1))
                            else:
                                groups = cfg.RPN_STAGE.BACKBONE.CONVGROUPS[i] if j != layer_num-1 else 1
                                block.add(Conv2d(num_filters[i], num_filters[i], 3, padding=1, groups=groups))
                                block.add(BatchNorm2d(num_filters[i]))
                                block.add(RELU(inplace=True))
                    blocks.append(block)

                else:
                    #deploy = False if self.training else True
                    deploy = False
                    rep_block = []
                    rep_block.append(RepVGGBlock(in_channels=in_filters[i], out_channels=num_filters[i], kernel_size=3,
                                      stride=layer_strides[i], padding=1, groups=1, deploy=deploy))
                    for j in range(layer_num):
                        rep_block.append(RepVGGBlock(in_channels=num_filters[i], out_channels=num_filters[i], kernel_size=3,
                                      stride=1, padding=1, groups=1, deploy=deploy))
                    blocks.append(Sequential(*rep_block))

                if cfg.RPN_STAGE.BACKBONE.FPN.PP_FPN:
                    #BatchNorm2d = partial(GroupNorm, num_groups=int(num_upsample_filters[i]/16), eps=1e-3)
                    if use_deconv:
                        deconv = ConvTranspose2d(num_filters[i], num_upsample_filters[i], upsample_strides[i],
                                            stride=upsample_strides[i])
                        upsample = Empty()
                    else:
                        deconv = Conv2d(num_filters[i], num_upsample_filters[i], kernel_size=3, padding=1)
                        upsample = nn.Upsample(scale_factor=upsample_strides[i])
                    if pre_conv and not use_deconv:
                        if self.split_fpn:
                            deblock = Sequential(
                                deconv,
                                BatchNorm2d(num_upsample_filters[i]),
                                RELU(inplace=True)
                            )
                        else:
                            deblock = Sequential(
                                deconv,
                                BatchNorm2d(num_upsample_filters[i]),
                                RELU(inplace=True),
                                upsample
                            )
                    if not pre_conv and not use_deconv:
                        if self.split_fpn:
                            assert False, 'Post-conv does not support split_fpn'
                        else:
                            deblock = Sequential(
                                upsample,
                                deconv,
                                BatchNorm2d(num_upsample_filters[i]),
                                RELU(inplace=True)
                            )
                    if use_deconv:
                        if self.split_fpn:
                            assert False, 'Deconv does not support split_fpn'
                        else:
                            if i < 2:
                                deblock = Sequential(
                                    deconv,
                                    #BatchNorm2d(num_upsample_filters[i]),
                                    RELU(inplace=True),
                                    upsample
                                )
                            else:
                                deblock = Sequential(
                                    ConvTranspose2d(num_filters[i], num_filters[i], 2,
                                            stride=2),RELU(inplace=True),
                                    ConvTranspose2d(num_filters[i], num_upsample_filters[i], 2,
                                            stride=2),
                                    #BatchNorm2d(num_upsample_filters[i]),
                                    RELU(inplace=True),
                                    upsample
                                )
                    deblocks.append(deblock)

                    if self.split_fpn:
                        if upsample_strides[i] == 4: #hardcode for rpnbase split_fpn
                            upsample_blocks.append(Sequential(nn.Upsample(scale_factor=2),nn.Upsample(scale_factor=2)))
                        else:
                            upsample_blocks.append(nn.Upsample(scale_factor=upsample_strides[i]))

                elif cfg.RPN_STAGE.BACKBONE.FPN.PP_FPN_HEAVY:
                    if use_deconv:
                        deconv = ConvTranspose2d(num_filters[i], num_upsample_filters[i], upsample_strides[i],
                                            stride=upsample_strides[i])
                        upsample = Empty()
                    else:
                        deconv = Conv2d(num_filters[i], num_upsample_filters[i], kernel_size=3, padding=1)
                        upsample = nn.Upsample(scale_factor=upsample_strides[i])
                    if pre_conv and not use_deconv:
                        deblock = Sequential(
                            deconv,
                            BatchNorm2d(num_upsample_filters[i]),
                            RELU(inplace=True),
                            upsample
                        )
                    if not pre_conv and not use_deconv:
                        deblock = Sequential(
                            upsample,
                            deconv,
                            BatchNorm2d(num_upsample_filters[i]),
                            RELU(inplace=True)
                        )
                    if use_deconv:
                        if i == 0:
                            deblock = Sequential(
                                Conv2d(num_filters[i], num_upsample_filters[i], kernel_size=3, padding=1),
                                BatchNorm2d(num_upsample_filters[i]),
                                nn.ReLU(inplace=True)
                            )
                        elif i == 1:
                            if cfg.RPN_STAGE.BACKBONE.FPN.PP_FPN_HEAVY_upsample:
                                deblock = Sequential(
                                    Conv2d(num_filters[i], num_upsample_filters[i], kernel_size=3, padding=1),
                                    BatchNorm2d(num_upsample_filters[i]),
                                    nn.ReLU(inplace=True),
                                    nn.Upsample(scale_factor=2,mode='nearest')
                                )
                            else:
                                deblock = Sequential(
                                    Conv2d(num_filters[i], num_upsample_filters[i], kernel_size=3, padding=1),
                                    BatchNorm2d(num_upsample_filters[i]),
                                    nn.ReLU(inplace=True),
                                    ConvTranspose2d(num_upsample_filters[i], num_upsample_filters[i], 2, stride=2)
                                )
                        else:
                            if cfg.RPN_STAGE.BACKBONE.FPN.PP_FPN_HEAVY_upsample: 
                                deblock = Sequential(
                                    Conv2d(num_filters[i], num_upsample_filters[i], kernel_size=3, padding=1),
                                    BatchNorm2d(num_upsample_filters[i]),
                                    nn.ReLU(inplace=True),
                                    nn.Upsample(scale_factor=2,mode='nearest'),
                                    Conv2d(num_upsample_filters[i], num_upsample_filters[i], kernel_size=3, padding=1),
                                    BatchNorm2d(num_upsample_filters[i]),
                                    nn.ReLU(inplace=True),
                                    nn.Upsample(scale_factor=2,mode='nearest')
                                )
                            else:
                                deblock = Sequential(
                                    Conv2d(num_filters[i], num_upsample_filters[i], kernel_size=3, padding=1),
                                    BatchNorm2d(num_upsample_filters[i]),
                                    nn.ReLU(inplace=True),
                                    ConvTranspose2d(num_upsample_filters[i], num_upsample_filters[i], 2, stride=2),
                                    Conv2d(num_upsample_filters[i], num_upsample_filters[i], kernel_size=3, padding=1),
                                    BatchNorm2d(num_upsample_filters[i]),
                                    nn.ReLU(inplace=True),
                                    ConvTranspose2d(num_upsample_filters[i], num_upsample_filters[i], 2, stride=2)
                                )
                    deblocks.append(deblock)

                elif cfg.RPN_STAGE.BACKBONE.FPN.PP_FPN_ORI:
                    if use_deconv:
                        deconv = ConvTranspose2d(num_filters[i], num_upsample_filters[i], upsample_strides[i],
                                            stride=upsample_strides[i])
                        upsample = Empty()
                    else:
                        deconv = Conv2d(num_filters[i], num_upsample_filters[i], kernel_size=3, padding=1)
                        upsample = nn.Upsample(scale_factor=upsample_strides[i])
                    if pre_conv or use_deconv:
                        deblock = Sequential(
                            deconv,
                            BatchNorm2d(num_upsample_filters[i]),
                            nn.ReLU(inplace=True),
                            upsample
                        )
                    else:
                        deblock = Sequential(
                            upsample,
                            deconv,
                            BatchNorm2d(num_upsample_filters[i]),
                            nn.ReLU(inplace=True)
                        )
                    deblocks.append(deblock)

                elif cfg.RPN_STAGE.BACKBONE.FPN.FPGA_FPNv1:
                    self.fpn_stack = cfg.RPN_STAGE.BACKBONE.FPN.FPN_STACK_NUM
                    self.upsample_block_s2 = nn.Upsample(scale_factor=2, mode='nearest')
                    self.pool_block_s2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                    for k in range(self.fpn_stack):
                        laterals = []
                        # conv down
                        if i == 0:
                            c_in = num_filters[i] if k == 0 else num_upsample_filters[i]
                            block1 = Sequential(
                                Conv2d(c_in, num_upsample_filters[i], 3, padding=1),
                                BatchNorm2d(num_upsample_filters[i]),
                                RELU(inplace=True)
                            )
                        else:
                            if k == 0:
                                block1 = Sequential(
                                    Conv2d(num_filters[i], num_upsample_filters[i], 3, padding=1),
                                    BatchNorm2d(num_upsample_filters[i]),
                                    RELU(inplace=True)
                                )
                            else:
                                block1 = Empty()
                        laterals.append(block1)
                        if i > 0:
                            block2 = Sequential(
                                Conv2d(num_upsample_filters[i], num_upsample_filters[i], 3, padding=1),
                                BatchNorm2d(num_upsample_filters[i]),
                                RELU(inplace=True)
                            )
                        else:
                            block2 = Empty()
                        laterals.append(block2)
                        # conv up
                        if i < len(layer_nums) - 1 and i > 0:
                            c_in = num_upsample_filters[i]
                            block3 = Sequential(
                                Conv2d(c_in, num_upsample_filters[i], 3, padding=1),
                                BatchNorm2d(num_upsample_filters[i]),
                                RELU(inplace=True)
                            )
                        else:
                            block3 = Empty()
                        laterals.append(block3) 
                        fpn_blocks.append(laterals)

                elif cfg.RPN_STAGE.BACKBONE.FPN.FPGA_FPNv2:
                    self.fpn_stack = cfg.RPN_STAGE.BACKBONE.FPN.FPN_STACK_NUM
                    if self.fpn_stack == 1:
                        deconvBN = Sequential(
                            Conv2d(num_filters[i], num_upsample_filters[i] // 2, 3, padding=1),
                            BatchNorm2d(num_upsample_filters[i] // 2)
                        )
                        deblocks.append(deconvBN)
                    else:
                        deconvs = [] 
                        for s in range(self.fpn_stack):
                            deconvBN = Sequential(
                                Conv2d(num_filters[i] if s == 0 else num_upsample_filters[i], num_upsample_filters[i] // 2, 3, padding=1),
                                BatchNorm2d(num_upsample_filters[i] // 2)
                            )
                            deconvs.append(deconvBN)
                        deblocks.append(deconvs)

                    self.upsample_block = nn.Upsample(scale_factor=2, mode='nearest')
                    self.pool_block = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                    for k in range(self.fpn_stack):
                        laterals = []
                        # conv down
                        if i > 0 and i < len(layer_nums) - 1:
                            c_in = num_upsample_filters[i - 1] // 2 + num_upsample_filters[i] // 2
                            c_out = num_upsample_filters[i] // 2
                            node_op = Sequential(
                                Conv2d(c_in, c_out, 3, padding=1),
                                BatchNorm2d(c_out)
                            )
                        elif i == len(layer_nums) - 1:
                            c_in = num_upsample_filters[i - 1] // 2 + num_upsample_filters[i] // 2
                            c_out = num_upsample_filters[i]
                            node_op = Sequential(
                                Conv2d(c_in, c_out, 3, padding=1),
                                BatchNorm2d(c_out),
                                RELU(inplace=True)
                            )
                        else:
                            node_op = nn.Sequential()
                        laterals.append(node_op)
                        # conv eq
                        if i < len(layer_nums) - 1:
                            c_in = c_out = num_upsample_filters[i]
                            node_op = Sequential(
                                Conv2d(c_in, c_out, 3, padding=1),
                                BatchNorm2d(c_out),
                                RELU(inplace=True)
                            )
                        else:
                            node_op = nn.Sequential()
                        laterals.append(node_op)
                        # conv up
                        if i < len(layer_nums) - 1:
                            c_in = num_upsample_filters[i + 1]
                            c_out = num_upsample_filters[i] // 2
                            node_op = Sequential(
                                Conv2d(c_in, c_out, 3, padding=1),
                                BatchNorm2d(c_out)
                            )
                        else:
                            node_op = nn.Sequential()
                        laterals.append(node_op)
                        fpn_blocks.append(laterals)

                else:
                    if cfg.RPN_STAGE.BACKBONE.FPN.FCONV3:
                        fconv = Conv2d(num_filters[i], num_upsample_filters[i], kernel_size=3, padding=1)
                        fblock = Sequential(
                            fconv,
                            BatchNorm2d(num_upsample_filters[i]),
                            RELU(inplace=True)
                        )
                    else:
                        fconv = Conv2d(num_filters[i], num_upsample_filters[i], kernel_size=1, padding=0)
                        # fblock = Sequential(
                        #     fconv,
                        # )
                        fblock = Sequential(
                            fconv,
                            BatchNorm2d(num_upsample_filters[i]),
                            RELU(inplace=True)
                        )
                    fblocks.append(fblock)

        if cfg.TO_CAFFE:
            self.blocks = nn.Sequential()
            for idx, module in enumerate(blocks):
                self.blocks.add_module(str(idx), module)
            self.deblocks = nn.Sequential()
            for idx, module in enumerate(deblocks):
                self.deblocks.add_module(str(idx), module)
            self.upsample_blocks = nn.Sequential()
            for idx, module in enumerate(upsample_blocks):
                self.upsample_blocks.add_module(str(idx), module)
        else:
            self.blocks = nn.ModuleList(blocks)
            if cfg.RPN_STAGE.BACKBONE.FPN.PP_FPN:
                self.deblocks = nn.ModuleList(deblocks)
                if self.split_fpn:
                    self.upsample_blocks = nn.ModuleList(upsample_blocks)
            elif cfg.RPN_STAGE.BACKBONE.FPN.PP_FPN_ORI or cfg.RPN_STAGE.BACKBONE.FPN.PP_FPN_HEAVY:
                self.deblocks = nn.ModuleList(deblocks)
                if cfg.RPN_STAGE.BACKBONE.FPN.FPN_SUM:
                    self.fpn_merge = Conv2d(num_upsample_filters[0], num_upsample_filters[0], kernel_size=1, padding=0)
            elif cfg.RPN_STAGE.BACKBONE.FPN.FPGA_FPNv1:
                fpn_blocks = [nn.ModuleList(l) for l in fpn_blocks]
                self.fpn_blocks = nn.ModuleList(fpn_blocks)
            elif cfg.RPN_STAGE.BACKBONE.FPN.FPGA_FPNv2:
                fpn_blocks = [nn.ModuleList(l) for l in fpn_blocks]
                self.fpn_blocks= nn.ModuleList(fpn_blocks)
                if self.fpn_stack > 1:
                    deblocks = [nn.ModuleList(l) for l in deblocks]
                self.deblocks = nn.ModuleList(deblocks)
            else:
                self.fupsample = nn.Upsample(scale_factor=2, mode='nearest')
                self.fblocks = nn.ModuleList(fblocks)

        # if cfg.RPN_STAGE.BACKBONE.FIX_BACKBONE:
        #     for param in self.parameters():
        #         param.requires_grad = False

    def forward(self, x, sparse_masks=None):
        cfg = self.cfg
        if cfg.RPN_STAGE.BACKBONE.FPN.PP_FPN:
            if self.split_fpn:
                ups_stride2 = []
                ups_stride4 = []
                for i in range(len(self.blocks)):
                    x = self.blocks[i](x, sparse_masks=sparse_masks)
                    features = self.deblocks[i](x, sparse_masks=sparse_masks)
                    if i == 0:
                        ups_stride2.append(self.upsample_blocks[i](features))
                    elif i == 1:
                        if self.split_fpn == 1:
                            ups_stride4.append(features)
                        features = self.upsample_blocks[i](features)
                        ups_stride2.append(features)
                    elif i == 2:
                        features = self.upsample_blocks[i][0](features)
                        ups_stride4.append(features)
                        features = self.upsample_blocks[i][1](features)
                        ups_stride2.append(features)
                ups_stride4 = torch.cat(ups_stride4, dim=1)
                ups_stride2 = torch.cat(ups_stride2, dim=1)
                return [ups_stride2, ups_stride4]
            else:
                ups_stride2 = []
                for i in range(len(self.blocks)):
                    x = self.blocks[i](x)
                    features = self.deblocks[i](x)
                    ups_stride2.append(features)
                if not cfg.RPN_STAGE.BACKBONE.FPN.FPN_SUM:
                    ups_stride2_final = torch.cat(ups_stride2, dim=1)
                else:
                    #ups_stride2_final = ups_stride2[0] + ups_stride2[1] + ups_stride2[2]
                    y = ups_stride2[1] + ups_stride2[2]
                    ups_stride2_final = ups_stride2[0] + self.fpn_merge(y)
                return ups_stride2_final

        elif cfg.RPN_STAGE.BACKBONE.FPN.PP_FPN_ORI or cfg.RPN_STAGE.BACKBONE.FPN.PP_FPN_HEAVY:
            ups_stride2 = []
            for i in range(len(self.blocks)):
                x = self.blocks[i](x)
                features = self.deblocks[i](x)
                ups_stride2.append(features)
            if not cfg.RPN_STAGE.BACKBONE.FPN.FPN_SUM:
                ups_stride2_final = torch.cat(ups_stride2, dim=1)
            else:
                #ups_stride2_final = ups_stride2[0] + ups_stride2[1] + ups_stride2[2]
                y = ups_stride2[1] + ups_stride2[2]
                ups_stride2_final = ups_stride2[0] + self.fpn_merge(y)
            return ups_stride2_final

        elif cfg.RPN_STAGE.BACKBONE.FPN.FPGA_FPNv1:
            features = []
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, sparse_masks=sparse_masks)
                features.append(x)
            # FPN
            for k in range(self.fpn_stack):
                # bottom-up path
                for i in range(len(features)):
                    x = features[i]
                    if i == 0:
                        if k == 0:
                            features[i] = self.fpn_blocks[i * self.fpn_stack + k][0](x)
                        else:
                            x = self.upsample_block_s2(features[i + 1]) + x
                            features[i] = self.fpn_blocks[i * self.fpn_stack + k][0](x)
                    else:
                        if k == 0:
                            x = self.fpn_blocks[i * self.fpn_stack + k][0](x)
                        x = self.pool_block_s2(features[i - 1]) + x
                        features[i] = self.fpn_blocks[i * self.fpn_stack + k][1](x)
                # top-down path
                for i in range(len(features))[::-1]:
                    x = features[i]
                    if i < len(features) - 1 and i > 0:
                        x = self.upsample_block_s2(features[i + 1]) + x
                        features[i] = self.fpn_blocks[i * self.fpn_stack + k][2](x)
            ups_stride2 = torch.cat((features[0], self.upsample_block_s2(features[1]), self.upsample_block_s2(self.upsample_block_s2(features[2]))), dim=1)
            return ups_stride2

        elif cfg.RPN_STAGE.BACKBONE.FPN.FPGA_FPNv2:
            features = []
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, sparse_masks=sparse_masks)
                features.append(x)
            # FPN
            for s in range(self.fpn_stack):
                # bottom-up path
                for i in range(len(features)):
                    x = features[i]
                    if self.fpn_stack == 1:
                        x = self.deblocks[i](x)
                    else:
                        x = self.deblocks[i][s](x)
                    if i > 0:
                        x = torch.cat((self.pool_block(features[i - 1]), x), dim=1)
                    features[i] = self.fpn_blocks[i * self.fpn_stack + s][0](x)
                # top-down path
                for i in range(len(features))[::-1]:
                    x = features[i]
                    if i < len(features) - 1:
                        x = torch.cat((self.upsample_block(self.fpn_blocks[i * self.fpn_stack + s][2](features[i + 1])), x), dim=1)
                    x = self.fpn_blocks[i * self.fpn_stack + s][1](x)
                    features[i] = x
            ups_stride2 = torch.cat((features[0], self.upsample_block(features[1])), dim=1)
            ups_stride2 = torch.cat((ups_stride2, self.upsample_block(self.upsample_block(features[2]))), dim=1)
            return ups_stride2

        else:
            fblocks_f = []
            for i in range(len(self.blocks)):
                x = self.blocks[i](x)
                features = self.fblocks[i](x)
                fblocks_f.append(features)
            if cfg.RPN_STAGE.BACKBONE.FPN.FPN_CONCAT:
                for i in range(len(self.blocks)-1,0,-1):
                    if i == (len(self.blocks)-1):
                        up_f = self.fupsample(fblocks_f[i])
                    else:
                        ups_feature = torch.cat([fblocks_f[i],up_f], dim=1)
                        up_f = self.fupsample(ups_feature)
                ups_stride2 = torch.cat([fblocks_f[0],up_f], dim=1)
                return ups_stride2
            elif cfg.RPN_STAGE.BACKBONE.FPN.FPN_SUM:
                for i in range(len(self.blocks)-1,0,-1):
                    if i == (len(self.blocks)-1):
                        up_f = self.fupsample(fblocks_f[i])
                    else:
                        ups_feature = fblocks_f[i] + up_f
                        if i == 1:
                            ups_stride4 = ups_feature.clone()
                        up_f = self.fupsample(ups_feature)
                ups_stride2 = fblocks_f[0] + up_f
                if cfg.RPN_STAGE.BACKBONE.FPN.FPN_OUT2:
                    return ups_stride2, ups_stride4
                else:
                    return ups_stride2
            else:
                assert False, "Ori FPN type wrong"


class RPNBase_VoVNet(nn.Module):
    """
    RPN components without head.
    """
    def __init__(self,
                 cfg,
                 use_norm=True,
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2),
                 num_filters=(128, 128, 256),
                 upsample_strides=(1, 2, 4),
                 num_upsample_filters=(256, 256, 256),
                 num_input_features=128,
                 use_groupnorm=False,
                 num_groups=32,
                 use_deconv=False,
                 pre_conv=True,
                 split_fpn=False,
                 **kwargs):
        super(RPNBase_VoVNet, self).__init__()
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        self.split_fpn = split_fpn
        self.cfg = cfg
        if use_norm:
            if use_groupnorm:
                BatchNorm2d = partial(GroupNorm, num_groups=num_groups, eps=1e-3)
            else:
                if cfg.RPN_STAGE.BACKBONE.USE_BNET:
                    BatchNorm2d = BNET2d
                else:
                    BatchNorm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
            Conv2d = partial(nn.Conv2d, bias=False)
            ConvTranspose2d = partial(nn.ConvTranspose2d, bias=False)
        else:
            BatchNorm2d = Empty
            Conv2d = partial(nn.Conv2d, bias=True)
            ConvTranspose2d = partial(nn.ConvTranspose2d, bias=True)

        in_filters = [num_input_features, *num_filters[:-1]]
        blocks = []
        deblocks = []
        upsample_blocks = []
        fblocks = []
        fpn_blocks = []
        osa_trans = []
        osa_SE = []

        for i, layer_num in enumerate(layer_nums):
            block = nn.ModuleList()
            block.append(Sequential(
                Conv2d(in_filters[i], num_filters[i], 3, stride=layer_strides[i], padding=1),
                BatchNorm2d(num_filters[i]),
                RELU(inplace=True)
            ))
            for j in range(layer_num):
                block.append(Sequential(
                    Conv2d(num_filters[i], num_filters[i], 3, padding=1),
                    BatchNorm2d(num_filters[i]),
                    RELU(inplace=True)
                ))
            blocks.append(block)

            if not cfg.RPN_STAGE.BACKBONE.VoV_TRANS_CONCAT:
                osa_trans.append(Sequential(
                    Conv2d(num_filters[i], num_filters[i], 1, padding=0),
                    BatchNorm2d(num_filters[i]),
                    RELU(inplace=True)
                ))
            else:
                osa_trans.append(Sequential(
                    Conv2d(num_filters[i]*(layer_num+1), num_filters[i], 1, padding=0),
                    BatchNorm2d(num_filters[i]),
                    RELU(inplace=True)
                ))

            if cfg.RPN_STAGE.BACKBONE.VoV_SE:
                osa_SE.append(eSEModule(num_filters[i]))

            if cfg.RPN_STAGE.BACKBONE.FPN.PP_FPN:
                if use_deconv:
                    deconv = ConvTranspose2d(num_filters[i], num_upsample_filters[i], upsample_strides[i],
                                        stride=upsample_strides[i])
                    upsample = Empty()
                else:
                    deconv = Conv2d(num_filters[i], num_upsample_filters[i], kernel_size=3, padding=1)
                    upsample = nn.Upsample(scale_factor=upsample_strides[i])
                if pre_conv and not use_deconv:
                    if self.split_fpn:
                        deblock = Sequential(
                            deconv,
                            BatchNorm2d(num_upsample_filters[i]),
                            RELU(inplace=True)
                        )
                    else:
                        deblock = Sequential(
                            deconv,
                            BatchNorm2d(num_upsample_filters[i]),
                            RELU(inplace=True),
                            upsample
                        )
                if not pre_conv and not use_deconv:
                    if self.split_fpn:
                        assert False, 'Post-conv does not support split_fpn'
                    else:
                        deblock = Sequential(
                            upsample,
                            deconv,
                            BatchNorm2d(num_upsample_filters[i]),
                            RELU(inplace=True)
                        )
                if use_deconv:
                    if self.split_fpn:
                        assert False, 'Deconv does not support split_fpn'
                    else:
                        deblock = Sequential(
                            deconv,
                            BatchNorm2d(num_upsample_filters[i]),
                            RELU(inplace=True),
                            upsample
                        )
                deblocks.append(deblock)

                if self.split_fpn:
                    if upsample_strides[i] == 4: #hardcode for rpnbase split_fpn
                        upsample_blocks.append(Sequential(nn.Upsample(scale_factor=2),nn.Upsample(scale_factor=2)))
                    else:
                        upsample_blocks.append(nn.Upsample(scale_factor=upsample_strides[i]))

            elif cfg.RPN_STAGE.BACKBONE.FPN.FPGA_FPNv1:
                self.fpn_stack = cfg.RPN_STAGE.BACKBONE.FPN.FPN_STACK_NUM
                self.upsample_block_s2 = nn.Upsample(scale_factor=2, mode='nearest')
                self.pool_block_s2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                for k in range(self.fpn_stack):
                    laterals = []
                    # conv down
                    if i == 0:
                        c_in = num_filters[i] if k == 0 else num_upsample_filters[i]
                        block1 = Sequential(
                            Conv2d(c_in, num_upsample_filters[i], 3, padding=1),
                            BatchNorm2d(num_upsample_filters[i]),
                            RELU(inplace=True)
                        )
                    else:
                        if k == 0:
                            block1 = Sequential(
                                Conv2d(num_filters[i], num_upsample_filters[i], 3, padding=1),
                                BatchNorm2d(num_upsample_filters[i]),
                                RELU(inplace=True)
                            )
                        else:
                            block1 = Empty()
                    laterals.append(block1)
                    if i > 0:
                        block2 = Sequential(
                            Conv2d(num_upsample_filters[i], num_upsample_filters[i], 3, padding=1),
                            BatchNorm2d(num_upsample_filters[i]),
                            RELU(inplace=True)
                        )
                    else:
                        block2 = Empty()
                    laterals.append(block2)
                    # conv up
                    if i < len(layer_nums) - 1 and i > 0:
                        c_in = num_upsample_filters[i]
                        block3 = Sequential(
                            Conv2d(c_in, num_upsample_filters[i], 3, padding=1),
                            BatchNorm2d(num_upsample_filters[i]),
                            RELU(inplace=True)
                        )
                    else:
                        block3 = Empty()
                    laterals.append(block3) 
                    fpn_blocks.append(laterals)

            elif cfg.RPN_STAGE.BACKBONE.FPN.FPGA_FPNv2:
                self.fpn_stack = cfg.RPN_STAGE.BACKBONE.FPN.FPN_STACK_NUM
                if self.fpn_stack == 1:
                    deconvBN = Sequential(
                        Conv2d(num_filters[i], num_upsample_filters[i] // 2, 3, padding=1),
                        BatchNorm2d(num_upsample_filters[i] // 2)
                    )
                    deblocks.append(deconvBN)
                else:
                    deconvs = [] 
                    for s in range(self.fpn_stack):
                        deconvBN = Sequential(
                            Conv2d(num_filters[i] if s == 0 else num_upsample_filters[i], num_upsample_filters[i] // 2, 3, padding=1),
                            BatchNorm2d(num_upsample_filters[i] // 2)
                        )
                        deconvs.append(deconvBN)
                    deblocks.append(deconvs)

                self.upsample_block = nn.Upsample(scale_factor=2, mode='nearest')
                self.pool_block = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                for k in range(self.fpn_stack):
                    laterals = []
                    # conv down
                    if i > 0 and i < len(layer_nums) - 1:
                        c_in = num_upsample_filters[i - 1] // 2 + num_upsample_filters[i] // 2
                        c_out = num_upsample_filters[i] // 2
                        node_op = Sequential(
                            Conv2d(c_in, c_out, 3, padding=1),
                            BatchNorm2d(c_out)
                        )
                    elif i == len(layer_nums) - 1:
                        c_in = num_upsample_filters[i - 1] // 2 + num_upsample_filters[i] // 2
                        c_out = num_upsample_filters[i]
                        node_op = Sequential(
                            Conv2d(c_in, c_out, 3, padding=1),
                            BatchNorm2d(c_out),
                            RELU(inplace=True)
                        )
                    else:
                        node_op = nn.Sequential()
                    laterals.append(node_op)
                    # conv eq
                    if i < len(layer_nums) - 1:
                        c_in = c_out = num_upsample_filters[i]
                        node_op = Sequential(
                            Conv2d(c_in, c_out, 3, padding=1),
                            BatchNorm2d(c_out),
                            RELU(inplace=True)
                        )
                    else:
                        node_op = nn.Sequential()
                    laterals.append(node_op)
                    # conv up
                    if i < len(layer_nums) - 1:
                        c_in = num_upsample_filters[i + 1]
                        c_out = num_upsample_filters[i] // 2
                        node_op = Sequential(
                            Conv2d(c_in, c_out, 3, padding=1),
                            BatchNorm2d(c_out)
                        )
                    else:
                        node_op = nn.Sequential()
                    laterals.append(node_op)
                    fpn_blocks.append(laterals)

            else:
                if cfg.RPN_STAGE.BACKBONE.FPN.FCONV3:
                    fconv = Conv2d(num_filters[i], num_upsample_filters[i], kernel_size=3, padding=1)
                    fblock = Sequential(
                        fconv,
                        BatchNorm2d(num_upsample_filters[i]),
                        RELU(inplace=True)
                    )
                else:
                    fconv = Conv2d(num_filters[i], num_upsample_filters[i], kernel_size=1, padding=0)
                    # fblock = Sequential(
                    #     fconv,
                    # )
                    fblock = Sequential(
                        fconv,
                        BatchNorm2d(num_upsample_filters[i]),
                        RELU(inplace=True)
                    )
                fblocks.append(fblock)

        if cfg.TO_CAFFE:
            self.blocks = nn.Sequential()
            for idx, module in enumerate(blocks):
                self.blocks.add_module(str(idx), module)
            self.deblocks = nn.Sequential()
            for idx, module in enumerate(deblocks):
                self.deblocks.add_module(str(idx), module)
            self.upsample_blocks = nn.Sequential()
            for idx, module in enumerate(upsample_blocks):
                self.upsample_blocks.add_module(str(idx), module)
        else:
            self.blocks = nn.ModuleList(blocks)
            self.osa_trans = nn.ModuleList(osa_trans)
            if cfg.RPN_STAGE.BACKBONE.VoV_SE:
                self.osa_SE = nn.ModuleList(osa_SE)
            if cfg.RPN_STAGE.BACKBONE.FPN.PP_FPN:
                self.deblocks = nn.ModuleList(deblocks)
                if self.split_fpn:
                    self.upsample_blocks = nn.ModuleList(upsample_blocks)
            elif cfg.RPN_STAGE.BACKBONE.FPN.FPGA_FPNv1:
                fpn_blocks = [nn.ModuleList(l) for l in fpn_blocks]
                self.fpn_blocks = nn.ModuleList(fpn_blocks)
            elif cfg.RPN_STAGE.BACKBONE.FPN.FPGA_FPNv2:
                fpn_blocks = [nn.ModuleList(l) for l in fpn_blocks]
                self.fpn_blocks= nn.ModuleList(fpn_blocks)
                if self.fpn_stack > 1:
                    deblocks = [nn.ModuleList(l) for l in deblocks]
                self.deblocks = nn.ModuleList(deblocks)
            else:
                self.fupsample = nn.Upsample(scale_factor=2, mode='nearest')
                self.fblocks = nn.ModuleList(fblocks)

        # if cfg.RPN_STAGE.BACKBONE.FIX_BACKBONE:
        #     for param in self.parameters():
        #         param.requires_grad = False

    def forward(self, x, sparse_masks=None):
        cfg = self.cfg
        features = []
        for i, block in enumerate(self.blocks):
            output = []
            for j, fb in enumerate(block):
                x = fb(x)
                if j == 0:
                    identity_feat = x
                output.append(x)
            if cfg.RPN_STAGE.BACKBONE.VoV_TRANS_CONCAT:
                x = torch.cat(output, dim=1)
            else:
                x = sum(output)
            x = self.osa_trans[i](x)
            if cfg.RPN_STAGE.BACKBONE.VoV_SE:
                x = self.osa_SE[i](x)
            if cfg.RPN_STAGE.BACKBONE.VoV_IDMAP:
                x = x + identity_feat
            features.append(x)

        if cfg.RPN_STAGE.BACKBONE.FPN.PP_FPN:
            ups_stride2 = []
            for i in range(len(self.blocks)):
                f = self.deblocks[i](features[i], sparse_masks=sparse_masks)
                ups_stride2.append(f)
            ups_stride2 = torch.cat(ups_stride2, dim=1)
            return ups_stride2

        elif cfg.RPN_STAGE.BACKBONE.FPN.FPGA_FPNv1:
            # FPN
            for k in range(self.fpn_stack):
                # bottom-up path
                for i in range(len(features)):
                    x = features[i]
                    if i == 0:
                        if k == 0:
                            features[i] = self.fpn_blocks[i * self.fpn_stack + k][0](x)
                        else:
                            x = self.upsample_block_s2(features[i + 1]) + x
                            features[i] = self.fpn_blocks[i * self.fpn_stack + k][0](x)
                    else:
                        if k == 0:
                            x = self.fpn_blocks[i * self.fpn_stack + k][0](x)
                        x = self.pool_block_s2(features[i - 1]) + x
                        features[i] = self.fpn_blocks[i * self.fpn_stack + k][1](x)
                # top-down path
                for i in range(len(features))[::-1]:
                    x = features[i]
                    if i < len(features) - 1 and i > 0:
                        x = self.upsample_block_s2(features[i + 1]) + x
                        features[i] = self.fpn_blocks[i * self.fpn_stack + k][2](x)
            ups_stride2 = torch.cat((features[0], self.upsample_block_s2(features[1]), self.upsample_block_s2(self.upsample_block_s2(features[2]))), dim=1)
            return ups_stride2

        elif cfg.RPN_STAGE.BACKBONE.FPN.FPGA_FPNv2:
            # FPN
            for s in range(self.fpn_stack):
                # bottom-up path
                for i in range(len(features)):
                    x = features[i]
                    if self.fpn_stack == 1:
                        x = self.deblocks[i](x)
                    else:
                        x = self.deblocks[i][s](x)
                    if i > 0:
                        x = torch.cat((self.pool_block(features[i - 1]), x), dim=1)
                    features[i] = self.fpn_blocks[i * self.fpn_stack + s][0](x)
                # top-down path
                for i in range(len(features))[::-1]:
                    x = features[i]
                    if i < len(features) - 1:
                        x = torch.cat((self.upsample_block(self.fpn_blocks[i * self.fpn_stack + s][2](features[i + 1])), x), dim=1)
                    x = self.fpn_blocks[i * self.fpn_stack + s][1](x)
                    features[i] = x
            ups_stride2 = torch.cat((features[0], self.upsample_block(features[1])), dim=1)
            ups_stride2 = torch.cat((ups_stride2, self.upsample_block(self.upsample_block(features[2]))), dim=1)
            return ups_stride2

        else:
            fblocks_f = []
            for i in range(len(self.blocks)):
                f = self.fblocks[i](features[i])
                fblocks_f.append(f)
            if cfg.RPN_STAGE.BACKBONE.FPN.FPN_CONCAT:
                for i in range(len(self.blocks)-1,0,-1):
                    if i == (len(self.blocks)-1):
                        up_f = self.fupsample(fblocks_f[i])
                    else:
                        ups_feature = torch.cat([fblocks_f[i],up_f], dim=1)
                        up_f = self.fupsample(ups_feature)
                ups_stride2 = torch.cat([fblocks_f[0],up_f], dim=1)
                return ups_stride2
            elif cfg.RPN_STAGE.BACKBONE.FPN.FPN_SUM:
                for i in range(len(self.blocks)-1,0,-1):
                    if i == (len(self.blocks)-1):
                        up_f = self.fupsample(fblocks_f[i])
                    else:
                        ups_feature = fblocks_f[i] + up_f
                        if i == 1:
                            ups_stride4 = ups_feature.clone()
                        up_f = self.fupsample(ups_feature)
                ups_stride2 = fblocks_f[0] + up_f
                if cfg.RPN_STAGE.BACKBONE.FPN.FPN_OUT2:
                    return ups_stride2, ups_stride4
                else:
                    return ups_stride2
            else:
                assert False, "Ori FPN type wrong"


class RPNV2(RPNBase):
    """Compare with RPN, RPNV2 support arbitrary number of stage.
    """
    def __init__(self,
                 num_class=2,
                 num_upsample_filters=(256, 256, 256),
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 box_code_size=7,
                 use_rc_net=False,
                 **rpn_base_args):
        super(RPNV2, self).__init__(**rpn_base_args)
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        self._use_rc_net = use_rc_net

        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        self.conv_cls = nn.Conv2d(sum(num_upsample_filters), num_cls, 1)

        reg_channels = num_anchor_per_loc * box_code_size
        self.conv_box = nn.Conv2d(sum(num_upsample_filters), reg_channels, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(sum(num_upsample_filters), num_anchor_per_loc * 2, 1)

        if self._use_rc_net:
            self.conv_rc = nn.Conv2d(sum(num_upsample_filters), num_anchor_per_loc * box_code_size, 1)

    def forward(self, x, bev=None):
        # t = time.time()
        # torch.cuda.synchronize()
        x = super(RPNV2, self).forward(x)
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds
        if self._use_rc_net:
            rc_preds = self.conv_rc(x)
            rc_preds = rc_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["rc_preds"] = rc_preds
        # torch.cuda.synchronize()
        # print("rpn forward time", time.time() - t)
        return ret_dict

# if cfg.RPN_BASE_MODEL == 'RPNBase':
#     RPN_BASE_MODEL = RPNBase
# elif cfg.RPN_BASE_MODEL == 'RPNBase_ResNet':
#     RPN_BASE_MODEL = RPNBase_ResNet
# elif cfg.RPN_BASE_MODEL == 'HourglassNet':
#     RPN_BASE_MODEL = HourglassNet
# elif cfg.RPN_BASE_MODEL == 'RPNBase_DenseNet':
#     RPN_BASE_MODEL = RPNBase_DenseNet
# elif cfg.RPN_BASE_MODEL == 'RPNBase_VoVNet':
#     RPN_BASE_MODEL = RPNBase_VoVNet
# else:
#     assert False, "RPN_BASE_MODEL not available"
class MultiHeadRPN(RPNBase):
    def __init__(self,
                 cfg,
                 is_distiller=False,
                 num_class=2,
                 num_anchor_per_cls=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 box_code_size=7,
                 num_direction_bins=2,
                 drop_ratio=0.,
                 rpn_head_cfgs=None,
                 **rpn_base_args):
        """
            upsample_strides support float: [0.25, 0.5, 1]
            if upsample_strides < 1, conv2d will be used instead of convtranspose2d.
        """
        super(MultiHeadRPN, self).__init__(cfg, **rpn_base_args)
        self.cfg = cfg
        self._num_direction_bins = num_direction_bins
        self._num_class = num_class
        self._drop_ratio = drop_ratio
        self._use_direction_classifier = use_direction_classifier
        self._box_code_size = box_code_size
        self._split_fpn = rpn_base_args['split_fpn']

        if len(rpn_base_args['num_upsample_filters']) == 0:
            final_num_filters = self._num_out_filters
        else:
            # if cfg.RPN_BASE_MODEL == 'RPNBase' or cfg.RPN_BASE_MODEL == 'RPNBase_ResNet':
            #     final_num_filters = sum(rpn_base_args['num_upsample_filters'])
            if cfg.RPN_BASE_MODEL == 'HourglassNet':
                final_num_filters = rpn_base_args['num_hourglass_feats'] * 2
            else:
                if not cfg.RPN_STAGE.BACKBONE.FPN.FPN_SUM:
                    final_num_filters = sum(rpn_base_args['num_upsample_filters'])
                else:
                    final_num_filters = rpn_base_args['num_upsample_filters'][0]

        self.is_distiller = is_distiller
        if cfg.DISTILLER_1x1CONV:
            if cfg.CONCAT0:
                self.concat_conv2d = nn.Conv2d(96, 192, kernel_size=1, stride=1, bias=True)
            if cfg.CONV2127:
                self.conv21_conv2d = nn.Conv2d(32, 128, kernel_size=1, stride=1, bias=True)
                self.conv27_conv2d = nn.Conv2d(32, 128, kernel_size=1, stride=1, bias=True)

        self._make_head(
                        rpn_head_cfgs,
                        final_num_filters,
                        num_anchor_per_cls,
                        encode_background_as_zeros,
                        use_direction_classifier,
                        box_code_size,
                        num_direction_bins,
                        rpn_base_args["use_norm"])
        if cfg.MODEL.get('FUSER', None) is not None:
            self.fuser = fuser.__all__[cfg.MODEL.FUSER.NAME](
                model_cfg=cfg.MODEL.FUSER
            )
    def _make_head(self,
                   rpn_head_cfgs,
                   final_num_filters,
                   num_anchor_per_cls,
                   encode_background_as_zeros,
                   use_direction_classifier,
                   box_code_size,
                   num_direction_bins,
                   use_norm):
        cfg = self.cfg
        rpn_heads = []
        self.rpn_head_input_key = []
        for i, rpn_head_cfg in enumerate(rpn_head_cfgs):
            self.rpn_head_input_key.append('out')
            in_num_filters = final_num_filters
            stride = rpn_head_cfg.downsample_level // 2
            if i == 1 and self._split_fpn == 1:
                in_num_filters = final_num_filters * 2 // 3
                stride = 1  # this is a hard code here
            elif i == 1 and self._split_fpn == 2:
                in_num_filters = final_num_filters // 3
                stride = 1  # this is a hard code here
            elif i == 1 and (cfg.RPN_STAGE.BACKBONE.FPN.FPN_OUT2 and cfg.RPN_STAGE.BACKBONE.FPN.FPN_SUM):
                stride = 1

            rpn_heads.append(
                ConvHead(
                    cfg=cfg,
                    in_filters=in_num_filters,
                    num_class=self._num_class,
                    num_anchor_per_cls=num_anchor_per_cls,
                    encode_background_as_zeros=encode_background_as_zeros,
                    use_direction_classifier=use_direction_classifier,
                    box_code_size=box_code_size,
                    num_direction_bins=num_direction_bins,
                    num_filters=rpn_head_cfg.num_filters,
                    head_cls_num=len(rpn_head_cfg.class_name),
                    conv_nums=rpn_head_cfg.conv_nums,
                    dilations=rpn_head_cfg.dilation,
                    stride=stride,
                    use_se=rpn_head_cfg.use_se,
                    use_res=rpn_head_cfg.use_res,
                    use_glore=rpn_head_cfg.use_glore,
                    use_shuffle=rpn_head_cfg.use_shuffle,
                    ratio=rpn_head_cfg.ratio,
                    se_ratio=rpn_head_cfg.se_ratio,
                    num_res=rpn_head_cfg.num_res,
                    ds_keep_ratio=rpn_head_cfg.ds_keep_ratio,
                    use_norm=use_norm,
                    split_class=rpn_head_cfg.split_class if 'split_class' in rpn_head_cfg else False,
                    split_class_branch=rpn_head_cfg.split_class_branch if 'split_class_branch' in rpn_head_cfg else False,
                    split_class_branch_convnum=rpn_head_cfg.split_class_branch_convnum if 'split_class_branch_convnum' in rpn_head_cfg else 1,
                    split_class_branch_convfeats=rpn_head_cfg.split_class_branch_convfeats if 'split_class_branch_convfeats' in rpn_head_cfg else 32,
                    num_anchor_per_cls_list=rpn_head_cfg.num_anchor_per_cls_list if 'num_anchor_per_cls_list' in rpn_head_cfg else [num_anchor_per_cls]*len(rpn_head_cfg.class_name),
                    ks1=rpn_head_cfg.ks1 if 'ks1' in rpn_head_cfg else 1,
                    firstBN=rpn_head_cfg.firstBN if 'firstBN' in rpn_head_cfg else False,
                    DH_scale=rpn_head_cfg.DH_scale if 'DH_scale' in rpn_head_cfg else False,
                    ssratio=rpn_head_cfg.ssratio if 'ssratio' in rpn_head_cfg else None,
                    decoupled_head=rpn_head_cfg.decoupled_head if 'decoupled_head' in rpn_head_cfg else False,
                    decoupled_ch=rpn_head_cfg.decoupled_ch if 'decoupled_ch' in rpn_head_cfg else None
                )
            )
        if cfg.TO_CAFFE:
            self.rpn_heads = nn.Sequential()
            for idx, module in enumerate(rpn_heads):
                self.rpn_heads.add_module(str(idx), module)
        else:
            self.rpn_heads = nn.ModuleList(rpn_heads)

    def forward(self, x, sparse_masks=None, img_ft=None, image_idx=-1):
        res = super().forward(x, sparse_masks=sparse_masks)
        cfg = self.cfg
        if cfg.CONCAT0:
            distiller_res = [res]
        else:
            distiller_res = []
        ret_dicts = []
        if cfg.MODEL.get('FUSER', None) is not None: 
            res = self.fuser(res, img_ft)
        for i, rpn_head in enumerate(self.rpn_heads):
            if self._split_fpn or (cfg.RPN_STAGE.BACKBONE.FPN.FPN_OUT2 and cfg.RPN_STAGE.BACKBONE.FPN.FPN_SUM):
                head_input = res[i]
            else:
                head_input = res
            ret_dict, distiller_x = rpn_head(head_input, sparse_masks=sparse_masks, head_idx=i, image_idx=image_idx)
            ret_dicts.append(ret_dict)
            if cfg.CONV2127:
                distiller_res.append(distiller_x)

        if not cfg.TO_CAFFE:
            ret = {
                "box_preds": torch.cat([ret_dict["box_preds"] for ret_dict in ret_dicts], dim=1),
                "cls_preds": torch.cat([ret_dict["cls_preds"] for ret_dict in ret_dicts], dim=1),
            }
            # box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
            # cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
            if self._use_direction_classifier:
                ret["dir_cls_preds"] = torch.cat([ret_dict["dir_cls_preds"] for ret_dict in ret_dicts], dim=1)
            if cfg.RPN_STAGE.IOU_HEAD.USE:
                ret["iou_preds"] = torch.cat([ret_dict["iou_preds"] for ret_dict in ret_dicts], dim=1)
            if cfg.RPN_STAGE.RPN_HEAD.REG_VARIANCE.OUTPUT:
                ret["var_preds"] = torch.cat([ret_dict["var_preds"] for ret_dict in ret_dicts], dim=1)
        else:
            ret = ret_dicts

        if cfg.DISTILLER_1x1CONV:
            if cfg.CONCAT0 and distiller_res[0].shape[1] == 96:
                distiller_res[0] = self.concat_conv2d(distiller_res[0])
            if cfg.CONV2127 and distiller_res[-1].shape[1] == 32:
                distiller_res[-1] = self.conv27_conv2d(distiller_res[-1])
            if cfg.CONV2127 and distiller_res[-2].shape[1] == 32:
                distiller_res[-2] = self.conv21_conv2d(distiller_res[-2])

        return ret, distiller_res


class MultiHeadRPN_CAM(nn.Module):
    def __init__(self,
                 cfg,
                 is_distiller=False,
                 num_class=2,
                 num_anchor_per_cls=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 box_code_size=7,
                 num_direction_bins=2,
                 drop_ratio=0.,
                 rpn_head_cfgs=None,
                 **rpn_base_args):
        """
            upsample_strides support float: [0.25, 0.5, 1]
            if upsample_strides < 1, conv2d will be used instead of convtranspose2d.
        """
        super(MultiHeadRPN_CAM, self).__init__()
        self.cfg = cfg
        self._num_direction_bins = num_direction_bins
        self._num_class = num_class
        self._drop_ratio = drop_ratio
        self._use_direction_classifier = use_direction_classifier
        self._box_code_size = box_code_size
        self._split_fpn = rpn_base_args['split_fpn']

        if len(rpn_base_args['num_upsample_filters']) == 0:
            final_num_filters = self._num_out_filters
        else:
            if cfg.RPN_BASE_MODEL == 'HourglassNet':
                final_num_filters = rpn_base_args['num_hourglass_feats'] * 2
            else:
                if not cfg.RPN_STAGE.BACKBONE.FPN.FPN_SUM:
                    final_num_filters = sum(rpn_base_args['num_upsample_filters'])
                else:
                    final_num_filters = rpn_base_args['num_upsample_filters'][0]

        self.is_distiller = is_distiller

        self._make_head(
                        rpn_head_cfgs,
                        final_num_filters,
                        num_anchor_per_cls,
                        encode_background_as_zeros,
                        use_direction_classifier,
                        box_code_size,
                        num_direction_bins,
                        rpn_base_args["use_norm"])
        self.multihead_fs = (cfg.MODEL.get('FUSER', None) is not None) and (cfg.MODEL.get('FUSER_CAM', None) is not None)
        if cfg.MODEL.get('FUSER', None) is not None:
            if not self.multihead_fs:
                self.fuser = fuser.__all__[cfg.MODEL.FUSER.NAME](
                    model_cfg=cfg.MODEL.FUSER
                )
            else:
                self.fuser = fuser.__all__[cfg.MODEL.FUSER_CAM.NAME](
                    model_cfg=cfg.MODEL.FUSER_CAM
                )
    def _make_head(self,
                   rpn_head_cfgs,
                   final_num_filters,
                   num_anchor_per_cls,
                   encode_background_as_zeros,
                   use_direction_classifier,
                   box_code_size,
                   num_direction_bins,
                   use_norm):
        cfg = self.cfg
        rpn_heads = []
        self.rpn_head_input_key = []
        for i, rpn_head_cfg in enumerate(rpn_head_cfgs):
            self.rpn_head_input_key.append('out')
            in_num_filters = final_num_filters
            stride = rpn_head_cfg.downsample_level // 2
            if i == 1 and self._split_fpn == 1:
                in_num_filters = final_num_filters * 2 // 3
                stride = 1  # this is a hard code here
            elif i == 1 and self._split_fpn == 2:
                in_num_filters = final_num_filters // 3
                stride = 1  # this is a hard code here
            elif i == 1 and (cfg.RPN_STAGE.BACKBONE.FPN.FPN_OUT2 and cfg.RPN_STAGE.BACKBONE.FPN.FPN_SUM):
                stride = 1

            rpn_heads.append(
                ConvHead(
                    cfg=cfg,
                    in_filters=in_num_filters,
                    num_class=self._num_class,
                    num_anchor_per_cls=num_anchor_per_cls,
                    encode_background_as_zeros=encode_background_as_zeros,
                    use_direction_classifier=use_direction_classifier,
                    box_code_size=box_code_size,
                    num_direction_bins=num_direction_bins,
                    num_filters=rpn_head_cfg.num_filters,
                    head_cls_num=len(rpn_head_cfg.class_name),
                    conv_nums=rpn_head_cfg.conv_nums,
                    dilations=rpn_head_cfg.dilation,
                    stride=stride,
                    use_se=rpn_head_cfg.use_se,
                    use_res=rpn_head_cfg.use_res,
                    use_glore=rpn_head_cfg.use_glore,
                    use_shuffle=rpn_head_cfg.use_shuffle,
                    ratio=rpn_head_cfg.ratio,
                    se_ratio=rpn_head_cfg.se_ratio,
                    num_res=rpn_head_cfg.num_res,
                    ds_keep_ratio=rpn_head_cfg.ds_keep_ratio,
                    use_norm=use_norm,
                    split_class=rpn_head_cfg.split_class if 'split_class' in rpn_head_cfg else False,
                    split_class_branch=rpn_head_cfg.split_class_branch if 'split_class_branch' in rpn_head_cfg else False,
                    split_class_branch_convnum=rpn_head_cfg.split_class_branch_convnum if 'split_class_branch_convnum' in rpn_head_cfg else 1,
                    split_class_branch_convfeats=rpn_head_cfg.split_class_branch_convfeats if 'split_class_branch_convfeats' in rpn_head_cfg else 32,
                    num_anchor_per_cls_list=rpn_head_cfg.num_anchor_per_cls_list if 'num_anchor_per_cls_list' in rpn_head_cfg else [num_anchor_per_cls]*len(rpn_head_cfg.class_name),
                    ks1=rpn_head_cfg.ks1 if 'ks1' in rpn_head_cfg else 1,
                    firstBN=rpn_head_cfg.firstBN if 'firstBN' in rpn_head_cfg else False,
                    DH_scale=rpn_head_cfg.DH_scale if 'DH_scale' in rpn_head_cfg else False,
                    ssratio=rpn_head_cfg.ssratio if 'ssratio' in rpn_head_cfg else None,
                    decoupled_head=rpn_head_cfg.decoupled_head if 'decoupled_head' in rpn_head_cfg else False,
                    decoupled_ch=rpn_head_cfg.decoupled_ch if 'decoupled_ch' in rpn_head_cfg else None
                )
            )
        if cfg.TO_CAFFE:
            self.rpn_heads = nn.Sequential()
            for idx, module in enumerate(rpn_heads):
                self.rpn_heads.add_module(str(idx), module)
        else:
            self.rpn_heads = nn.ModuleList(rpn_heads)

    def forward(self, img_ft, sparse_masks=None, image_idx=-1):
        cfg = self.cfg
        if cfg.CONCAT0:
            distiller_res = [img_ft]
        else:
            distiller_res = []
        ret_dicts = []
        if cfg.MODEL.get('FUSER', None) is not None: 
            img_ft = self.fuser(img_ft)
        for i, rpn_head in enumerate(self.rpn_heads):
            head_input = img_ft
            ret_dict, distiller_x = rpn_head(head_input, sparse_masks=sparse_masks, head_idx=i, image_idx=image_idx)
            ret_dicts.append(ret_dict)
            if cfg.CONV2127:
                distiller_res.append(distiller_x)

        if not cfg.TO_CAFFE:
            ret = {
                "box_preds": torch.cat([ret_dict["box_preds"] for ret_dict in ret_dicts], dim=1),
                "cls_preds": torch.cat([ret_dict["cls_preds"] for ret_dict in ret_dicts], dim=1),
            }
            # box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
            # cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
            if self._use_direction_classifier:
                ret["dir_cls_preds"] = torch.cat([ret_dict["dir_cls_preds"] for ret_dict in ret_dicts], dim=1)
            if cfg.RPN_STAGE.IOU_HEAD.USE:
                ret["iou_preds"] = torch.cat([ret_dict["iou_preds"] for ret_dict in ret_dicts], dim=1)
            if cfg.RPN_STAGE.RPN_HEAD.REG_VARIANCE.OUTPUT:
                ret["var_preds"] = torch.cat([ret_dict["var_preds"] for ret_dict in ret_dicts], dim=1)
        else:
            ret = ret_dicts

        if cfg.DISTILLER_1x1CONV:
            if cfg.CONCAT0 and distiller_res[0].shape[1] == 96:
                distiller_res[0] = self.concat_conv2d(distiller_res[0])
            if cfg.CONV2127 and distiller_res[-1].shape[1] == 32:
                distiller_res[-1] = self.conv27_conv2d(distiller_res[-1])
            if cfg.CONV2127 and distiller_res[-2].shape[1] == 32:
                distiller_res[-2] = self.conv21_conv2d(distiller_res[-2])

        return ret, distiller_res



class MultiHeadRPN_fsmul(RPNBase):
    # for multihead fusion
    def __init__(self,
                 cfg,
                 is_distiller=False,
                 num_class=2,
                 num_anchor_per_cls=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 box_code_size=7,
                 num_direction_bins=2,
                 drop_ratio=0.,
                 rpn_head_cfgs=None,
                 **rpn_base_args):
        """
            upsample_strides support float: [0.25, 0.5, 1]
            if upsample_strides < 1, conv2d will be used instead of convtranspose2d.
        """
        super(MultiHeadRPN_fsmul, self).__init__(cfg, **rpn_base_args)
        self.cfg = cfg
        self._num_direction_bins = num_direction_bins
        self._num_class = num_class
        self._drop_ratio = drop_ratio
        self._use_direction_classifier = use_direction_classifier
        self._box_code_size = box_code_size
        self._split_fpn = rpn_base_args['split_fpn']

        if len(rpn_base_args['num_upsample_filters']) == 0:
            final_num_filters = self._num_out_filters
        else:
            # if cfg.RPN_BASE_MODEL == 'RPNBase' or cfg.RPN_BASE_MODEL == 'RPNBase_ResNet':
            #     final_num_filters = sum(rpn_base_args['num_upsample_filters'])
            if cfg.RPN_BASE_MODEL == 'HourglassNet':
                final_num_filters = rpn_base_args['num_hourglass_feats'] * 2
            else:
                if not cfg.RPN_STAGE.BACKBONE.FPN.FPN_SUM:
                    final_num_filters = sum(rpn_base_args['num_upsample_filters'])
                else:
                    final_num_filters = rpn_base_args['num_upsample_filters'][0]

        self.is_distiller = is_distiller
        if cfg.DISTILLER_1x1CONV:
            if cfg.CONCAT0:
                self.concat_conv2d = nn.Conv2d(96, 192, kernel_size=1, stride=1, bias=True)
            if cfg.CONV2127:
                self.conv21_conv2d = nn.Conv2d(32, 128, kernel_size=1, stride=1, bias=True)
                self.conv27_conv2d = nn.Conv2d(32, 128, kernel_size=1, stride=1, bias=True)

        self._make_head(
                        rpn_head_cfgs,
                        final_num_filters,
                        num_anchor_per_cls,
                        encode_background_as_zeros,
                        use_direction_classifier,
                        box_code_size,
                        num_direction_bins,
                        rpn_base_args["use_norm"])
        self._make_head_lidar(
                        rpn_head_cfgs,
                        final_num_filters,
                        num_anchor_per_cls,
                        encode_background_as_zeros,
                        use_direction_classifier,
                        box_code_size,
                        num_direction_bins,
                        rpn_base_args["use_norm"])
        if cfg.MODEL.get('FUSER', None) is not None:
            self.fuser = fuser.__all__[cfg.MODEL.FUSER.NAME](
                model_cfg=cfg.MODEL.FUSER
            )
    def _make_head(self,
                   rpn_head_cfgs,
                   final_num_filters,
                   num_anchor_per_cls,
                   encode_background_as_zeros,
                   use_direction_classifier,
                   box_code_size,
                   num_direction_bins,
                   use_norm):
        cfg = self.cfg
        rpn_heads = []
        self.rpn_head_input_key = []
        for i, rpn_head_cfg in enumerate(rpn_head_cfgs):
            self.rpn_head_input_key.append('out')
            in_num_filters = final_num_filters
            stride = rpn_head_cfg.downsample_level // 2
            if i == 1 and self._split_fpn == 1:
                in_num_filters = final_num_filters * 2 // 3
                stride = 1  # this is a hard code here
            elif i == 1 and self._split_fpn == 2:
                in_num_filters = final_num_filters // 3
                stride = 1  # this is a hard code here
            elif i == 1 and (cfg.RPN_STAGE.BACKBONE.FPN.FPN_OUT2 and cfg.RPN_STAGE.BACKBONE.FPN.FPN_SUM):
                stride = 1

            rpn_heads.append(
                ConvHead(
                    cfg=cfg,
                    in_filters=in_num_filters,
                    num_class=self._num_class,
                    num_anchor_per_cls=num_anchor_per_cls,
                    encode_background_as_zeros=encode_background_as_zeros,
                    use_direction_classifier=use_direction_classifier,
                    box_code_size=box_code_size,
                    num_direction_bins=num_direction_bins,
                    num_filters=rpn_head_cfg.num_filters,
                    head_cls_num=len(rpn_head_cfg.class_name),
                    conv_nums=rpn_head_cfg.conv_nums,
                    dilations=rpn_head_cfg.dilation,
                    stride=stride,
                    use_se=rpn_head_cfg.use_se,
                    use_res=rpn_head_cfg.use_res,
                    use_glore=rpn_head_cfg.use_glore,
                    use_shuffle=rpn_head_cfg.use_shuffle,
                    ratio=rpn_head_cfg.ratio,
                    se_ratio=rpn_head_cfg.se_ratio,
                    num_res=rpn_head_cfg.num_res,
                    ds_keep_ratio=rpn_head_cfg.ds_keep_ratio,
                    use_norm=use_norm,
                    split_class=rpn_head_cfg.split_class if 'split_class' in rpn_head_cfg else False,
                    split_class_branch=rpn_head_cfg.split_class_branch if 'split_class_branch' in rpn_head_cfg else False,
                    split_class_branch_convnum=rpn_head_cfg.split_class_branch_convnum if 'split_class_branch_convnum' in rpn_head_cfg else 1,
                    split_class_branch_convfeats=rpn_head_cfg.split_class_branch_convfeats if 'split_class_branch_convfeats' in rpn_head_cfg else 32,
                    num_anchor_per_cls_list=rpn_head_cfg.num_anchor_per_cls_list if 'num_anchor_per_cls_list' in rpn_head_cfg else [num_anchor_per_cls]*len(rpn_head_cfg.class_name),
                    ks1=rpn_head_cfg.ks1 if 'ks1' in rpn_head_cfg else 1,
                    firstBN=rpn_head_cfg.firstBN if 'firstBN' in rpn_head_cfg else False,
                    DH_scale=rpn_head_cfg.DH_scale if 'DH_scale' in rpn_head_cfg else False,
                    ssratio=rpn_head_cfg.ssratio if 'ssratio' in rpn_head_cfg else None,
                    decoupled_head=rpn_head_cfg.decoupled_head if 'decoupled_head' in rpn_head_cfg else False,
                    decoupled_ch=rpn_head_cfg.decoupled_ch if 'decoupled_ch' in rpn_head_cfg else None
                )
            )
        if cfg.TO_CAFFE:
            self.rpn_heads = nn.Sequential()
            for idx, module in enumerate(rpn_heads):
                self.rpn_heads.add_module(str(idx), module)
        else:
            self.rpn_heads = nn.ModuleList(rpn_heads)

    def _make_head_lidar(self,
                   rpn_head_cfgs,
                   final_num_filters,
                   num_anchor_per_cls,
                   encode_background_as_zeros,
                   use_direction_classifier,
                   box_code_size,
                   num_direction_bins,
                   use_norm):
        cfg = self.cfg
        rpn_heads = []
        self.rpn_head_input_key = []
        for i, rpn_head_cfg in enumerate(rpn_head_cfgs):
            self.rpn_head_input_key.append('out')
            in_num_filters = final_num_filters
            stride = rpn_head_cfg.downsample_level // 2
            if i == 1 and self._split_fpn == 1:
                in_num_filters = final_num_filters * 2 // 3
                stride = 1  # this is a hard code here
            elif i == 1 and self._split_fpn == 2:
                in_num_filters = final_num_filters // 3
                stride = 1  # this is a hard code here
            elif i == 1 and (cfg.RPN_STAGE.BACKBONE.FPN.FPN_OUT2 and cfg.RPN_STAGE.BACKBONE.FPN.FPN_SUM):
                stride = 1

            rpn_heads.append(
                ConvHead(
                    cfg=cfg,
                    in_filters=in_num_filters,
                    num_class=self._num_class,
                    num_anchor_per_cls=num_anchor_per_cls,
                    encode_background_as_zeros=encode_background_as_zeros,
                    use_direction_classifier=use_direction_classifier,
                    box_code_size=box_code_size,
                    num_direction_bins=num_direction_bins,
                    num_filters=rpn_head_cfg.num_filters,
                    head_cls_num=len(rpn_head_cfg.class_name),
                    conv_nums=rpn_head_cfg.conv_nums,
                    dilations=rpn_head_cfg.dilation,
                    stride=stride,
                    use_se=rpn_head_cfg.use_se,
                    use_res=rpn_head_cfg.use_res,
                    use_glore=rpn_head_cfg.use_glore,
                    use_shuffle=rpn_head_cfg.use_shuffle,
                    ratio=rpn_head_cfg.ratio,
                    se_ratio=rpn_head_cfg.se_ratio,
                    num_res=rpn_head_cfg.num_res,
                    ds_keep_ratio=rpn_head_cfg.ds_keep_ratio,
                    use_norm=use_norm,
                    split_class=rpn_head_cfg.split_class if 'split_class' in rpn_head_cfg else False,
                    split_class_branch=rpn_head_cfg.split_class_branch if 'split_class_branch' in rpn_head_cfg else False,
                    split_class_branch_convnum=rpn_head_cfg.split_class_branch_convnum if 'split_class_branch_convnum' in rpn_head_cfg else 1,
                    split_class_branch_convfeats=rpn_head_cfg.split_class_branch_convfeats if 'split_class_branch_convfeats' in rpn_head_cfg else 32,
                    num_anchor_per_cls_list=rpn_head_cfg.num_anchor_per_cls_list if 'num_anchor_per_cls_list' in rpn_head_cfg else [num_anchor_per_cls]*len(rpn_head_cfg.class_name),
                    ks1=rpn_head_cfg.ks1 if 'ks1' in rpn_head_cfg else 1,
                    firstBN=rpn_head_cfg.firstBN if 'firstBN' in rpn_head_cfg else False,
                    DH_scale=rpn_head_cfg.DH_scale if 'DH_scale' in rpn_head_cfg else False,
                    ssratio=rpn_head_cfg.ssratio if 'ssratio' in rpn_head_cfg else None,
                    decoupled_head=rpn_head_cfg.decoupled_head if 'decoupled_head' in rpn_head_cfg else False,
                    decoupled_ch=rpn_head_cfg.decoupled_ch if 'decoupled_ch' in rpn_head_cfg else None
                )
            )
        if cfg.TO_CAFFE:
            self.rpn_heads_lidar = nn.Sequential()
            for idx, module in enumerate(rpn_heads):
                self.rpn_heads_lidar.add_module(str(idx), module)
        else:
            self.rpn_heads_lidar = nn.ModuleList(rpn_heads)

    def forward(self, x, sparse_masks=None, img_ft=None, image_idx=-1):
        res = super().forward(x, sparse_masks=sparse_masks)
        cfg = self.cfg
        if cfg.CONCAT0:
            distiller_res = [res]
        else:
            distiller_res = []
        ret_dicts_ld, ret_dicts = [], []
        # lidar only
        for i, rpn_head in enumerate(self.rpn_heads_lidar):
            if self._split_fpn or (cfg.RPN_STAGE.BACKBONE.FPN.FPN_OUT2 and cfg.RPN_STAGE.BACKBONE.FPN.FPN_SUM):
                head_input = res[i]
            else:
                head_input = res
            ret_dict_ld, distiller_x_ld = rpn_head(head_input, sparse_masks=sparse_masks, head_idx=i, image_idx=image_idx)
            ret_dicts_ld.append(ret_dict_ld)
            if cfg.CONV2127:
                distiller_res.append(distiller_x)

        if not cfg.TO_CAFFE:
            ret_ld = {
                "box_preds": torch.cat([ret_dict["box_preds"] for ret_dict in ret_dicts_ld], dim=1),
                "cls_preds": torch.cat([ret_dict["cls_preds"] for ret_dict in ret_dicts_ld], dim=1),
            }
            # box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
            # cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
            if self._use_direction_classifier:
                ret_ld["dir_cls_preds"] = torch.cat([ret_dict["dir_cls_preds"] for ret_dict in ret_dicts_ld], dim=1)
            if cfg.RPN_STAGE.IOU_HEAD.USE:
                ret_ld["iou_preds"] = torch.cat([ret_dict["iou_preds"] for ret_dict in ret_dicts_ld], dim=1)
            if cfg.RPN_STAGE.RPN_HEAD.REG_VARIANCE.OUTPUT:
                ret_ld["var_preds"] = torch.cat([ret_dict["var_preds"] for ret_dict in ret_dicts_ld], dim=1)
        else:
            ret_ld = ret_dicts_ld
        # fusion
        if cfg.MODEL.get('FUSER', None) is not None: 
            res = self.fuser(res, img_ft)
        for i, rpn_head in enumerate(self.rpn_heads):
            if self._split_fpn or (cfg.RPN_STAGE.BACKBONE.FPN.FPN_OUT2 and cfg.RPN_STAGE.BACKBONE.FPN.FPN_SUM):
                head_input = res[i]
            else:
                head_input = res
            ret_dict, distiller_x = rpn_head(head_input, sparse_masks=sparse_masks, head_idx=i, image_idx=image_idx)
            ret_dicts.append(ret_dict)
            if cfg.CONV2127:
                distiller_res.append(distiller_x)

        if not cfg.TO_CAFFE:
            ret = {
                "box_preds": torch.cat([ret_dict["box_preds"] for ret_dict in ret_dicts], dim=1),
                "cls_preds": torch.cat([ret_dict["cls_preds"] for ret_dict in ret_dicts], dim=1),
            }
            # box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
            # cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
            if self._use_direction_classifier:
                ret["dir_cls_preds"] = torch.cat([ret_dict["dir_cls_preds"] for ret_dict in ret_dicts], dim=1)
            if cfg.RPN_STAGE.IOU_HEAD.USE:
                ret["iou_preds"] = torch.cat([ret_dict["iou_preds"] for ret_dict in ret_dicts], dim=1)
            if cfg.RPN_STAGE.RPN_HEAD.REG_VARIANCE.OUTPUT:
                ret["var_preds"] = torch.cat([ret_dict["var_preds"] for ret_dict in ret_dicts], dim=1)

        if cfg.DISTILLER_1x1CONV:
            if cfg.CONCAT0 and distiller_res[0].shape[1] == 96:
                distiller_res[0] = self.concat_conv2d(distiller_res[0])
            if cfg.CONV2127 and distiller_res[-1].shape[1] == 32:
                distiller_res[-1] = self.conv27_conv2d(distiller_res[-1])
            if cfg.CONV2127 and distiller_res[-2].shape[1] == 32:
                distiller_res[-2] = self.conv21_conv2d(distiller_res[-2])

        return ret, ret_ld, distiller_res