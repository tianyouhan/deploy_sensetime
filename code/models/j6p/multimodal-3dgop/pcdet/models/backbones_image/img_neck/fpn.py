# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...model_utils.basic_block_2d import BasicBlock2D
from pcdet.utils.common_utils import save_np
import os

class CustomFPN(nn.Module):

    def __init__(self, model_cfg):
        super().__init__()
        self.cnt = 0
        self.model_cfg = model_cfg
        in_channels =  self.model_cfg.IN_CHANNELS
        out_channels = self.model_cfg.OUT_CHANNELS
        self.num_ins = len(in_channels)
        num_outs = self.model_cfg.NUM_OUTS
        start_level = self.model_cfg.START_LEVEL
        end_level = self.model_cfg.END_LEVEL
        add_extra_convs = False
        self.in_channels = in_channels
        self.num_outs = num_outs
        self.relu_before_extra_convs = False
        self.no_norm_on_lateral = False
        self.fp16_enabled = False
        self.upsample_cfg = dict(mode='nearest')
        self.out_ids = self.model_cfg.OUT_IDS
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            # assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = nn.Conv2d(
                in_channels[i],
                out_channels, kernel_size=1, bias = False
            )
            self.lateral_convs.append(l_conv)
            if i in self.out_ids:
                fpn_conv = nn.Conv2d(out_channels,out_channels, kernel_size=3, padding=1, bias = False)
                self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = nn.Conv2d(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1, bias=False)
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, batch_dict):
        """Forward function."""
        inputs = batch_dict['image_features']
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [self.fpn_convs[i](laterals[j - self.start_level]) for i, j in enumerate(self.out_ids)]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        if os.getenv("EXPORT_FP16") == 'True':
            for idx in range(len(outs)):
                outs[idx] = torch.clamp(outs[idx], min=-65504, max=65504)
        batch_dict['image_fpn'] = tuple(outs)

        if os.getenv("CALIB") == 'True':
            calib_path = os.getenv("CALIB_PATH")
            save_dir = os.path.join(calib_path, 'cam-backbone')
            save_np(os.path.join(save_dir, "outputs/image_fpn/{}".format(self.cnt)), batch_dict['image_fpn'][0])
        self.cnt += 1
        return batch_dict
    

class FeatCat(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        in_channels =  self.model_cfg.IN_CHANNELS
        out_channels = self.model_cfg.OUT_CHANNELS
        num_ins = len(in_channels)
        num_outs = self.model_cfg.NUM_OUTS
        start_level = self.model_cfg.START_LEVEL
        end_level = self.model_cfg.END_LEVEL
        self.out_ids = self.model_cfg.OUT_IDS

        self.in_channels = in_channels

        if end_level == -1:
            self.backbone_end_level = num_ins
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = nn.Sequential(
                BasicBlock2D(in_channels[i],out_channels, kernel_size=1, bias = False),
                BasicBlock2D(out_channels,out_channels, kernel_size=3, padding=1, bias = False)
            )
            self.lateral_convs.append(l_conv)
        self.final_layer = BasicBlock2D(out_channels * len(self.lateral_convs), out_channels, kernel_size=1, bias = False)

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                image_features (list[tensor]): Multi-stage features from image backbone.
        Returns:
            batch_dict:
                image_fpn (list(tensor)): FPN features.
        """
        inputs = batch_dict['image_features']
        assert len(inputs) == len(self.in_channels)
        assert len(self.out_ids) == 1

        # build laterals
        laterals = [inputs[i] for i in range(self.start_level, self.backbone_end_level)]

        fusions = []
        for i in range(len(laterals)):
            laterals[i] = self.lateral_convs[i](laterals[i])
            x = F.interpolate(
                laterals[i],
                size=laterals[self.out_ids[0] - self.start_level].shape[2:],
                mode='bilinear', align_corners=False,
            )
            fusions.append(x)

        # build outputs
        outs = self.final_layer(torch.cat(fusions, dim=1))
        batch_dict['image_fpn'] = tuple([outs])
        return batch_dict
