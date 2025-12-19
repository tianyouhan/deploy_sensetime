import numpy as np
import torch
import torch.nn as nn
from pcdet.utils.common_utils import save_np
import os
from .base_bev_backbone import downsample_block
from torch.quantization import DeQuantStub

class BaseBEVBackbone_FPN_Qat(nn.Module):
    """general FPN with deconv"""
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.cnt = 0
        self.model_cfg = model_cfg
        self.downsample_input = self.model_cfg.get('DOWNSAMPLE_INPUT', None)
        use_deconv = self.model_cfg.get('USE_DECONV', False)
        layer_nums = self.model_cfg.LAYER_NUMS
        layer_strides = self.model_cfg.LAYER_STRIDES
        num_filters = self.model_cfg.NUM_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        self.use_cam = self.model_cfg.get('USE_CAM', False)
        self.use_ca = self.model_cfg.get('USE_CA', False)
        self.out_indices = self.model_cfg.get('OUT_INDICES', None)
        self.return_list = self.model_cfg.get('RETURN_LIST', ['cat'])
        self.use_forward_base = self.model_cfg.get('USE_FORWARD_BASE', True)
        if self.out_indices is not None:
            self.num_bev_features = sum([num_upsample_filters[i] for i in self.out_indices])
        else:
            self.num_bev_features = sum(num_upsample_filters)

        assert len(layer_nums) == len(num_filters) == len(num_upsample_filters)

        c_in_list = [input_channels, *num_filters[:-1]]
        self.downblocks = nn.ModuleList()
        self.upblocks = nn.ModuleList()
        self.lateral_layers = nn.ModuleList()
        self.smooth_layers = nn.ModuleList()
        if 'cat' in self.return_list:
            self.upfinal = nn.ModuleList()

        if self.downsample_input is not None:
            self.down_input = downsample_block(self.downsample_input.IN_CHANNEL, self.downsample_input.OUT_CHANNEL, stride=self.downsample_input.STRIDE)

        for idx in range(len(layer_nums)):
            cur_layers = [
                nn.Conv2d(c_in_list[idx], num_filters[idx], kernel_size=3, stride=layer_strides[idx], padding=1, bias=False),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True)
            ]

            for i in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True)
                ])
            self.downblocks.append(nn.Sequential(*cur_layers))

            lateral_layer = [
                nn.Conv2d(num_filters[idx], num_upsample_filters[idx], kernel_size=1, stride=1, padding=0, bias=False),
            ]
            self.lateral_layers.append(nn.Sequential(*lateral_layer))

            if 'cat' in self.return_list:
                if use_deconv:
                    upfinal = [nn.Conv2d(num_upsample_filters[idx], num_upsample_filters[idx], kernel_size=1, stride=1, padding=0, bias=False)]
                    for _ in range(upsample_strides[idx] // 2):
                        upfinal.extend([
                            nn.ConvTranspose2d(num_upsample_filters[idx], num_upsample_filters[idx], 2, 2, padding=0, bias=False),
                            nn.ReLU(inplace=True)
                        ])
                else:
                    upfinal = [
                        nn.Upsample(scale_factor=upsample_strides[idx], mode='nearest'),
                        ]
                self.upfinal.append(nn.Sequential(*upfinal))

            if idx > 0:
                if use_deconv:
                    cur_layers_up = [
                    nn.ConvTranspose2d(num_upsample_filters[idx], num_upsample_filters[idx], kernel_size=layer_strides[idx], stride=layer_strides[idx], padding=0, bias=False),
                    nn.ReLU(inplace=True)
                    ]
                else:
                    cur_layers_up = [
                    nn.Upsample(scale_factor=layer_strides[idx], mode='nearest'),
                    ]
                self.upblocks.append(nn.Sequential(*cur_layers_up))

                smooth_layer = [
                nn.Conv2d(num_upsample_filters[idx], num_upsample_filters[idx], kernel_size=3, stride=1, padding=1, bias=False),
                ]
                self.smooth_layers.append(nn.Sequential(*smooth_layer))
            
        if self.use_ca:
            self.CA_pooling = nn.AdaptiveAvgPool2d((1,1))
            self.CA = nn.Sequential(nn.Conv2d(self.num_bev_features, self.num_bev_features // 2, 1, 1, 0, bias=True),
                                    nn.ReLU(),
                                    nn.Conv2d(self.num_bev_features // 2, self.num_bev_features, 1, 1, 0, bias=True))
        if os.getenv("SPLIT_BKB") == 'True':
            print('split_bkb')
        else:
            self.spatial_features_2d_lidar_dequant = DeQuantStub()
                
    def forward_base(self, spatial_features):
        if self.downsample_input is not None:
            spatial_features = self.down_input(spatial_features)

        feats_down, feats_up, feat = [], [], spatial_features
        for i in range(len(self.downblocks)):
            feat = self.downblocks[i](feat)
            feats_down.append(feat)
        
        for i in range(len(feats_down) - 1, 0, -1):
            if i == len(feats_down) - 1:
                feat_up_add = self.lateral_layers[i](feats_down[i])
                feats_up.append(feat_up_add)
            feat_up_add = self.upblocks[i - 1](feat_up_add) + self.lateral_layers[i - 1](feats_down[i - 1])
            feats_up.append(self.smooth_layers[i - 1](feat_up_add))
        feats_up = feats_up[::-1]  # reverse scale to 1/2, 1/4, ...

        outs = []
        if 'scale' in self.return_list:
            out_scales = []
            for id in self.out_indices:
                out_scales.append(feats_up[id])
            outs.append(out_scales)

        if 'cat' in self.return_list:
            uplist = []
            out_ids = self.out_indices if self.out_indices is not None else range(len(feats_up))
            for i in out_ids:
                uplist.append(self.upfinal[i](feats_up[i]))
            concat_feat = torch.cat(uplist, dim=1)
            outs.append(concat_feat)
            if self.use_ca:
                outs[-1] = self.channel_attention(outs[-1])
        return outs
    
    def channel_attention(self, feature):
        weight = self.CA_pooling(feature) # （1，96，1，1）
        weight_out = self.CA(weight)
        weight_final = weight + weight_out
        out = torch.mul(feature, weight_final)
        return out

    def forward_lidar(self, data_dict):
        if os.getenv("CALIB") == 'True':
            calib_path = os.getenv("CALIB_PATH")
            save_dir = os.path.join(calib_path, 'lidar-bev-backbone')
            save_np(os.path.join(save_dir, "inputs/spatial_features/{}".format(self.cnt)), data_dict['spatial_features'])

        spatial_features = data_dict['spatial_features']
        if self.use_forward_base:
            outs = self.forward_base(spatial_features)
        else:
            outs = [spatial_features]
        if 'cat' in self.return_list or len(self.return_list) == 1:
            if os.getenv("SPLIT_BKB") == 'True':
                print('split_bkb_in_forward_lidar')
                spatial_features_2d = outs[-1]
            else:
                spatial_features_2d = self.spatial_features_2d_lidar_dequant(outs[-1])
            
            data_dict['spatial_features_2d'] = spatial_features_2d
            data_dict['spatial_features_2d_lidar'] = spatial_features_2d
        if 'scale' in self.return_list:
            data_dict['spatial_features_2d_lidar_scale'] = outs[0]

        if os.getenv("CALIB") == 'True':
            calib_path = os.getenv("CALIB_PATH")
            save_dir = os.path.join(calib_path, 'lidar-bev-backbone')
            save_np(os.path.join(save_dir, "outputs/spatial_features_2d_lidar/{}".format(self.cnt)), data_dict['spatial_features_2d_lidar'])
            save_dir = os.path.join(calib_path, 'lidar-branch')
            save_np(os.path.join(save_dir, "outputs/spatial_features_2d_lidar/{}".format(self.cnt)), data_dict['spatial_features_2d_lidar'])
        return data_dict

    def forward_camera(self, data_dict):
        spatial_features = data_dict['spatial_features_img']
        outs = self.forward_base(spatial_features)
        if 'cat' in self.return_list or len(self.return_list) == 1:
            data_dict['spatial_features_2d_cam'] = outs[-1]
        if 'scale' in self.return_list:
            data_dict['spatial_features_2d_cam_scale'] = outs[0]
        return data_dict

    def forward(self, data_dict):
        data_dict = self.forward_lidar(data_dict)
        if self.use_cam:
            data_dict = self.forward_camera(data_dict)
        self.cnt += 1
        return data_dict
    
