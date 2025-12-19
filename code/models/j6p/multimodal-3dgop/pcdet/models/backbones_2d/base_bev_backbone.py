import numpy as np
import torch
import torch.nn as nn
from pcdet.utils.common_utils import save_np
import os

class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride > 1 or (stride == 1 and not self.model_cfg.get('USE_CONV_FOR_NO_STRIDE', False)):
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x
        data_dict['spatial_features_2d_lidar'] = x
        return data_dict

class BaseBEVBackbone_cam(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride > 1 or (stride == 1 and not self.model_cfg.get('USE_CONV_FOR_NO_STRIDE', False)):
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in
        self.extra_net = self.model_cfg.get('EXTRA_NET', False)
        if self.extra_net:
            in_channel = self.extra_net.IN_CHANNEL
            out_channel = self.extra_net.OUT_CHANNEL
            self.extra_conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True)
            )
            deblock_inc = self.extra_net.DEBLOCK_INC
            deblock_ouc = self.extra_net.DEBLOCK_OUC
            self.extra_deblock = nn.Sequential(
                nn.ConvTranspose2d(
                    deblock_inc, deblock_ouc,
                    2,
                    stride=2, bias=False
                ),
                nn.BatchNorm2d(deblock_ouc, eps=1e-3, momentum=0.01),
                nn.ReLU()
            )
            self.num_bev_features = out_channel

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features_img']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)
        if self.extra_net:
            x = self.extra_deblock(x)
            x = self.extra_conv(x)
        data_dict['spatial_features_2d_cam'] = x
        return data_dict

class BaseBEVBackboneV1(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        layer_nums = self.model_cfg.LAYER_NUMS
        num_filters = self.model_cfg.NUM_FILTERS
        assert len(layer_nums) == len(num_filters) == 2

        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        assert len(num_upsample_filters) == len(upsample_strides)

        num_levels = len(layer_nums)
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    num_filters[idx], num_filters[idx], kernel_size=3,
                    stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['multi_scale_2d_features']

        x_conv4 = spatial_features['x_conv4']
        x_conv5 = spatial_features['x_conv5']

        ups = [self.deblocks[0](x_conv4)]

        x = self.blocks[1](x_conv5)
        ups.append(self.deblocks[1](x))

        x = torch.cat(ups, dim=1)
        x = self.blocks[0](x)

        data_dict['spatial_features_2d'] = x

        return data_dict
    

class BaseBEVBackbone_FPN(nn.Module):
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
        outs = self.forward_base(spatial_features)
        if 'cat' in self.return_list or len(self.return_list) == 1:
            data_dict['spatial_features_2d'] = outs[-1]
            data_dict['spatial_features_2d_lidar'] = outs[-1]
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
    

class BaseBEVBackbone_FPN_cam(nn.Module):
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
        self.use_ca = self.model_cfg.get('USE_CA', False)
        self.out_indices = self.model_cfg.get('OUT_INDICES', None)
        self.return_list = self.model_cfg.get('RETURN_LIST', ['cat'])
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

    def channel_attention(self, feature):
        weight = self.CA_pooling(feature) # （1，96，1，1）
        weight_out = self.CA(weight)
        weight_final = weight + weight_out
        out = torch.mul(feature, weight_final)
        return out

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features_img']

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

        if 'cat' in self.return_list or len(self.return_list) == 1:
            data_dict['spatial_features_2d'] = outs[-1]
            data_dict['spatial_features_2d_cam'] = outs[-1]
        if 'scale' in self.return_list:
            data_dict['spatial_features_2d_cam_scale'] = outs[0]

        if os.getenv("CALIB") == 'True':
            calib_path = os.getenv("CALIB_PATH")
            save_dir = os.path.join(calib_path, 'cam-bev-backbone')
            save_np(os.path.join(save_dir, "outputs/spatial_features_2d_cam/{}".format(self.cnt)), data_dict['spatial_features_2d_cam'])
            save_dir = os.path.join(calib_path, 'cam-branch')
            save_np(os.path.join(save_dir, "outputs/spatial_features_2d_cam/{}".format(self.cnt)), data_dict['spatial_features_2d_cam'])
        self.cnt += 1
        return data_dict


def downsample_block(in_channel, out_channel, stride):
    if stride >= 1:
        downsample = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
        )
    else:
        scale_factor = int(1 / stride)
        downsample = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.ConvTranspose2d(out_channel, out_channel, scale_factor, stride=scale_factor, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
        )
    return downsample

