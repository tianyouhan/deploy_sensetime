import torch.nn as nn
import torch
from torch.nn import functional as F


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict
    

class HeightCompression_Cat_Seg(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.smooth2 = nn.Conv2d(832, 256, kernel_size=1, padding=0, bias=False)
        self.smooth3 = nn.Conv2d(832, 256, kernel_size=1, padding=0, bias=False)
        self.smooth4 = nn.Conv2d(768, 256, kernel_size=1, padding=0, bias=False)
        self.smooth5 = nn.Conv2d(256, 256, kernel_size=1, padding=0, bias=False)
        self.smooth_out = nn.Conv2d(256, 32, kernel_size=3, padding=1, bias=False)

    def sparse_to_dense(self, feat):
        spatial_features = feat.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        return spatial_features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """

        feat_2 = self.smooth2(self.sparse_to_dense(batch_dict['multi_scale_3d_features']['x_conv2']))  # 1/2 scale
        feat_3 = self.smooth3(self.sparse_to_dense(batch_dict['multi_scale_3d_features']['x_conv3']))  # 1/4 scale
        feat_4 = self.smooth4(self.sparse_to_dense(batch_dict['multi_scale_3d_features']['x_conv4']))  # 1/8 scale
        feat_5 = self.smooth5(self.sparse_to_dense(batch_dict['encoded_spconv_tensor']))               # 1/8 scale

        feat_3 = F.interpolate(feat_3, scale_factor=2, mode='bilinear', align_corners=False)
        feat_4 = F.interpolate(feat_4, scale_factor=4, mode='bilinear', align_corners=False)
        feat_5 = F.interpolate(feat_5, scale_factor=4, mode='bilinear', align_corners=False)

        feat_cat = feat_2 + feat_3 + feat_4 + feat_5
        feat_cat = self.smooth_out(feat_cat)

        batch_dict['spatial_features'] = feat_cat
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict


class HeightCompression_Up_Seg(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.up = torch.nn.PixelShuffle(4)

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        spatial_features = self.up(spatial_features)

        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']

        return batch_dict