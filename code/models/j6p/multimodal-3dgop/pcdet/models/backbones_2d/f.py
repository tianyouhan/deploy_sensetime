import numpy as np
import torch
import torch.nn as nn


class IdentityBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        c_in = sum(num_upsample_filters)
        self.num_bev_features = c_in
    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        x = spatial_features
        data_dict['spatial_features_2d'] = x
        return data_dict
