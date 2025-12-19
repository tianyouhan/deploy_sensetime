import torch
from torch import nn


class IdentityFuser(nn.Module):
    def __init__(self, model_cfg) -> None:
        super().__init__()
        self.downsample = nn.Identity()
        
    def forward(self,batch_dict):
        """
        Args:
            batch_dict:
                spatial_features_img (tensor): Bev features from image modality
                spatial_features (tensor): Bev features from lidar modality

        Returns:
            batch_dict:
                spatial_features (tensor): Bev features after muli-modal fusion
        """
        img_bev = batch_dict['spatial_features_img']
        batch_dict['spatial_features'] = self.downsample(img_bev)
        return batch_dict