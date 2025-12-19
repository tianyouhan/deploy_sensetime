import torch
from torch import nn
from .attention import CFAMBlock, CBAM, NONLocalBlock2D

class SE_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.att(x)



class ConvFuser(nn.Module):
    def __init__(self,model_cfg) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        in_channel = self.model_cfg.IN_CHANNEL
        out_channel = self.model_cfg.OUT_CHANNEL
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
            )
        self.use_se = self.model_cfg.SE if 'SE' in self.model_cfg else False
        self.use_cam = self.model_cfg.CAM if 'CAM' in self.model_cfg else False
        self.use_cbam = self.model_cfg.CBAM if 'CBAM' in self.model_cfg else False
        self.use_non_local = self.model_cfg.NON_LOCAL if 'NON_LOCAL' in self.model_cfg else False
        if self.use_se:
            self.se = SE_Block(out_channel)
        if self.use_cam:
            self.cam = CFAMBlock(out_channel, out_channel)
        if self.use_cbam:
            self.conv1 = nn.Conv2d(out_channel, out_channel, 1)
            self.bn1 = nn.BatchNorm2d(out_channel)
            self.cbam = CBAM(out_channel)
            self.relu1 = nn.ReLU(True)
        if self.use_non_local:
            self.non_local = NONLocalBlock2D(out_channel)
        
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
        lidar_bev = batch_dict['spatial_features']
        cat_bev = torch.cat([img_bev,lidar_bev],dim=1)
        mm_bev = self.conv(cat_bev)
        if self.use_se:
            mm_bev = self.se(mm_bev)
        elif self.use_cam:
            mm_bev = self.cam(mm_bev)
        elif self.use_cbam:
            out = self.bn1(self.conv1(mm_bev))
            out = self.cbam(out)
            mm_bev = mm_bev + out
            mm_bev = self.relu1(mm_bev)
        elif self.use_non_local:
            mm_bev = self.non_local(mm_bev)
        batch_dict['spatial_features'] = mm_bev
        return batch_dict