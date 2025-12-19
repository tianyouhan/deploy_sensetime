import torch
from torch import nn
from .attention import CFAMBlock, CBAM, NONLocalBlock2D, PAM
from pcdet.utils.common_utils import save_np
import os

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



class ConvFuser_later(nn.Module):
    def __init__(self,model_cfg) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        in_channel = self.model_cfg.IN_CHANNEL
        out_channel = self.model_cfg.OUT_CHANNEL
        if type(in_channel) not in [list, tuple]:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True)
                )
        self.use_se = self.model_cfg.SE if 'SE' in self.model_cfg else False
        self.use_cam = self.model_cfg.CAM if 'CAM' in self.model_cfg else False
        self.use_cbam = self.model_cfg.CBAM if 'CBAM' in self.model_cfg else False
        self.use_non_local = self.model_cfg.NON_LOCAL if 'NON_LOCAL' in self.model_cfg else False
        if 'PAM' in self.model_cfg and self.model_cfg.PAM.ENABLE:
            self.use_pam = True
            self.pam_cfg = self.model_cfg.PAM
        else:
            self.use_pam = False
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
        if self.use_pam:
            pam_outc = self.pam_cfg.get('PAM_OUTC', in_channel[0])
            reduction = self.pam_cfg.get('REDUCTION', 1)
            self.pam = PAM(in_channel[0], in_channel[1], pam_outc, reduction)
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
            self.upconv = nn.ModuleList()
            for idx in range(len(upsample_strides)):
                self.upconv.append(
                    nn.Upsample(scale_factor=upsample_strides[idx], mode='nearest')
                )
        
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

        if self.use_pam:
            if self.pam_cfg.FUSE_MODE == 'multihead_lss_lidarbev':
                lidar_bev = batch_dict['spatial_features_2d_lidar_scale']
                img_bev = batch_dict['spatial_features_img_scale']
            else:
                lidar_bev = batch_dict['spatial_features_2d_lidar_scale']
                img_bev = [batch_dict['spatial_features_img']]
            
            lidar_bev[-1] = self.pam(lidar_bev[-1], img_bev[-1])
            uplist = []
            for i in range(len(lidar_bev)):
                uplist.append(self.upconv[i](lidar_bev[i]))
            mm_bev = torch.cat(uplist, dim=1)
        else:
            img_bev = batch_dict['spatial_features_2d_cam']
            lidar_bev = batch_dict['spatial_features_2d_lidar']
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
        batch_dict['spatial_features_2d'] = mm_bev
        batch_dict['spatial_features_2d_fusion'] = mm_bev
        return batch_dict


class ConvFuser_later_atten(nn.Module):
    def __init__(self,model_cfg) -> None:
        super().__init__()
        self.cnt = 0
        self.model_cfg = model_cfg
        in_channel = self.model_cfg.IN_CHANNEL
        out_channel = self.model_cfg.OUT_CHANNEL
        layers_conv = self.model_cfg.get('LAYERS_CONV',1)
        conv_list = []
        for i in range(layers_conv):
            if i==0:
                conv_list.extend([
                    nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(True)])
            else:
                conv_list.extend([
                    nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(True)])                
        self.conv = nn.Sequential(*conv_list)

        self.use_se = self.model_cfg.SE if 'SE' in self.model_cfg else False
        self.use_cam = self.model_cfg.CAM if 'CAM' in self.model_cfg else False
        self.use_cbam = self.model_cfg.CBAM if 'CBAM' in self.model_cfg else False
        self.use_non_local = self.model_cfg.NON_LOCAL if 'NON_LOCAL' in self.model_cfg else False
        if self.use_se:
            self.se = SE_Block(in_channel)
        if self.use_cam:
            self.cam = CFAMBlock(in_channel, in_channel)
        if self.use_cbam:
            self.conv1 = nn.Conv2d(in_channel, in_channel, 1)
            self.bn1 = nn.BatchNorm2d(in_channel)
            self.cbam = CBAM(in_channel)
            self.relu1 = nn.ReLU(True)
        if self.use_non_local:
            self.non_local = NONLocalBlock2D(in_channel)
        self.init_weights()

    def init_weights(self):
        """Initialize the transformer weights."""
        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
          
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
        if os.getenv("CALIB") == 'True':
            calib_path = os.getenv("CALIB_PATH")
            save_dir = os.path.join(calib_path, 'fuser-fusion-head')
            save_np(os.path.join(save_dir, "inputs/spatial_features_2d_lidar/{}".format(self.cnt)), batch_dict['spatial_features_2d_lidar'])
            save_np(os.path.join(save_dir, "inputs/spatial_features_2d_cam/{}".format(self.cnt)), batch_dict['spatial_features_2d_cam'])
        
        img_bev = batch_dict['spatial_features_2d_cam']
        lidar_bev = batch_dict['spatial_features_2d_lidar']
        cat_bev = torch.cat([img_bev,lidar_bev],dim=1)

        if self.use_se:
            mm_bev = self.se(cat_bev)
        elif self.use_cam:
            mm_bev = self.cam(cat_bev)
        elif self.use_cbam:
            out = self.bn1(self.conv1(cat_bev))
            out = self.cbam(out)
            mm_bev = cat_bev + out
            mm_bev = self.relu1(mm_bev)
        elif self.use_non_local:
            mm_bev = self.non_local(cat_bev)
        else:
            mm_bev = cat_bev

        mm_bev = self.conv(mm_bev)
        batch_dict['spatial_features_2d'] = mm_bev
        batch_dict['spatial_features_2d_fusion'] = mm_bev
        self.cnt += 1
        return batch_dict