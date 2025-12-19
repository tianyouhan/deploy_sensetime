import numpy as np
import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
from torchvision.models.resnet import BasicBlock, Bottleneck
from typing import Type, Any, Callable, Union, List, Optional

class ResNet_bev(resnet.ResNet):
    def __init__(self, model_cfg=None):
        self.model_cfg = model_cfg
        self.blocks = {'Bottleneck': Bottleneck}

        self.block = self.blocks[self.model_cfg.get('BLOCK', None)]
        self.layers = self.model_cfg.get('LAYERS', None)
        
        self.num_classes=1000
        self.zero_init_residual=False
        self.groups=1
        self.width_per_group=64
        self.replace_stride_with_dilation=None
        self.norm_layer=None
        super(ResNet_bev, self).__init__(self.block, self.layers, self.num_classes, 
                                         self.zero_init_residual, self.groups, 
                                         self.width_per_group, self.replace_stride_with_dilation, 
                                         self.norm_layer)
        self.num_stages = self.model_cfg.get('NUM_STAGES', 4)
        self.out_indices = self.model_cfg.get('OUT_INDICES', (2, 3))
        self.init_cfg = self.model_cfg.get('INIT_CFG', None)
        self.res_layers = []
        for i in range(self.num_stages):
            layer_name = f'layer{i + 1}'
            self.res_layers.append(layer_name)
        self.avgpool = nn.Identity()
        self.fc = nn.Identity()

    def init_weights(self):
        ckpt = torch.load(self.init_cfg.checkpoint, map_location='cpu')
        state_dict = ckpt
        self.load_state_dict(state_dict, False)
        return 

    def forward(self, batch_dict):
        x = batch_dict['camera_imgs']
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        
        # x1 = self.layer1(x)
        # x2 = self.layer2(x1)
        # x3 = self.layer3(x2)
        # x4 = self.layer4(x3)
        # outs = [x3, x4]
        batch_dict['image_features'] = outs
        return batch_dict
    
    def forward_onnx(self, batch_dict):
        x = batch_dict['camera_imgs']
        # B, N, C, H, W = x.size()
        # x = x.view(B * N, C, H, W)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        
        # x1 = self.layer1(x)
        # x2 = self.layer2(x1)
        # x3 = self.layer3(x2)
        # x4 = self.layer4(x3)
        # outs = [x3, x4]
        batch_dict['image_features'] = outs
        return batch_dict

