from .detector3d_template import Detector3DTemplate
from .. import backbones_image, view_transforms
from ..backbones_image import img_neck
from ..backbones_2d import fuser
import torch

class PVRCNNPlusPlus_fusion(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'image_backbone','neck','vtransform','fuser',
            'backbone_2d', 'dense_head',  'point_head', 'roi_head'
        ]
        self.module_list = self.build_networks()
        self.freeze(model_cfg)
        
    def build_neck(self,model_info_dict):
        if self.model_cfg.get('NECK', None) is None:
            return None, model_info_dict
        neck_module = img_neck.__all__[self.model_cfg.NECK.NAME](
            model_cfg=self.model_cfg.NECK
        )
        model_info_dict['module_list'].append(neck_module)

        return neck_module, model_info_dict
    
    def build_vtransform(self,model_info_dict):
        if self.model_cfg.get('VTRANSFORM', None) is None:
            return None, model_info_dict
        
        vtransform_module = view_transforms.__all__[self.model_cfg.VTRANSFORM.NAME](
            model_cfg=self.model_cfg.VTRANSFORM
        )
        model_info_dict['module_list'].append(vtransform_module)

        return vtransform_module, model_info_dict

    def build_image_backbone(self, model_info_dict):
        if self.model_cfg.get('IMAGE_BACKBONE', None) is None:
            return None, model_info_dict
        image_backbone_module = backbones_image.__all__[self.model_cfg.IMAGE_BACKBONE.NAME](
            model_cfg=self.model_cfg.IMAGE_BACKBONE
        )
        image_backbone_module.init_weights()
        model_info_dict['module_list'].append(image_backbone_module)

        return image_backbone_module, model_info_dict
    
    def build_fuser(self, model_info_dict):
        if self.model_cfg.get('FUSER', None) is None:
            return None, model_info_dict
    
        fuser_module = fuser.__all__[self.model_cfg.FUSER.NAME](
            model_cfg=self.model_cfg.FUSER
        )
        model_info_dict['module_list'].append(fuser_module)
        model_info_dict['num_bev_features'] = self.model_cfg.FUSER.OUT_CHANNEL
        return fuser_module, model_info_dict
    
    def freeze(self, model_cfg):
        if 'FREEZE_CAM' in model_cfg:
            print('FREEZE_CAM')
            for param in self.image_backbone.parameters():
                param.requires_grad = False
            for param in self.neck.parameters():
                param.requires_grad = False
            for param in self.vtransform.parameters():
                param.requires_grad = False
        elif 'FREEZE_CAM_P1' in model_cfg:
            print('FREEZE_CAM')
            for param in self.image_backbone.parameters():
                param.requires_grad = False
            for param in self.neck.parameters():
                param.requires_grad = False
            
        elif 'FREEZE_LIDAR' in model_cfg:
            print('FREEZE_LIDAR')
            for param in self.backbone_3d.parameters():
                param.requires_grad = False
            for param in self.map_to_bev_module.parameters():
                param.requires_grad = False
            for param in self.vfe.parameters():
                param.requires_grad = False
        elif 'FREEZE_ALL' in model_cfg:
            print('FREEZE_ALL')
            for param in self.image_backbone.parameters():
                param.requires_grad = False
            for param in self.neck.parameters():
                param.requires_grad = False
            for param in self.vtransform.parameters():
                param.requires_grad = False
            for param in self.backbone_3d.parameters():
                param.requires_grad = False
            for param in self.map_to_bev_module.parameters():
                param.requires_grad = False
            for param in self.vfe.parameters():
                param.requires_grad = False
        return
    
    def forward(self, batch_dict):
        torch.cuda.empty_cache()
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.image_backbone(batch_dict)
        batch_dict = self.neck(batch_dict)
        torch.cuda.empty_cache()
        batch_dict = self.vtransform(batch_dict)
        batch_dict = self.fuser(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        batch_dict = self.dense_head(batch_dict)
        torch.cuda.empty_cache()
        batch_dict = self.roi_head.proposal_layer(
            batch_dict, nms_config=self.roi_head.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.roi_head.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            batch_dict['roi_targets_dict'] = targets_dict
            num_rois_per_scene = targets_dict['rois'].shape[1]
            if 'roi_valid_num' in batch_dict:
                batch_dict['roi_valid_num'] = [num_rois_per_scene for _ in range(batch_dict['batch_size'])]

        batch_dict = self.pfe(batch_dict)
        batch_dict = self.point_head(batch_dict)
        batch_dict = self.roi_head(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        if self.point_head is not None:
            loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        else:
            loss_point = 0
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict
