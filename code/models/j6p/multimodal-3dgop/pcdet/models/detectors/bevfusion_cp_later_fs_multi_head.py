from .detector3d_template import Detector3DTemplate
from .. import backbones_image, view_transforms
from ..backbones_image import img_neck
from .. import backbones_2d, dense_heads
from ..backbones_2d import fuser
import torch
class BevFusion_cp_later_fs_multi_head(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset, freeze=True):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'image_backbone','neck','vtransform',
            'backbone_2d', 'backbone_2d_cam',
            'fuser', 'dense_head_fusion', 'dense_head_cam', 'dense_head_lidar', 'point_head', 'roi_head'
        ]
        self.module_list = self.build_networks()
        if freeze:
            self.freeze(model_cfg)

    def build_backbone_2d_cam(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_2D_CAM', None) is None:
            return None, model_info_dict
        if 'NUM_BEV_FEATURES' in self.model_cfg.BACKBONE_2D_CAM:
            input_channels = self.model_cfg.BACKBONE_2D_CAM['NUM_BEV_FEATURES']
        else:
            input_channels = model_info_dict.get('num_bev_features', None)
        backbone_2d_module = backbones_2d.__all__[self.model_cfg.BACKBONE_2D_CAM.NAME](
            model_cfg=self.model_cfg.BACKBONE_2D_CAM,
            input_channels=input_channels
        )
        model_info_dict['module_list'].append(backbone_2d_module)
        model_info_dict['num_bev_features'] = backbone_2d_module.num_bev_features
        return backbone_2d_module, model_info_dict

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
    
    def build_dense_head_fusion(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD_FUSION', None) is None:
            return None, model_info_dict
        dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD_FUSION.NAME](
            model_cfg=self.model_cfg.DENSE_HEAD_FUSION,
            input_channels=model_info_dict['num_bev_features'] if 'num_bev_features' in model_info_dict else self.model_cfg.DENSE_HEAD_FUSION.INPUT_FEATURES,
            num_class=self.num_class if not self.model_cfg.DENSE_HEAD_FUSION.CLASS_AGNOSTIC else 1,
            class_names=self.class_names,
            grid_size=model_info_dict['grid_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False),
            voxel_size=model_info_dict.get('voxel_size', False)
        )
        model_info_dict['module_list'].append(dense_head_module)
        return dense_head_module, model_info_dict

    def build_dense_head_lidar(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD_LIDAR', None) is None:
            return None, model_info_dict
        dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD_LIDAR.NAME](
            model_cfg=self.model_cfg.DENSE_HEAD_LIDAR,
            input_channels=model_info_dict['num_bev_features'] if 'num_bev_features' in model_info_dict else self.model_cfg.DENSE_HEAD_LIDAR.INPUT_FEATURES,
            num_class=self.num_class if not self.model_cfg.DENSE_HEAD_LIDAR.CLASS_AGNOSTIC else 1,
            class_names=self.class_names,
            grid_size=model_info_dict['grid_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False),
            voxel_size=model_info_dict.get('voxel_size', False)
        )
        model_info_dict['module_list'].append(dense_head_module)
        return dense_head_module, model_info_dict

    def build_dense_head_cam(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD_CAM', None) is None:
            return None, model_info_dict
        dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD_CAM.NAME](
            model_cfg=self.model_cfg.DENSE_HEAD_CAM,
            input_channels=model_info_dict['num_bev_features'] if 'num_bev_features' in model_info_dict else self.model_cfg.DENSE_HEAD_CAM.INPUT_FEATURES,
            num_class=self.num_class if not self.model_cfg.DENSE_HEAD_CAM.CLASS_AGNOSTIC else 1,
            class_names=self.class_names,
            grid_size=model_info_dict['grid_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False),
            voxel_size=model_info_dict.get('voxel_size', False)
        )
        model_info_dict['module_list'].append(dense_head_module)
        return dense_head_module, model_info_dict

    def freeze(self, model_cfg):
        if 'FREEZE_CAM' in model_cfg:
            print('FREEZE_CAM')
            for param in self.image_backbone.parameters():
                param.requires_grad = False
            for param in self.neck.parameters():
                param.requires_grad = False
            for param in self.vtransform.parameters():
                param.requires_grad = False
            for param in self.backbone_2d_cam.parameters():
                param.requires_grad = False
        if 'FREEZE_CAM_P' in model_cfg:
            print('FREEZE_CAM_P')
            for param in self.image_backbone.parameters():
                param.requires_grad = False
            for param in self.neck.parameters():
                param.requires_grad = False
            for param in self.vtransform.parameters():
                param.requires_grad = False

        elif 'FREEZE_LIDAR' in model_cfg:
            print('FREEZE_LIDAR')
            for param in self.backbone_3d.parameters():
                param.requires_grad = False
            for param in self.map_to_bev_module.parameters():
                param.requires_grad = False
            for param in self.vfe.parameters():
                param.requires_grad = False
            for param in self.backbone_2d.parameters():
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
            for param in self.backbone_2d_cam.parameters():
                param.requires_grad = False
            for param in self.backbone_2d.parameters():
                param.requires_grad = False
        return

    def forward(self, batch_dict):
        torch.cuda.empty_cache()
        for i,cur_module in enumerate(self.module_list):
            batch_dict = cur_module(batch_dict)
            torch.cuda.empty_cache()
        if self.training:
            disp_dict = {}

            loss_fs, tb_dict = self.dense_head_fusion.get_loss()
            tb_dict = {
                'loss_rpn': loss_fs.item(),
                **tb_dict
            }
            loss_ld, tb_dict_ld = self.dense_head_lidar.get_loss()
            loss_cam, tb_dict_cam = self.dense_head_cam.get_loss()
            
            loss = loss_fs + loss_ld + loss_cam
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            if self.model_cfg.POST_PROCESSING.get('LIDAR', False):
                batch_dict['final_box_dicts'] = batch_dict['final_box_dicts_lidar']
            if self.model_cfg.POST_PROCESSING.get('CAM', False):
                batch_dict['final_box_dicts'] = batch_dict['final_box_dicts_cam']
            if self.model_cfg.POST_PROCESSING.get('FUSION', False):
                batch_dict['final_box_dicts'] = batch_dict['final_box_dicts_fusion']
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )
            if 'EVAL_RANGE' in post_process_cfg:
                recall_dict_range = {}
                recall_dict_range = self.generate_recall_record_range(
                    box_preds=pred_boxes,
                    recall_dict=recall_dict_range, batch_index=index, data_dict=batch_dict,
                    thresh_list=post_process_cfg.RECALL_THRESH_LIST,
                    eval_range_list=post_process_cfg.EVAL_RANGE
                )
                recall_dict.update(recall_dict_range)

        return final_pred_dict, recall_dict
