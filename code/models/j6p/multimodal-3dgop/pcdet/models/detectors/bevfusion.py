import torch
import numpy as np
from .detector3d_template import Detector3DTemplate
from .. import backbones_image, view_transforms
from ..backbones_image import img_neck
from ..backbones_2d import fuser
from .. import backbones_2d, dense_heads
import os
from pcdet.utils.common_utils import save_np

class BevFusion(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'image_backbone','neck','vtransform','fuser',
            'backbone_2d', 'dense_head',  'point_head', 'roi_head'
        ]
        self.module_list = self.build_networks()
       
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

    def forward(self, batch_dict):

        for i,cur_module in enumerate(self.module_list):
            batch_dict = cur_module(batch_dict)
        
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self,batch_dict):
        disp_dict = {}

        loss_trans, tb_dict = batch_dict['loss'],batch_dict['tb_dict']
        tb_dict = {
            'loss_trans': loss_trans.item(),
            **tb_dict
        }

        loss = loss_trans
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

        return final_pred_dict, recall_dict
    

class BevFusion_Seg(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.cnt = 0
        if model_cfg.get('FUSE_LATER', False):
            self.module_topology = [
                'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
                'image_backbone','neck','vtransform',
                'backbone_2d', 'backbone_2d_cam', 'fuser', 
                'dense_head', 
                'dense_head_seg', 'dense_head_seg_lidar', 'dense_head_seg_cam', 'dense_head_seg_fusion',
                'dense_head_det', 'dense_head_det_lidar', 'dense_head_det_cam', 'dense_head_det_fusion', 
                'point_head', 'roi_head'
            ]
        else:
            self.module_topology = [
                'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
                'image_backbone','neck','vtransform', 'fuser', 
                'backbone_2d', 'backbone_2d_cam', 
                'dense_head', 
                'dense_head_seg', 'dense_head_seg_lidar', 'dense_head_seg_cam', 'dense_head_seg_fusion',
                'dense_head_det', 'dense_head_det_lidar', 'dense_head_det_cam', 'dense_head_det_fusion',
                'point_head', 'roi_head'
            ]
        self.build_dense_head_seg_multitask()
        self.build_dense_head_det_multitask()
        self.module_list = self.build_networks()
        if model_cfg.get('FREEZE', False):
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
        # image_backbone_module.init_weights()
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
    
    def dense_head_base(self, model_info_dict, head_name=None):
        if self.model_cfg.get(head_name, None) is None:
            return None, model_info_dict
        head_cfg = getattr(self.model_cfg, head_name)
        if 'pp_heavy' not in head_cfg.NAME:
            dense_head_module = dense_heads.__all__[head_cfg.NAME](
                model_cfg=head_cfg,
                input_channels=model_info_dict['num_bev_features'] if 'num_bev_features' in model_info_dict else head_cfg.INPUT_FEATURES,
                num_class=self.num_class if not head_cfg.CLASS_AGNOSTIC else 1,
                class_names=self.class_names if 'CLASS_NAMES' not in head_cfg else head_cfg.CLASS_NAMES,
                grid_size=model_info_dict['grid_size'],
                point_cloud_range=model_info_dict['point_cloud_range'],
                predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False),
                voxel_size=model_info_dict.get('voxel_size', False)
            )
        else:
            # pp heavy head
            # TODO cfg
            dense_head_module = dense_heads.__all__[head_cfg.NAME](
                cfg=head_cfg,
                dataset=self.dataset,
                num_class=self.num_class if not head_cfg.CLASS_AGNOSTIC else 1,)
        model_info_dict['module_list'].append(dense_head_module)
        return dense_head_module, model_info_dict
    
    def build_dense_head_seg(self, model_info_dict):
        dense_head_module, model_info_dict = self.dense_head_base(model_info_dict, 'DENSE_HEAD_SEG')
        return dense_head_module, model_info_dict
        
    def build_dense_head_seg_multitask(self):
        for module_name in ['dense_head_seg_lidar', 'dense_head_seg_cam', 'dense_head_seg_fusion']:
            setattr(self, f'build_{module_name}', lambda model_info_dict, head_name=module_name.upper(): self.dense_head_base(model_info_dict, head_name))
        
    def build_dense_head_det(self, model_info_dict):
        dense_head_module, model_info_dict = self.dense_head_base(model_info_dict, 'DENSE_HEAD_DET')
        return dense_head_module, model_info_dict
    
    def build_dense_head_det_multitask(self):
        for module_name in ['dense_head_det_lidar', 'dense_head_det_cam', 'dense_head_det_fusion']:
            setattr(self, f'build_{module_name}', lambda model_info_dict, head_name=module_name.upper(): self.dense_head_base(model_info_dict, head_name))
    
    def freeze(self, model_cfg):
        fix_modules = model_cfg.FREEZE.FIX_MODULES
        ignore_keys = model_cfg.FREEZE.IGNORE_KEYs
        print('FREEZE:', fix_modules)
        unfix = set()
        for module in fix_modules:
            for key, param in getattr(self, module).named_parameters():
                if ignore_keys is not None and any([(ig_key.split('.')[0] == module and ig_key[len(ig_key.split('.')[0] + '.'):] in key) for ig_key in ignore_keys]):
                    unfix.add(f'{module}.{key}')
                    continue
                param.requires_grad = False
        print('Not FREEZE:', list(unfix))
        return

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = 0, {}, {}
            for module in ['DENSE_HEAD_DET', 'DENSE_HEAD_DET_FUSION', 'DENSE_HEAD_DET_LIDAR', 'DENSE_HEAD_DET_CAM']:
                if self.model_cfg.get(module, None) is not None:
                    module = module.lower()
                    task = module.split('_')[-1]
                    loss_, tb_dict_ = getattr(self, module).get_loss()
                    tb_dict_ = {f'loss_{module}': {f'loss_{task}': loss_.item(), **tb_dict_}}
                    loss += loss_
                    tb_dict.update(tb_dict_)
            for module in ['DENSE_HEAD', 'DENSE_HEAD_SEG', 'DENSE_HEAD_SEG_FUSION', 'DENSE_HEAD_SEG_LIDAR', 'DENSE_HEAD_SEG_CAM']:
                if self.model_cfg.get(module, None) is not None:
                    module = module.lower()
                    loss_, tb_dict_, = getattr(self, module).get_loss(batch_dict)
                    tb_dict_ = {f'loss_{module}': {**tb_dict_}}
                    loss += loss_
                    tb_dict.update(tb_dict_)
            if 'depth_loss' in batch_dict:
                loss += batch_dict['depth_loss']
                tb_dict.update({'loss_depth': batch_dict['depth_loss'].item()})

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, label_dicts = {}, {}
            for module in ['DENSE_HEAD_DET', 'DENSE_HEAD_DET_FUSION', 'DENSE_HEAD_DET_LIDAR', 'DENSE_HEAD_DET_CAM']:
                if self.model_cfg.get(module, None) is not None:
                    task = module.split('_')[-1].lower()
                    if task in ['lidar', 'cam', 'fusion']: 
                        det_pred_dict, det_recall_dict = self.post_processing(batch_dict, task_key=f'det_pred_dicts_{task}')
                        _task = f'_{task}'
                    else:
                        det_pred_dict, det_recall_dict = self.post_processing(batch_dict)
                        _task = ''

                    pred_dicts[f'det_pred_dicts{_task}'] = det_pred_dict
                    label_dicts[f'det_recall_dicts{_task}'] = det_recall_dict
                    if os.getenv("CALIB") == 'True':
                        calib_path = os.getenv("CALIB_PATH")
                        save_dir = os.path.join(calib_path, f"final-pred/")
                        save_np(os.path.join(save_dir, "det/{}/pred_boxes/{}".format(task, self.cnt)), det_pred_dict[0]['pred_boxes'])
                        save_np(os.path.join(save_dir, "det/{}/pred_scores/{}".format(task, self.cnt)), det_pred_dict[0]['pred_scores'])
                        save_np(os.path.join(save_dir, "det/{}/pred_labels/{}".format(task, self.cnt)), det_pred_dict[0]['pred_labels'])
            
            for module in ['DENSE_HEAD', 'DENSE_HEAD_SEG', 'DENSE_HEAD_SEG_FUSION', 'DENSE_HEAD_SEG_LIDAR', 'DENSE_HEAD_SEG_CAM']:
                if self.model_cfg.get(module, None) is not None:
                    task = module.split('_')[-1].lower()
                    if task in ['lidar', 'cam', 'fusion']:
                        seg_pred_dicts, seg_label_dicts = batch_dict[f'seg_pred_dicts_{task}'], batch_dict[f'seg_label_dicts_{task}']
                        _task = f'_{task}'
                    else:
                        seg_pred_dicts, seg_label_dicts = batch_dict['seg_pred_dicts'], batch_dict['seg_label_dicts']
                        _task = '_unknown'

                    pred_dicts[f'seg_pred_dicts{_task}'] = seg_pred_dicts
                    label_dicts[f'seg_label_dicts{_task}'] = seg_label_dicts
                    if os.getenv("CALIB") == 'True':
                        calib_path = os.getenv("CALIB_PATH")
                        save_dir = os.path.join(calib_path, f"final-pred/")
                        save_np(os.path.join(save_dir, "seg/{}/pred_segmap/{}".format(task, self.cnt)), seg_pred_dicts[f'{task}_pred'])
            if os.getenv("CALIB") == 'True':
                calib_path = os.getenv("CALIB_PATH")
                check_path = os.path.join(calib_path, 'cam-backbone/inputs/camera_imgs/')
                len_save = len(os.listdir(check_path))
                if len_save >= int(os.getenv("NUM")):
                    assert(0)
            self.cnt += 1
            return pred_dicts, label_dicts
        
    def forward_onnx_bev_cam(self, batch_dict, bev_cam_module):
        bev_h, bev_w = bev_cam_module.model_cfg.INPUT_BEV_SIZE
        slots = batch_dict['cam_bev_backbone_input']
        slots = slots.view(1, bev_h, bev_w, -1)
        slots = slots.permute(0, 3, 1, 2).contiguous()
        if self.vtransform.output_proj_cfg is not None:
            slots = self.vtransform.output_proj(slots)
        batch_dict['spatial_features_img'] = slots
        return bev_cam_module.forward(batch_dict)
        
    def forward_onnx(self, batch_dict, onnx_module=None, onnx_outputs=None, task=None):
        for cur_module in self.module_list:
            cur_module_name = type(cur_module).__name__
            if cur_module_name in onnx_module:
                
                if cur_module_name in ['SegHead_pcseg', 'pp_heavy_head']:  # 根据task区分不同head
                    if not (task is not None and cur_module.task == task):
                        continue
                
                print(f'{cur_module_name} forward')
                if cur_module_name in ['AttentionTransform', 'AttentionTransform_Lidaraug', 'PillarVFE', 'PointPillarScatter_Seg', 'BaseBEVBackbone_FPN_cam', 'BaseBEVBackbone_cam', \
                                       'ResNet_bev', 'DLANet', 'DLANet_', 'pp_heavy_head']:
                    if cur_module_name in ['BaseBEVBackbone_FPN_cam', 'BaseBEVBackbone_cam']:
                        batch_dict = self.forward_onnx_bev_cam(batch_dict, cur_module)
                    else:
                        batch_dict = cur_module.forward_onnx(batch_dict)
                else:
                    batch_dict = cur_module(batch_dict)

        if onnx_outputs is not None:
            out_dict = {}
            for name in onnx_outputs:
                out_dict[name] = batch_dict[name]
        else:
            out_dict = batch_dict
        return out_dict

    def get_training_loss(self, batch_dict=None):
        disp_dict = {}
        if batch_dict is not None:
            loss_sum, tb_dict = self.dense_head.get_loss(batch_dict)
        else:
            loss_sum, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_sum': loss_sum.item(),
            **tb_dict
        }

        loss = loss_sum
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict, task_key=None):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        if task_key is None:
            final_pred_dict = batch_dict['final_box_dicts']
        else:
            final_pred_dict = batch_dict[task_key]['final_box_dicts']
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
