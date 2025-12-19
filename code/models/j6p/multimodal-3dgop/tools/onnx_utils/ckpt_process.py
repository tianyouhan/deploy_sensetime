import os
import torch
import numpy as np


def merge_gop_pvb_ckpt():
    ckpt_gop_path = '/mnt/afs2/xuzhiyong/code/stchery-pvb-gop/work_dirs/T68/spetr_dla34v3_gop_v7.4.0/iter_433212.pth'
    ckpt_pvb_path = '/mnt/afs2/xuzhiyong/code/stchery-pvb-gop/B3P90S90_iter_55599.pth'
    new_ckpt_path = '/mnt/afs2/xuzhiyong/code/stchery-pvb-gop/work_dirs/T68/spetr_dla34v3_gop_v7.4.0/B3_0218_iter_433212_gop_iter_55599_pvb.pth'
    ckpt_gop = torch.load(ckpt_gop_path, map_location='cpu')
    ckpt_pvb = torch.load(ckpt_pvb_path, map_location='cpu')
    same_modules = ['img_backbone_pinhole']
    pvb_modules = ['img_rpn_head_pinhole_pvb', 'pts_bbox_head_pvb']
    gop_modules = ['img_rpn_head_pinhole_gop', 'pts_bbox_head_gop']

    new_ckpt = {}
    new_ckpt = {'state_dict':{}}
    for key, val in ckpt_pvb['state_dict'].items():
        for module in same_modules:
            if module in key:
                assert torch.sum(val - ckpt_gop['state_dict'][key]) == 0.
                new_ckpt['state_dict'][key] = val
                print(f'add {key}')

    for key, val in ckpt_pvb['state_dict'].items():
        for module in pvb_modules:
            if module in key:
                new_ckpt['state_dict'][key] = val
                print(f'add {key}')

    for key, val in ckpt_gop['state_dict'].items():
        for module in gop_modules:
            if module in key:
                new_ckpt['state_dict'][key] = val
                print(f'add {key}')

    torch.save(new_ckpt, new_ckpt_path)

def check_same():
    ckpt_gop_path = '/mnt/lustre/xuzhiyong/code/multimodal-3dgop-xuzhiyong/tools/onnx_utils/checkpoints/MM3DGOP-A02-V1.9-3heads_A02_3dgop_alldet_2V_V1.9_No1/checkpoint_epoch_40_3heads.pth'
    ckpt_pvb_path = 'output/A02/lidar_A02_3dgop_alldet_2V_V1.9.1_No1/default/ckpt/checkpoint_epoch_60.pth'
    ckpt_gop = torch.load(ckpt_gop_path, map_location='cpu')
    ckpt_pvb = torch.load(ckpt_pvb_path, map_location='cpu')
    # same_modules = ['image_backbone', 'neck', 'vtransform', 'backbone_2d_cam', 'dense_head_det_cam']
    same_modules = ['vfe', 'map_to_bev_module', 'backbone_2d', 'dense_head_det_lidar']

    new_ckpt = {}
    new_ckpt = {'model_state':{}}
    for key, val in ckpt_pvb['model_state'].items():
        for module in same_modules:
            if module in key:
                assert torch.sum(val - ckpt_gop['model_state'][key]) == 0.
                new_ckpt['model_state'][key] = val
                print(f'add {key}')


if __name__ == '__main__':
    # merge_gop_pvb_ckpt()
    check_same()

