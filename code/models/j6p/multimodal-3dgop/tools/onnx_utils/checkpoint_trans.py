import torch
import sys
sys.path.append('../')
import os
def checkpoint_transform(path_in, path_out):
    """
    Running in torch version >= 1.6, to transform zip archive checkpoint to torch version < 1.6. 
    """
    state_dict = torch.load(path_in, map_location='cpu')
    torch.save(state_dict, path_out, _use_new_zipfile_serialization=False)
    return None


def fuse_checkpoint(fix_id=None):
    for id_ in range(1, 41):
        if fix_id is not None and id_ != fix_id:
            continue
        
        cam_ckpt = 'output/A02_ceph/camera_A02_3dgop_alldet_2V_V1.9.1_No1_1024x576_virtual_0715/default/ckpt/checkpoint_epoch_40.pth'
        lidar_ckpt = 'output/A02_ceph/lidar_A02_3dgop_alldet_2V_V1.9.1_No1_0711/default/ckpt/checkpoint_epoch_60.pth'
        fusion_ckpt = f'output/A02_ceph/fusion_A02_3dgop_alldet_2V_V1.9.1_No1_1024x576_virtual_0715/default/ckpt/checkpoint_epoch_40.pth'
        save_path = f'tools/onnx_utils/checkpoints/fusion_A02_3dgop_alldet_2V_V1.9.1_No1_1024x576_virtual_0715/checkpoint_epoch_{id_}_3heads.pth'
        
        ckpt_fusion = torch.load(fusion_ckpt, map_location='cpu')
        ckpt_cam = torch.load(cam_ckpt, map_location='cpu')
        ckpt_lidar = torch.load(lidar_ckpt, map_location='cpu')
        for key, val in ckpt_cam['model_state'].items():
            if 'dense_head_det_cam' in key or 'dense_head_seg_cam' in key:
                ckpt_fusion['model_state'][key] = val
        for key, val in ckpt_lidar['model_state'].items():
            if 'dense_head_det_lidar' in key or 'dense_head_seg_lidar' in key:
                ckpt_fusion['model_state'][key] = val
        print(ckpt_fusion['model_state'].keys())
        basedir = os.path.dirname(save_path)
        os.makedirs(basedir, exist_ok=True)
        torch.save(ckpt_fusion, save_path)


if __name__ == '__main__':
    # path_in = 'tools/checkpoints/occ_DLA34_backbone.pth'
    # path_out = 'tools/checkpoints/occ_DLA34_backbone.pth'
    # checkpoint_transform(path_in, path_out)

    fuse_checkpoint(fix_id=40)