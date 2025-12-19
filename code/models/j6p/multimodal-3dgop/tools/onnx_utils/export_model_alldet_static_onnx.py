# coding: utf-8
import os
from os import path
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import onnx
from onnxsim import simplify
from onnx.shape_inference import infer_shapes

from pcdet.config import cfg_from_yaml_file, cfg
from pcdet.models import build_network
from fuse_conv_bn import fuse_conv_bn
from onnx_infer_utils import ONNXModel


def register_op():
    ## replace functions and register custom symbolics ##
    def identity(x):
        return x
    nan_to_num = torch.nan_to_num
    torch.nan_to_num = identity

    def maximum_symbolic(g, lhs, rhs):
        return g.op("Max", lhs, rhs)

    def register_op_symbolic(symbolic_name, symbolic_fn, opset_version):
        ns, op_name = symbolic_name.split('::')
        import torch.onnx.symbolic_registry as sym_registry
        try:
            from torch.onnx.symbolic_helper import _onnx_stable_opsets, _onnx_main_opset
            for version in _onnx_stable_opsets + [_onnx_main_opset]:
                if version >= opset_version:
                    sym_registry.register_op(op_name, symbolic_fn, ns, opset_version)
        except:
            from torch.onnx._constants import onnx_stable_opsets, onnx_main_opset
            for version in list(onnx_stable_opsets) + [onnx_main_opset]:
                if version >= opset_version:
                    sym_registry.register_op(op_name, symbolic_fn, ns, opset_version)

    register_op_symbolic('onnx::maximum', maximum_symbolic, 6)
    GridSampleFuction.replace()


class GridSampleFuction(torch.autograd.Function):
    original_grid_sample = None

    @staticmethod
    def symbolic(g, img, grid, mode, padding_mode, align_corners):
        mode_str = mode
        padding_mode_str = 'zeros'
        align_corners = True
        return g.op("com.microsoft::GridSample", img, grid,
                    mode_s=mode_str,
                    padding_mode_s=padding_mode_str,
                    align_corners_i=align_corners)

    @staticmethod
    def forward(ctx, img, grid, mode, padding_mode, align_corners):
        output = GridSampleFuction.original_grid_sample(img, grid, mode, padding_mode, align_corners=align_corners)
        # output = GridSampleFuction.original_grid_sample(img, grid, mode='nearest', padding_mode='zeros', align_corners=False)
        return output

    instance = None
    
    @classmethod
    def replace(cls):
        cls.original_grid_sample = torch.nn.functional.grid_sample
        if cls.instance is None:
            cls.instance = GridSampleFuction()
        #mode = bilinear nearest
        def grid_sample(input, grid, mode='nearest', padding_mode='zeros', align_corners=True):
            return cls.instance.apply(input, grid, mode, padding_mode, align_corners)

        torch.nn.functional.grid_sample = grid_sample
    
    @classmethod
    def restore(cls):
        torch.nn.functional.grid_sample = cls.original_grid_sample


def load_checkpoint(model, checkpoint_path, map_location):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model_state = checkpoint['model_state']
    model.load_state_dict(model_state)
    return model_state


def bulid_model():
    ## build model ##
    config = cfg_from_yaml_file(cfg_file, cfg)
    model = build_network(model_cfg=config.MODEL, num_class=len(config.CLASS_NAMES), dataset=config.ONNX_CONFIG)
    if checkpoint_path is not None:
        checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.to(device)
    if fuse_conv_bn_flag:
        model = fuse_conv_bn(model)
    model.eval()
    if export_fp16:
        model.half()
    return model


def simplify_model(model_path):
    model = onnx.load(model_path)
    # due to a bug in onnx-simplifier, we have to infer shape first and then remove tensor value
    # infos with unknown dimension by hand. 
    model = infer_shapes(model)
    def is_valid_info(info):
        return all(item.HasField("dim_value") for item in info.type.tensor_type.shape.dim)
    model.graph.ClearField("value_info")
    valid_value_infos = [item for item in model.graph.value_info if is_valid_info(item)]
    model.graph.value_info.extend(valid_value_infos)
    # model, check = simplify(model, tensor_size_threshold="512KB", skip_shape_inference=True)
    model, check = simplify(model, tensor_size_threshold="512KB", skip_shape_inference=False)
    assert check
    base_name, ext_name = path.splitext(model_path)
    onnx.save(model, f"{base_name}-sim{ext_name}")
    return f"{base_name}-sim{ext_name}"


def cal_cosine_similarity(val1, val2):
    val_arr = val1.cpu().numpy()
    val_arr = np.array(val_arr, dtype=np.float32)
    val2 = np.array(val2, dtype=np.float32)
    cosine_numpy = (np.sum(val_arr * val2) + 1e-8) / (np.linalg.norm(val_arr) * np.linalg.norm(val2) + 1e-8)
    print('cosine_similarity:', cosine_numpy)
    return cosine_numpy


def check_results(model, onnx_path, cur_batch_dict, cur_module, output_names, task):
    result_pytorch = model.forward_onnx(cur_batch_dict, cur_module, output_names, task)
    return_onnx = ONNXModel(onnx_path).forward(cur_batch_dict)
    for i, ele in enumerate(return_onnx):
        print(f'onnx output:{i} shape:', ele.shape)
    count = 0
    for idx, (key, val) in enumerate(result_pytorch.items()):
        print(f'-----------------------{cur_module} similarity ---------------------------')
        if type(val) in [list, tuple]:
            for j in range(len(val)):
                cal_cosine_similarity(val[j], return_onnx[count])
                count += 1
        elif type(val) in [dict]:
            for j, (key2, val2) in enumerate(val.items()):
                cal_cosine_similarity(val2, return_onnx[count])
                count += 1
        else:
            cal_cosine_similarity(val, return_onnx[idx])
            count += 1

    
def apply_onnx_transfer(batch_dict, module_list, onnx_file_name):
    assert len(module_list) == len(onnx_file_name) == len(onnx_inputs)
    register_op()
    model = bulid_model()
    print(model)
    with torch.no_grad():
        for idx in range(len(module_list)):
            cur_module = module_list[idx]
            cur_onnx_file_name = onnx_file_name[idx]

            original_forward = model.forward
            model.forward = model.forward_onnx
            onnx_model_name = os.path.join(save_onnx_root, f"{cur_onnx_file_name}.onnx")
            cur_batch_dict = {name: batch_dict[name] for name in onnx_inputs[idx]}

            if cur_onnx_file_name in ['fuser-fusion-head', 'lidar-head', 'cam-head']:  # 因为输出是二维字典，而onnx只输出val，因此需要val数量的output_names
                output_names = onnx_outputs_head[cur_onnx_file_name]
                task = cur_onnx_file_name.split('-')[-2]
            else:
                output_names = onnx_outputs[idx]
                task = None

            dynamic_axes = None
            if 'PillarVFE' in cur_module:
                dynamic_axes = {'vfe_input': {0:'N'}, 'voxel_coords': {0:'N'}}
            if 'AttentionTransform_Lidaraug' in cur_module:
                dynamic_axes = {'gridsample_ref_points': {1:'MAX_N'}, 'gridsample_indexes_0': {0:'N'}, 'gridsample_indexes_1': {0:'M'}}
            if not export_dynamic:
                dynamic_axes = None
            torch.onnx.export(
                model,
                (cur_batch_dict, cur_module, onnx_outputs[idx], task),
                onnx_model_name,
                verbose=False,  # 如果想要看日志
                opset_version=11,
                input_names=onnx_inputs[idx],
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
            model.forward = original_forward
            onnx_model_name_sim = simplify_model(onnx_model_name)

            check_results(model, onnx_model_name_sim, cur_batch_dict, cur_module, onnx_outputs[idx], task)


if __name__ == '__main__':
    device = torch.device("cuda:0")
    fuse_conv_bn_flag = True
    export_fp16 = False
    export_dynamic = False
    dtype = torch.float16 if export_fp16 else torch.float32
    cfg_file = "../tools/cfgs/A02_ceph/MM3DGOP-A02-V1.9-3heads_A02_3dgop_alldet_2V_V1.9_No1.yaml"
    checkpoint_path = "../data/baseline_ckpts/baseline_checkpoint_epoch_40_3heads.pth"  # 在这里指定checkpoint_path
    save_onnx_root = "onnx_utils/onnx_output/MM3DGOP-A02-V1.9_alldet-8m_static_n3"
    os.makedirs(save_onnx_root, exist_ok=True)
    max_len = 3580
    max_len2 = 2851
    batch_dict = {
            "vfe_input": torch.rand((20000, 9, 32, 1), dtype=dtype, device=device),  # lidar-vfe input
            "voxel_coords": torch.rand((20000, 2), dtype=dtype, device=device) * 50,  # lidar-vfe input, x y
            "voxel_valid_flag": torch.rand((20000, 1), dtype=dtype, device=device),  # lidar-vfe input, 1 for valid

            "camera_imgs": torch.rand(2, 3, 512, 1024, dtype=dtype, device=device),  # cam-backbone input
            "center_camera_fov120": torch.rand(1, 3, 512, 1024, dtype=dtype, device=device),  # cam-backbone input
            "center_camera_fov30": torch.rand(1, 3, 512, 1024, dtype=dtype, device=device),  # cam-backbone input

            "spatial_features": torch.rand(1, 32, 96, 640, dtype=dtype, device=device),  # lidar-bev-backbone input

            "cam_bev_backbone_input": torch.rand(1, 1, 3840, 64, dtype=dtype, device=device),  # cam-bev-backbone input

            "spatial_features_2d_cam": torch.rand(1, 96, 48, 320, dtype=dtype, device=device),  # fuser-multihead cam input
            "spatial_features_2d_lidar": torch.rand(1, 96, 48, 320, dtype=dtype, device=device),  # fuser-multihead lidar input
            
            "gridsample_input": torch.rand(2, 256, 64, 128, dtype=dtype, device=device),  # cam-atttransform feature input
            "gridsample_ref_points": torch.rand(2, max_len, 1, 1, 15, 2, dtype=dtype, device=device),  # 3578为fov120和fov30中投影到相机上点的max_len,随着数据会发生变化
            "gridsample_indexes_0": torch.rand((max_len), dtype=dtype, device=device) * 1000,  # center_camera_fov120在3840中的index
            "gridsample_indexes_1": torch.rand((max_len2), dtype=dtype, device=device) * 1000,  # center_camera_fov30在3840中的index
            "ref_points_valid_num": torch.ones((1, 3840), dtype=dtype, device=device) * 2,  # cam-atttransform input, 参考点投影到N个相机上，在range范围内就+1
        }
    module_list = [
        ['PillarVFE', 'PointPillarScatter_Seg', 'BaseBEVBackbone_FPN'],
        ['DLANet_', 'CustomFPN', 'AttentionTransform_Lidaraug', 'BaseBEVBackbone_FPN_cam'],
        ['pp_heavy_head'],  # lidar, 要和yaml文件中lidar, cam, fusion的顺序保持一致
        ['pp_heavy_head'],  # cam
        ['ConvFuser_later_atten', 'pp_heavy_head'],  # fusion
    ]
    onnx_inputs = [
        ['vfe_input', 'voxel_coords'],
        ['center_camera_fov120', 'center_camera_fov30', 'gridsample_ref_points', 'gridsample_indexes_0', 'gridsample_indexes_1', 'ref_points_valid_num'],
        ['spatial_features_2d_lidar'],
        ['spatial_features_2d_cam'],
        ['spatial_features_2d_lidar', 'spatial_features_2d_cam'],
    ]
    onnx_outputs = [
        ['spatial_features_2d_lidar'],
        ['spatial_features_2d_cam'],
        ['det_pred_dicts_lidar'],
        ['det_pred_dicts_cam'],
        ['det_pred_dicts_fusion'],
    ]
    
    onnx_outputs_head = {
        'lidar-head': ['det_pred_dicts_lidar_cls', 'det_pred_dicts_lidar_box', 'det_pred_dicts_lidar_dir_cls'],
        'cam-head': ['det_pred_dicts_cam_cls', 'det_pred_dicts_cam_box', 'det_pred_dicts_cam_dir_cls'],
        'fuser-fusion-head': ['det_pred_dicts_fusion_cls', 'det_pred_dicts_fusion_box', 'det_pred_dicts_fusion_dir_cls'],
    }
    
    onnx_file_name = [
        'lidar-branch',  
        'cam-branch', 
        'lidar-head', 
        'cam-head',
        'fuser-fusion-head', 
    ]
    
    apply_onnx_transfer(batch_dict, module_list, onnx_file_name)

