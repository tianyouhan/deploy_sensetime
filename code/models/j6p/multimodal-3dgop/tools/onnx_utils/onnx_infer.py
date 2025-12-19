# -*-coding: utf-8 -*-
 
import os
import onnxruntime
import numpy as np


class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))
 
    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name
 
    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name
 
    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed
 
    def forward(self, image_numpy):
        '''
        # image_numpy = image.transpose(2, 0, 1)
        # image_numpy = image_numpy[np.newaxis, :]
        # onnx_session.run([output_name], {input_name: x})
        # :param image_numpy:
        # :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_numpy})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: iimage_numpy})
        input_feed = {}
        for idx in range(len(self.input_name)):
            input_feed.update(self.get_input_feed([self.input_name[idx]], image_numpy[idx]))
        outputs = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return outputs
 
 
 
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


## lidar-branch
# input_names = ['vfe_input', 'voxel_coords']
# inputs_shape = [(20000, 9, 32, 1), (20000, 2)]

# output_names = ['spatial_features_2d_lidar']
# outputs_shape = [(1, 96, 48, 320)]

# model_path = f"tools/onnx_utils/onnx_output/MM3DGOP-A02-V1.9_alldet-8m_dynamic/lidar-branch-sim.onnx"
# quant_data_inputs = f'output/dump_data_alldet_v1.9_dynamic/lidar-branch/inputs'
# quant_data_outputs = f'output/dump_data_alldet_v1.9_dynamic/lidar-branch/outputs'

## cam-branch
input_names = ['center_camera_fov120', 'center_camera_fov30', 'gridsample_ref_points', 'gridsample_indexes_0', 'gridsample_indexes_1', 'ref_points_valid_num']
inputs_shape = [(1, 3, 512, 1024), (1, 3, 512, 1024), (2, 3578, 1, 1, 15, 2), (3578), (3578), (1, 3840)]

output_names = ['spatial_features_2d_cam']
outputs_shape = [(1, 96, 48, 320)]

model_path = f"tools/onnx_utils/onnx_output/MM3DGOP-A02-V1.9_alldet-8m_dynamic/cam-branch-sim.onnx"
quant_data_inputs = f'output/dump_data_alldet_v1.9_dynamic/cam-branch/inputs'
quant_data_outputs = f'output/dump_data_alldet_v1.9_dynamic/cam-branch/outputs'


## vfe
# input_names = ['vfe_input']
# inputs_shape = [(70000, 9, 32, 1)]

# output_names = ['vfe_output']
# outputs_shape = [(70000, 32, 32)]

# model_path = f"tools/onnx_utils/onnx_output/MM3DGOP-A02-V1.6-8m/lidar-vfe-sim.onnx"
# quant_data_inputs = f'output/dump_data/lidar-vfe/inputs'
# quant_data_outputs = f'output/dump_data/lidar-vfe/outputs'

## cam-backbone
# input_names = ['camera_imgs']
# inputs_shape = [(2, 3, 512, 1024)]

# output_names = ['image_fpn']
# outputs_shape = [(2, 256, 64, 128)]

# model_path = f"tools/onnx_utils/onnx_output/MM3DGOP-A02-V1.9_alldet-8m_clip2fp16/cam-backbone-sim.onnx"
# quant_data_inputs = f'output/dump_data_alldet_v1.9/cam-backbone/inputs'
# quant_data_outputs = f'output/dump_data_alldet_v1.9/cam-backbone/outputs'

# model_path = f"tools/onnx_utils/onnx_output/MM3DGOP-A02-V1.6-8m/cam-backbone-sim.onnx"
# quant_data_inputs = f'output/dump_data_v1.6/cam-backbone/inputs'
# quant_data_outputs = f'output/dump_data_v1.6/cam-backbone/outputs'

## cam-atttransform
# input_names = ['gridsample_input', 'gridsample_ref_points']
# inputs_shape = [(2, 256, 64, 128), (2, 3578, 1, 1, 15, 2)]

# output_names = ['gridsample_output']
# outputs_shape = [(1, 2, 3578, 64)]

# model_path = f"tools/onnx_utils/onnx_output/MM3DGOP-A02-V1.9_alldet-8m_clip2fp16/cam-atttransform-sim.onnx"
# quant_data_inputs = f'output/dump_data_alldet_v1.9/cam-atttransform/inputs'
# quant_data_outputs = f'output/dump_data_alldet_v1.9/cam-atttransform/outputs'

## cam-bev-backbone
# input_names = ['cam_bev_backbone_input']
# inputs_shape = [(1, 1, 3840, 64)]

# output_names = ['spatial_features_2d_cam']
# outputs_shape = [(1, 96, 48, 320)]

# model_path = f"tools/onnx_utils/onnx_output/MM3DGOP-A02-V1.9_alldet-8m_clip2fp16/cam-bev-backbone-sim.onnx"
# quant_data_inputs = f'output/dump_data_alldet_v1.9/cam-bev-backbone/inputs'
# quant_data_outputs = f'output/dump_data_alldet_v1.9/cam-bev-backbone/outputs'

## lidar-bev-backbone
# input_names = ['spatial_features']
# inputs_shape = [(1, 32, 96, 640)]

# output_names = ['spatial_features_2d_lidar']
# outputs_shape = [(1, 96, 48, 320)]

# model_path = f"tools/onnx_utils/onnx_output/MM3DGOP-A02-V1.6-8m/lidar-bev-backbone-sim.onnx"
# quant_data_inputs = f'output/dump_data/lidar-bev-backbone/inputs'
# quant_data_outputs = f'output/dump_data/lidar-bev-backbone/outputs'

## lidar-head
# input_names = ['spatial_features_2d_lidar']
# inputs_shape = [(1, 96, 48, 320)]

# output_names = ['det_pred_dicts_lidar_cls', 'det_pred_dicts_lidar_box', 'det_pred_dicts_lidar_dir_cls', 'seg_pred_dicts_lidar']
# outputs_shape = [(1, 1, 153600, 5), (1, 1, 153600, 7), (1, 1, 153600, 2), (1, 6, 96, 640)]

# model_path = f"tools/onnx_utils/onnx_output/MM3DGOP-A02-V1.6-8m/lidar-head-sim.onnx"
# quant_data_inputs = f'output/dump_data/lidar-head/inputs'
# quant_data_outputs = f'output/dump_data/lidar-head/outputs'

## cam-head
# input_names = ['spatial_features_2d_cam']
# inputs_shape = [(1, 96, 48, 320)]

# output_names = ['det_pred_dicts_cam_cls', 'det_pred_dicts_cam_box', 'det_pred_dicts_cam_dir_cls', 'seg_pred_dicts_cam']
# outputs_shape = [(1, 1, 153600, 5), (1, 1, 153600, 7), (1, 1, 153600, 2), (1, 6, 96, 640)]

# model_path = f"tools/onnx_utils/onnx_output/MM3DGOP-A02-V1.6-8m/cam-head-sim.onnx"
# quant_data_inputs = f'output/dump_data/cam-head/inputs'
# quant_data_outputs = f'output/dump_data/cam-head/outputs'

## fuser-fusion-head
# input_names = ['spatial_features_2d_lidar', 'spatial_features_2d_cam']
# inputs_shape = [(1, 96, 48, 320), (1, 96, 48, 320)]

# output_names = ['det_pred_dicts_fusion_cls', 'det_pred_dicts_fusion_box', 'det_pred_dicts_fusion_dir_cls', 'seg_pred_dicts_fusion']
# outputs_shape = [(1, 1, 153600, 5), (1, 1, 153600, 7), (1, 1, 153600, 2), (1, 6, 96, 640)]

# model_path = f"tools/onnx_utils/onnx_output/MM3DGOP-A02-V1.6-8m/fuser-fusion-head-sim.onnx"
# quant_data_inputs = f'output/dump_data/fuser-fusion-head/inputs'
# quant_data_outputs = f'output/dump_data/fuser-fusion-head/outputs'

# onnx init：
rnet1 = ONNXModel(model_path)


# cosine
def cal_cosine_similarity(val1, val2):
    if export_fp16:
        val1 = np.clip(val1, a_min=-65504, a_max=65504)
    cosine_numpy = (np.sum(val1 * val2) + 1e-8) / (np.linalg.norm(val1) * np.linalg.norm(val2) + 1e-8)
    return cosine_numpy


def export_onnx_result(fix_file_name=None, save_onnx_path=None):
    if fix_file_name is None:
        file_name_list = sorted(os.listdir(os.path.join(quant_data_inputs, input_names[0])))
    else:
        file_name_list = [fix_file_name]
    
    for file in file_name_list:
        file_name = file[:-4]          
        inputs = []
        for idx, input_name in enumerate(input_names):
            if postfix == 'bin':
                input_arr_2 = np.fromfile(os.path.join(quant_data_inputs, input_name, f'{file_name}.bin'), np.float32)
                input_arr = input_arr_2.reshape(inputs_shape[idx])
            elif postfix == 'npy':
                input_arr = np.load(os.path.join(quant_data_inputs, input_name, f'{file_name}.npy')).astype(np.float32)
            if export_fp16:
                input_arr = np.array(input_arr, dtype=np.float16)
            inputs.append(input_arr)
        inputs = tuple(inputs)

        ## onnx outputs
        onnx_outputs = rnet1.forward(inputs)

        ## pytorch outputs
        pytorch_outputs = []
        for idx, output_name in enumerate(output_names):
            if postfix == 'bin':
                output_arr_2 = np.fromfile(os.path.join(quant_data_outputs, output_name, f'{file_name}.bin'), np.float32)
                output_arr = output_arr_2.reshape(outputs_shape[idx])
            elif postfix == 'npy':
                output_arr = np.load(os.path.join(quant_data_outputs, output_name, f'{file_name}.npy'))
            pytorch_outputs.append(output_arr)

        print(f'***************************** {file} ******************************')
        for i, output_name in enumerate(output_names):
            # print(output_name, np.sum(onnx_outputs[i]), np.sum(pytorch_outputs[i]))
            # print(np.sum(np.abs(onnx_outputs[i]) > 2^16), onnx_outputs[i].shape)
            cos_sim = cal_cosine_similarity(onnx_outputs[i], pytorch_outputs[i])
            print(output_name, cos_sim)

        if save_onnx_path is not None:
            for i, output_name in enumerate(output_names):
                save_out_path = os.path.join(save_onnx_path, output_name)
                os.makedirs(save_out_path, exist_ok=True)
                onnx_outputs[i].tofile(f'{save_out_path}/{file_name}.bin')


if __name__ == '__main__':
    postfix = 'npy'
    save_onnx_path = None  # 'work_dirs/adela/calib_0218_GOP_data_T68_GOP_T68_30_Quant/spetr_rpn/outputs'
    export_fp16 = False

    export_onnx_result(save_onnx_path=save_onnx_path)



