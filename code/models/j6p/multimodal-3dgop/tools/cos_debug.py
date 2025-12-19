# 导入debug模块
import horizon_nn.quantizer.debugger as dbg
# 导入log日志模块
import logging

# 若verbose=True时，需要先设置log level为INFO
logging.getLogger().setLevel(logging.INFO)
# 获取节点量化敏感度
node_message = dbg.get_sensitivity_of_nodes(
        model_or_file='/iag_ad_01/ad/zhanghongcheng/code/multimodal-3dgop-Main_LidarPVBGOP_v1.11.0/tools/cfgs/A02_ceph/pure-bkb-int8-ep58/lidar-backbone-vpu_ptq_model.onnx',
        metrics=['cosine-similarity', 'mse'],
        calibrated_data='/iag_ad_01/ad/zhanghongcheng/code/multimodal-3dgop-Main_LidarPVBGOP_v1.11.0/output/A02_ceph/GAC-P643M1ATX_PVBGOP-J6M_volc_qat_smallrange_v0_0831_atx/calib_dump_ep58_new_npy/lidar-bev-backbone/inputs',
        output_node=None,
        node_type='node',
        data_num=None,
        verbose=True,
        interested_nodes=None)