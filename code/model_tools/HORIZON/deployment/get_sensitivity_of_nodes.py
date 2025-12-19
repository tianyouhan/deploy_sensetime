# 导入debug模块
import hmct.quantizer.debugger as dbg
# 导入log日志模块
import logging

# 若verbose=True时，需要先设置log level为INFO
logging.getLogger().setLevel(logging.INFO)
# 获取节点量化敏感度
node_message = dbg.get_sensitivity_of_nodes(
        model_or_file='./quant484_oriadd_rm/torch-jit-export_subnet0_calibrated_model.onnx',
        metrics=['cosine-similarity', 'mse'],
        calibrated_data='../quant_1/',
        output_node=None,
        node_type='node',
        data_num=None,
        verbose=True,
        interested_nodes=None)

