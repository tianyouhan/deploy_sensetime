# 导入debug模块
import hmct.quantizer.debugger as dbg

dbg.sensitivity_analysis(model_or_file='quant484_oriadd_rm/torch-jit-export_subnet0_calibrated_model.onnx',
                         calibrated_data='quant_1',
                         pick_threshold=0.9999,
                         data_num=1,
                         qtype="float32",
                         sensitive_nodes=[])
