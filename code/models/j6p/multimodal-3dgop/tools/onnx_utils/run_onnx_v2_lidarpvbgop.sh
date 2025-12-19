

## 主线-PVBGOP
## static onnx. 按lidar分支导出，静态维度
cd tools
ONNX_BRANCH=True EXPORT_FP16=True DYNAMIC_ONNX=True \
python -u onnx_utils/export_model_lidarpvbgp_static_onnx_1024x576.py