
## step 1, fusion pth + lidar pth + cam pth
# python tools/onnx_utils/checkpoint_trans.py
## 东风
## dynamic onnx. 按lidar，cam分支导出，动态维度
cd tools
ONNX_BRANCH=True EXPORT_FP16=True DYNAMIC_ONNX=True \
python -u onnx_utils/export_model_alldet_dynamic_onnx_1024x576.py