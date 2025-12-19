PARTITION=ad_lidar

# srun -p ${PARTITION} \
#     --job-name=train \
#     --ntasks=1 \
#     --gres=gpu:1 \
#     --ntasks-per-node=1 \
#     --cpus-per-task=1 \
#     --kill-on-bad-exit=1 \
#     python -u onnx_utils/export_model_onnx.py

## alldet
# EXPORT_FP16=True \
# srun -p ${PARTITION} \
#     --job-name=train \
#     --ntasks=1 \
#     --gres=gpu:1 \
#     --ntasks-per-node=1 \
#     --cpus-per-task=1 \
#     --kill-on-bad-exit=1 \
#     python -u onnx_utils/export_model_alldet_onnx.py

## alldet 按lidar，cam分支导出，静态维度
# ONNX_BRANCH=True EXPORT_FP16=True \
# srun -p ${PARTITION} \
#     --job-name=train \
#     --ntasks=1 \
#     --gres=gpu:1 \
#     --ntasks-per-node=1 \
#     --cpus-per-task=1 \
#     --kill-on-bad-exit=1 \
#     python -u onnx_utils/export_model_alldet_one_onnx.py

## dynamic onnx. 按lidar，cam分支导出，动态维度
ONNX_BRANCH=True EXPORT_FP16=True DYNAMIC_ONNX=True \
srun -p ${PARTITION} \
    --job-name=train \
    --ntasks=1 \
    --gres=gpu:1 \
    --ntasks-per-node=1 \
    --cpus-per-task=1 \
    --kill-on-bad-exit=1 \
    python -u onnx_utils/export_model_alldet_dynamic_onnx.py