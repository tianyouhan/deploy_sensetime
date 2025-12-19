PARTITION=pat_libra

## test
# CALIB_PATH='/mnt/lustre/xuzhiyong/code/multimodal-3dgop-xuzhiyong/output/dump_data' \
# CALIB=True SR=1 NUM=20 saving_format=npy \
# srun -p ${PARTITION} \
#     --job-name=test \
#     --ntasks=1 \
#     --gres=gpu:1 \
#     --ntasks-per-node=1 \
#     --cpus-per-task=1 \
#     --kill-on-bad-exit=1 \
#     python -u test.py \
#         --launcher slurm \
#         --tcp_port 13450 \
#         --cfg_file=cfgs/A02/MM3DGOP-A02-V1.6-3heads_A02_3dgop_seg_det_2V_V1.6_No2.yaml \
#         --workers 0 \
#         --eval_type "seg_det" \
#         --sync_bn \
#         --output_folder output \
#         --ckpt ../output/A02/MM3DGOP-A02-V1.6-3heads_A02_3dgop_seg_det_2V_V1.6_No2/checkpoint_epoch_39_3heads.pth \
#         # --vis_dir onnx_utils/checkpoints/MM3DGOP-A02-V1.5-3heads_A02_3dgop_seg_det_2V_V1.5_No2/fusion_A02_3dgop_seg_det_2V_V1.5_No2_3heads.yaml


# CALIB_PATH='/mnt/lustre/xuzhiyong/code/multimodal-3dgop-xuzhiyong/output/dump_data_v1.9' \
# CALIB=True SR=1 NUM=20 saving_format=npy det_cls_with_sigmoid=True \
# srun -p ${PARTITION} \
#     --job-name=test \
#     --ntasks=1 \
#     --gres=gpu:1 \
#     --ntasks-per-node=1 \
#     --cpus-per-task=1 \
#     --kill-on-bad-exit=1 \
#     python -u test.py \
#         --launcher slurm \
#         --tcp_port 13450 \
#         --cfg_file=cfgs/A02/MM3DGOP-A02-V1.9-3heads_A02_3dgop_seg_det_2V_V1.9_No1.yaml \
#         --workers 0 \
#         --eval_type "seg_det" \
#         --sync_bn \
#         --output_folder output \
#         --ckpt ../output/A02/MM3DGOP-A02-V1.9-3heads_A02_3dgop_seg_det_2V_V1.9_No1/checkpoint_epoch_17_3heads.pth \
#         # --vis_dir onnx_utils/checkpoints/MM3DGOP-A02-V1.5-3heads_A02_3dgop_seg_det_2V_V1.5_No2/fusion_A02_3dgop_seg_det_2V_V1.5_No2_3heads.yaml

# dump
# quant: bin
# PADDING=False \ 
cd tools
export CALIB_PATH='/mnt/lustrenew/zhanghongcheng/zhc/multimodal-GOP/output/A02/MM3DGOP-A02-V1.9-3heads_A02_3dgop_alldet_2V_V1.9_No1/dump_0606_pad10'
export CALIB=True
export SR=1
export NUM=600
export PADDING=True
export saving_format=bin 


python -u test.py \
    --launcher pytorch \
    --tcp_port 12880 \
    --cfg_file=cfgs/A02/MM3DGOP-A02-V1.9-3heads_A02_3dgop_alldet_2V_V1.9_No1.yaml \
    --workers 0 \
    --eval_type "seg_det" \
    --sync_bn \
    --output_folder output \
    --ckpt ../output/A02/MM3DGOP-A02-V1.9-3heads_A02_3dgop_alldet_2V_V1.9_No1/checkpoint_epoch_40_3heads.pth \
    --vis_dir xxx
