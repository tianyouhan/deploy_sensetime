PARTITION=ad_lidar

## train
srun -p ${PARTITION} -w SH-IDC1-10-5-36-[220,221,222,223,224,242,243,245] \
    --job-name=train \
    --ntasks=64 \
    --gres=gpu:8 \
    --ntasks-per-node=8 \
    --cpus-per-task=1 \
    --kill-on-bad-exit=1 \
    python -u train.py \
        --launcher slurm \
        --tcp_port 13592 \
        --cfg_file=cfgs/A02/fusion_A02_3dgop_seg_det_2V_V1.7_No1.yaml \
        --workers 1 \
        --eval_type "seg_det" \
        --sync_bn \
        --fup \
        --output_folder output \
        --load_from_cam ../output/A02/camera_A02_3dgop_seg_det_2V_V1.7_No1/default/ckpt/checkpoint_epoch_40.pth \
        --pretrained_model ../output/A02/lidar_A02_3dgop_seg_det_2V_V1.7_No1/default/ckpt/checkpoint_epoch_80.pth


## test
# srun -p ${PARTITION} -w SH-IDC1-10-5-40-[121-124] \
#     --quotatype=spot \
#     --job-name=test \
#     --ntasks=32 \
#     --gres=gpu:8 \
#     --ntasks-per-node=8 \
#     --cpus-per-task=1 \
#     --kill-on-bad-exit=1 \
#     python -u test.py \
#         --launcher slurm \
#         --tcp_port 13450 \
#         --cfg_file=cfgs/A02/fusion_A02_3dgop_seg_det_2V_V1.6_No2.yaml \
#         --workers 1 \
#         --eval_type "seg_det" \
#         --sync_bn \
#         --output_folder output \
#         --ckpt ../output/A02/fusion_A02_3dgop_seg_det_2V_V1.6_No2/default/ckpt/checkpoint_epoch_39.pth \
#         # --vis_dir onnx_utils/checkpoints/MM3DGOP-A02-V1.5-3heads_A02_3dgop_seg_det_2V_V1.5_No2/fusion_A02_3dgop_seg_det_2V_V1.5_No2_3heads.yaml
