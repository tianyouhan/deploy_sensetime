PARTITION=ad_lidar

## train
srun -p ${PARTITION} -x SH-IDC1-10-5-36-[244,250,252,254] \
    --job-name=train \
    --ntasks=32 \
    --gres=gpu:8 \
    --ntasks-per-node=8 \
    --cpus-per-task=8 \
    --kill-on-bad-exit=1 \
    python -u train.py \
        --launcher slurm \
        --tcp_port 13591 \
        --cfg_file=cfgs/A02/camera_A02_3dgop_seg_det_2V_V1.9.1_No1.yaml \
        --workers 1 \
        --eval_type "seg_det" \
        --sync_bn \
        --fup \
        --output_folder output


## test
# srun -p ${PARTITION} -w SH-IDC1-10-5-40-[148,149] \
#     --quotatype=spot \
#     --job-name=test \
#     --ntasks=16 \
#     --gres=gpu:8 \
#     --ntasks-per-node=8 \
#     --cpus-per-task=1 \
#     --kill-on-bad-exit=1 \
#     python -u test.py \
#         --launcher slurm \
#         --tcp_port 13450 \
#         --cfg_file=cfgs/A02/camera_A02_3dgop_seg_det_2V_V1.5_No1.yaml \
#         --workers 1 \
#         --eval_type "seg_det" \
#         --sync_bn \
#         --output_folder output \
#         --ckpt ../output/A02/camera_A02_3dgop_seg_det_2V_V1.5_No1/default/ckpt/checkpoint_epoch_26.pth \
#         # --vis_dir unknown_vis_tmp/camera_A02_3dgop_seg_det_2V_No2
