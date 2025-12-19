PARTITION=ad_lidar

## train
srun -p ${PARTITION} -w SH-IDC1-10-5-36-[230,231,233,242,245,246,247,248] \
    --job-name=train \
    --ntasks=64 \
    --gres=gpu:8 \
    --ntasks-per-node=8 \
    --cpus-per-task=8 \
    --kill-on-bad-exit=1 \
    python -u train.py \
        --launcher slurm \
        --tcp_port 13592 \
        --cfg_file=cfgs/A02/fusion_A02_3dgop_alldet_2V_V1.9.1_No1.yaml \
        --workers 1 \
        --eval_type "seg_det" \
        --sync_bn \
        --fup \
        --output_folder output \
        --load_from_cam ../output/A02/camera_A02_3dgop_alldet_2V_V1.9.1_No1/default/ckpt/checkpoint_epoch_40.pth \
        --pretrained_model ../output/A02/lidar_A02_3dgop_alldet_2V_V1.9.1_No1/default/ckpt/checkpoint_epoch_60.pth


## test
# srun -p ${PARTITION} -w SH-IDC1-10-5-36-[248,249] \
#     --job-name=test \
#     --ntasks=16 \
#     --gres=gpu:8 \
#     --ntasks-per-node=8 \
#     --cpus-per-task=1 \
#     --kill-on-bad-exit=1 \
#     python -u test.py \
#         --launcher slurm \
#         --tcp_port 13450 \
#         --cfg_file=cfgs/A02/fusion_A02_3dgop_seg_det_2V_V1.9_No1.yaml \
#         --workers 1 \
#         --eval_type "seg_det" \
#         --sync_bn \
#         --output_folder output \
#         --ckpt ../output/A02/fusion_A02_3dgop_seg_det_2V_V1.9_No1/default/ckpt/checkpoint_epoch_17.pth \
#         --vis_dir unknown_vis_lustrenew/fusion_A02_3dgop_seg_det_2V_V1.9_No1_rainy_batch_10
#         # --eval_all \