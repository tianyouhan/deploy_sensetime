PARTITION=ad_lidar

## train
srun -p ${PARTITION} -w SH-IDC1-10-5-36-[218,227,228,229] \
    --job-name=train \
    --ntasks=32 \
    --gres=gpu:8 \
    --ntasks-per-node=8 \
    --cpus-per-task=8 \
    --kill-on-bad-exit=1 \
    python -u train.py \
        --launcher slurm \
        --tcp_port 13451 \
        --cfg_file=cfgs/A02/lidar_A02_3dgop_alldet_2V_V1.9.1_No1.yaml \
        --workers 1 \
        --eval_type "seg_det" \
        --sync_bn \
        --fup \
        --output_folder output \
        # --pretrained_model ../tmp_pkl/gacbaidu_lidar_No5_checkpoint_epoch_78.pth


## train
# srun -p ${PARTITION} -w SH-IDC1-10-5-36-[229-233,242,243,245]  \
#     --job-name=train \
#     --ntasks=64 \
#     --gres=gpu:8 \
#     --ntasks-per-node=8 \
#     --cpus-per-task=1 \
#     --kill-on-bad-exit=1 \
#     python -u train.py \
#         --launcher slurm \
#         --tcp_port 13451 \
#         --cfg_file=cfgs/A02/lidar_A02_3dgop_seg_det_2V_V1.6.1_No1.yaml \
#         --workers 1 \
#         --eval_type "seg_det" \
#         --sync_bn \
#         --fup \
#         --output_folder output \
#         --pretrained_model ../tmp_pkl/gacbaidu_lidar_No5_checkpoint_epoch_78.pth


## test
# srun -p ${PARTITION} -w SH-IDC1-10-5-40-[145,147,148,149] \
#     --job-name=test \
#     --ntasks=32 \
#     --gres=gpu:8 \
#     --ntasks-per-node=8 \
#     --cpus-per-task=1 \
#     --kill-on-bad-exit=1 \
#     python -u test.py \
#         --launcher slurm \
#         --tcp_port 13450 \
#         --cfg_file=cfgs/A02/lidar_A02_3dgop_seg_det_2V_V1.6_No1.yaml \
#         --workers 1 \
#         --eval_type "seg_det" \
#         --sync_bn \
#         --output_folder output \
#         --ckpt ../output/A02/lidar_A02_3dgop_seg_det_2V_V1.5_No4/default/ckpt/checkpoint_epoch_71.pth \
#         # --vis_dir unknown_vis_tmp/camera_A02_3dgop_seg_det_2V_No2

