PARTITION=ad_lidar

## train
srun -p ${PARTITION} \
    --job-name=train \
    --ntasks=32 \
    --gres=gpu:8 \
    --ntasks-per-node=8 \
    --cpus-per-task=1 \
    --kill-on-bad-exit=1 \
    python -u train.py \
        --launcher slurm \
        --tcp_port 13450 \
        --cfg_file=cfgs/gacbaidu/lidar_gacbaidu_3dgop_seg_det_2V.yaml \
        --workers 1 \
        --eval_type "seg_det" \
        --sync_bn \
        --fup \
        --output_folder output


## test
# srun -p ${PARTITION} \
#     --job-name=test \
#     --ntasks=32 \
#     --gres=gpu:8 \
#     --ntasks-per-node=8 \
#     --cpus-per-task=1 \
#     --kill-on-bad-exit=1 \
#     python -u test.py \
#         --launcher slurm \
#         --tcp_port 13450 \
#         --cfg_file=cfgs/gacbaidu/lidar_gacbaidu_3dgop_seg_det_2V.yaml \
#         --workers 1 \
#         --eval_type "seg_det" \
#         --sync_bn \
#         --output_folder output \
#         --ckpt xx.pth
