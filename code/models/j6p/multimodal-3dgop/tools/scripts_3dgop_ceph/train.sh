
CONFIG=$1
GPUS=$2
MODE=$3
WORKS=$4
CKPT=$5
LIDAR_PTH=$6
CAMERA_PTH=$7

export CONFIG=${CONFIG}
export GPUS=${GPUS}
export OMP_NUM_THREADS=12
export CUDA_LAUNCH_BLOCKING=1

## train
if [ $MODE == multi ]; then
    echo "Multi GPUS training start"
    python -m torch.distributed.run --master_port $MLP_WORKER_0_PORT --nnodes $MLP_WORKER_NUM --node_rank $MLP_ROLE_INDEX --master_addr $MLP_WORKER_0_HOST --nproc_per_node $MLP_WORKER_GPU \
    train.py --cfg_file ${CONFIG} --workers ${WORKS} --eval_type "seg_det" --sync_bn --fup --output_folder output --pretrained_model ${LIDAR_PTH} --load_from_cam ${CAMERA_PTH} --launcher pytorch
        
else
    echo "Single GPUS training start"
    python -m torch.distributed.run --master_port ${MLP_WORKER_0_PORT:-29500} --nproc_per_node ${GPUS} \
    train.py --cfg_file ${CONFIG} --workers ${WORKS} --eval_type "seg_det" --sync_bn --fup --output_folder output --pretrained_model ${LIDAR_PTH} --load_from_cam ${CAMERA_PTH} --launcher pytorch
fi

