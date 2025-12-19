
CONFIG=$1
GPUS=$2
MODE=$3
WORKS=$4
CKPT=$5

export CONFIG=${CONFIG}
export GPUS=${GPUS}
export OMP_NUM_THREADS=12
export CUDA_LAUNCH_BLOCKING=1

## test
if [ $MODE == multi ]; then
    echo "Multi GPUS testing start"
    python -m torch.distributed.run --master_port $MLP_WORKER_0_PORT --nnodes $MLP_WORKER_NUM --node_rank $MLP_ROLE_INDEX --master_addr $MLP_WORKER_0_HOST --nproc_per_node $MLP_WORKER_GPU \
    test.py --cfg_file ${CONFIG} --workers ${WORKS} --eval_type "seg_det" --sync_bn --output_folder output --ckpt ${CKPT} --launcher pytorch
        
else
    echo "Single GPUS testing start"
    python -m torch.distributed.run --master_port ${MLP_WORKER_0_PORT:-28888} --nproc_per_node ${GPUS} \
    test.py --cfg_file ${CONFIG} --workers ${WORKS} --eval_type "seg_det" --sync_bn --output_folder output --ckpt ${CKPT} --launcher pytorch
fi

