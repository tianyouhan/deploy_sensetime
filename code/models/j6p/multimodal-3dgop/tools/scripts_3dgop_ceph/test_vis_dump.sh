
CONFIG=$1
GPUS=$2
MODE=$3
WORKS=$4
CKPT=$5
VIS_DIR=$6
CALIB_DIR=$7
SR=$8

export CONFIG=${CONFIG}
export GPUS=${GPUS}
export OMP_NUM_THREADS=12
export CUDA_LAUNCH_BLOCKING=1

## test
if [ $MODE == multi ]; then
    echo "Multi GPUS testing start"
    PADDING=True SR=$SR CALIB_PATH=$CALIB_DIR CALIB=True TEST_FLAG=True python3 -m torch.distributed.run --master_port $MLP_WORKER_0_PORT --nnodes $MLP_WORKER_NUM --node_rank $MLP_ROLE_INDEX --master_addr $MLP_WORKER_0_HOST --nproc_per_node $MLP_WORKER_GPU \
    test.py --cfg_file ${CONFIG} --workers ${WORKS} --eval_type "seg_det" --sync_bn --output_folder output --ckpt ${CKPT} --vis_dir ${VIS_DIR} --launcher pytorch
        
else
    echo "Single GPUS testing start"
    echo "CONFIG $CONFIG"
    echo "GPUS $GPUS"
    echo "MODE $MODE"
    echo "WORKS $WORKS"
    echo "CKPT $CKPT"
    echo "VIS_DIR $VIS_DIR"
    echo "CALIB_DIR $CALIB_DIR"
    echo "SR $SR"
    # '''
    # saving_format: bin
    # '''
    PADDING=True SR=$SR CALIB_PATH=$CALIB_DIR CALIB=True TEST_FLAG=True python3 -m torch.distributed.run --master_port ${MLP_WORKER_0_PORT:-29500} --nproc_per_node ${GPUS} \
    test.py --cfg_file ${CONFIG} --workers ${WORKS} --eval_type "seg_det" --sync_bn --output_folder output --ckpt ${CKPT} --vis_dir ${VIS_DIR} --launcher pytorch

    # '''
    # saving_format: npy
    # '''
    # PADDING=True SR=$SR CALIB_PATH=$CALIB_DIR CALIB=True TEST_FLAG=True saving_format=npy python -m torch.distributed.run --master_port ${MLP_WORKER_0_PORT:-29500} --nproc_per_node ${GPUS} \
    # test.py --cfg_file ${CONFIG} --workers ${WORKS} --eval_type "seg_det" --sync_bn --output_folder output --ckpt ${CKPT} --vis_dir ${VIS_DIR} --launcher pytorch
fi

