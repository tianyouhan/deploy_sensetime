# 工作主目录
rootDir=${1:-/data/horizon_j6/data/work/lgx/mnt/model/logistics_vehicle/road/6v_multi_task_rclane/gridsample_e2e_hnop_v3.10.33_guangxiong}
# 模型编译结果文件夹
modelDir=${2:-quant484_oriadd_rm}
# 量化数据集，也用来做为评估数据集
quantDir=${3:-calibration_data_dir}
# 模型图像输入tensor个数，例如input_type_train: bgr; bgr; bgr; bgr; bgr; bgr; featuremap; featuremap，这里是6个bgr图像输入
cameraNum=${4:-6}

# indexOri：评估数据集的起始索引，默认从1开始
# length：评估数据集的样本数量，默认20个
# mean：图像归一化的均值，默认[103.53, 116.28, 123.675]，这里是BGR通道的均值
# scale：图像归一化的缩放因子，默认[0.017429193899782137, 0.01750700280112045, 0.017124753831663668]，这里是BGR通道的缩放因子
# python3 02_ptq_verify_avg.py \
python3 ./horizon_scripts/02_ptq_verify_avg.py \
    --rootDir ${rootDir} \
    --dataDir ${quantDir} \
    --modelDir ${modelDir} \
    --cameraNum ${cameraNum} \
    --modelPrefix torch-jit-export_subnet0 \
    --indexOri 1 \
    --length 20 \
    --mean 103.53 116.28 123.675 \
    --scale 0.017429193899782137 0.01750700280112045 0.017124753831663668



