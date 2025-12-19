# 有些模型需要换onnx op的版本号，例如gridsample算子
python3 exchange_onnx.py model.onnx

# 转对应的J6P模型以及中间产生的文件
result_folder=road_rclane_quant_result
mkdir ${result_folder}
hb_compile -c road_compile_config.yml 2>&1 | tee ${result_folder}/log_${result_folder}_compile.txt

# 量化集
quantdir=./calibration_data_dir

# 相机数量
camera_num=6

# 量化相似度
bash ./horizon_scripts/ptq_verify_386.bash $PWD ${result_folder} ${quantdir} ${camera_num} 2>&1 | tee ${result_folder}/log_${result_folder}_verify.txt

# 敏感层分析
#bash ./horizon_scripts/run_get_sensitivity.bash ${result_folder} 2>&1 | tee ${result_folder}/log_${result_folder}_sensitivity.txt
