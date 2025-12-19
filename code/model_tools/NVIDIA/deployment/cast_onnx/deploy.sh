export model_name=cam-branch-sim
export model_precision=fp16
precision_flag=""
if [[ $model_precision != "fp32" ]]; then
    precision_flag="--${model_precision}"
fi

export dst=thor_dev_01@10.155.172.184:/mnt/external_disk/ghb/lidar_model

# pc 或 thor 执行 pip3 install onnxconverter_common 
python3 gen_stronly_typed_onnx.py \
    --skip-node-names Conv_124,Add_123,Add_387,Concat_382,Mul_388,Conv_386,GlobalAveragePool_383,Conv_384,Relu_385 \
    --path ${model_name}.onnx --path2 subgraph.onnx --opath ${model_name}_${model_precision}.onnx
python3 insert_cast.py --model ${model_name}_${model_precision}.onnx --node Slice_251  --output ${model_name}_${model_precision}_t.onnx 
python3 insert_cast.py --model ${model_name}_${model_precision}_t.onnx --node Slice_289  --output ${model_name}_${model_precision}_y.onnx
python3 node_to_f32.py --onnx ${model_name}_${model_precision}_y.onnx --tensor spatial_features_2d_cam --out ${model_name}_${model_precision}_z.onnx
scp cam-branch-sim_fp16_z.onnx ${dst} 
# thor 执行
trtexec \
    --onnx=${model_name}_${model_precision}_z.onnx \
    --saveEngine=${model_name}_${model_precision}.trt \
    --timingCacheFile=${model_name}_timing_cache \
    --exportTimes=${model_name}_${model_precision}_timing.json \
    --exportProfile=${model_name}_${model_precision}_profile.json \
    --exportLayerInfo=${model_name}_${model_precision}_layerinfo.json \
    --useManagedMemory \
    --separateProfileRun \
    --profilingVerbosity=detailed \
    --stronglyTyped \
    --minShapes=gridsample_ref_points:2x1x1x1x15x2,gridsample_indexes_0:1,gridsample_indexes_1:1 \
    --optShapes=gridsample_ref_points:2x3840x1x1x15x2,gridsample_indexes_0:3840,gridsample_indexes_1:3840 \
    --maxShapes=gridsample_ref_points:2x3840x1x1x15x2,gridsample_indexes_0:3840,gridsample_indexes_1:3840 \
    2>&1 | tee ./build_${model_name}_${model_precision}.log

trtexec \
    --onnx=${model_name}.onnx \
    --saveEngine=${model_name}_${model_precision}.trt \
    --timingCacheFile=${model_name}_timing_cache \
    --exportTimes=${model_name}_${model_precision}_timing.json \
    --exportProfile=${model_name}_${model_precision}_profile.json \
    --exportLayerInfo=${model_name}_${model_precision}_layerinfo.json \
    --useManagedMemory \
    --separateProfileRun \
    --profilingVerbosity=detailed \
    --minShapes=gridsample_ref_points:2x1x1x1x15x2,gridsample_indexes_0:1,gridsample_indexes_1:1 \
    --optShapes=gridsample_ref_points:2x3840x1x1x15x2,gridsample_indexes_0:3840,gridsample_indexes_1:3840 \
    --maxShapes=gridsample_ref_points:2x3840x1x1x15x2,gridsample_indexes_0:3840,gridsample_indexes_1:3840 \
    2>&1 | tee ./build_${model_name}_${model_precision}.log



export model_name=lidar-branch-sim
export model_precision=fp16
precision_flag=""
if [[ $model_precision != "fp32" ]]; then
    precision_flag="--${model_precision}"
fi

python3 gen_stronly_typed_onnx.py \
    --skip-node-names Add_97,Mul_98,Gather_8,Mul_10,Gather_12,Add_13,Cast_14,Add_18,Equal_24,Where_25,Unsqueeze_27,Unsqueeze_35,Reshape_38,ScatterND_39 \
    --path ${model_name}.onnx --opath ${model_name}_${model_precision}.onnx
python3 node_to_f32.py --onnx ${model_name}_${model_precision}.onnx --tensor spatial_features_2d_lidar --out ${model_name}_${model_precision}_y.onnx
python3 node_to_f32.py --onnx ${model_name}_${model_precision}_y.onnx --tensor vfe_input --out ${model_name}_${model_precision}.onnx


scp ${model_name}_${model_precision}.onnx ${dst} 


trtexec \
    --onnx=${model_name}_${model_precision}.onnx \
    --saveEngine=${model_name}_${model_precision}.trt \
    --timingCacheFile=${model_name}_timing_cache \
    --exportTimes=${model_name}_${model_precision}_timing.json \
    --exportProfile=${model_name}_${model_precision}_profile.json \
    --exportLayerInfo=${model_name}_${model_precision}_layerinfo.json \
    --useManagedMemory \
    --separateProfileRun \
    --profilingVerbosity=detailed \
    --stronglyTyped \
    --minShapes=voxel_coords:0x2,vfe_input:0x9x32x1 \
    --optShapes=voxel_coords:10000x2,vfe_input:10000x9x32x1 \
    --maxShapes=voxel_coords:20000x2,vfe_input:20000x9x32x1 \
    2>&1 | tee ./build_${model_name}_${model_precision}.log

trtexec \
    --onnx=ma_${model_name}.onnx \
    --saveEngine=${model_name}_${model_precision}.trt \
    --timingCacheFile=${model_name}_timing_cache \
    --exportTimes=${model_name}_${model_precision}_timing.json \
    --exportProfile=${model_name}_${model_precision}_profile.json \
    --exportLayerInfo=${model_name}_${model_precision}_layerinfo.json \
    --useManagedMemory \
    --separateProfileRun \
    --profilingVerbosity=detailed \
    ${precision_flag} \
    --minShapes=voxel_coords:0x2,vfe_input:0x9x32x1 \
    --optShapes=voxel_coords:10000x2,vfe_input:10000x9x32x1 \
    --maxShapes=voxel_coords:20000x2,vfe_input:20000x9x32x1 \
    2>&1 | tee ./build_${model_name}_${model_precision}.log

export model_name=fuser-fusion-head-sim
export model_precision=fp16
precision_flag=""
if [[ $model_precision != "fp32" ]]; then
    precision_flag="--${model_precision}"
fi
python3 gen_stronly_typed_onnx.py \
    --skip-node-names Add_97,Mul_98,Gather_8,Mul_10,Gather_12,Add_13,Cast_14,Add_18,Equal_24,Where_25,Unsqueeze_27,Unsqueeze_35,Reshape_38,ScatterND_39 \
    --path ${model_name}.onnx --opath ${model_name}_${model_precision}.onnx
python3 node_to_f32.py --onnx ${model_name}_${model_precision}.onnx --tensor spatial_features_2d_lidar --out ${model_name}_${model_precision}_y.onnx

trtexec \
    --onnx=${model_name}_${model_precision}_y.onnx \
    --saveEngine=${model_name}_${model_precision}.trt \
    --timingCacheFile=${model_name}_timing_cache \
    --exportTimes=${model_name}_${model_precision}_timing.json \
    --exportProfile=${model_name}_${model_precision}_profile.json \
    --exportLayerInfo=${model_name}_${model_precision}_layerinfo.json \
    --useManagedMemory \
    --separateProfileRun \
    --profilingVerbosity=detailed \
    --stronglyTyped \
    2>&1 | tee ./build_${model_name}_${model_precision}.log


python3 ../scripts/gen_polygraphy_input.py  --input_dir dump_data_1024/lidar-branch/inputs  --output_file ${model_name}_inputs.json  --onnx ${model_name}.onnx
polygraphy run ${model_name}.onnx --onnxrt    --load-inputs ${model_name}_inputs.json     --warm-up 1     --save-results ${model_name}_fp32_outputs_onnx.json     --verbose     --profiling-verbosity detailed
scp ${model_name}_inputs.json ${dst} 
scp ${model_name}_fp32_outputs_onnx.json ${dst} 

polygraphy run ${model_name}_${model_precision}.trt --model-type engine --trt \
    --load-inputs ${model_name}_inputs.json \
    --verbose \
    --save-results ${model_name}_${model_precision}_outputs.json \
    --profiling-verbosity detailed | tee debug_onnx.log
python3 ../scripts/cmp_trt_outputs.py ${model_name}_fp32_outputs.json ${model_name}_fp16_outputs.json | tee cmp.log
python3 ../scripts/cmp_trt_outputs.py ${model_name}_fp32_outputs_onnx.json ${model_name}_fp16_outputs.json | tee cmp.log
python3 ../scripts/cmp_trt_outputs.py outputs ${model_name}_fp16_outputs.json | tee cmp.log
python3 ../scripts/cmp_trt_outputs.py outputsl ${model_name}_fp16_outputs.json | tee cmp.log
python3 ../scripts/cmp_trt_outputs.py outputs ${model_name}_fp32_outputs.json | tee cmp.log
python3 ../scripts/cmp_trt_outputs.py /mnt/afs3/guhanbin/model_opt/lidar_model/dump_data_1024/cam-branch/outputs ${model_name}_fp32_outputs_onnx.json | tee cmp.log
python3 ../scripts/cmp_trt_outputs.py /mnt/afs3/guhanbin/model_opt/lidar_model/dump_data_1024/lidar-branch/outputs ${model_name}_fp32_outputs_onnx.json | tee cmp.log

python3 ../scripts/gen_polygraphy_input.py  --input_dir ${model_name}-dump/inputs --output ${model_name}_inputs.json
