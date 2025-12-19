#! /bin/bash
set -e
# export LD_LIBRARY_PATH=/iag_ad_vepfs_volc/iag_ad_vepfs_volc/hantianyou/TensorRT-10.8.0.43/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CXX=g++
export CC=gcc

 echo "执行 vd backbone 模型量化..."
    python3 script.py \
    -o /mnt/data/hantianyou/road_compare_tool/quant/spetr_step1.onnx \
    -c /mnt/data/hantianyou/winshare/backbone_bin_in \
    -q /mnt/data/hantianyou/road_compare_tool/quant/model_quantized.onnx \
    # -n "Conv_64,Conv_46,Conv_66,Conv_68,Conv_106,Conv_108,Conv_110,Conv_112"\
    -op "Conv"