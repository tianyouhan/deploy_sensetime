rootDir=${1:-/home/mnt/zengyuqian/code/gridsample_e2e/gridsample_e2e_v0317_AD_regnetx400_bevg8mk7}
python3 ../horizon_scripts/02_ptq_verify_avg.py \
    --rootDir ${rootDir} \
    --dataDir /mnt/afs/lirunze/workspace/experiment/quant_600_withT28 \
    --modelDir quant_600withT28_lite_0321 \
    --modelPrefix torch-jit-export_subnet0 \
    --indexOri 0 \
    --length 20 \
    --mean 103.53 116.28 123.675 \
    --scale 0.017429193899782137 0.01750700280112045 0.017124753831663668