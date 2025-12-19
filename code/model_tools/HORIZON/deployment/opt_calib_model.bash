python 05_opt_calib_model_horizon_nn.py \
    -c /home/mnt/zengyuqian/code/gridsample_e2e/gridsample_e2e_v0325_AD_regnetx400_bevg8mk7_bevneck64exp2fixfreeze_e8/calibration_data \
    -r /home/mnt/zengyuqian/code/gridsample_e2e/gridsample_e2e_v0325_AD_regnetx400_bevg8mk7_bevneck64exp2fixfreeze_e8/ \
    -md quant_386_lite_0321 \
    -mp torch-jit-export_subnet0 \
    -nk Conv