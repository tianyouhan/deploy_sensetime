set -x

cd tools

c_name=GAC-P643M1ATX_PVBGOP-J6M_volc_qat_smallrange_v0_0909_atx_splitbkb
c_output_dir=../output/A02_ceph/${c_name}_v15_1
c_vis_dir=$c_output_dir/vis
export SPLIT_BKB=True
ckpts=(
    /iag_ad_vepfs_volc/iag_ad_vepfs_volc/zhanghongcheng/code/multimodal-3dgop-Main_LidarPVBGOP_v1.11.0/checkpoints/V1.15.0_checkpoint_epoch_58.pth
)
for ckpt in "${ckpts[@]}"; do
    now=$(date +"%Y%m%d_%H%M%S")
    bash scripts_3dgop_ceph/calib_vis.sh \
    cfgs/A02_ceph/${c_name}.yaml \
    1 \
    single \
    0 \
    $ckpt \
    $c_vis_dir \
    2>&1|tee $c_output_dir/calib-$now.log
done

