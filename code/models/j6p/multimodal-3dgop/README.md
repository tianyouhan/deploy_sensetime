

# OpenPCDet-LidarPVBGOP
Lidar PVB+GOP联合模型

## Overview
- [Basic usage of OpenPCDet](README_ori.md)

## Environment
选择火山云镜像：iag-ad-01-cn-shanghai.cr.volces.com/ad/lidar_ptq:v1

# QAT部署
对于模型中的VFE scatter部分采取QAT部署

calib -> calib ckpt
```
scripts/v1.15.0/x_single_qat_calib_vis_splitbkb.sh
```

compile -> hbm model
```
scripts/v1.15.0/x_single_qat_compile_splitbkb.sh
```

dump data
```
scripts/v1.15.0/x_single_qat_test_vis_dumpcalibdata_splitbkb.sh
```

# PTQ 部署
对于模型中的 Lidar branch + Head 采取PTQ部署

```
# lidar backbone 部署
hb_compile -c scripts/v1.15.0/GAC-P643M1ATX_PVBGOP-J6M_volc-BKB_PTQ.yaml
# lidar gop head 部署
hb_compile -c scripts/v1.15.0/GAC-P643M1ATX_PVBGOP-J6M_volc-GOPHead_PTQ_int16.yaml
# lidar pvb head 部署
hb_compile -c scripts/v1.15.0/GAC-P643M1ATX_PVBGOP-J6M_volc-PVBHead_PTQ_int16.yaml
```
# 数据路径
## checkpoint + onnx
```
# 火山云
/iag_ad_vepfs_volc/iag_ad_vepfs_volc/zhanghongcheng/code/multimodal-3dgop-Main_LidarPVBGOP_v1.11.0/checkpoints
```
## 量化数据集（PTQ）
```
# 火山云
/iag_ad_vepfs_volc/iag_ad_vepfs_volc/zhanghongcheng/code/multimodal-3dgop-Main_LidarPVBGOP_v1.11.0/output/A02_ceph/GAC-P643M1ATX_PVBGOP-J6M_volc_qat_smallrange_v0_0909_atx_splitbkb/calib_dump_V15ep58_splitbkb
```