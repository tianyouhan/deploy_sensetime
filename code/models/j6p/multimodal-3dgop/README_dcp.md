

# OpenPCDet-BEVFusion


## Overview
- [Basic usage of OpenPCDet](README_ori.md)
- [BEV-Cam-3D](#bev-cam-3d)
- [BEVFusion](#bevfusion)

## Environment

Step 0: 激活 SH40/SH36 虚拟环境，进入工作路径  
`SH40:`
```
source env_sh40.sh
```
`SH36:`
```
source env_sh36.sh
```
```
cd tools
```

## BEVFusion on GAC-Baidu
Step 1: 训练Lidar分支: `pointpillar: VFE + Scatter + BEV-backbone + (Seg-head & Det-head)`
```
sh scripts_3dgop/train_lidar_gacbaidu_3dgop_seg_det_2V.sh
```
Step 2: 训练Camera分支: `Image-backbone + FPN-neck + LSS/GridSampling + BEV-backbone + (Seg-head & Det-head)`
```
sh scripts_3dgop/train_camera_gacbaidu_3dgop_seg_det_2V.sh
```
Step 3: 训练Fusion (load cam & lidar weight): `(Lidar & Camera) + Fuser + (Seg-head & Det-head)`
```
sh scripts_3dgop/train_fusion_gacbaidu_3dgop_seg_det_2V.sh
```

## BEVFusion on A02
Step 1: 训练Lidar分支 (load gacbaidu lidar pretrain): `pointpillar: VFE + Scatter + BEV-backbone + (Seg-head & Det-head)`
```
sh scripts_3dgop/train_lidar_A02_3dgop_seg_det_2V.sh
```
Step 2: 训练Camera分支: `Image-backbone + FPN-neck + LSS/GridSampling + BEV-backbone + (Seg-head & Det-head)`
```
sh scripts_3dgop/train_camera_A02_3dgop_seg_det_2V.sh
```
Step 3: 训练Fusion (load lidar & cam weight): `(Lidar & Camera) + Fuser + (Seg-head & Det-head)`
```
sh scripts_3dgop/train_fusion_A02_3dgop_seg_det_2V.sh
```



