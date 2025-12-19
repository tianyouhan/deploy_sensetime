import cv2
import os
from glob import glob
import numpy as np

# 两组图片路径（支持通配符 *.png、*.png）
left_dir = "/iag_ad_01/ad/zhanghongcheng/code/multimodal-3dgop/output/A02_ceph/camera_A02_3dgop_alldet_2V_V1.9.1_No1_1024x576_virtual_visdump/dump_0630/vis3/det_vis/cam/2025_06_20_11_02_39_AutoCollect/center_camera_fov30_cam"
right_dir = "/iag_ad_01/ad/zhanghongcheng/code/multimodal-3dgop/output/A02_ceph/camera_A02_3dgop_alldet_2V_V1.9.1_No1_1024x576_virtual_visdump/dump_0630/vis3/det_vis/cam/2025_06_20_11_02_39_AutoCollect/center_camera_fov120_cam"

left_imgs = sorted(glob(os.path.join(left_dir, '*.png')))
right_imgs = sorted(glob(os.path.join(right_dir, '*.png')))

# 确保数量相同
assert len(left_imgs) == len(right_imgs), "图片数量不一致"

# 读取第一张图确定尺寸
img1 = cv2.imread(left_imgs[0])
img2 = cv2.imread(right_imgs[0])

assert img1.shape[0] == img2.shape[0], "两图高度不一致"

height = img1.shape[0]
width = img1.shape[1] + img2.shape[1]
fps = 10  # 你想要的视频帧率

# 输出视频路径
out_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

# 拼接并写入视频
for l_path, r_path in zip(left_imgs, right_imgs):
    img1 = cv2.imread(l_path)
    img2 = cv2.imread(r_path)

    # 横向拼接
    combined = np.hstack((img1, img2))

    video_writer.write(combined)

video_writer.release()
print(f"✅ 视频已保存：{out_path}")