import glob
import os
import pickle as pkl
import cv2
import numpy as np



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='multi')
    parser.add_argument('-r', '--root', type=str, default=None, help='root_dir')
    args = parser.parse_args()
    basepath = args.root
    case_name = basepath.rstrip('/').split('/')[-1]

    with_postfix = True
    if with_postfix:
        path_A = f'{basepath}/center_camera_fov120_cam'
        path_B = f'{basepath}/center_camera_fov30_cam'
    else:
        path_A = f'{basepath}/center_camera_fov120'
        path_B = f'{basepath}/center_camera_fov30'
    path_C = f'{basepath}/bev'
    output_video = f'{basepath}/{case_name}.mp4'
    print(output_video)
    # 目标 resize 尺寸
    
    resize_width = 640
    

    # 获取图片文件（按名称排序对齐）
    images_A = sorted(glob.glob(os.path.join(path_A, '*.png')))
    images_B = sorted(glob.glob(os.path.join(path_B, '*.png')))
    images_C = sorted(glob.glob(os.path.join(path_C, '*.png')))
    if not len(images_A):
        images_A = sorted(glob.glob(os.path.join(path_A, '*.jpg')))
        images_B = sorted(glob.glob(os.path.join(path_B, '*.jpg')))
        # images_C = sorted(glob.glob(os.path.join(path_C, '*.jpg')))
    tmp_img = cv2.imread(images_A[0])
    print(tmp_img.shape)
    h, w, _ = tmp_img.shape
    resize_height = int(resize_width/w*h)
    # 检查数量一致
    print(len(images_A), len(images_B), len(images_C))
    assert len(images_A) == len(images_B) == len(images_C), "图片数量不一致"

    # 视频参数
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_size = (1280, resize_height + 512)  # 横1280，纵872
    fps = 10
    writer = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

    # 遍历每组图像进行处理
    for img_A_path, img_B_path, img_C_path in zip(images_A, images_B, images_C):
        img_A = cv2.imread(img_A_path)
        img_B = cv2.imread(img_B_path)
        img_C = cv2.imread(img_C_path)

        # resize A、B 图像
        img_A_resized = cv2.resize(img_A, (resize_width, resize_height))
        img_B_resized = cv2.resize(img_B, (resize_width, resize_height))

        # 横向拼接 A 和 B
        top_concat = np.hstack((img_A_resized, img_B_resized))  # shape: (360,1280)

        # 确保 C 是 (512,1280)
        img_C_resized = cv2.resize(img_C, (1280, 512)) if img_C.shape[1] != 1280 or img_C.shape[0] != 512 else img_C

        # 纵向拼接 Top + C
        final_img = np.vstack((top_concat, img_C_resized))  # shape: (872,1280)

        # 写入视频帧
        writer.write(final_img)

    writer.release()
    print("finish:", output_video)