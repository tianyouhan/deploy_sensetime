
import json
import numpy as np
import os
import argparse
import tqdm
import open3d as o3d

def load_pcd_file(file_path):
    cloud = pypcd.PointCloud.from_path(file_path)
    points = np.zeros([cloud.width, 4], dtype=np.float32)
    points[:, 0] = cloud.pc_data['x'].copy()
    points[:, 1] = cloud.pc_data['y'].copy()
    points[:, 2] = cloud.pc_data['z'].copy()
    points[:, 3] = cloud.pc_data['intensity'].copy().astype(np.float32) / 255.0
    return points

def load_pcd_file2(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    # points_m1_0 = np.asarray(pcd.points).astype(np.float32)
    xyz = np.asarray(pcd.points)
    print(dir(pcd))
    # 先假设 intensity 也在 point cloud属性里
    # 有些版本的 Open3D 可能能直接读 intensity（PCD 格式字段支持）
    try:
        intensity = np.asarray(pcd.intensity).reshape(-1, 1)
    except:
        # 某些文件格式下 intensity 可能读不到
        print("intensity 属性未找到，需确认文件格式是否支持")
        intensity = np.zeros((xyz.shape[0], 1))  # 默认补 0

    # 拼接为 (N, 4)
    points_m1_0 = np.hstack((xyz, intensity))
    return points_m1_0

def load_bin_file(file_path):
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1,4)
    return points

def save_bin_file(points, file_path):
    points.astype(np.float32).tofile(file_path)
    return 

def robust_ground_height(points, lower_percentile=0.5, upper_percentile=2):
    z_vals = points[:, 2]
    lower = np.percentile(z_vals, lower_percentile)
    upper = np.percentile(z_vals, upper_percentile)
    ground_candidates = z_vals[(z_vals >= lower) & (z_vals <= upper)]
    return np.mean(ground_candidates)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('-i', '--input', type=str, default=None, help='input pcd path')
    parser.add_argument('-o', '--output', type=str, default=None, help='output bin path')
    args = parser.parse_args()
    M1_pcd_path = args.input
    M1_list = os.listdir(M1_pcd_path)
    M1_list.sort()

    M1_bin_path = args.output
    os.makedirs(M1_bin_path, exist_ok=True)
    print('start converting ', M1_pcd_path)
    for i in range(len(M1_list)):
        if M1_list[i].endswith('.txt'):
            continue
        cur_m1_pcd_path = os.path.join(M1_pcd_path, M1_list[i])
        print(cur_m1_pcd_path)
        points = load_pcd_file2(cur_m1_pcd_path)
        print(points.shape)
        z = robust_ground_height(points)
        # if z > -1.5:
        #     print('-1.6m')
        print(z)
        points[:, 2] -= 1.6
        cur_m1_bin_path = os.path.join(M1_bin_path, M1_list[i].replace('pcd', 'bin'))
        save_bin_file(points, cur_m1_bin_path)
    print('finish converting, saving to', M1_bin_path)

