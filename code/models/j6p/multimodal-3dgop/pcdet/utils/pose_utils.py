import os
import numpy as np


def get_lidar_pose(lines):
    pose_dict = {}
    for line in lines:
        line = line.strip().split()
        if len(line) == 0:
            continue
        assert len(line) >= 13, f'{line}: pose deficient!!! '  # format like: frame_name + 12 pose params
        pose_dict[os.path.splitext(line[-13])[0]] = list(map(float, line[-12:]))
    return pose_dict


def get_trans_matrix(src_pose, tar_pose):
    ## pose: [x1, x2 ... x12]
    src_pose = np.array(src_pose, dtype=np.float64).reshape(3, 4)
    src_pose = np.vstack((src_pose, np.array([0., 0., 0., 1.])))
    tar_pose = np.array(tar_pose, dtype=np.float64).reshape(3, 4)
    tar_pose = np.vstack((tar_pose, np.array([0., 0., 0., 1.])))
    trans_matrix = np.dot(np.linalg.inv(tar_pose), src_pose)  # src_pose to tar_pose
    return trans_matrix


def unnormalized_cloud(cloud_xyz, ground_height, normlized_height = -1.60, xy_drection = None):
    z_offset = normlized_height - ground_height
    cloud_points = np.zeros_like(cloud_xyz)
    if xy_drection == 'front_x_left_y':
        # The x-axis in point cloud is front
        cloud_points[:, 0] = cloud_xyz[:, 0]
        cloud_points[:, 1] = cloud_xyz[:, 1]
        cloud_points[:, 2] = cloud_xyz[:, 2] - z_offset
    elif xy_drection == 'right_x_front_y':
        # The x-axis in point cloud is right
        cloud_points[:, 0] = cloud_xyz[:, 1] * -1.0
        cloud_points[:, 1] = cloud_xyz[:, 0]
        cloud_points[:, 2] = cloud_xyz[:, 2] - z_offset
    elif xy_drection == 'left_x_back_y':
        # The x-axis in point cloud is left
        cloud_points[:, 0] = cloud_xyz[:, 1]
        cloud_points[:, 1] = cloud_xyz[:, 0] * -1.0
        cloud_points[:, 2] = cloud_xyz[:, 2] - z_offset
    else:
        raise ValueError('Direction x_y not specified')
    return cloud_points
    

def normalized_cloud(cloud_xyz, ground_height, normlized_height = -1.60, xy_drection = None):
    z_offset = normlized_height - ground_height
    cloud_points = np.zeros_like(cloud_xyz)
    if xy_drection == 'front_x_left_y':
        # The x-axis in point cloud is front
        cloud_points[:, 0] = cloud_xyz[:, 0]
        cloud_points[:, 1] = cloud_xyz[:, 1]
        cloud_points[:, 2] = cloud_xyz[:, 2] + z_offset
    elif xy_drection == 'right_x_front_y':
        # The x-axis in point cloud is right
        cloud_points[:, 0] = cloud_xyz[:, 1]
        cloud_points[:, 1] = cloud_xyz[:, 0] * -1.0
        cloud_points[:, 2] = cloud_xyz[:, 2] + z_offset
    elif xy_drection == 'left_x_back_y':
        # The x-axis in point cloud is left
        cloud_points[:, 0] = cloud_xyz[:, 1] * -1.0
        cloud_points[:, 1] = cloud_xyz[:, 0]
        cloud_points[:, 2] = cloud_xyz[:, 2] + z_offset
    else:
        raise ValueError('Direction x_y not specified')
    return cloud_points


def warp_cloud(src_cloud, src_pose, tar_pose, unnormalized_pose=True):
    src_cloud_xyz, intensity = src_cloud[:, 0:3], src_cloud[:, 3:]
    new_cloud = np.zeros_like(src_cloud)

    trans_matrix = get_trans_matrix(src_pose, tar_pose)  # 3D-GOP pose is unnormalized

    if unnormalized_pose:  # trans cloud to unnormalized, 3DGOP: groud_height=-2.0, direction=left_x_back_y
        src_cloud_xyz = unnormalized_cloud(src_cloud_xyz, ground_height=-2.0, xy_drection='left_x_back_y')

    src_cloud_hom = np.hstack((src_cloud_xyz, np.ones((src_cloud_xyz.shape[0], 1))))
    warp_cloud = np.transpose(np.dot(trans_matrix, np.transpose(src_cloud_hom)))  # p1 = (R|T)p0

    if unnormalized_pose:  # normalized cloud, 3DGOP: groud_height=-2.0, direction=left_x_back_y
        warp_cloud = normalized_cloud(warp_cloud, ground_height=-2.0, xy_drection='left_x_back_y')

    new_cloud[:, 0:3] = warp_cloud[:, 0:3]
    new_cloud[:, 3:] = intensity
    return new_cloud