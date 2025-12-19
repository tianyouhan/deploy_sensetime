import numpy as np
from ...utils import common_utils


def random_flip_along_x(label, points, return_flip=False, enable=None, gt_boxes=None, probability=None):
    """
    Args:
        label: (M, 1)
        points: (M, 3 + C)
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
    Returns:
    """
    if enable is None:
        if probability is None:
            enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
        else:
            enable = np.random.choice([False, True], replace=False, p=probability)
    if enable:
        points[:, 1] = -points[:, 1]
        if gt_boxes is not None:
            gt_boxes[:, 1] = -gt_boxes[:, 1]
            gt_boxes[:, 6] = -gt_boxes[:, 6]
            if gt_boxes.shape[1] > 7:
                gt_boxes[:, 8] = -gt_boxes[:, 8]
    return label, points, enable, gt_boxes


def random_flip_along_y(label, points, return_flip=False, enable=None, gt_boxes=None):
    """
    Args:
        label: (M, 1)
        points: (M, 3 + C)
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
    Returns:
    """
    if enable is None:
        enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        points[:, 0] = -points[:, 0]
        if gt_boxes is not None:
            gt_boxes[:, 0] = -gt_boxes[:, 0]
            gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
            if gt_boxes.shape[1] > 7:
                gt_boxes[:, 7] = -gt_boxes[:, 7]
    return label, points, enable, gt_boxes


def global_rotation(label, points, rot_range, return_rot=False, noise_rotation=None, gt_boxes=None, rot_clockwise=True):
    """
    Args:
        label: (M, 1)
        points: (M, 3 + C),
        rot_range: [min, max]
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
    Returns:
    """
    if noise_rotation is None: 
        noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
    if gt_boxes is not None:
        gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
        if not rot_clockwise:
            gt_boxes[:, 6] += noise_rotation  # 默认points是 x==>y 逆时针增加
        else:
            gt_boxes[:, 6] -= noise_rotation  # TODO, 广汽车 x前y左，yaw是沿着y的负半轴，顺时针增加。这里取减
        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
                np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
                np.array([noise_rotation])
            )[0][:, 0:2]
    return label, points, noise_rotation, gt_boxes


def global_scaling(label, points, scale_range, return_scale=False, gt_boxes=None):
    """
    Args:
        label: (M, 1)
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    if gt_boxes is not None:
        gt_boxes[:, :6] *= noise_scale
        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7:] *= noise_scale
    return label, points, noise_scale, gt_boxes