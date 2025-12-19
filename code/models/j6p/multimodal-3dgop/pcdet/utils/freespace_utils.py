import numpy as np
import cv2


def create_drivable_mask(points_list, voxel_size, pc_range, ego_flag=False):
    w, h = int((pc_range[3] - pc_range[0]) / voxel_size[0]), int((pc_range[4] - pc_range[1]) / voxel_size[1])
    mask = np.zeros((h, w), dtype=np.uint8)  # [y-->down, x-->right]
    center_point = [int(-pc_range[0] / voxel_size[0]), int(-pc_range[1] / voxel_size[1])]

    grid_ego, grid = [], []
    for points in points_list:
        points = ((points - pc_range[0:2]) / voxel_size[0:2]).astype(np.int32)
        grid.append(points)
        if ego_flag and cv2.pointPolygonTest(points, tuple(center_point), False) >= 0:
            grid_ego.append(points)

    valid_grid = grid_ego if len(grid_ego) > 0 else grid
    cv2.fillPoly(mask, valid_grid, color=(1))

    return mask


def get_mask(batch_dict, label=None):
    mask = None
    if label is not None:
        mask = (~(label == 255)).int()
    if 'freespace_label' in batch_dict:
        if mask is not None:
            mask = mask * batch_dict['freespace_label']
        else:
            mask = batch_dict['freespace_label']
    return mask


