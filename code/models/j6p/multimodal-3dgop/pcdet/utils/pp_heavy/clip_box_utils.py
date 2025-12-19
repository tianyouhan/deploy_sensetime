"""
Copyright (c) Kui XU, SenseTime Group. All Rights Reserved
"""

import numpy as np
import os


class Object3dST(object):
    """ 3d object label (sensetime version)"""

    def __init__(self, data):
        # extract 3d bounding box information
        self.t = (
            (data[0] + data[3]) / 2,
            (data[1] + data[4]) / 2,
            data[2],
        )  # location (x,y,z) in camera coord.
      
        self.w = data[3] - data[0]
        self.l = data[4] - data[1]
        self.h = data[5] - data[2]
        self.ry = data[6]  # yaw angle, actually is z

class Object3d(object):
    """ 3d object label """

    def __init__(self, data):

        # extract label, truncation, occlusion
        self.w = data[3]  # box height
        self.l = data[4]  # box width
        self.h = data[5]  # box length (in meters)
        self.t = (data[0], data[1], data[2])  # location (x,y,z) in camera coord.
        self.ry = (
            data[6]  # - np.pi / 2
        )  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

    def print_object(self):
        
        print("3d bbox h,w,l: %f, %f, %f" % (self.h, self.w, self.l))
        print(
            "3d bbox location, ry: (%f, %f, %f), %f"
            % (self.t[0], self.t[1], self.t[2], self.ry)
        )
        # print("Difficulty of estimation: {}".format(self.estimate_diffculty()))

    def draw_bbox(self, fig=None, color=(0, 1, 0)):
        from utils import visualize_utils
        corners3d = compute_box_3d(self)
        fig = visualize_utils.draw_corners3d(np.array([corners3d]), fig=fig, color=color)
        return fig

def rotx(t):
    """ 3D Rotation about the x-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotz(t):
    """ Rotation about the z-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def read_label(preds_tensor):
    lines = []
    for i in range(preds_tensor.shape[0]):
        one_pred = []
        one_pred.append(preds_tensor[i][0].item())
        one_pred.append(preds_tensor[i][1].item())
        one_pred.append(preds_tensor[i][2].item())
        one_pred.append(preds_tensor[i][3].item())
        one_pred.append(preds_tensor[i][4].item())
        one_pred.append(preds_tensor[i][5].item())
        one_pred.append(preds_tensor[i][6].item()) 
        lines.append(one_pred)
    objects = [Object3d(line) for line in lines]
    return objects



def load_velo_scan(velo_filename, dtype=np.float32, n_vec=4):
    scan = np.fromfile(velo_filename, dtype=dtype)
    scan = scan.reshape((-1, n_vec))
    return scan


def in_hull(p, hull):
    from scipy.spatial import Delaunay

    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0

def extract_pc_in_box3d(pc, box3d):
    """ pc: (N,3), box3d: (8,3) """
    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds

def compute_box_3d(obj):
    """ Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
        Returns:
            corners_3d: (8,3) array in in rect camera coord.
    """
    # compute rotational matrix around yaw axis
    R = rotz(-obj.ry)

    EPS = 1e-6
    # 3d bounding box dimensions
    l = obj.l + EPS
    w = obj.w + EPS
    h = obj.h + EPS

    # 3d bounding box corners
    # x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    # y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    # z_corners = [h, h, h, h, 0, 0, 0, 0]

    y_corners = [-l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2.]
    x_corners = [w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.]
    z_corners = [h, h, h, h, 0, 0, 0, 0]
   
    # rotate and translate 3d bounding box
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d = np.dot(R, corners_3d)
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2]


    return np.transpose(corners_3d)


def get_box_center(box):
    mi = np.min(box, axis=0)
    ma = np.max(box, axis=0)
    data = np.array([mi[0], mi[1], mi[2], ma[0], ma[1], ma[2]])
    return ((data[0] + data[3]) / 2, (data[1] + data[4]) / 2, (data[2] + data[5]) / 2)

def get_center(pc):
    mi = np.min(pc, axis=0)
    ma = np.max(pc, axis=0)
    data = np.array([mi[0], mi[1], mi[2], ma[0], ma[1], ma[2]])
    return ((data[0] + data[3]) / 2, (data[1] + data[4]) / 2, (data[2] + data[5]) / 2)

def rotate_pc(pc, ry):
    cen = get_center(pc)
    pc_t = pc[:, :3] - cen
    R = rotz(-ry)
    pc_r = np.dot(pc_t, R)
    pc = pc_r + cen
    return pc

def pc_to_box(pc, raw_obj, raw_center):
    # import pdb; pdb.set_trace()
    mi = np.min(pc, axis=0)
    ma = np.max(pc, axis=0)
    obj = Object3dST([mi[0], mi[1], mi[2], ma[0], ma[1], ma[2], raw_obj.ry])

    xyz = obj.t
    d_xyz = np.array(xyz) - raw_center
    R = rotz(raw_obj.ry)
    d_xyz = np.dot(d_xyz, R)
    obj.t = tuple(d_xyz + raw_center)

    box = compute_box_3d(obj)
    return obj, box

def get_clipped_boxes3d(pc_velo, obj, raw_box):
    # box3d_pts_3d_velo = compute_box_3d(obj)
    # print("box3d_pts_3d_velo:",box3d_pts_3d_velo.shape)
    # print(box3d_pts_3d_velo)

    box3droi_pc_velo, _ = extract_pc_in_box3d(pc_velo, raw_box)

    if len(box3droi_pc_velo) > 0 :
        # draw_lidar(box3droi_pc_velo, fig=fig, pc_label=False)

        # clipped_box  = pc_to_box(box3droi_pc_velo, obj)
        # draw_gt_boxes3d([clipped_box], color=(0, 0, 1), label="")
        rot_pc = rotate_pc(box3droi_pc_velo, obj.ry)
        cen = get_center(box3droi_pc_velo)
        clipped_rot_obj, clipped_rot_box = pc_to_box(rot_pc, obj, cen)
        # draw_lidar(rot_pc, fig=fig, pc_label=False)
    else:
        # print ('Warning: There is a empty bbox!')
        clipped_rot_obj, clipped_rot_box = None, None

    return clipped_rot_obj, clipped_rot_box


def clip_boxes_with_lidar(pc_velo, preds_tensor):
    """ 
    clip boxes with lidar
    """
    clipped_boxes_lines = []

    objects = read_label(preds_tensor)

    for i in range(len(objects)):
        obj = objects[i]
        box3d_pts_3d_velo = compute_box_3d(obj)

        clipped_rot_obj, clipped_rot_box = get_clipped_boxes3d(
            pc_velo, obj, raw_box=box3d_pts_3d_velo
        )
        if clipped_rot_obj is None:
            continue

        # clipped_rot_obj.ry += (np.pi / 2)
        preds_tensor[i][0] = clipped_rot_obj.t[0]
        preds_tensor[i][1] = clipped_rot_obj.t[1]
        preds_tensor[i][2] = clipped_rot_obj.t[2]
        preds_tensor[i][3] = clipped_rot_obj.w
        preds_tensor[i][4] = clipped_rot_obj.l
        preds_tensor[i][5] = clipped_rot_obj.h
        preds_tensor[i][6] = clipped_rot_obj.ry

    return preds_tensor


