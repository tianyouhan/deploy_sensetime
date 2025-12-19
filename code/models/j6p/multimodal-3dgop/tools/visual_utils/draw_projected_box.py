import numpy as np
import pickle as pkl
import cv2

def valid_pts(p1, p2, width, height):
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    flag1 = (x1 < 0) | (x1 > width) | (y1 < 0) | (y1 > height)
    flag2 = (x2 < 0) | (x2 > width) | (y2 < 0) | (y2 > height)
    return ~(flag1 & flag2)

def trunc_line(p1, p2, width, height):
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    return (x1, y1), (x2, y2)

def draw_projected_box(gt_box, P0_image, P0, Trv2c, rect, flag=False, window_name=None, thickness=2, color_list=None):
    H, W, _ = P0_image.shape
    for i in range(len(gt_box)):
        box = gt_box[i]
        x, y, z, l, w, h, yaw = box[:7]
        if not flag:
            z = z + h/2
        half_l = l / 2
        half_w = w / 2
        half_h = h / 2

        offsets = np.array([[half_l, half_w, half_h],
                        [-half_l, half_w, half_h],
                        [-half_l, -half_w, half_h],
                        [half_l, -half_w, half_h],
                        [half_l, half_w, -half_h],
                        [-half_l, half_w, -half_h],
                        [-half_l, -half_w, -half_h],
                        [half_l, -half_w, -half_h],
                        [0, 0, 0]])

        rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                    [np.sin(yaw), np.cos(yaw), 0],
                                    [0, 0, 1]])

        rotated_offsets = np.dot(rotation_matrix, offsets.T).T
        corner_coordinates = rotated_offsets + np.array([x, y, z])

        # Apply the transformation from vehicle coordinates to camera coordinates
        transformation_matrix = np.dot(Trv2c, np.hstack((corner_coordinates, np.ones((9, 1)))).T).T
        
        if transformation_matrix[8,2] < 0:
            # print("remove the boxes behind camera")
            continue

        for idx in range(8):
            if transformation_matrix[idx,2] < 0:
                transformation_matrix[idx,2] = 0.15
        projection_matrix = np.dot(P0, np.dot(rect, transformation_matrix.T)).T
        projection_matrix[:, 0] /= projection_matrix[:, 2]
        projection_matrix[:, 1] /= projection_matrix[:, 2]

        # Convert to pixel coordinates
        pixel_coordinates = projection_matrix[:, :2]

        # Draw 3D box on the image, Green color for the box
        pixel_coordinates = pixel_coordinates.astype(np.int32)
        
        # Connect the corners to form the box
        point_pairs = [[0,1], [1,2], [2,3], [3,0],
                       [4,5], [5,6], [6,7], [7,4],
                       [0,4], [1,5], [2,6], [3,7]]
        
        if color_list is None:
            color = (255, 0, 0)
        else:
            color = tuple([int(c) for c in color_list[i]])

        for pairs in point_pairs:
            i0, i1 = pairs
            p0, p1 = trunc_line(pixel_coordinates[i0], pixel_coordinates[i1], W, H)
            if valid_pts(p0, p1, W, H):
                cv2.line(P0_image, p0, p1, color, thickness)
    return P0_image
    

def draw_projected_points(points, P0_image, P0, Trv2c, rect, flag=False, window_name=None, thickness=2, color=(0, 255, 0)):
    H, W, _ = P0_image.shape

    corner_coordinates = points[:, :3]
    Nums = len(points)
    # Apply the transformation from vehicle coordinates to camera coordinates
    transformation_matrix = np.dot(Trv2c, np.hstack((corner_coordinates, np.ones((Nums, 1)))).T).T
    
    # mask = transformation_matrix[:, 2] >= 0
    
    projection_matrix = np.dot(P0, np.dot(rect, transformation_matrix.T)).T
    projection_matrix[:, 0] /= projection_matrix[:, 2]
    projection_matrix[:, 1] /= projection_matrix[:, 2]

    # Convert to pixel coordinates
    sample_points_cam = projection_matrix[:, :2]

    # Draw 3D box on the image, Green color for the box
    sample_points_cam = sample_points_cam.astype(np.int32)
    homo = projection_matrix[:, 2]
    # sample_points_cam[..., 0] /= W
    # sample_points_cam[..., 1] /= H

    # check if out of image
    valid_mask = ((homo > 1e-5) \
        & (sample_points_cam[:, 1]/H > 0.0)
        & (sample_points_cam[:, 1]/H < 1.0)
        & (sample_points_cam[:, 0]/W > 0.0)
        & (sample_points_cam[:, 0]/W < 1.0)
    )

    masked_points = sample_points_cam[valid_mask]
    for point in masked_points:
        x, y = point
        cv2.circle(P0_image, (x, y), radius=1, color=(0, 255, 0), thickness=-1)
    
    return P0_image

if __name__ == '__main__':
    with open("res0.pkl", "rb") as f:
        res = pkl.load(f)

    annotation = res['annos']

    point_color = [0.5,0.5,0.5]
    rot_axis=2
    center_mode = 'lidar_bottom'
    bbox_color=(1, 0, 0) 
    points_in_box_color=(0, 0, 1),
    dimensions = annotation['dimensions']
    locations = annotation['location']
    yaw_angles = annotation['rotation_y']
    rect = res['calib']['R0_rect']
    Trv2c = res['calib']['Tr_velo_to_cam']
    P0 = res['calib']['P0'][:3,:]

    P0_image = cv2.imread("imgs/0000000_0.png")
    H, W, _ = P0_image.shape
    with open("save.pkl", "rb") as f:
        gt_box = pkl.load(f)
    
    draw_projected_box(gt_box, P0_image, P0, Trv2c, rect)

    



    