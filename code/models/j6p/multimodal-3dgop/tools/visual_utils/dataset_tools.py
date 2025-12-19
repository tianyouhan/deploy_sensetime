import numpy as np
import pickle as pkl
import cv2
import os
import math
import json
from draw_projected_box import draw_projected_box, draw_projected_points
from pcdet.utils import pose_utils, ceph_utils
import open3d as o3d

        

PATH_MAPPING = dict({
    # "/mnt/": "aoss-gt-gop:s3://aoss-gt/sh36/mnt/",
    "/mnt/": "aoss-zhc-v2:s3://dcp36_lustre_aoss_v2_xuzhiyong/",
    "s3://aoss-test-data": "aoss-test-data:s3://aoss-test-data",
    "s3:/aoss-test-data": "aoss-test-data:s3://aoss-test-data",
    "s3://aoss-gt/": "aoss-gt:s3://aoss-gt/",
    "s3://sdc3-adas-3": "ad_system_common_auto:s3://sdc3-adas-3"
})

def limit_period(val, offset=0.5, period=np.pi):
    return val - np.floor(val / period + offset) * period

def draw_circle(bev_image, bev_height, bev_width, bev_resolution, distance=25, center_coords=None):
    circle_radius_meter = distance  # 圆的半径，单位为米
    circle_radius_pixel = int(circle_radius_meter / bev_resolution)

    # 绘制圆形
    if center_coords is None:
        center_w, center_h = [bev_width // 2, bev_height // 2]
    else:
        center_w, center_h = center_coords
    circle_center = tuple([center_w, center_h])  # 圆心位置
    circle_color = (111,144,249)  # 圆的颜色，使用 BGR 格式

    cv2.circle(bev_image, circle_center, circle_radius_pixel, circle_color, thickness=1)

    # 添加文字
    text = '%sm' % str(distance)
    text_offset_x = circle_radius_pixel-20  # 文字相对于圆心的 x 偏移量
    text_offset_y = 0  # 文字相对于圆心的 y 偏移量
    text_position = (circle_center[0] + text_offset_x, circle_center[1] + text_offset_y)  # 文字位置
    text_font = cv2.FONT_HERSHEY_SIMPLEX  # 字体类型
    text_scale = 0.5  # 字体缩放系数
    text_color = (0, 0, 0)  # 文字颜色，使用 BGR 格式
    text_thickness = 2  # 文字粗细

    cv2.putText(bev_image, text, text_position, text_font, text_scale, text_color, text_thickness)
    return 

class_color_map = np.array([
                            [128, 128, 128],  # gray
                            [0,     0, 255],  # red
                            [  0, 255,   0],  # green
                            [255,   0,   0],  # blue
                            [  0, 255, 255],  # yellow
                            # [255,   0, 255],  # purple
                            [0,     0, 128],  # deep red
                            [0,   128,   0],  # deep green
                            [128,   0,   0],  # deep blue  
                            [0,   128, 128],  # deep yellow
                            [128,   0, 128],  # deep purple
                            [255, 255,   0],  # sky blue
                            [255, 255, 255],  # white
                            [24,  81,  172],  # seg keep: cls error, brown
                            [0,     0,   0],  # seg keep: defect error, black
                                            ],  # bgr format for cv2
                            dtype=np.uint8)

def draw_texts_right_bottom(img, lines, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5,
                            thickness=1, color=(0, 0, 0), padding=10, char_width=8):

    # 每个字符大概宽度（像素），根据字体估算。你可调大点（中文会更宽）
    max_characters = 28
    max_text_width = char_width * max_characters

    # 获取文字高度信息
    _, text_height = cv2.getTextSize("Test", font, font_scale, thickness)[0]
    # 起始纵坐标（从底往上画）
    y = img.shape[0] - padding
    for line in reversed(lines):
        # 计算文字宽度
        (text_width, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        x = img.shape[1] - max_text_width - padding  # 保证右侧预留28字符宽度

        # 不超出顶部才画
        if y - text_height >= 0:
            cv2.putText(img, line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
            y -= text_height + 4  # 行距
        else:
            break

def draw_bev(points, boxes, bev_height, bev_width, center_coords=None,
             bev_resolution=0.16, show_img=False, with_dir=True,
             radius_list=[], save_dir='', filename='', color_list=None, seg_pred=None, gt_names=None):
    bev_image = np.zeros((bev_height, bev_width, 3), dtype=np.uint8) + 255
    # draw points in bev
    if center_coords is None:
        center_w, center_h = [bev_width // 2, bev_height // 2]
    else:
        center_w, center_h = center_coords
    image_coordinates = np.round((points[:, :2] / bev_resolution) + [center_w, center_h]).astype(int)
    valid_indices = np.logical_and(image_coordinates[:, 0] >= 0, image_coordinates[:, 0] < bev_width)
    valid_indices = np.logical_and(valid_indices, np.logical_and(image_coordinates[:, 1] >= 0, image_coordinates[:, 1] < bev_height))
    image_coordinates = image_coordinates[valid_indices]
    bev_image[image_coordinates[:, 1], image_coordinates[:, 0]] = [128, 128, 128]  # x_right_y_back (256, 640
    if seg_pred is not None:
        bev_image[seg_pred != 0] = class_color_map[seg_pred][seg_pred != 0]
    
    # boxes = m1_gtbox.copy()
    for j, gt in enumerate(boxes):
        x, y, z = gt[0], gt[1], gt[2]
        l, w, h = gt[3], gt[4], gt[5]
        yaw = gt[6] #TODO
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        # import ipdb; ipdb.set_trace()
        corner_points = np.array([[-l / 2, -w / 2, 0], [l / 2, -w / 2, 0], [l / 2, w / 2, 0], [-l / 2, w / 2, 0]])
        rotated_corner_points = np.zeros_like(corner_points)
        rotated_corner_points[:, 0] = cos_yaw * corner_points[:, 0] - sin_yaw * corner_points[:, 1] + x
        rotated_corner_points[:, 1] = sin_yaw * corner_points[:, 0] + cos_yaw * corner_points[:, 1] + y
        rotated_corner_points[:, 2] = corner_points[:, 2]

        bev_x = np.round((rotated_corner_points[:, 0] / bev_resolution) + center_w).astype(int)
        bev_y = np.round((rotated_corner_points[:, 1] / bev_resolution) + center_h).astype(int)

        if color_list is None:
            color = (255, 0, 0)
        else:
            color = tuple([int(c) for c in color_list[j]])
        
        for ii in range(4):
            cv2.line(bev_image, (bev_x[ii], bev_y[ii]), (bev_x[(ii + 1) % 4], bev_y[(ii + 1) % 4]), color, 2)
        if with_dir:
            ct_x = 0.5 * (bev_x[0] + bev_x[2])    
            ct_y = 0.5 * (bev_y[0] + bev_y[2]) 
            cv2.arrowedLine(bev_image, (int(ct_x), int(ct_y)), (int(ct_x+25*cos_yaw), int(ct_y+25*sin_yaw)), (0, 0, 0), 1, cv2.LINE_AA, tipLength=0.2)
            
    bev_image = cv2.flip(bev_image, 0)
    str_lines = []
    for j, gt in enumerate(boxes):
        x, y, z = gt[0], gt[1], gt[2]
        l, w, h = gt[3], gt[4], gt[5]
        yaw = gt[6] #TODO
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        corner_points = np.array([[-l / 2, -w / 2, 0], [l / 2, -w / 2, 0], [l / 2, w / 2, 0], [-l / 2, w / 2, 0]])
        rotated_corner_points = np.zeros_like(corner_points)
        rotated_corner_points[:, 0] = cos_yaw * corner_points[:, 0] - sin_yaw * corner_points[:, 1] + x
        rotated_corner_points[:, 1] = sin_yaw * corner_points[:, 0] + cos_yaw * corner_points[:, 1] + y
        rotated_corner_points[:, 2] = corner_points[:, 2]
        bev_x = np.round((rotated_corner_points[:, 0] / bev_resolution) + center_w).astype(int)
        bev_y = np.round((rotated_corner_points[:, 1] / bev_resolution) + center_h).astype(int)
        ct_x = 0.5 * (bev_x[0] + bev_x[2])    
        ct_y = 0.5 * (bev_y[0] + bev_y[2]) 
        h, w, _ = bev_image.shape
        cv2.putText(bev_image, str(j), (int(ct_x), h-int(ct_y)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,255), thickness=2)
        if gt_names is not None:
            cur_gtname = gt_names[j]
            cur_str = f"Obj ID: {j},    {cur_gtname}"
            str_lines.append(cur_str)
    if gt_names is not None:
        draw_texts_right_bottom(bev_image, str_lines)
    for radius in radius_list:
        draw_circle(bev_image, bev_height, bev_width, bev_resolution, radius, center_coords=center_coords)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    bev_name = os.path.join(save_dir, filename)
    cv2.imwrite(bev_name, bev_image)
    if show_img:
        cv2.imshow("frame %s" % (filename), bev_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return


def draw_bev_o(points, boxes, bev_height, bev_width, 
             bev_resolution=0.1, offset_x=0, show_img=False,
             radius_list=[], save_dir='', filename=''):
    bev_image = np.zeros((bev_height, bev_width, 3), dtype=np.uint8) + 255
    # draw points in bev
    points[:, 0] -= offset_x
    image_coordinates = np.round((points[:, :2] / bev_resolution) + [bev_width / 2, bev_height / 2]).astype(int)
    valid_indices = np.logical_and(image_coordinates[:, 0] >= 0, image_coordinates[:, 0] < bev_width)
    valid_indices = np.logical_and(valid_indices, np.logical_and(image_coordinates[:, 1] >= 0, image_coordinates[:, 1] < bev_height))
    image_coordinates = image_coordinates[valid_indices]
    bev_image[image_coordinates[:, 1], image_coordinates[:, 0]] = [128, 128, 128]
    # boxes = m1_gtbox.copy()
    for j, gt in enumerate(boxes):
        x, y, z = gt[0], gt[1], gt[2]
        l, w, h = gt[3], gt[4], gt[5]
        x -= offset_x
        yaw = -gt[6]
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        corner_points = np.array([[-l / 2, -w / 2, 0], [l / 2, -w / 2, 0], [l / 2, w / 2, 0], [-l / 2, w / 2, 0]])
        rotated_corner_points = np.zeros_like(corner_points)
        rotated_corner_points[:, 0] = cos_yaw * corner_points[:, 0] - sin_yaw * corner_points[:, 1] + x
        rotated_corner_points[:, 1] = sin_yaw * corner_points[:, 0] + cos_yaw * corner_points[:, 1] + y
        rotated_corner_points[:, 2] = corner_points[:, 2]

        bev_x = np.round((rotated_corner_points[:, 0] / bev_resolution) + (bev_width / 2)).astype(int)
        bev_y = np.round((rotated_corner_points[:, 1] / bev_resolution) + (bev_height / 2)).astype(int)

        for ii in range(4):
            cv2.line(bev_image, (bev_x[ii], bev_y[ii]), (bev_x[(ii + 1) % 4], bev_y[(ii + 1) % 4]), (0, 255, 0), 2)
            
    bev_image = cv2.flip(bev_image, 0)
    for radius in radius_list:
        draw_circle(bev_image, bev_height, bev_width, bev_resolution, radius, offset_x=-offset_x)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    bev_name = os.path.join(save_dir, filename)
    cv2.imwrite(bev_name, bev_image)
    if show_img:
        cv2.imshow("frame %s" % (filename), bev_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return

def mapping_path_to_ceph(PATH_MAPPING, path):
    new_path = path
    if new_path.count(':') >= 2:
        return new_path
    for key, val in PATH_MAPPING.items():
        if key in path:
            new_path = path.replace(key, val)
            break
    return new_path

def load_intrinsics_calib(calib_json_path, ceph_client=None, cam_K=False):
    calib_json_path = mapping_path_to_ceph(PATH_MAPPING, calib_json_path)
    intrinsic_calib = ceph_utils.ceph_read(calib_json_path, None, True, ceph_client)
    key = [ele for ele in intrinsic_calib.keys()][0]
    camera_intrinsic_dict = intrinsic_calib[key]

    # camera_dist = np.array(camera_intrinsic_dict['param']['cam_dist']['data'][0])
    # cam_k_new_resize
    print(camera_intrinsic_dict['param'])
    if cam_K and cam_K in camera_intrinsic_dict['param']:
        camera_intrinsic = np.array(camera_intrinsic_dict['param'][cam_K]['data'])
        print(cam_K)
    else:
        if not cam_K and 'cam_k_new_resize' in camera_intrinsic_dict['param']:
            camera_intrinsic = np.array(camera_intrinsic_dict['param']['cam_k_new_resize']['data'])
            print('cam_k_new_resize')
        elif not cam_K and 'cam_k_new' in camera_intrinsic_dict['param']:
            camera_intrinsic = np.array(camera_intrinsic_dict['param']['cam_k_new']['data'])
            print('cam_k_new')
        elif not cam_K and 'cam_K_new' in camera_intrinsic_dict['param']:
            camera_intrinsic = np.array(camera_intrinsic_dict['param']['cam_K_new']['data'])
            print('cam_K_new')
        else:
            camera_intrinsic = np.array(camera_intrinsic_dict['param']['cam_K']['data'])
            print('cam_K')
    return camera_intrinsic
        
def load_extrinsic_calib(calib_json_path, normalize_calib='A02', lidar2camera=False, ceph_client=None):
    calib_json_path = mapping_path_to_ceph(PATH_MAPPING, calib_json_path)
    extrinsic_calib = ceph_utils.ceph_read(calib_json_path, None, True, ceph_client)
    print(extrinsic_calib)
    key = [ele for ele in extrinsic_calib.keys()][0]
    lidar_extrinsic = np.array(
        extrinsic_calib[key]['param']['sensor_calib']['data'], dtype=np.float32)
    if (normalize_calib is not None) or (type(normalize_calib) is bool and normalize_calib):
        # (lidar_extrinsic @ trans_mat) @ point_xyz, points "front_x_left_y, ground=-1.6" to "left_x_back_y, ground=-2.0"
        if type(normalize_calib) is bool and normalize_calib:
            normalize_calib = 'gacbaidu'

        if type(normalize_calib) is str:
            if normalize_calib == 'gacbaidu':
                trans_mat = np.array(
                    [[ 0,   1,   0,    0],
                        [-1,   0,   0,    0],
                        [ 0,   0,   1, -0.4],
                        [ 0,   0,   0,    1]], dtype=np.float32
                )
            elif normalize_calib == 'A02':
                trans_mat = np.array(
                    [[ 1,   0,   0,    0],
                        [ 0,   1,   0,    0],
                        [ 0,   0,   1,  1.6],
                        [ 0,   0,   0,    1]], dtype=np.float32
                )
            else:
                raise ValueError(f'Not support calib:{normalize_calib}')
        elif type(normalize_calib) is list:
            trans_mat = np.array(normalize_calib, dtype=np.float32)
        else:
            raise ValueError(f'Not support calib:{normalize_calib}')
        
        if lidar2camera:
            lidar_extrinsic = lidar_extrinsic @ trans_mat
        else:
            lidar_extrinsic = np.linalg.inv(lidar_extrinsic) @ trans_mat
    return lidar_extrinsic

def extend_matrix(mat):
    mat = np.hstack((mat, np.zeros((3, 1))))
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat

def read_calib_fromfile_a02(calib_path, ceph_client, cam_K=False):
    intrinsic, extrinsic = [], []
    cam_list = ['center_camera_fov120', 'center_camera_fov30']#, 'left_front_camera', 'right_front_camera', 'left_rear_camera', 'right_rear_camera', 'rear_camera']
    for camera_name in cam_list:
        intrinsic_file = f'{camera_name}-intrinsic.json'
        extrinsic_file = f'{camera_name}-to-car_center-extrinsic.json'
        # extrinsic_file = f'{camera_name}-to-car-center.json'
        camera_intrinsics_path = os.path.join(calib_path, camera_name, intrinsic_file)
        camera_extrinsic_path = os.path.join(calib_path, camera_name, extrinsic_file)
        
        extrinsic_cam = load_extrinsic_calib(camera_extrinsic_path, ceph_client=ceph_client)
        intrinsic_cam = extend_matrix(load_intrinsics_calib(camera_intrinsics_path, ceph_client=ceph_client, cam_K=cam_K))
        intrinsic.append(np.expand_dims(intrinsic_cam, axis=0))
        extrinsic.append(np.expand_dims(extrinsic_cam, axis=0))
    extrinsic = np.concatenate(extrinsic)
    intrinsic = np.concatenate(intrinsic)
    return extrinsic, intrinsic

def read_calib_fromfile_a02_t(calib_path, ceph_client, cam_K=False):
    intrinsic, extrinsic = [], []
    cam_list = ['center_camera_fov120', 'center_camera_fov30', 'left_front_camera', 'right_front_camera', 'left_rear_camera', 'right_rear_camera', 'rear_camera']
    for camera_name in cam_list:
        intrinsic_file = f'{camera_name}-intrinsic.json'
        # extrinsic_file = f'{camera_name}-to-car_center-extrinsic.json'
        extrinsic_file = f'{camera_name}-to-car-center.json'
        camera_intrinsics_path = os.path.join(calib_path, camera_name, intrinsic_file)
        camera_extrinsic_path = os.path.join(calib_path, camera_name, extrinsic_file)
        
        extrinsic_cam = load_extrinsic_calib(camera_extrinsic_path, ceph_client=ceph_client)
        intrinsic_cam = extend_matrix(load_intrinsics_calib(camera_intrinsics_path, ceph_client=ceph_client, cam_K=cam_K))
        intrinsic.append(np.expand_dims(intrinsic_cam, axis=0))
        extrinsic.append(np.expand_dims(extrinsic_cam, axis=0))
    extrinsic = np.concatenate(extrinsic)
    intrinsic = np.concatenate(intrinsic)
    return extrinsic, intrinsic

def load_extrinsic_lidar(calib_json_path, ceph_client=None):
    calib_json_path = mapping_path_to_ceph(PATH_MAPPING, calib_json_path)
    extrinsic_calib = ceph_utils.ceph_read(calib_json_path, None, True, ceph_client)
    key = [ele for ele in extrinsic_calib.keys()][0]
    lidar_extrinsic = np.array(
        extrinsic_calib[key]['param']['sensor_calib']['data'], dtype=np.float32)
    return lidar_extrinsic


def read_lidar2carcenter(calib_path, ceph_client):
    extrinsic_file = 'top_center_lidar-to-car_center-extrinsic.json'
    camera_extrinsic_path = os.path.join(calib_path, 'top_center_lidar', extrinsic_file)
    extrinsic_lidar = load_extrinsic_lidar(camera_extrinsic_path, ceph_client=ceph_client)
    return extrinsic_lidar

def robust_ground_height(points, lower_percentile=0.5, upper_percentile=2):
    z_vals = points[:, 2]
    lower = np.percentile(z_vals, lower_percentile)
    upper = np.percentile(z_vals, upper_percentile)
    ground_candidates = z_vals[(z_vals >= lower) & (z_vals <= upper)]
    return np.mean(ground_candidates)

# CLS_MAPPING = {
#     'STONE POLE': 'pole',
#     'POLE': 'pole',
#     'CONSTRUCTION_SIGN': 'construction_sign',
#     'CONE': 'cone',
#     'TRAFFIC LIGHT': 'traffic_light'
# }

# CLS_MAPPING = {
#     'STONE POLE': 'pole',#
#     'POLE': 'pole',#
#     'CONSTRUCTION_SIGN': 'construction_sign',#
#     'CONE': 'cone',#
#     'TRAFFIC LIGHT': 'traffic_light',
#     'BARRIER': 'barrier',#
#     'PERMANENT BARRICADE': 'permanent_barricade',
#     # 'CEMENT PIER': 'barrier',#
#     'BARRIER_GATE': 'Gate_rod',#
#     'ISOLATION_BARRER': 'isolation_barrel',#
#     'OBSTACLES': 'obstacles',
#     'TEMPORARY BARRICADE': 'temporary_barricade',
#     'RETRACTABLE DOOR': 'retractable_door',
#     'SPEED BUMP': 'speed_bump',
#     'VEHICLE_CAR': 'Car',
#     'PEDESTRIAN_NORMAL': 'Pedestrian',
#     'CYCLIST_MOTOR': 'Truck',
#     'VEHICLE_BUS': 'Truck',
#     'CYCLIST_MOTOR': 'Cyclist'
# }
CLS_MAPPING = {
    'STONE POLE': 'pole',
    'POLE': 'pole',
    'CONSTRUCTION_SIGN': 'construction_sign',
    'CONE': 'cone',
    'TRAFFIC LIGHT': 'traffic_light',
    'BARRIER': 'barrier',
    'PERMANENT BARRICADE': 'permanent_barricade',
    'CEMENT PIER': 'cement_pier',
    'BARRIER_GATE': 'gate_rod',
    'ISOLATION_BARRER': 'isolation_barrel',
    'OBSTACLES': 'obstacles',
    'TEMPORARY BARRICADE': 'temporary_barricade',
    'RETRACTABLE DOOR': 'retractable_door',
    'SPEED BUMP': 'speed_bump',
    "COLUMN": 'pole',
    "Car": "Car", 
    "Pedestrian": "Pedestrian", 
    "Cyclist": "Cyclist", 
    "Truck": "Truck",
    'VEHICLE_CAR': "Car",
    'VEHICLE_TRAILER': "Truck", 
    'VEHICLE_SUV': "Car", 
    'VEHICLE_TRIKE': "Cyclist", 
    'VEHICLE_PICKUP': "Car", 
    'VEHICLE_SPECIAL': "Truck", 
    'CYCLIST_BICYCLE': "Cyclist", 
    'VEHICLE_TRUCK_SMALL': "Truck",
    'CYCLIST_MOTOR': 'Cyclist',
    'VEHICLE_BUS': 'Truck', 
    'PEDESTRIAN_NORMAL': 'Pedestrian', 
    'CYCLIST_OTHERS': 'Cyclist',
}
if __name__ == '__main__':
    func = 'vis_pkl'
    # debugpy.listen(("10.5.36.226", 38758))
    # print('Waitting for debuger attach')
    # # 等待debug工具连接
    # debugpy.wait_for_client()
    if func == 'stat':
        # 统计脚本
        pkl_path = '/mnt/lustre/datatag/songxiao/GAC/3M1-sample/GAC_case/SelfB3/pkls/pvb.pkl'
        info = pkl.load(open(pkl_path, 'rb'))
        cls_map = dict()
        frame_num_map ={'cone': 0, 'obstacles': 0, 'pole': 0, 'barrier': 0, 'temporary_barricade': 0, 'permanent_barricade': 0, 'isolation_barrel': 0, 
'gate_rod': 0, 'construction_sign': 0, 'triangle_warning': 0, 'animal': 0}
        cnt = 0
        for seq, val in info.items():
            for i, frame in enumerate(val):
                cnt += 1
                annos = frame['annos']
                gt_names = annos['name']
                for key in frame_num_map.keys():
                    if key in gt_names:
                        frame_num_map[key] += 1
                for cur_name in gt_names:
                    if cur_name not in cls_map:
                        cls_map[cur_name] = 1
                    else:
                        cls_map[cur_name] += 1
                cam_list = ['center_camera_fov120', 'center_camera_fov30']
                map_ = frame['img_path']
                rect = np.eye(4)
                for k, cam in enumerate(cam_list):
                    img_path = map_[cam]
                    
                    img = cv2.imread(img_path)
                    print(cam, img.shape)
        with open("gt_names_lidargop_map_v0.json", "w") as file:
            json.dump(cls_map, file)
        print(cls_map)
        print(cnt)
        print(frame_num_map)
    
    if func == 'stat_pvb':
        # 统计脚本
        
        cls_map = dict()
        frame_num_map ={'Car': 0, 'Truck': 0, 'Pedestrian': 0, 'Cyclist': 0}
        cnt = 0
        pkl_str = """
/mnt/lustre/share/songxiao/3ddet/pkl_20w_backbone_internal/hard_bg/internal_18W-PK1-2_maxgt-class-assemble-6W-hardBG-2W_around.pkl;/mnt/lustre/share/songxiao/3ddet/internal_21-0410-HQ_around.pkl;/mnt/lustre/share/songxiao/data/hesai_test_1101/hesai_1101_train_val_unified.pkl;/mnt/lustre/share/songxiao/data/hesai_test_1224/hesai_1224_around.pkl;/mnt/lustre/share/songxiao/3ddet/internal_1219_around.pkl;/mnt/lustre/share/songxiao/3ddet/internal_lg_issue_20-0608_around.pkl;/mnt/lustre/share/songxiao/3ddet/internal_lg_issue_20-0701_around.pkl;/mnt/lustre/share/songxiao/3ddet/internal_lg_issue_20-1208_around.pkl;/mnt/lustre/share/songxiao/3ddet/internal_lg_issue_21-0225_around.pkl;/mnt/lustre/share/songxiao/3ddet/internal_lg_issue_21-0409_around.pkl;/mnt/lustre/share/songxiao/3ddet/internal_1212_zhouhui_fusion_around.pkl
"""
        pkl_paths = pkl_str.strip().split(';')
        for pkl_path in pkl_paths:
            info = pkl.load(open(pkl_path, 'rb'))
            for frame in info:
                cnt += 1
                annos = frame['annos']
                gt_names = annos['name']
                for key in frame_num_map.keys():
                    if key in gt_names:
                        frame_num_map[key] += 1
                for cur_name in gt_names:
                    if cur_name not in cls_map:
                        cls_map[cur_name] = 1
                    else:
                        cls_map[cur_name] += 1
            
        with open("gt_names_lidarpvb_map_v0.json", "w") as file:
            json.dump(cls_map, file)
        print(cls_map)
        print(cnt)
        print(frame_num_map)
#     {'Truck': 192387, 'Car': 848794, 'Pedestrian': 289654, 'Cyclist': 185093}
# 97260
# {'Car': 92409, 'Truck': 65376, 'Pedestrian': 44478, 'Cyclist': 45599}
    if func == 'plot':
        import matplotlib.pyplot as plt
        # {'cone': 13690164, 'obstacles': 2563100, 'pole': 23904397, 'barrier': 6182331, 'temporary_barricade': 5115153, 'permanent_barricade': 34396840, 'isolation_barrel': 1193635, 
        # 'gate_rod': 719411, 'construction_sign': 487745, 'triangle_warning': 5123, 'animal': 12861}
        map_ ={'cone': 13690164, 'pole': 23904397, 'barrier': 6182331, 'isolation_barrel': 1193635, 'gate_rod': 719411, 'construction_sign': 487745, 'triangle_warning': 5123, 'animal': 12861}
        labels = list(map_.keys())
        
        sizes = list(map_.values())
        sorted_indices = sorted(range(len(sizes)), key=lambda i: -sizes[i])
        labels_sorted = [labels[i] for i in sorted_indices]
        sizes_sorted = [sizes[i] for i in sorted_indices]

        fig, ax = plt.subplots(figsize=(8, 8))  # 设置画布大小

        # 使用 autopct 显示百分比, 通过 labeldistance=None 避免标签重叠
        wedges, texts, autotexts = ax.pie(
            sizes_sorted, labels=labels_sorted, autopct='%1.1f%%',
            pctdistance=1.0,  # 控制百分比文本的距离
            labeldistance=None,  # 避免标签重叠
            startangle=140  # 旋转角度
        )

        # 使用 plt.legend() 代替标签，避免重叠
        new_labels = [labels_sorted[i]+' '+autotexts[i]._text for i in range(len(labels))]
        ax.legend(wedges, new_labels, title="Classes", loc="center left", bbox_to_anchor=(1, 0.5))

        # 设置百分比字体大小
        for autotext in autotexts:
            autotext.set_fontsize(1)

        # 自动调整布局
        plt.tight_layout()

        # 保存图片
        plt.savefig('lidargop_cls_1.png', dpi=1000, bbox_inches='tight')

    if func == 'gen_demo_dataset':
        lidar_path = '/iag_ad_01/ad/zhanghongcheng/code/multimodal-3dgop/data/gop_data_0717/lidar/stitching_lidar_bin'
        cam_names = ['center_camera_fov120', 'center_camera_fov30']
        camera_path = '/iag_ad_01/ad/zhanghongcheng/code/multimodal-3dgop/data/gop_data_0717/object'
        lidar_txt = '/iag_ad_01/ad/zhanghongcheng/code/multimodal-3dgop/data/gop_data_0717/lidar/stitching_lidar/lidar.txt'
        
        calib_path = camera_pose_path = '/iag_ad_01/ad/zhanghongcheng/code/multimodal-3dgop/data/gop_data_0717/calib'
        with open(lidar_txt, 'r') as f:
            lidarlist = f.readlines()
        map_ = {}
        map_['lidar'] = lidarlist

        # pkl_path = '/mnt/lustrenew/share_data/xuzhiyong/3DGOP_THOR/batch_test/pkls/3DGOP_batch_test_normalized.pkl'
        # info_ = pkl.load(open(pkl_path, 'rb'))
        for cam in cam_names:
            img_txt = f'{camera_path}/{cam}_0/img.txt'
            with open(img_txt, 'r') as f:
                imglist = f.readlines()
            map_[cam] = imglist
        infos = dict()
        frame_list = []
        for i in range(len(lidarlist)):
            frame_id = i
            lidar_name = lidarlist[i].split(' ')[1].strip()
            # import ipdb; ipdb.set_trace()
            frame_name = lidar_name.split('.')[0]
            annos = {'name': [], 
                     'location': np.array([], dtype=np.float32), 
                     'dimensions': np.array([], dtype=np.float32), 
                     'rotation_y': np.array([], dtype=np.float32), 
                     'num_points_in_box': None, 
                     'trackID': np.array([], dtype=np.int32), 
                     'valid': np.array([], dtype=bool)}
            velodyne_path = os.path.join(lidar_path, lidar_name.replace('pcd', 'bin'))
            img_path_map = dict()
            for cam in cam_names:
                camlist = map_[cam]
                cam_name = camlist[i].split(' ')[1].strip()
                img_path = os.path.join(camera_path, cam+'_0', cam_name)
                img_path_map[cam] = img_path
            
            cur_frame_dict = dict(
                frame_id=frame_id,
                frame_name=frame_name,
                annos=annos,
                velodyne_path=velodyne_path,
                label_path='',
                img_path=img_path_map,
                lidar_pose_path='',
                camera_pose_path=camera_pose_path,
                calib_path=calib_path,
                drivable=[], 
                non_drivable=[],
                blocked_radio={},
            )
            frame_list.append(cur_frame_dict)
        infos['0717_demo'] = frame_list
        with open('data/demo_pkl/demo_data_0717_v1.pkl', 'wb') as f:
            pkl.dump(infos, f)
        print('finish')
    
    if func == 'convert_pvb':
        # convert_pvb_dataset
        pkl_dir = 'data/LidarPVB_pkls_3M1_eval'
        pkl_paths = [os.path.join(pkl_dir, f) for f in os.listdir(pkl_dir)]
        print(pkl_paths)
        new_pkls = []
        
        for pkl_path in pkl_paths:
            infos = dict()
            print('cur_pkl: ', pkl_path)
            use_ceph = 's3' in pkl_path
            res = pkl.load(open(pkl_path, 'rb'))
            pkl_basename = os.path.basename(pkl_path).split('.')[0]
            print(pkl_basename)
            if type(res) == dict:
                print("skip db infos ", pkl_path)
                continue
            frame_list = []
            for i, frame in enumerate(res):
                frame_id = frame['image_idx']
                lidar_name = frame['velodyne_path']
                frame_name = os.path.basename(lidar_name).split('.')[0]
                annos_ = frame['annos']
                N = len(annos_['name'])
                # print(N)
                try:
                    annos = {'name': annos_['name'],
                            'location': annos_['location'], 
                            'dimensions': annos_['dimensions'], 
                            'rotation_y': annos_['rotation_y'], 
                            'num_points_in_box': annos_['num_points_in_gt'] if 'num_points_in_gt' in annos_ else np.array([10 for _ in range(N)]), 
                            'trackID': annos_['index'], 
                            'valid': np.ones((N, ), dtype=bool)}
                except:
                    import ipdb; ipdb.set_trace()
                velodyne_path = frame['velodyne_path']
                img_path_map = dict()
                camera_pose_path = None
                calib_path = None
                cur_frame_dict = dict(
                    frame_id=frame_id,
                    frame_name=frame_name,
                    annos=annos,
                    velodyne_path=velodyne_path,
                    label_path='',
                    img_path=img_path_map,
                    lidar_pose_path='',
                    camera_pose_path=camera_pose_path,
                    calib_path=calib_path,
                    drivable=[], 
                    non_drivable=[],
                    blocked_radio={},
                    pointcloud_num_features=frame['pointcloud_num_features'],
                )
                frame_list.append(cur_frame_dict)
            infos[pkl_basename] = frame_list
            with open(f'data/LidarPVB_convert_3M1_eval/{pkl_basename}.pkl', 'wb') as f:
                pkl.dump(infos, f)
        print('finish')
    if func == 'cat_video':
        import glob
        basepath = 'output/A02_ceph/lidar_A02_3dgop_alldet_2V_V1.9.1_No1_0711/0_2025_06_08_22_22_47_pilotGtParser/vis1/det_vis/lidar/0_2025_06_08_22_22_47_pilotGtParser'
        with_postfix = True
        if with_postfix:
            path_A = f'{basepath}/center_camera_fov120_cam'
            path_B = f'{basepath}/center_camera_fov30_cam'
        else:
            path_A = f'{basepath}/center_camera_fov120'
            path_B = f'{basepath}/center_camera_fov30'
        path_C = f'{basepath}/bev'
        output_video = f'{basepath}/lidar.mp4'
        
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
    
    if func == 'vis_bev':
        M1_bin_path = "/iag_ad_01/ad/zhanghongcheng/code/multimodal-3dgop/output/A02_ceph/MM3DGOP-A02-V1.9-3heads_A02_3dgop_alldet_2V_V1.9_No1_1024x576/dump_data_0624/lidar-points"
        M1_list = os.listdir(M1_bin_path)
        M1_list.sort()
        for i, filename in enumerate(M1_list):
            cur_m1_bin_path = os.path.join(M1_bin_path, M1_list[i])
            basename = os.path.basename(cur_m1_bin_path)
            print(basename)
            points_m1_0 = np.fromfile(cur_m1_bin_path, dtype=np.float32).reshape(-1,4)
            # 读取 pcd 文件
            # pcd = o3d.io.read_point_cloud(cur_m1_bin_path)
            # points_m1_0 = np.asarray(pcd.points)
            # print(points_m1_0.shape)
            # P128 points transform
            
            bev_save_dir = os.path.join('./data/dump_data_0624', 'bev')
            bev_height, bev_width = 512, 1280
            pc_range = [0, -20.48, -4, 102.4, 20.48, 4]
            voxel_size = [0.16, 0.16, 8]
            center_w, center_h = int(-pc_range[0] / voxel_size[0]), int(-pc_range[1] / voxel_size[1])
            draw_bev(points_m1_0, [], bev_height, bev_width, bev_resolution=0.08, center_coords=[center_w*2, center_h*2], radius_list=[0.1, 10, 25, 50, 75], 
                    save_dir=bev_save_dir, filename=basename.replace('npy', 'png'))
                
    if func == 'vis_pkl':
        # LidarGOP train data
        ceph_client = ceph_utils.ceph_init('/iag_ad_01/ad/zhanghongcheng/code/multimodal-3dgop/petreloss.conf')
        # pkl_path = '/iag_ad_01/ad/zhanghongcheng/code/multimodal-3dgop/data/A02_data/train_infos_unknown_3DGOP_A02_v1.9.1_blocked.pkl'
        pkl_path = 'data/train_ATX_0729/pvb_0726_lidar_pvb_1.pkl'
        info = pkl.load(open(pkl_path, 'rb'))
        tar_list = ['cone', 'pole', 'isolation_barrel', 'triangle_warning', 'animal', 'gate_rod', 'barrier', 'construction_sign',
                    'VEHICLE_TRAILER', 
                    'VEHICLE_SUV', 
                    'VEHICLE_TRIKE', 
                    'VEHICLE_PICKUP', 
                    'VEHICLE_SPECIAL', 
                    'CYCLIST_BICYCLE', 
                    'VEHICLE_TRUCK_SMALL',
                    'CYCLIST_MOTOR',
                    'VEHICLE_BUS', 
                    'PEDESTRIAN_NORMAL', 
                    "Car", 
                    "Pedestrian", 
                    "Cyclist", 
                    "Truck",]
        for seq, val in info.items():
            # import ipdb; ipdb.set_trace()
            if 'calib_path' in val[0]:
                calib_path = val[0]['calib_path']
            else:
                calib_path = calib_path = '/'.join(val[0]['velodyne_path'].split('/')[:-2]) + '/calib'
            # calib_path = mapping_path_to_ceph(PATH_MAPPING, calib_path)
            # cam_K = False
            cam_K = 'cam_K_new'
            # calib_path = 'data/epai7-957_origin'
            extrinsic, intrinsic = read_calib_fromfile_a02(calib_path, ceph_client, cam_K=cam_K)
            # case_name = 'LidarGOP_standard_data_v1'
            case_name = 'pvb_0726_lidar_pvb_1_v1'
            bev_height, bev_width = 512, 1280
            for i, frame in enumerate(val):
                annos = frame['annos']
                # import ipdb; ipdb.set_trace()
                gt_names = annos['name']
                l = []
                for cur_cls in gt_names:
                    if cur_cls in CLS_MAPPING:
                        l.append(CLS_MAPPING[cur_cls])
                    else:
                        l.append(cur_cls)
                        print(cur_cls)
                gt_names =np.array(l)
                N = annos['location'].shape[0]
                gt_boxes = np.zeros((N, 7), dtype=float)

                if N > 0:
                    x, y, z = annos['location'][:, 0:1], annos['location'][:, 1:2], annos['location'][:, 2:3]
                    w, h, l = annos['dimensions'][:, 0:1], annos['dimensions'][:, 1:2], annos['dimensions'][:, 2:3]
                    yaw = np.expand_dims(annos['rotation_y'], axis=-1) # atx -; atxpvb +
                    gt_boxes = np.concatenate((x,y,z,l,w,h,yaw), axis=-1)
                    # gt_boxes[:, 2] -= 1.6
                    valid_cls_mask = np.isin(gt_names, tar_list)
                    gt_boxes = gt_boxes[valid_cls_mask]
                    num_pts_in_box = annos['num_points_in_box']
                    num_pts_in_box = num_pts_in_box[valid_cls_mask]
                    point_mask = num_pts_in_box > 0
                    gt_boxes = gt_boxes[point_mask]
                pcd_path = frame['velodyne_path']
                # import ipdb; ipdb.set_trace()
                basename = pcd_path.split('/')[-1]
                pcd_path = mapping_path_to_ceph(PATH_MAPPING, pcd_path)
                flag_ = 's3' in pcd_path
                try:
                    points_m1_0 = ceph_utils.ceph_read(pcd_path, np.float32, flag_, client=ceph_client).reshape(-1,4)
                    draw_bev_flag = True
                except:
                    print('loading points failed')
                    points_m1_0 = np.ones((10, 4))
                    draw_bev_flag = False
                # # points_m1_0[:, 2] += 1.6
                # N = points_m1_0.shape[0]
                # pad = np.ones((N, 1))
                # points_new = np.concatenate((points_m1_0[:, :3], pad), axis=1)
                # points_ = (lidar2carcenter @ points_new.T).T
                points_m1_0[:, 2] -= 1.6
                points_ = points_m1_0.copy()
                zm = robust_ground_height(points_m1_0)
                # print(zm)
                bev_save_dir = os.path.join(f'./data/{case_name}', 'bev')
                pc_range = [0, -20.48, -4, 102.4, 20.48, 4]
                voxel_size = [0.16, 0.16, 8]
                center_w, center_h = int(-pc_range[0] / voxel_size[0]), int(-pc_range[1] / voxel_size[1])
                # gt_boxes = np.array([[30, 3.0, -0.1, 4.08, 2.5, 3.1, 0.1]])
                if draw_bev_flag:
                    draw_bev(points_m1_0, gt_boxes.copy(), bev_height, bev_width, bev_resolution=0.08, center_coords=[center_w*2, center_h*2], radius_list=[25, 50, 75], 
                        save_dir=bev_save_dir, filename=basename.replace('bin', 'png'))
                
                cam_list = ['center_camera_fov120', 'center_camera_fov30']
                map_ = frame['img_path']
                rect = np.eye(4)
                flag_ = False
                for k, cam in enumerate(cam_list):
                    img_name = map_[cam]
                    img_path = mapping_path_to_ceph(PATH_MAPPING, map_[cam])
                    if '1750932281194171942' in img_path:
                        flag_ = True
                    if flag_:
                        print(map_)
                        print(pcd_path)
                    img = ceph_utils.ceph_read(img_path, np.int8, True, client=ceph_client)
                    if i % 100 == 0:
                        print(cam, img.shape)
                    P0_image = draw_projected_box(gt_boxes.copy(), img.copy(), intrinsic[k], extrinsic[k], rect, flag=False, window_name=img_name)
                    P0_image_pc = draw_projected_points(points_m1_0.copy(), P0_image.copy(), intrinsic[k], extrinsic[k], rect, flag=True, window_name=img_name)
                    # P0_image = cv2.resize(P0_image, (P0_image.shape[1]//2, P0_image.shape[0]//2))
                    # P0_image_pc = cv2.resize(P0_image_pc, (P0_image_pc.shape[1]//2, P0_image_pc.shape[0]//2))
                    
                    cam_dir = os.path.join(f'./data/{case_name}', cam)
                    if not os.path.exists(cam_dir):
                        os.makedirs(cam_dir)
                    cam_dir_pc = os.path.join(f'./data/{case_name}', cam+'_pc')
                    if not os.path.exists(cam_dir_pc):
                        os.makedirs(cam_dir_pc)
                    basename = img_name.split('/')[-1]
                    img_save_name = os.path.join(cam_dir, basename)
                    cv2.imwrite(img_save_name, P0_image)
                    img_save_name = os.path.join(cam_dir_pc, basename)
                    cv2.imwrite(img_save_name, P0_image_pc)

    if func == 'vis_pvbgt':
        ceph_client = ceph_utils.ceph_init('petreloss.conf')
        pkl_path = 'data/demo_pkl/2025_06_10_10_59_04_L2.pkl'
        # pkl_path = '/mnt/lustrenew/share_data/xuzhiyong/3DGOP_A02/pkls/V1.9.1/train_infos_unknown_3DGOP_A02_v1.9.1_blocked.pkl'
        info = pkl.load(open(pkl_path, 'rb'))
        if type(info) == dict:
            info = info[list(info.keys())[0]]
        for frame in info:
        
            case_name = '2025_06_10_10_59_04_L2_v1'
            bev_height, bev_width = 512, 1280
            
            annos = frame['annos']
            gt_names = annos['name']
            l = []
            for cur_cls in gt_names:
                if cur_cls in CLS_MAPPING:
                    l.append(CLS_MAPPING[cur_cls])
                else:
                    l.append(cur_cls)
                    # print(cur_cls)
            gt_names =np.array(l)
            N = annos['location'].shape[0]
            gt_boxes = np.zeros((N, 7), dtype=float)

            if N > 0:
                x, y, z = annos['location'][:, 0:1], annos['location'][:, 1:2], annos['location'][:, 2:3]
                l, h, w = annos['dimensions'][:, 0:1], annos['dimensions'][:, 1:2], annos['dimensions'][:, 2:3] # songxiao lidar
                yaw = - np.pi/2 - np.expand_dims(annos['rotation_y'], axis=-1) # songxiao lidar
                # w, h, l = annos['dimensions'][:, 0:1], annos['dimensions'][:, 1:2], annos['dimensions'][:, 2:3]
                # yaw = - np.expand_dims(annos['rotation_y'], axis=-1)
                gt_boxes = np.concatenate((x,y,z,l,w,h,yaw), axis=-1)
            gt_boxes[:, 6] = limit_period(gt_boxes[:, 6], offset=0.5, period=2 * np.pi)
            pcd_path = frame['velodyne_path']
            basename = pcd_path.split('/')[-1]
            pcd_path = mapping_path_to_ceph(PATH_MAPPING, pcd_path)
            if 'songxiao' in pcd_path:
                pcd_path = pcd_path.replace('dcp36_lustre_aoss_v2_xuzhiyong', 'dcp36_lustre_aoss_v2')
            points_m1_0 = ceph_utils.ceph_read(pcd_path, np.float32, True, client=ceph_client).reshape(-1,4)
            # # points_m1_0[:, 2] += 1.6
            # N = points_m1_0.shape[0]
            # pad = np.ones((N, 1))
            # points_new = np.concatenate((points_m1_0[:, :3], pad), axis=1)
            # points_ = (lidar2carcenter @ points_new.T).T
            # points_[:, 2] -= 1.6
            # points_m1_0[:, 2] -= 1.6
            points_ = points_m1_0.copy()
            zm = robust_ground_height(points_m1_0)
            print(zm)
            bev_save_dir = os.path.join(f'./data/{case_name}', 'bev')
            pc_range = [0, -20.48, -4, 102.4, 20.48, 4]
            voxel_size = [0.16, 0.16, 8]
            center_w, center_h = int(-pc_range[0] / voxel_size[0]), int(-pc_range[1] / voxel_size[1])
            draw_bev(points_m1_0, gt_boxes.copy(), bev_height, bev_width, bev_resolution=0.08, center_coords=[center_w*2, center_h*2], radius_list=[25, 50, 75], 
                save_dir=bev_save_dir, filename=basename.replace('bin', 'png'))
        
            
    if func == 'vis_pkl_newdata':
        # new LidarGOP train data
        # pcd -1.6
        ceph_client = ceph_utils.ceph_init('/iag_ad_01/ad/zhanghongcheng/code/multimodal-3dgop/petreloss.conf')
        pkl_path = 'data/atx_traindata/all_delivery_ATX_lidar_0623+0624_cleaned_data.pkl'
        info = pkl.load(open(pkl_path, 'rb'))
        tar_list = ['cone', 'pole', 'isolation_barrel', 'triangle_warning', 'animal', 'gate_rod', 'barrier', 'construction_sign']
        for seq, val in info.items():
            if 'calib_path' in val[0]:
                calib_path = val[0]['calib_path']
            else:
                calib_path = calib_path = '/'.join(val[0]['velodyne_path'].split('/')[:-2]) + '/calib'
            
            extrinsic, intrinsic = read_calib_fromfile_a02(calib_path, ceph_client, cam_K=False)
            # lidar2carcenter = read_lidar2carcenter(calib_path, ceph_client)
            case_name = 'all_delivery_ATX_lidar_0623+0624_cleaned_data_v7'
            bev_height, bev_width = 1024, 2560
            for i, frame in enumerate(val):
                annos = frame['annos']
                gt_names = annos['name']
                l = []
                for cur_cls in gt_names:
                    if cur_cls in CLS_MAPPING:
                        l.append(CLS_MAPPING[cur_cls])
                    else:
                        l.append(cur_cls)
                        print(cur_cls)
                gt_names =np.array(l)
                N = annos['location'].shape[0]
                gt_boxes = np.zeros((N, 7), dtype=float)

                if N > 0:
                    x, y, z = annos['location'][:, 0:1], annos['location'][:, 1:2], annos['location'][:, 2:3]
                    w, h, l = annos['dimensions'][:, 0:1], annos['dimensions'][:, 1:2], annos['dimensions'][:, 2:3]
                    yaw = np.expand_dims(annos['rotation_y'], axis=-1)
                    gt_boxes = np.concatenate((x,y,z,l,w,h,yaw), axis=-1)
                    # gt_boxes[:, 2] -= 1.6
                    valid_cls_mask = np.isin(gt_names, tar_list)
                    gt_boxes = gt_boxes[valid_cls_mask]
                    gt_names = gt_names[valid_cls_mask]
                    num_pts_in_box = annos['num_points_in_box']
                    num_pts_in_box = num_pts_in_box[valid_cls_mask]
                    point_mask = num_pts_in_box > 0
                    gt_boxes = gt_boxes[point_mask]
                    gt_names = gt_names[point_mask]
                pcd_path = frame['velodyne_path']
                basename = pcd_path.split('/')[-1]
                pcd_path = mapping_path_to_ceph(PATH_MAPPING, pcd_path)
                points_m1_0 = ceph_utils.ceph_read(pcd_path, np.float32, True, client=ceph_client).reshape(-1,4)
                print(pcd_path)
                points_ = points_m1_0.copy()
                points_[:, 2] -= 1.6
                zm = robust_ground_height(points_m1_0)
                bev_save_dir = os.path.join(f'./data/{case_name}', 'bev')
                pc_range = [0, -20.48, -4, 102.4, 20.48, 4]
                voxel_size = [0.08, 0.08, 8]
                center_w, center_h = int(-pc_range[0] / voxel_size[0]), int(-pc_range[1] / voxel_size[1])
                draw_bev(points_m1_0, gt_boxes.copy(), bev_height, bev_width, bev_resolution=0.04, center_coords=[center_w*2, center_h*2], radius_list=[25, 50, 75], 
                    save_dir=bev_save_dir, filename=basename.replace('bin', 'png'), gt_names=gt_names)
                
                cam_list = ['center_camera_fov120', 'center_camera_fov30']
                map_ = frame['img_path']
                rect = np.eye(4)
                for k, cam in enumerate(cam_list):
                    img_name = map_[cam]
                    img_path = mapping_path_to_ceph(PATH_MAPPING, map_[cam])
                
                    img = ceph_utils.ceph_read(img_path, np.int8, True, client=ceph_client)
                    
                    P0_image = draw_projected_box(gt_boxes.copy(), img.copy(), intrinsic[k], extrinsic[k], rect, flag=False, window_name=img_name)
                    P0_image_pc = draw_projected_points(points_.copy(), P0_image.copy(), intrinsic[k], extrinsic[k], rect, flag=True, window_name=img_name)
                    P0_image = cv2.resize(P0_image, (P0_image.shape[1]//2, P0_image.shape[0]//2))
                    P0_image_pc = cv2.resize(P0_image_pc, (P0_image_pc.shape[1]//2, P0_image_pc.shape[0]//2))
                    
                    cam_dir = os.path.join(f'./data/{case_name}', cam)
                    if not os.path.exists(cam_dir):
                        os.makedirs(cam_dir)
                    cam_dir_pc = os.path.join(f'./data/{case_name}', cam+'_pc')
                    if not os.path.exists(cam_dir_pc):
                        os.makedirs(cam_dir_pc)
                    basename = img_name.split('/')[-1]
                    img_save_name = os.path.join(cam_dir, basename)
                    cv2.imwrite(img_save_name, P0_image)
                    img_save_name = os.path.join(cam_dir_pc, basename)
                    cv2.imwrite(img_save_name, P0_image_pc)


    if func == 'vis_pkl_t68':
        # T68数据GT,PCD
        ceph_client = ceph_utils.ceph_init('/mnt/lustrenew/zhanghongcheng/zhc/multimodal-GOP/petreloss.conf')
        pkl_path = 'data/all_delivery_0521_caiceyiti_T68-PV159_v2pkl/0_dataset_infos.pkl'
        info = pkl.load(open(pkl_path, 'rb'))
        for seq, val in info.items():
            # val = val['0_2025_06_01_13_26_35_pilotGtParser']
            calib_path = val[0]['calib_path']
            
            extrinsic, intrinsic = read_calib_fromfile_a02(calib_path, ceph_client, cam_K=False)
            lidar2carcenter = read_lidar2carcenter(calib_path, ceph_client)
            case_name = 'all_delivery_0521_caiceyiti_T68-PV159_v13'
            for i, frame in enumerate(val):
                annos = frame['annos']
                gt_names = annos['name']
                pcd_path = frame['velodyne_path']
                basename = pcd_path.split('/')[-1]
                pcd_path = mapping_path_to_ceph(PATH_MAPPING, pcd_path)
                points_m1_0 = ceph_utils.ceph_read(pcd_path, np.float32, True, client=ceph_client).reshape(-1,4)
                N = annos['location'].shape[0]
                gt_boxes = np.zeros((N, 7), dtype=float)

                if N > 0:
                    x, y, z = annos['location'][:, 0:1], annos['location'][:, 1:2], annos['location'][:, 2:3]
                    w, h, l = annos['dimensions'][:, 0:1], annos['dimensions'][:, 1:2], annos['dimensions'][:, 2:3]
                    yaw = np.expand_dims(annos['rotation_y'], axis=-1)
                    gt_boxes = np.concatenate((x,y,z,l,w,h,yaw), axis=-1)
                    # z - 1.6 - h/2
                N = points_m1_0.shape[0]
                # points_m1_0[:, 2] -= 3.2
                # pad = np.ones((N, 1))
                # points_new = np.concatenate((points_m1_0[:, :3], pad), axis=1)
                # points_ = (np.linalg.inv(lidar2carcenter) @ points_new.T).T
                # points_[:, 2] -= 1.6
                points_m1_0[:, 2] -= 1.6
                points_ = points_m1_0.copy()
                zm = robust_ground_height(points_m1_0)
                bev_save_dir = os.path.join(f'./data/{case_name}', 'bev')
                pc_range = [0, -20.48, -4, 102.4, 20.48, 4]
                voxel_size = [0.16, 0.16, 8]
                center_w, center_h = int(-pc_range[0] / voxel_size[0]), int(-pc_range[1] / voxel_size[1])
                bev_height, bev_width = 512, 1280
                draw_bev(points_m1_0, gt_boxes.copy(), bev_height, bev_width, bev_resolution=0.08, center_coords=[center_w*2, center_h*2], radius_list=[25, 50, 75], 
                    save_dir=bev_save_dir, filename=basename.replace('bin', 'png'))
                
                cam_list = ['center_camera_fov120', 'center_camera_fov30']
                map_ = frame['img_path']
                rect = np.eye(4)
                for k, cam in enumerate(cam_list):
                    img_name = map_[cam]
                    img_path = mapping_path_to_ceph(PATH_MAPPING, map_[cam])
        
                    img = ceph_utils.ceph_read(img_path, np.int8, True, client=ceph_client)
                    # pred_boxes = np.array([[30, 3.0, -0.1, 4.08, 2.5, 3.1, 0.1]])
                    P0_image = draw_projected_box(gt_boxes.copy(), img.copy(), intrinsic[k], extrinsic[k], rect, flag=False, window_name=img_name)
                    P0_image_pc = draw_projected_points(points_.copy(), P0_image.copy(), intrinsic[k], extrinsic[k], rect, flag=True, window_name=img_name)
                    P0_image = cv2.resize(P0_image, (P0_image.shape[1]//2, P0_image.shape[0]//2))
                    P0_image_pc = cv2.resize(P0_image_pc, (P0_image_pc.shape[1]//2, P0_image_pc.shape[0]//2))
                    
                    cam_dir = os.path.join(f'./data/{case_name}', cam)
                    if not os.path.exists(cam_dir):
                        os.makedirs(cam_dir)
                    cam_dir_pc = os.path.join(f'./data/{case_name}', cam+'_pc')
                    if not os.path.exists(cam_dir_pc):
                        os.makedirs(cam_dir_pc)
                    basename = img_name.split('/')[-1]
                    img_save_name = os.path.join(cam_dir, basename)
                    cv2.imwrite(img_save_name, P0_image)
                    img_save_name = os.path.join(cam_dir_pc, basename)
                    cv2.imwrite(img_save_name, P0_image_pc)

    if func == 'check_dumpdata':
        # LidarGOP train data
        ceph_client = ceph_utils.ceph_init('/iag_ad_01/ad/zhanghongcheng/code/multimodal-3dgop/petreloss.conf')
        # pkl_path = 'data/2025_06_01_13_26_35_pilotGtParser_v3/0_dataset_infos.pkl'
        campath = 'output/A02_ceph/camera_A02_3dgop_alldet_2V_V1.9.1_No1_0620/dump_virtual_v2'
        gt_path = 'output/A02_ceph/camera_A02_3dgop_alldet_2V_V1.9.1_No1_0620/dump_virtual_v2/gt_boxes'
        case_name = 'origin_dump_v0'
        for i in range(10):
            gt_ = os.path.join(gt_path, f'{i}.npy')
            gt_boxes = np.load(gt_)
            gt_boxes[:, 6] *= -1
            cam_list = ['center_camera_fov120', 'center_camera_fov30']
            rect = np.eye(4)
            for k, cam in enumerate(cam_list):
                
                ext_path = os.path.join(campath, cam, 'exts', f'{i}.npy')
                int_path = os.path.join(campath, cam, 'ints', f'{i}.npy')
                ext = np.load(ext_path)
                int_ = np.load(int_path)
                img_path = os.path.join(campath, cam, 'fig', f'{i}.npy')
                img = np.load(img_path)
                
                P0_image = draw_projected_box(gt_boxes.copy(), img.copy(), int_, ext, rect, flag=False, window_name=str(i))
                
                cam_dir = os.path.join(f'./data/{case_name}', cam)
                if not os.path.exists(cam_dir):
                    os.makedirs(cam_dir)

                basename = img_path.split('/')[-1]
                img_save_name = os.path.join(cam_dir, basename.replace('npy', 'png'))
                cv2.imwrite(img_save_name, P0_image)



    if func == 'check_v00':
        pkl_path = '/mnt/lustrenew/share_data/xuzhiyong/3DGOP_A02/pkls/V1.9.1/train_infos_unknown_3DGOP_A02_v1.9.1_blocked.pkl'
        info = pkl.load(open(pkl_path, 'rb'))
        invalid_list = []
        tar = 'A02_MM11V'
        for seq, val in info.items():
            invalid_seq = False
            for i, frame in enumerate(val):
                lidar_path = frame['velodyne_path']
                lidar_pose_path = frame['lidar_pose_path']
                if tar in lidar_path or tar in lidar_pose_path:
                    invalid_seq = True
                    break
                invalid_cam_flag = False
                for cam, cam_path in frame['img_path'].items():
                    if tar in cam_path:
                        invalid_cam_flag = True
                        break
                if invalid_cam_flag:
                    invalid_seq = True
                    break
            if invalid_seq:
                invalid_list.append(seq)    
        print(invalid_list)
        json.dump(invalid_list, open('./invalid_list.json', 'w'))

    if func == 'cp10':
        # 统计脚本
        pkl_path = 'data/3DGOP_batch_test_normalized.pkl'
        info = pkl.load(open(pkl_path, 'rb'))
        cls_map = dict()
        cnt = 0
        dst = 'data/quant_data_bin_10/'
        for seq, val in info.items():
            for i, frame in enumerate(val):
                pcd_path = frame['velodyne_path']
                cmd = 'cp %s %s' % (pcd_path, dst)
                os.system(cmd)
                cnt += 1
                if cnt >= 10:
                    break
    
    if func == 'split_dataset':
        def ceph_write(data, path, use_ceph=False, client=None, update_cache=True):
            postfix = os.path.splitext(path)[1].lower()
            assert postfix == '.pkl', 'Only .pkl saving is supported currently'

            file_bytes = pkl.dumps(data)

            if use_ceph:
                assert client is not None, 'client should not be None'
                client.put(path, file_bytes, update_cache=update_cache)
            else:
                with open(path, 'wb') as f:
                    f.write(file_bytes)
    
        res = {}
        pkl_paths = [
            'aoss-zhc-v2:s3://zhc-v2/lidargop/dataset/all_delivery_0720-ATX-15s-shuima/all_delivery_0720-ATX-15s-shuima_S.pkl', 
            'aoss-zhc-v2:s3://zhc-v2/lidargop/dataset/all_delivery_0720-ATX-15s-zhuitong/all_delivery_0720-ATX-15s-zhuitong_S.pkl', 
            'aoss-zhc-v2:s3://zhc-v2/lidargop/dataset/all_delivery_0720-ATX-15s-gelizhu/all_delivery_0720-ATX-15s-gelizhu_S.pkl',
        ]
        ceph_client = ceph_utils.ceph_init('/iag_ad_01/ad/zhanghongcheng/code/multimodal-3dgop/petreloss.conf')
        for pkl_path in pkl_paths:
            cur_pkl = ceph_utils.ceph_read(pkl_path, None, True, client=ceph_client)
            print(len(cur_pkl.keys()))
            res.update(cur_pkl)
        
        val_set, train_set = {}, {}
        for i, (seq, val) in enumerate(res.items()):
            if i % 10 == 0:
                val_set[seq] = val
            else:
                train_set[seq] = val
            
        pkl_save_name_val = 'aoss-zhc-v2:s3://zhc-v2/lidargop/dataset/all_delivery_0720-ATX-15s-merge_val_S.pkl'
        pkl_save_name_train = 'aoss-zhc-v2:s3://zhc-v2/lidargop/dataset/all_delivery_0720-ATX-15s-merge_train_S.pkl'
        print(len(val_set.keys()))
        print(len(train_set.keys()))
        ceph_write(val_set, pkl_save_name_val, True, ceph_client)
        ceph_write(train_set, pkl_save_name_train, True, ceph_client)