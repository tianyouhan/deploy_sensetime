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


CLS_MAPPING = {
    'STONE POLE': 'pole',#
    'POLE': 'pole',#
    'CONSTRUCTION_SIGN': 'construction_sign',#
    'CONE': 'cone',#
    'TRAFFIC LIGHT': 'traffic_light',
    'BARRIER': 'barrier',#
    'PERMANENT BARRICADE': 'permanent_barricade',
    # 'CEMENT PIER': 'barrier',#
    'BARRIER_GATE': 'Gate_rod',#
    'ISOLATION_BARRER': 'isolation_barrel',#
    'OBSTACLES': 'obstacles',
    'TEMPORARY BARRICADE': 'temporary_barricade',
    'RETRACTABLE DOOR': 'retractable_door',
    'SPEED BUMP': 'speed_bump',
    'VEHICLE_CAR': 'Car',
    'PEDESTRIAN_NORMAL': 'Pedestrian',
    'VEHICLE_TRUCK': 'Truck',
    'VEHICLE_BUS': 'Truck',
    'CYCLIST_MOTOR': 'Cyclist'
}
if __name__ == '__main__':
    func = 'vis_pkl'
    if func == 'cat_video':
        # 拼接视频
        import glob
        basepath = 'output/A02_ceph/fusion_A02_3dgop_alldet_2V_V1.9.1_No1_planB_infer/dump_0704/vis1/det_vis/fusion/2025_07_02_12_17_42_planb_v2_0704'
        with_postfix = True
        if with_postfix:
            path_A = f'{basepath}/center_camera_fov120_cam'
            path_B = f'{basepath}/center_camera_fov30_cam'
        else:
            path_A = f'{basepath}/center_camera_fov120'
            path_B = f'{basepath}/center_camera_fov30'
        path_C = f'{basepath}/bev'
        output_video = f'{basepath}/video_baseline_1024x512.mp4'
        
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
    
    if func == 'vis_pkl':
        # 生产BEV, 点云GT投影可视化图片
        ceph_client = ceph_utils.ceph_init('/iag_ad_01/ad/zhanghongcheng/code/multimodal-3dgop/petreloss.conf')
        pkl_path = '/iag_ad_01/ad/zhanghongcheng/code/multimodal-3dgop/data/demo_pkl/0_PVB_0630_ATX_LIDAR_dataset_infos.pkl'
        info = pkl.load(open(pkl_path, 'rb'))
        tar_list = ['cone', 'pole', 'isolation_barrel', 'triangle_warning', 'animal', 'gate_rod', 'barrier', 'construction_sign']
        for seq, val in info.items():
            if 'calib_path' in val[0]:
                calib_path = val[0]['calib_path']
            else:
                calib_path = calib_path = '/'.join(val[0]['velodyne_path'].split('/')[:-2]) + '/calib'
            cam_K = 'cam_K_new'

            extrinsic, intrinsic = read_calib_fromfile_a02(calib_path, ceph_client, cam_K=cam_K)
            case_name = '0_PVB_0630_ATX_LIDAR_dataset_infos'
            bev_height, bev_width = 512, 1280
            for i, frame in enumerate(val):
                annos = frame['annos']
                gt_names = annos['name']
                l = []
                for cur_cls in gt_names:
                    if cur_cls in CLS_MAPPING:
                        l.append(CLS_MAPPING[cur_cls])
                    else:
                        l.append(cur_cls)
                        import ipdb; ipdb.set_trace()
                        print(cur_cls)
                gt_names =np.array(l)
                N = annos['location'].shape[0]
                gt_boxes = np.zeros((N, 7), dtype=float)

                if N > 0:
                    x, y, z = annos['location'][:, 0:1], annos['location'][:, 1:2], annos['location'][:, 2:3]
                    w, h, l = annos['dimensions'][:, 0:1], annos['dimensions'][:, 1:2], annos['dimensions'][:, 2:3]
                    yaw = -np.expand_dims(annos['rotation_y'], axis=-1)
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
                points_m1_0[:, 2] -= 1.6
                points_ = points_m1_0.copy()
                zm = robust_ground_height(points_m1_0)
                bev_save_dir = os.path.join(f'./data/{case_name}', 'bev')
                pc_range = [0, -20.48, -4, 102.4, 20.48, 4]
                voxel_size = [0.16, 0.16, 8]
                center_w, center_h = int(-pc_range[0] / voxel_size[0]), int(-pc_range[1] / voxel_size[1])
                if draw_bev_flag:
                    draw_bev(points_m1_0, gt_boxes.copy(), bev_height, bev_width, bev_resolution=0.08, center_coords=[center_w*2, center_h*2], radius_list=[25, 50, 75], 
                        save_dir=bev_save_dir, filename=basename.replace('bin', 'png'))
                
                cam_list = ['center_camera_fov120', 'center_camera_fov30']
                map_ = frame['img_path']
                rect = np.eye(4)

                for k, cam in enumerate(cam_list):
                    img_name = map_[cam]
                    img_path = mapping_path_to_ceph(PATH_MAPPING, map_[cam])
                    
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

     