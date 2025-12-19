import numpy as np
import pickle as pkl
import cv2
import os
import math
import json
from draw_projected_box import draw_projected_box
from tqdm import tqdm
from multiprocessing import Process
from pcdet.utils import ceph_utils

PATH_MAPPING = dict({
                "/mnt/": "aoss-zhc-v2:s3://dcp36_lustre_aoss_v2/",
                "s3://aoss-test-data": "aoss-test-data:s3://aoss-test-data",
                "s3:/aoss-test-data": "aoss-test-data:s3://aoss-test-data",
                "s3://aoss-gt/": "aoss-gt:s3://aoss-gt/",
            })
CEPH_CLIENT = ceph_utils.ceph_init('/iag_ad_01/ad/zhanghongcheng/code/multimodal-3dgop/petreloss.conf')
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

def load_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def limit_period(val, offset=0.5, period=2*np.pi):
    return val - np.floor(val / period + offset) * period

def extend_matrix(mat):
    mat = np.hstack((mat, np.zeros((3, 1))))
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat

def load_intrinsics_calib(calib_json_path, cam_K=False):
    use_ceph = 's3:' in calib_json_path
    if use_ceph:
        calib_json_path = mapping_path_to_ceph(calib_json_path)
    intrinsic_calib = ceph_utils.ceph_read(calib_json_path, None, use_ceph, CEPH_CLIENT)
    key = [ele for ele in intrinsic_calib.keys()][0]
    camera_intrinsic_dict = intrinsic_calib[key]
    if cam_K and cam_K in camera_intrinsic_dict['param']:
        camera_intrinsic = np.array(camera_intrinsic_dict['param'][cam_K]['data'])
        print(cam_K)
    else:
        # camera_dist = np.array(camera_intrinsic_dict['param']['cam_dist']['data'][0])
        if 'cam_K_new' in camera_intrinsic_dict['param']:
            camera_intrinsic = np.array(camera_intrinsic_dict['param']['cam_K_new']['data'])
        else:
            camera_intrinsic = np.array(camera_intrinsic_dict['param']['cam_K']['data'])
    return camera_intrinsic

def load_extrinsic_calib(calib_json_path, normalize_calib, lidar2camera=True):
    use_ceph = 's3:' in calib_json_path
    if use_ceph:
        calib_json_path = mapping_path_to_ceph(calib_json_path)
    extrinsic_calib = ceph_utils.ceph_read(calib_json_path, None, use_ceph, CEPH_CLIENT)
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
            lidar_extrinsic = lidar_extrinsic @ trans_mat  # gacbaidu
        else:
            lidar_extrinsic = np.linalg.inv(lidar_extrinsic) @ trans_mat  # A02
    return lidar_extrinsic

def read_calib(calib_path, cam_K=False):
    intrinsic, extrinsic = [], []
    for camera_name in cam_list:
        if data_source in ['gacbaidu', 'top_center_lidar']:
            intrinsic_file = f'{camera_name}-intrinsic.json'
            extrinsic_file = f'top_center_lidar-to-{camera_name}-extrinsic.json'
            camera_intrinsics_path = os.path.join(calib_path, camera_name, intrinsic_file)
            camera_extrinsic_path = os.path.join(calib_path, 'top_center_lidar', extrinsic_file)
            extrinsic_cam = load_extrinsic_calib(camera_extrinsic_path, 'gacbaidu', lidar2camera=True)  # gacbaidu
        elif data_source in ['A02', 'concat_M1_lidar']:
            intrinsic_file = f'{camera_name}-intrinsic.json'
            extrinsic_file = f'{camera_name}-to-car_center-extrinsic.json'
            camera_intrinsics_path = os.path.join(calib_path, camera_name, intrinsic_file)
            camera_extrinsic_path = os.path.join(calib_path, camera_name, extrinsic_file)
            extrinsic_cam = load_extrinsic_calib(camera_extrinsic_path, 'A02', lidar2camera=False)  # A02
        intrinsic_cam = extend_matrix(load_intrinsics_calib(camera_intrinsics_path, cam_K=cam_K))
        intrinsic.append(np.expand_dims(intrinsic_cam, axis=0))
        extrinsic.append(np.expand_dims(extrinsic_cam, axis=0))
    extrinsic = np.concatenate(extrinsic)
    intrinsic = np.concatenate(intrinsic)
    return extrinsic, intrinsic

def mapping_path_to_ceph(path):
    new_path = path
    if path is None or path == "":
        return None
    for key, val in PATH_MAPPING.items():
        if key in path:
            if '/mnt' not in key:
                new_path = path.replace(key, val)
                break
            else:
                if 'xuzhiyong' in path:
                    val = 'aoss-zhc-v2:s3://dcp36_lustre_aoss_v2_xuzhiyong/'
                else:
                    val = 'aoss-zhc-v2:s3://dcp36_lustre_aoss_v2/'
                new_path = path.replace(key, val)
                break
    return new_path
def get_seq_from_res(res, seq_name):
    infos = []
    for info in res:
        if 'seq' in info['metadata'] and info['metadata']['seq'] == seq_name:
            infos.append(info)
        else:
            cur_velopath = info['metadata']['velodyne_path']#.split('/')[-4]
            if seq_name in cur_velopath:
                infos.append(info)
    return infos

# def draw_bev(points, boxes, bev_height, bev_width, center_coords=None,
#              bev_resolution=0.16, show_img=False,
#              radius_list=[], save_dir='', filename='', color_list=None, seg_pred=None):
#     bev_image = np.zeros((bev_height, bev_width, 3), dtype=np.uint8) + 255
#     # draw points in bev
#     if center_coords is None:
#         center_w, center_h = [bev_width // 2, bev_height // 2]
#     else:
#         center_w, center_h = center_coords
#     image_coordinates = np.round((points[:, :2] / bev_resolution) + [center_w, center_h]).astype(int)
#     valid_indices = np.logical_and(image_coordinates[:, 0] >= 0, image_coordinates[:, 0] < bev_width)
#     valid_indices = np.logical_and(valid_indices, np.logical_and(image_coordinates[:, 1] >= 0, image_coordinates[:, 1] < bev_height))
#     image_coordinates = image_coordinates[valid_indices]
#     bev_image[image_coordinates[:, 1], image_coordinates[:, 0]] = [128, 128, 128]  # x_right_y_back (256, 640
#     if seg_pred is not None:
#         bev_image[seg_pred != 0] = class_color_map[seg_pred][seg_pred != 0]
    
#     # boxes = m1_gtbox.copy()
#     for j, gt in enumerate(boxes):
#         x, y, z = gt[0], gt[1], gt[2]
#         l, w, h = gt[3], gt[4], gt[5]
#         yaw = -gt[6]
#         cos_yaw = math.cos(yaw)
#         sin_yaw = math.sin(yaw)
#         corner_points = np.array([[-l / 2, -w / 2, 0], [l / 2, -w / 2, 0], [l / 2, w / 2, 0], [-l / 2, w / 2, 0]])
#         rotated_corner_points = np.zeros_like(corner_points)
#         rotated_corner_points[:, 0] = cos_yaw * corner_points[:, 0] - sin_yaw * corner_points[:, 1] + x
#         rotated_corner_points[:, 1] = sin_yaw * corner_points[:, 0] + cos_yaw * corner_points[:, 1] + y
#         rotated_corner_points[:, 2] = corner_points[:, 2]

#         bev_x = np.round((rotated_corner_points[:, 0] / bev_resolution) + center_w).astype(int)
#         bev_y = np.round((rotated_corner_points[:, 1] / bev_resolution) + center_h).astype(int)

#         if color_list is None:
#             color = (255, 0, 0)
#         else:
#             color = tuple([int(c) for c in color_list[j]])
        
#         for ii in range(4):
#             cv2.line(bev_image, (bev_x[ii], bev_y[ii]), (bev_x[(ii + 1) % 4], bev_y[(ii + 1) % 4]), color, 2)
            
#     bev_image = cv2.flip(bev_image, 0)
#     for radius in radius_list:
#         draw_circle(bev_image, bev_height, bev_width, bev_resolution, radius, center_coords=center_coords)
    
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     bev_name = os.path.join(save_dir, filename)
#     cv2.imwrite(bev_name, bev_image)
#     if show_img:
#         cv2.imshow("frame %s" % (filename), bev_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     return

def draw_texts_right_bottom(img, lines, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.25,
                            thickness=1, color=(50, 50, 50), padding=10, char_width=4):

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
            y -= text_height + 3  # 行距
        else:
            break

def draw_bev(points, boxes, bev_height, bev_width, center_coords=None,
             bev_resolution=0.16, show_img=False, with_dir=False,
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
        ct_x = 0.5 * (bev_x[0] + bev_x[2]) + 6    
        ct_y = 0.5 * (bev_y[0] + bev_y[2]) + 6
        h, w, _ = bev_image.shape
        cv2.putText(bev_image, str(j), (int(ct_x), h-int(ct_y)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.25, color=(0,0,185), thickness=1)
        if gt_names is not None:
            cur_gtname = gt_names[j]
            cur_str = f"Obj id: {j}, {cur_gtname}"
            str_lines.append(cur_str)
            # print(cur_str)
    str_lines.append(filename)
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


def get_seg_label(input_path, filename, bev_width, bev_height):
    path = os.path.join(input_path,'{}.png'.format(filename))
    if os.path.exists(path):
        label = cv2.imread(path, -1)
    else:
        path = os.path.join(input_path,'{}.label'.format(filename))
        label = np.fromfile(path, dtype = np.uint8)
    label = label.reshape(bev_height, bev_width)  # [y-->down, x-->right], [256, 640]
    return label


def split_list_n_list(origin_list, n):
    n_list = []
    if len(origin_list) % n == 0:
        cnt = len(origin_list) // n
    else:
        cnt = len(origin_list) // n + 1
    for i in range(0, n):
        n_list.append(origin_list[i*cnt:(i+1)*cnt])
    return n_list


def main(vis_gt, score, vis_seg, rm_empty_gt_box, seq_name_list):
    global class_color_map
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
    
    det_class_names, seg_class_names = list(det_classes.keys()), list(seg_classes.keys())
    
    bev_width, bev_height = int((pc_range[3] - pc_range[0]) / voxel_size[0]), int((pc_range[4] - pc_range[1]) / voxel_size[1])
    center_w, center_h = int(-pc_range[0] / voxel_size[0]), int(-pc_range[1] / voxel_size[1])
    seq_name_list = dataset.keys()
    save_bbox_txt = False
    for seq_name in tqdm(seq_name_list):
        print(f'process {seq_name}')
        results = dataset[seq_name]
        # import ipdb; ipdb.set_trace()
        if len(results) > 1000:
            results = results[:1000]
        if type(results[0]) == str:
            results_ = []
            for r in results:
                res = ceph_utils.ceph_read(r, None, True, CEPH_CLIENT)
                results_.append(res)
            results = results_
        try:
            preds = get_seq_from_res(all_preds, seq_name)
        except:
            preds = all_preds
        # import ipdb; ipdb.set_trace()
        calib_path = '/'.join(results[0]['velodyne_path'].split('/')[:-2]) + '/calib'
        if not os.path.exists(calib_path):
            if 'calib_path' in results[0]:
                calib_path = results[0]['calib_path']
            else:
                calib_path = mapping_path_to_ceph(calib_path)
                print(calib_path)
        if calib_path is None:
            bev_only = True
        else:
            bev_only = False
            extrinsic, intrinsic = read_calib(calib_path, cam_K=CAM_K)

        results_frame_list = []
        for i in tqdm(range(len(results))):
            velodyne_path = results[i]['velodyne_path']
            frame_name = os.path.basename(velodyne_path)[:-4]
            results_frame_list.append(frame_name)

        rect = np.eye(4)
        for i in tqdm(range(len(preds))):
            # import ipdb; ipdb.set_trace()
            cur_m1_pcd_path = preds[i]['metadata']['velodyne_path']
            frame_name = os.path.basename(cur_m1_pcd_path)[:-4]
            # import ipdb; ipdb.set_trace()
            if frame_name not in results_frame_list:
                print(i, frame_name)
                continue
            results_index = results_frame_list.index(frame_name)
            res = results[results_index]
            # import ipdb; ipdb.set_trace()
            # res = results[i]
            pcd_name = preds[i]['metadata']['frame_name']
            cur_cloud_path = os.path.join(cloud_path, str(pcd_name) + '.bin')
            points_m1_0 = np.fromfile(cur_cloud_path, dtype=np.float32).reshape(-1,4)
            # import ipdb; ipdb.set_trace()
            # read seg pred
            if vis_seg:
                seg_pred = get_seg_label(seg_pred_path, frame_name, bev_width, bev_height)
                seg_label = get_seg_label(seg_label_path, frame_name, bev_width, bev_height)
                seg_pred[seg_pred != 0] += 5  # deep color
                seg_label[seg_label != 0] += 5  # deep color
            else:
                seg_pred = None

            # P128 points transform
            if vis_gt:
                N = res['annos']['location'].shape[0]
                m1_gtbox = np.zeros((N, 7))
                m1_gtbox[:, 0:3] = res['annos']['location'][:, 0:3]
                m1_gtbox[:, 3] = res['annos']['dimensions'][:, 2]
                m1_gtbox[:, 4] = res['annos']['dimensions'][:, 0]
                m1_gtbox[:, 5] = res['annos']['dimensions'][:, 1]
                m1_gtbox[:, 6] = res['annos']['rotation_y']
                m1_gtbox[:, 6] = limit_period(m1_gtbox[:,6])
                # draw_pcd_o3d(points_m1_0, m1_gtbox.copy(), mode='M1')
                img_gtbox = m1_gtbox.copy()
                img_gtbox[:, 6] = -img_gtbox[:, 6]
                try:
                    mask = np.ones(img_gtbox.shape[0], dtype=np.bool_)
                except:
                    mask = np.ones(img_gtbox.shape[0], dtype=np.bool)
                # rm empty box
                if rm_empty_gt_box:
                    empty_box_mask = res['annos']['num_points_in_box'] > 0
                    mask = mask & empty_box_mask
                # class mask
                mask_class = np.array([class_name in det_class_names for class_name in res['annos']['name']]).astype(mask.dtype)
                mask = mask & mask_class
                # apply mask
                img_gtbox = img_gtbox[mask]
                valid_class = np.array(res['annos']['name'])[mask]
                # color list
                color_list_gt = class_color_map[[det_classes[name] for name in valid_class]]
            # get pred_boxes
            if 'boxes_lidar' in preds[i]:
                pred_boxes = preds[i]['boxes_lidar']
            else:
                pred_boxes = preds[i]['pred_boxes']
            # add score mask
            if type(score) is not list:
                score = [score for _ in range(len(det_classes))]
            try:
                mask = np.zeros(preds[i]['score'].shape[0], dtype=np.bool_)
            except:
                mask = np.zeros(preds[i]['score'].shape[0], dtype=np.bool)
            for idx, (cur_class, cur_score) in enumerate(zip(preds[i]['name'], preds[i]['score'])):
                thresh_score = score[det_class_names.index(cur_class)]
                if cur_class in ['cone', 'pole', 'isolation_barrel', 'triangle_warning', 'animal', 'gate_rod', 'barrier', 'construction_sign']:
                    if pred_boxes[idx, 0] > 50:  # 大于50m score阈值给0.2
                        thresh_score = 0.3
                if cur_score > thresh_score:
                    mask[idx] = True
            # score mask
            pred_boxes = pred_boxes[mask]
            pred_boxes[:, 6] = -pred_boxes[:, 6]
            valid_class = preds[i]['name'][mask]
            valid_score = preds[i]['score'][mask]
            
            ## write pred box
            box_pred_path = os.path.join(root_dir, 'box_pred', task, seq_name)
            os.makedirs(box_pred_path, exist_ok=True)
            txt_writer = open(os.path.join(box_pred_path, f'{frame_name}.txt'), 'w')
            for idx in range(pred_boxes.shape[0]):
                cur_box = pred_boxes[idx]
                line = '{} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f}'.format(valid_class[idx], cur_box[3], cur_box[4], cur_box[5], cur_box[0], cur_box[1], cur_box[2], -cur_box[6], valid_score[idx])
                txt_writer.write(line + '\n')
            txt_writer.close()

            # color list
            color_list_pred = class_color_map[[det_classes[name] for name in valid_class]]
            
            if vis_gt:
                pred_boxes = np.concatenate([img_gtbox, pred_boxes], axis=0)
                color_list_gt_new = color_list_gt // 2
                color_list_pred = np.concatenate([color_list_gt_new, color_list_pred], axis=0)
                if vis_seg:
                    cls_error = (seg_pred != seg_label) & (seg_pred != 0)
                    defect_error = (seg_pred != seg_label) & (seg_pred == 0)
                    seg_pred[cls_error] = len(class_color_map) - 2  # last two: brown
                    seg_pred[defect_error] = len(class_color_map) - 1  # last one: black
            bev_save_dir = os.path.join(det_dir, seq_name, 'bev')
            # print(valid_class)
            # if save_bbox_txt:
            # import ipdb; ipdb.set_trace()
            draw_bev(points_m1_0, pred_boxes.copy(), bev_height*2, bev_width*2, bev_resolution=voxel_size[0]/2, center_coords=[center_w*2, center_h*2], radius_list=[25, 50, 75], 
                    save_dir=bev_save_dir, filename=frame_name + '.png', color_list=color_list_pred, seg_pred=seg_pred, gt_names=valid_class)
            if not bev_only:
                for k, cam in enumerate(cam_list):
                    # img_name = 'data/%s/image_undistort/%s/%s' % (seq_name, cam_list[k], os.path.basename(res['img_path'][cam]))
                    img_name = res['img_path'][cam]
                    # img = cv2.imread(img_name)
                    # window_name = "frame %s, P%s" % (i, k)
                    img_path = mapping_path_to_ceph(img_name)
                    img = ceph_utils.ceph_read(img_path, np.int8, True, client=CEPH_CLIENT)
                    P0_image = draw_projected_box(pred_boxes.copy(), img, intrinsic[k], extrinsic[k], rect, flag=False, window_name=img_name, color_list=color_list_pred, thickness=4)
                    P0_image = cv2.resize(P0_image, (P0_image.shape[1]//2, P0_image.shape[0]//2))
                    
                    cam_dir = os.path.join(det_dir, seq_name, cam+'_cam')
                    if not os.path.exists(cam_dir):
                        os.makedirs(cam_dir)
                    
                    img_save_name = os.path.join(cam_dir, frame_name + '.png')
                    cv2.imwrite(img_save_name, P0_image)
                    
                    title = 'Camera result %s' % img_name
                    if show_img:
                        cv2.imshow(title, P0_image)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()  


def multi_process_main(vis_gt, det_score, vis_seg, rm_empty_gt_box, thread_num=20):
    seq_name_list = []
    if fix_seq_name is None:
        for info in all_preds:
            if 'seq' in info['metadata']:
                cur_seq = info['metadata']['seq']
            else:
                cur_seq = info['metadata']['velodyne_path'].split('/')[-3]
            seq_name_list.append(cur_seq)
        seq_name_list = list(set(seq_name_list))
    else:
        seq_name_list = [fix_seq_name]
    
    if thread_num > 1:
        n_list = split_list_n_list(seq_name_list, thread_num)
        process_list = []
        for cur_list in n_list:
            t = Process(target=main, args=(vis_gt, det_score, vis_seg, rm_empty_gt_box, cur_list, ))
            t.start()
            process_list.append(t)
        for t in process_list: # 等待所有进程运行完毕
            t.join()
    else:
        main(vis_gt, det_score, vis_seg, rm_empty_gt_box, seq_name_list)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='multi')
    parser.add_argument('--id', type=int, default=1, help='epoch id')
    parser.add_argument('-r', '--root', type=str, default=None, help='root_dir')
    parser.add_argument('-p', '--pkl', type=str, default=None, help='res_pkl')
    parser.add_argument('-t', '--task', type=str, default='fusion', help='task')
    parser.add_argument('-k', '--camk', type=str, default='', help='cam_Key')
    parser.add_argument('--pvb', action='store_true', help='Enable pvb mode')
    args = parser.parse_args()
    args.id = 25
    if args.camk:
        CAM_K = args.camk
    else:
        CAM_K = False
    data_source = 'A02'  # choice from ['gacbaidu', 'A02]
    task = args.task
    root_dir = args.root
    det_dir = f'{root_dir}/det_vis/{task}'
    if not os.path.exists(det_dir):
        os.makedirs(det_dir)
    res_pkl = args.pkl
    cloud_path = f'{root_dir}/cloud_bin/{task}'
    seg_pred_path = f'{root_dir}/pred/{task}'
    seg_label_path = f'{root_dir}/gt/{task}'
    show_img = False
    cam_list = ['center_camera_fov120', 'center_camera_fov30']
    fix_seq_name = None  # '2023_10_29_02_14_31_AutoCollect_0'
    # ground truth
    on_ceph = 's3:' in res_pkl
    if not on_ceph:
        with open(res_pkl, "rb") as f:
            dataset = pkl.load(f)
    else:
        dataset = ceph_utils.ceph_read(res_pkl, None, True, client=CEPH_CLIENT)
    # predictions
    pred_pkl = f'{root_dir}/result_{task}.pkl'
    with open(pred_pkl, "rb") as f:
        all_preds_pvb = pkl.load(f)
    pred_pkl_gop = f'{root_dir}/result_{task}gop.pkl'
    with open(pred_pkl_gop, "rb") as f:
        all_preds_gop = pkl.load(f)
    all_preds = []
    for k, pvb in enumerate(all_preds_pvb):
        gop = all_preds_gop[k]
        res = dict()
        # import ipdb; ipdb.set_trace()
        res['name'] = np.concatenate((pvb['name'], gop['name']), axis=0)
        res['score'] = np.concatenate((pvb['score'], gop['score']), axis=0)
        res['pred_boxes'] = np.concatenate((pvb['pred_boxes'], gop['pred_boxes']), axis=0)
        res['pred_labels'] = np.concatenate((pvb['pred_labels'], gop['pred_labels']+4), axis=0)
        res['frame_id'] = pvb['frame_id']
        res['metadata'] = pvb['metadata']
        # import ipdb; ipdb.set_trace()
        all_preds.append(res)
    pc_range = [0, -20.48, -4, 102.4, 20.48, 4]
    voxel_size = [0.16, 0.16, 8]

    det_score = [0.3, 0.3, 0.3, 0.3, 0.3]
    det_classes = {'cone': 1, 'pole': 2, 'isolation_barrel': 3, 'triangle_warning': 4, 'animal': 5}
    seg_classes = {'gate_rod': 6, 'barrier': 7, 'permanent_barricade': 8, 'temporary_barricade': 8, 'construction_sign': 9, 'obstacles':10}

    ## vis seg and det
    # multi_process_main(vis_gt=False, det_score=det_score, vis_seg=True, rm_empty_gt_box=True, thread_num=20)

    ## vis det
    det_score = [0.5, 0.3, 0.3, 0.5, 
                 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    
    det_classes = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Truck': 4,
                'cone': 5, 'pole': 6, 'isolation_barrel': 7, 'triangle_warning': 8, 'animal': 9, 'gate_rod': 10, 'barrier': 11, 'construction_sign': 12}
    
    multi_process_main(vis_gt=False, det_score=det_score, vis_seg=False, rm_empty_gt_box=True, thread_num=1)

     

