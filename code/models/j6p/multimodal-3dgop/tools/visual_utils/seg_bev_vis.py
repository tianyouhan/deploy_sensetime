import os
import copy
import cv2
import numpy as np
from tqdm import tqdm
from numba import jit
import json
import threading


def create_voxels(points, pc_range, voxel_size, label=None):
    points_min_x, points_min_y, points_min_z, points_max_x, points_max_y, points_max_z = pc_range
    x_scale, y_scale = voxel_size[0:2]
    img_height, img_width = round((pc_range[3] - pc_range[0]) / voxel_size[0]), round((pc_range[4] - pc_range[1]) / voxel_size[1])
    
    @jit(nopython=True)
    def points_to_voxels():
        voxel_occup = np.zeros((img_height, img_width), np.int32)
        for i in range(points.shape[0]):
            cur_pts = points[i, :]
            if cur_pts[0] < points_max_x and cur_pts[0] > points_min_x and \
                cur_pts[1] < points_max_y and cur_pts[1] > points_min_y and \
                cur_pts[2] < points_max_z and cur_pts[2] > points_min_z:
                x_index = int((cur_pts[0] - points_min_x) / x_scale)
                y_index = int((cur_pts[1] - points_min_y) / y_scale)
                if label is not None and label[i] > 0:
                    voxel_occup[x_index, y_index] = label[i] + 1  # to distinguish no empty points
                elif voxel_occup[x_index, y_index] == 0:
                    voxel_occup[x_index, y_index] = 1
        return voxel_occup, img_width, img_height
        
    voxel_occup, img_width, img_height = points_to_voxels()

    return voxel_occup, img_width, img_height

def vis(bev_occu, input_path, filename, label_H, label_W, class_color_map):
    path = os.path.join(input_path,'{}.png'.format(filename))
    mask = cv2.imread(path, -1)
    if not os.path.exists(path):
        path = os.path.join(input_path,'{}.label'.format(filename))
        mask = np.fromfile(path, dtype = np.uint8)  # [y-->down, x-->right], [864, 1024]
    mask = mask.reshape(label_H, label_W).transpose((1, 0))  # [x-->down, y-->right], [1024, 864]
    bev_occu[mask > 0] = mask[mask > 0] + 1  # no empty points is 1, so label +1
    bev_occu = class_color_map[bev_occu]
    bev_occu = bev_occu.astype(np.uint8)
    bev_occu = bev_occu[::-1, ::-1]  # But what we need is [x--> up, y-->left]
    return bev_occu

def main(file_list):
    class_color_map = np.array([[255, 255, 255],  # white
                                [128, 128, 128],  # gray
                                [0,     0, 255],  # red
                                [  0, 255,   0],  # green
                                [255,   0,   0],  # blue
                                [  0, 255, 255],  # yellow
                                [255,   0, 255],  # purple
                                [0,     0, 128],  # deep red
                                [0,   128,   0],  # deep green
                                [128,   0,   0],  # deep blue
                                [0,   128, 128],  # deep yellow
                                [128,   0, 128],  # deep purple
                                [255, 255,   0],  # sky blue
                                [24,  81,  172],  # seg keep: cls error, brown
                                [0,     0,   0],  # seg keep: defect error, black              
                                              ],  # bgr format for cv2
                                dtype=np.int) # bg_color, points, gt_noise, pred_noise

    if vis_drivable:
        drivable_poly = class_color_map[create_poly_mask(mask_id=7)].astype(np.uint8)
    if vis_merge:
        img_path_json = os.path.join(root_dir, 'image_path.json')
        if os.path.exists(img_path_json):
            img_path = json.load(open(os.path.join(root_dir, 'image_path.json'), 'r', encoding='utf-8'))
        else:
            img_path = raw_img_path

    for file in tqdm(file_list):
        # filename = os.path.splitext(file)[0]
        # if  os.path.exists(os.path.join(merge_photo_path,'{}.png'.format(filename))):
        #     continue
        filename = os.path.splitext(file)[0]
        point_cloud = np.fromfile(os.path.join(point_cloud_path, '{}.bin'.format(filename)), dtype = np.float32)
        # [x-->down, y-->right] [1024, 864]
        bev_occu, label_H, label_W = create_voxels(point_cloud.reshape(-1, 4), pc_range, voxel_size)

        if vis_points:
            bev_occu_points = class_color_map[bev_occu]
            bev_occu_points = bev_occu_points.astype(np.uint8)
            bev_occu_points = bev_occu_points[::-1, ::-1]  # But what we need is [x--> up, y-->left]
            cv2.imwrite(os.path.join(points_photo_path, '{}.png'.format(filename)), bev_occu_points)  # [x, y]

        if vis_pred:
            bev_occu_pred = copy.deepcopy(bev_occu)
            bev_occu_pred = vis(bev_occu_pred, model_output_path, filename, label_H, label_W, class_color_map)
            if vis_drivable:
                bev_occu_pred = cv2.addWeighted(bev_occu_points, 1.0, bev_occu_pred, 0.2, 0)
                bev_occu_pred = cv2.addWeighted(bev_occu_pred, 1.0, drivable_poly, 0.5, 0)
            cv2.imwrite(os.path.join(pred_photo_path, '{}.png'.format(filename)), bev_occu_pred)  # [x, y]

        if vis_gt:
            bev_occu_gt = copy.deepcopy(bev_occu)
            bev_occu_gt = vis(bev_occu_gt, gt_path, filename, label_H, label_W, class_color_map)
            if vis_drivable:
                bev_occu_gt = cv2.addWeighted(bev_occu_points, 1.0, bev_occu_gt, 0.2, 0)
                bev_occu_gt = cv2.addWeighted(bev_occu_gt, 1.0, drivable_poly, 0.5, 0)
            cv2.imwrite(os.path.join(gt_photo_path, '{}.png'.format(filename)), bev_occu_gt)  # [x, y]

        if vis_merge:
            if type(img_path) is dict:
                if cam_list is not None:
                    imgs = [cv2.imread(path) for key, path in img_path[filename].items() if key in cam_list]
                else:
                    imgs = [cv2.imread(path) for key, path in img_path[filename].items()]
            else:
                imgs = [cv2.imread(os.path.join(img_path, filename))]
            if only_pred:
                cat_bev = [bev_occu_pred]
            else:
                cat_bev = [bev_occu_pred, bev_occu_gt]
            bev_map = np.concatenate(cat_bev, axis=1).transpose(1, 0, 2)[:, ::-1, :]  # front_x_left_y -> right_x_frony_y
            bev_h, bev_w = bev_map.shape[0:2]
            img_h, img_w = imgs[0].shape[0:2]
            if len(imgs) == 1:
                resize_w = bev_w
                resize_h = int(img_h * (resize_w / img_w))
                imgs = [cv2.resize(img, (resize_w, resize_h)) for img in imgs]
                merge_photo = np.concatenate((imgs[0], bev_map), axis=0)
            elif len(imgs) == 2:
                """
                        ego_car  ego_car  center_fov120
                        ego_car  ego_car  center fov30
                """
                resize_w = bev_w // 2
                resize_h = int(img_h * (resize_w / img_w))
                imgs = [cv2.resize(img, (resize_w, resize_h)) for img in imgs]
                merge_photo = np.concatenate([np.concatenate(imgs, axis=1), bev_map], axis=0)  # center front fov120, fov30
            elif len(imgs) == 7:
                """
                       left_rear left_front
                    rear ego_car  ego_car  center_fov120
                         ego_car  ego_car  center fov30
                       right_rear right_front
                """
                resize_w = bev_w // 2
                resize_h = int(img_h * (resize_w / img_w))
                imgs = [cv2.resize(img, (resize_w, resize_h)) for img in imgs]
                board_h, board_w = 2 * resize_h + bev_h, 4 * resize_w
                merge_photo = np.zeros((board_h, board_w, 3), dtype=np.uint8)
                merge_photo[0:resize_h, board_w//2-resize_w:board_w//2+resize_w] = np.concatenate([imgs[3], imgs[2]], axis=1)  # left rear, left front
                merge_photo[board_h//2-resize_h:board_h//2+resize_h, board_w-resize_w:] = np.concatenate(imgs[0:2], axis=0)  # center front fov120, fov30
                merge_photo[board_h-resize_h:, board_w//2-resize_w:board_w//2+resize_w] = np.concatenate([imgs[6], imgs[5]], axis=1)  # right rear, right front
                merge_photo[board_h//2-resize_h//2:board_h//2+resize_h//2, 0:resize_w] = imgs[4]  # rear
                merge_photo[board_h//2-bev_h//2:board_h//2+bev_h//2, board_w//2-bev_w//2:board_w//2+bev_w//2] = bev_map
            cv2.imwrite(os.path.join(merge_photo_path,'{}.png'.format(filename)), merge_photo)
       
def create_poly_mask(mask_id, pixel_width=2):
    w, h = int((pc_range[3] - pc_range[0]) / voxel_size[0]), int((pc_range[4] - pc_range[1]) / voxel_size[1])
    mask = np.zeros((h, w), dtype=np.uint8)  # [y-->down, x-->right]
    x1, x2, y1, y2 = drivable_range[0], drivable_range[3], drivable_range[1], drivable_range[4]
    corner_points = [np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]], dtype=np.float32)]
    voxel = []
    for points in corner_points:
        points = ((points - pc_range[0:2]) / voxel_size[0:2]).astype(np.int32)
        voxel.append(points)
    cv2.polylines(mask, voxel, isClosed=True, color=(mask_id), thickness=pixel_width)
    mask = mask.transpose((1, 0))  # [x-->down, y-->right]
    mask = mask[::-1, ::-1]  # [x--> up, y-->left]
    return mask

def creat_dir(root_dir, folder):
    save_path = os.path.join(root_dir, folder)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    return save_path

def split_list_n_list(origin_list, n):
    n_list = []
    if len(origin_list) % n == 0:
        cnt = len(origin_list) // n
    else:
        cnt = len(origin_list) // n + 1
    for i in range(0, n):
        n_list.append(origin_list[i*cnt:(i+1)*cnt])
    return n_list

class MyThread(threading.Thread):

    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result  # 如果子线程不使用join方法，此处可能会报没有self.result的错误
        except Exception:
            return None

if __name__ == '__main__':
    """
    Run: sh vis/vis.sh
    """
    pc_range = [0, -20.48, -4, 102.4, 20.48, 4]
    drivable_range = [0, -20.48, -4, 102.4, 20.48, 4]
    voxel_size = [0.16, 0.16, 15]

    # ['center_camera_fov120', 'center_camera_fov30', 'left_front_camera', 'left_rear_camera', 'rear_camera', 'right_front_camera', 'right_rear_camera']
    cam_list = ['center_camera_fov120', 'center_camera_fov30']

    vis_points, vis_pred, vis_gt, vis_merge, only_pred, vis_drivable = True, True, True, False, True, False
    
    root_dir = 'unknown_vis_tmp/fusion_A02_3dgop_seg_det_NoA23_batch2/'
    folder_list = ['pred', 'gt', 'cloud_bin', 'image', 'img_points', 'img_pred', 'img_gt', 'img_merge']
    path_list = []
    for folder in folder_list:
        save_path = creat_dir(root_dir=root_dir, folder=folder)
        path_list.append(save_path)
    model_output_path, gt_path, point_cloud_path, raw_img_path, \
        points_photo_path, pred_photo_path, gt_photo_path, merge_photo_path = path_list
    
    file_list = sorted(os.listdir(point_cloud_path))
    print("File numbers:{}".format(len(file_list)))
    source_dir_n_list = split_list_n_list(file_list, 10)
    threading_list = []
    for part_dir in source_dir_n_list:
        t = MyThread(main, args=(part_dir, ))
        threading_list.append(t)
        t.start()
    for t in threading_list: # 等待所有线程运行完毕
        t.join()