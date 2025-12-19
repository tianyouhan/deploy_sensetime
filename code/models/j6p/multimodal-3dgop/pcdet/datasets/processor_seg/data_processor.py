from functools import partial
import cv2
import numpy as np
from numba import jit
from skimage import transform
import torch
import torchvision
from ...utils import box_utils, common_utils


@jit(nopython=True)
def VoxelGenerator(points, label, pc_range, voxel_size, grid_size, voxel_num_points, voxel_pos_num, coor_to_voxelindex, 
                   voxels, coors, coor, voxel_label, max_points, max_voxels, thresh=0):

    N = points.shape[0]
    ndim = 3
    ndim_minus_1 = ndim - 1

    voxel_num = 0
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - pc_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelindex = coor_to_voxelindex[coor[0], coor[1], coor[2]]
        if voxelindex == -1:
            voxelindex = voxel_num
            if voxelindex >= max_voxels:
                break
            coor_to_voxelindex[coor[0], coor[1], coor[2]] = voxelindex
            coors[voxelindex] = coor
            voxel_label[coor[0], coor[1], coor[2]] = 0

            voxel_num += 1

        num = voxel_num_points[voxelindex]
        if num < max_points:
            voxels[voxelindex, num] = points[i]
            voxel_num_points[voxelindex] += 1

        if label[i, 0] > 0:
            voxel_pos_num[voxelindex] += 1
            if voxel_pos_num[voxelindex] >= thresh:
                voxel_label[coor[0], coor[1], coor[2]] = label[i, 0]  # numba requires a[i, 0] not a[i]
    
    return voxel_num


class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        self.voxel_size = np.array(vsize_xyz, dtype=np.float32)
        self.pc_range = np.array(coors_range_xyz, dtype=np.float32)

        grid_size = (coors_range_xyz[3:]-coors_range_xyz[:3]) / self.voxel_size
        self.grid_size = tuple(np.round(grid_size).astype(np.int32).tolist())
        self.bev_shape = self.grid_size[::-1]

        self.max_points = max_num_points_per_voxel
        self.max_voxels = max_num_voxels
        self.num_point_features = num_point_features

    def generate(self, points, label):
        voxel_num_points = np.zeros(shape=(self.max_voxels, ), dtype=np.int32)
        coor_to_voxelindex = -np.ones(shape=self.bev_shape, dtype=np.int32)
        voxels = np.zeros(shape=(self.max_voxels, self.max_points, points.shape[-1]), dtype=points.dtype)
        coors = np.zeros(shape=(self.max_voxels, 3), dtype=np.int32)
        voxel_label = np.zeros(shape=self.bev_shape, dtype=np.int32)
        coor = np.zeros(shape=(3, ), dtype=np.int32)
        voxel_pos_num = np.zeros(shape=(self.max_voxels, ), dtype=np.int32)

        voxel_num = VoxelGenerator(points, label, self.pc_range, self.voxel_size, self.grid_size, voxel_num_points, voxel_pos_num, coor_to_voxelindex, 
                                   voxels, coors, coor, voxel_label, self.max_points, self.max_voxels)
        coors = coors[:voxel_num]
        voxels = voxels[:voxel_num]
        voxel_num_points = voxel_num_points[:voxel_num]
        voxel_output = voxels, coors, voxel_num_points, voxel_label

        return voxel_output


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.mode = 'train' if training else 'test'
        
        self.voxel_size = None
        self.grid_size = None
        for cur_cfg in processor_configs:
            if cur_cfg.NAME == 'transform_points_and_label_to_voxels' or \
               cur_cfg.NAME == 'transform_points_and_label_to_voxels_placeholder':
                self.voxel_size = np.array(cur_cfg.VOXEL_SIZE)
                grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / self.voxel_size
                self.grid_size = np.round(grid_size).astype(np.int64)
                break

        self.data_processor_queue = []

        self.voxel_generator = None

        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_label_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_label_outside_range, config=config)

        if data_dict.get('points', None) is not None:
            mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
            data_dict['points'] = data_dict['points'][mask]
            data_dict['label'] = data_dict['label'][mask]

        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_LABEL:
            gt_boxes_norm = box_utils.boxes_normalize(data_dict['gt_boxes'], is_bottom_center=True, is_clockwise=True)
            mask = box_utils.mask_boxes_outside_range_numpy(
                gt_boxes_norm, self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1), 
                use_center_to_filter=config.get('USE_CENTER_TO_FILTER', True)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
            if 'gt_names' in data_dict:
                data_dict['gt_names'] = data_dict['gt_names'][mask]
            if 'num_points_in_gt' in data_dict:
                data_dict['num_points_in_gt'] = data_dict['num_points_in_gt'][mask]
            if 'blocked_label' in data_dict:
                data_dict['blocked_label'] = data_dict['blocked_label'][mask]
        return data_dict
    
    def mask_points_and_label_outside_fov(self, data_dict=None, config=None):  # TODO, unfinished
        if data_dict is None:
            return partial(self.mask_points_and_label_outside_fov, config=config)

        camera_intrinsics, lidar2camera_rt = data_dict["camera_intrinsics"], data_dict["lidar2camera"]
        lidar_aug_matrix, img_aug_matrix = data_dict['lidar_aug_matrix'], data_dict["img_aug_matrix"]
        lidar_aug_matrix_inv = np.linalg.inv(lidar_aug_matrix).astype(np.float32)
        img_h, img_w = np.array(config.IMAGE_SIZE, np.int32)

        points_enable = (data_dict.get('points', None) is not None and config.REMOVE_OUTSIDE_POINTS_LABEL) or config.GT_DEPTH
        gt_boxes_enable = data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES_LABEL
        if points_enable:
            points_xyz = data_dict['points'][:, 0:3].copy()
            fov_points_mask = np.zeros(data_dict['points'].shape[0], dtype=np.bool_)
            if config.GT_DEPTH:
                gt_depth = np.zeros((len(data_dict['camera_imgs']), 1, img_h, img_w), dtype=np.float32)
            else:
                gt_depth = [None for _ in range(len(data_dict['camera_imgs']))]
        if gt_boxes_enable:
            fov_boxes_mask = np.zeros(data_dict['gt_boxes'].shape[0], dtype=np.bool_)
            gt_boxes = data_dict['gt_boxes'].copy()

        for idx in range(len(camera_intrinsics)):
            ins_mat, ext_mat, img_aug_mat = camera_intrinsics[idx].astype(np.float32), lidar2camera_rt[idx].astype(np.float32), img_aug_matrix[idx].astype(np.float32)
            # fuse lidar aug params
            ext_mat = ext_mat @ lidar_aug_matrix_inv  # inverse lidar aug
         
            if points_enable:
                mask = common_utils.mask_points_by_fov((img_w, img_h), points_xyz, ins_mat, ext_mat, img_aug_mat, gt_depth[idx])
                fov_points_mask |= mask
                
            if gt_boxes_enable:
                gt_boxes_norm = box_utils.boxes_normalize(gt_boxes, is_bottom_center=True, is_clockwise=True)
                corner_points = box_utils.boxes_to_corners_3d(gt_boxes_norm)
                corner_points = corner_points.reshape(-1, 3)  # (N, 8, 3) to (N*8, 3)
                mask = common_utils.mask_points_by_fov((img_w, img_h), corner_points, ins_mat, ext_mat, img_aug_mat)
                mask = mask.reshape(-1, 8)
                mask = np.sum(mask, axis=1) >= config.MIN_KEPT_POINTS_IN_BOX  # 有N个角点在fov内就保留
                fov_boxes_mask |= mask
        
        if points_enable:
            if config.REMOVE_OUTSIDE_POINTS_LABEL:
                data_dict['points'] = data_dict['points'][fov_points_mask]
                data_dict['label'] = data_dict['label'][fov_points_mask]
            if config.GT_DEPTH:
                data_dict['gt_depth'] = gt_depth
        if gt_boxes_enable:
            data_dict['gt_boxes'] = data_dict['gt_boxes'][fov_boxes_mask]
            if 'gt_names' in data_dict:
                data_dict['gt_names'] = data_dict['gt_names'][fov_boxes_mask]
            if 'num_points_in_gt' in data_dict:
                data_dict['num_points_in_gt'] = data_dict['num_points_in_gt'][fov_boxes_mask]
            if 'blocked_label' in data_dict:
                data_dict['blocked_label'] = data_dict['blocked_label'][fov_boxes_mask]
        return data_dict

    def shuffle_points_and_label(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points_and_label, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points
            if data_dict.get('label', None) is not None:
                data_dict['label'] = data_dict['label'][shuffle_idx]

        return data_dict

    def transform_points_and_label_to_voxels_placeholder(self, data_dict=None, config=None):
        # just calculate grid size
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_and_label_to_voxels_placeholder, config=config)
        
        return data_dict

    def transform_points_and_label_to_voxels(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_and_label_to_voxels, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        points, label = data_dict['points'], data_dict['label']
        voxel_output = self.voxel_generator.generate(points, label)
        voxels, coordinates, num_points, voxel_label = voxel_output
        
        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        if config.get('PILLAR_LABEL', False):
            voxel_label = np.max(voxel_label, axis=0, keepdims=True)  # voxel to pillar

        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        data_dict['voxel_label'] = voxel_label
        return data_dict
    
    def dilate_and_erode_voxel_label(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.dilate_and_erode_voxel_label, config=config)

        if data_dict.get('voxel_label', None) is not None:
            ori_dtype = data_dict['voxel_label'].dtype
            voxel_label = data_dict['voxel_label'].squeeze().astype(np.uint8).copy()
            id_choice, pixel_width = config.get('CHOICE_ID', None), config.PIXEL_WIDTH
            voxel_label_valid = voxel_label.copy()
            if id_choice is not None:
                ignore_index = ~(np.logical_or.reduce([voxel_label == idx for idx in id_choice], axis=0))
            else:
                ignore_index = voxel_label == 0
            voxel_label_valid[ignore_index] = 0
            keep_index = ignore_index & (voxel_label != 0)  # 除了id_choice之外的正样本需要被保持

            kernel = np.ones((abs(pixel_width), abs(pixel_width)), dtype=np.uint8)
            if pixel_width >= 0:
                voxel_label_valid = cv2.dilate(voxel_label_valid, kernel, iterations=1)
            else:
                voxel_label_valid = cv2.erode(voxel_label_valid, kernel, iterations=1)
            voxel_label_valid[keep_index] = voxel_label[keep_index]

            if config.get('TASK_IGNORE', None) is not None:
                for ignore_task in config.TASK_IGNORE:
                    data_dict[f'{ignore_task}_label'] = data_dict.get(f'{ignore_task}_label', data_dict['voxel_label'])
            data_dict['voxel_label'] = voxel_label_valid[None, :, :].astype(ori_dtype)
        return data_dict
    
    def get_convexHull_voxel_label(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.get_convexHull_voxel_label, config=config)
        
        def boxes_to_bev_points(boxes, lidar_aug_matrix):
            boxes_norm = box_utils.boxes_normalize(boxes, is_bottom_center=True, is_clockwise=True)
            corner_points = box_utils.boxes_to_corners_3d(boxes_norm).reshape(-1, 3)  # (N, 8, 3) to (N*8, 3)
            corner_points_xy = (corner_points @ lidar_aug_matrix[:3, :3].T  + lidar_aug_matrix[:3, -1])[:, 0:2]
            corner_points_xy = ((corner_points_xy - self.point_cloud_range[0:2]) / self.voxel_size[0:2]).astype(np.int32)  # (N*8, 2)
            x, y = corner_points_xy[..., 0], corner_points_xy[..., 1]
            valid_mask = (0 <= x) & (x < W) & (0 <= y) & (y < H)
            valid_mask = np.any(valid_mask.reshape(-1, 8), axis=1)
            corner_points_xy = corner_points_xy.reshape(-1, 8, 2)[valid_mask]  # (M, 8, 2)
            return corner_points_xy

        if data_dict.get('voxel_label', None) is not None:
            voxel_label = data_dict['voxel_label'].copy().squeeze()
            id_choice = list(config.CLASS_ID.values())
            id_choice = list(set(np.unique(voxel_label).tolist()) & set(id_choice))
            class_choice = [key for key, val in config.CLASS_ID.items() if val in id_choice]
            ignore_zero_box = config.get('IGNORE_ZERO_BOX', False)
            ignore_empty_in_box = ('IGNORE_EMPTY_IN_BOX' in config) and config.IGNORE_EMPTY_IN_BOX.ENABLE

            H, W = voxel_label.shape[-2:]
            location, dimensions, rotation_y = data_dict['metadata']['annos']['location'].reshape(-1, 3), data_dict['metadata']['annos']['dimensions'].reshape(-1, 3), data_dict['metadata']['annos']['rotation_y'].reshape(-1, 1)
            names = data_dict['metadata']['annos']['name']
            num_points_in_box = data_dict['metadata']['annos']['num_points_in_box']
            lidar_aug_matrix = data_dict['lidar_aug_matrix']
            boxes_ori = np.concatenate([location, dimensions[:, (2, 0, 1)], rotation_y], axis=1)  # x y bottom_z w l h yaw

            if len(id_choice) > 0 and boxes_ori.shape[0] > 0:
                mask = np.array([name in class_choice for name in names], dtype=np.bool_)
                if config.MODE != 'zerobox':
                    mask &= (num_points_in_box > 0)
                boxes = boxes_ori[mask]
                corner_points_xy = boxes_to_bev_points(boxes, lidar_aug_matrix)
                voxel_label_neg = voxel_label == 0
                for id_ in id_choice:
                    voxel_label_id = common_utils.get_convexHull((voxel_label == id_).astype(np.uint8), corner_points_xy, config)
                    pos_mask = (voxel_label_id > 0) & voxel_label_neg
                    # if not self.training:  # 测试时把补的地方设置成ignore
                    #     id_ = 255
                    voxel_label[pos_mask] = voxel_label_id[pos_mask] * id_

            if ignore_empty_in_box and boxes_ori.shape[0] > 0:
                maks_range = config.IGNORE_EMPTY_IN_BOX.RANGE
                if maks_range is not None:
                    x1, y1, x2, y2 = maks_range
                nonempty_boxes = boxes_ori[num_points_in_box > 0]
                nonempty_corner_points_xy = boxes_to_bev_points(nonempty_boxes, lidar_aug_matrix)
                voxel_label_ori = voxel_label.copy()
                if maks_range is None:
                    voxel_label = common_utils.fill_box_in_bev_label(voxel_label, nonempty_corner_points_xy, config)
                else:
                    voxel_label[y1:y2, x1:x2] = common_utils.fill_box_in_bev_label(voxel_label[y1:y2, x1:x2], nonempty_corner_points_xy, config)
                pos_mask = voxel_label_ori > 0
                voxel_label[pos_mask] = voxel_label_ori[pos_mask]

            if ignore_zero_box and boxes_ori.shape[0] > 0:
                zero_boxes = boxes_ori[num_points_in_box == 0]
                zero_corner_points_xy = boxes_to_bev_points(zero_boxes, lidar_aug_matrix)
                voxel_label = common_utils.fill_box_in_bev_label(voxel_label, zero_corner_points_xy, config)

            data_dict['voxel_label'] = voxel_label[None, ...]
        return data_dict

    def calculate_grid_size(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.calculate_grid_size, config=config)
        return data_dict

    def downsample_depth_map(self, data_dict=None, config=None):
        if data_dict is None:
            self.depth_downsample_factor = config.DOWNSAMPLE_FACTOR
            return partial(self.downsample_depth_map, config=config)

        data_dict['depth_maps'] = transform.downscale_local_mean(
            image=data_dict['depth_maps'],
            factors=(self.depth_downsample_factor, self.depth_downsample_factor)
        )
        return data_dict
    
    def image_normalize(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_normalize, config=config)
        mean = config.mean
        std = config.std
        compose = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )
        data_dict["camera_imgs"] = [compose(img) for img in data_dict["camera_imgs"]]
        return data_dict
    
    def image_calibrate(self,data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_calibrate, config=config)
        img_process_infos = data_dict['img_process_infos']
        transforms = []
        for img_process_info in img_process_infos:
            resize, crop, flip, rotate = img_process_info

            rotation = torch.eye(2)
            translation = torch.zeros(2)
            # post-homography transformation
            rotation *= resize
            translation -= torch.Tensor(crop[:2])
            if flip:
                A = torch.Tensor([[-1, 0], [0, 1]])
                b = torch.Tensor([crop[2] - crop[0], 0])
                rotation = A.matmul(rotation)
                translation = A.matmul(translation) + b
            theta = rotate / 180 * np.pi
            A = torch.Tensor(
                [
                    [np.cos(theta), np.sin(theta)],
                    [-np.sin(theta), np.cos(theta)],
                ]
            )
            b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
            b = A.matmul(-b) + b
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            transforms.append(transform.numpy())
        data_dict["img_aug_matrix"] = transforms
        return data_dict

    def forward(self, data_dict, disable_keys=None):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                label: (N, 1)
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            if disable_keys is not None and cur_processor.keywords['config']['NAME'] in disable_keys:
                continue
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
