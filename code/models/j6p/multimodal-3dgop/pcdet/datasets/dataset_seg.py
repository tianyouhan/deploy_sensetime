from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as torch_data

from ..utils import common_utils
from .augmentor_seg.data_augmentor import DataAugmentor
from .processor_seg.data_processor import DataProcessor
from .processor_seg.point_feature_encoder import PointFeatureEncoder


class DatasetTemplateSeg(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.cur_epoch = 0
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        self.logger = logger
        if self.dataset_cfg is None or class_names is None:
            return

        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        self.data_augmentor = DataAugmentor(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger
        ) if self.training else None
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range,
            training=self.training, num_point_features=self.point_feature_encoder.num_point_features
        )

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

        if hasattr(self.data_processor, "depth_downsample_factor"):
            self.depth_downsample_factor = self.data_processor.depth_downsample_factor
        else:
            self.depth_downsample_factor = None
            
    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        try:
            del d['logger']
        except:
            pass
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def update_epoch(self, epoch):
        self.cur_epoch = epoch

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7 or 9), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            box_dim = 9 if self.dataset_cfg.get('TRAIN_WITH_SPEED', False) else 7
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'pred_boxes': np.zeros([num_samples, box_dim]), 'pred_labels': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict
            
            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['pred_boxes'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            if 'metadata' in batch_dict:
                single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    def set_lidar_aug_matrix(self, data_dict):
        """
            Get lidar augment matrix (4 x 4), which are used to recover orig point coordinates.
        """
        lidar_aug_matrix = np.eye(4)
        if 'flip_x' in data_dict.keys():
            flip_x = data_dict['flip_x']
            if flip_x:
                lidar_aug_matrix[:3,:3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ lidar_aug_matrix[:3,:3]
        if 'flip_y' in data_dict.keys():
            flip_y = data_dict['flip_y']
            if flip_y:
                lidar_aug_matrix[:3,:3] = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ lidar_aug_matrix[:3,:3]
        if 'noise_rot' in data_dict.keys():
            noise_rot = data_dict['noise_rot']
            # lidar_aug_matrix[:3,:3] = common_utils.angle2matrix(torch.tensor(noise_rot)) @ lidar_aug_matrix[:3,:3]  # ori setting
            lidar_aug_matrix[:3,:3] = common_utils.angle2matrix(torch.tensor(noise_rot)) @ torch.tensor(lidar_aug_matrix[:3,:3], dtype=torch.float32)
        if 'noise_scale' in data_dict.keys():
            noise_scale = data_dict['noise_scale']
            lidar_aug_matrix[:3,:3] *= noise_scale
        if 'noise_translate' in data_dict.keys():
            noise_translate = data_dict['noise_translate']
            lidar_aug_matrix[:3,3:4] = noise_translate.T
        data_dict['lidar_aug_matrix'] = lidar_aug_matrix
        return data_dict
    
    def get_task_label(self, data_dict, force_replace=False):
        if self.dataset_cfg.get('TASKS', None) is None:
            return data_dict
        if 'voxel_label' not in data_dict:
            return data_dict
        for task in self.dataset_cfg.TASKS:
            if force_replace or f'{task}_label' not in data_dict:
                data_dict[f'{task}_label'] = data_dict['voxel_label']
        data_dict.pop('voxel_label')
        
        return data_dict
    
    def transform_coordinate(self, data_dict=None):
        if self.dataset_cfg.get('TRANSFORM_COORD', False):
            points = data_dict['points']
            trans_mat = np.array(self.dataset_cfg.TRANSFORM_COORD.TRANS_MAT, dtype=np.float32)
            trans_points = np.hstack((points[:, 0:3], np.ones_like(points[:, 0, None]))) @ trans_mat.T
            data_dict['points'][:, 0:3] = trans_points[:, 0:3]
        return data_dict

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: optional, (N, 3 + C_in)
                label: optional, (N, 1)
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        data_dict = self.transform_coordinate(data_dict)

        if self.training:
            assert 'label' in data_dict, 'label should be provided for training'
            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                }
            )

        data_dict = self.set_lidar_aug_matrix(data_dict)

        if data_dict.get('points', None) is not None:
            data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        data_dict = self.get_task_label(data_dict)
        if 'gt_boxes' in data_dict and data_dict['gt_boxes'] is None:
            data_dict.pop('gt_boxes')
        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        batch_size_ratio = 1

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    if isinstance(val[0], list):
                        batch_size_ratio = len(val[0])
                        val = [i for item in val for i in item]
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    if isinstance(val[0], list):
                        val =  [i for item in val for i in item]
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes', 'gt_boxes_gop']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['gt_boxes2d']:
                    max_boxes = 0
                    max_boxes = max([len(x) for x in val])
                    batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        if val[k].size > 0:
                            batch_boxes2d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_boxes2d
                elif key in ["images", "depth_maps"]:
                    # Get largest image size (H, W)
                    max_h = 0
                    max_w = 0
                    for image in val:
                        max_h = max(max_h, image.shape[0])
                        max_w = max(max_w, image.shape[1])

                    # Change size of images
                    images = []
                    for image in val:
                        pad_h = common_utils.get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                        pad_w = common_utils.get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                        pad_width = (pad_h, pad_w)
                        pad_value = 0

                        if key == "images":
                            pad_width = (pad_h, pad_w, (0, 0))
                        elif key == "depth_maps":
                            pad_width = (pad_h, pad_w)

                        image_pad = np.pad(image,
                                           pad_width=pad_width,
                                           mode='constant',
                                           constant_values=pad_value)

                        images.append(image_pad)
                    ret[key] = np.stack(images, axis=0)
                elif key in ['calib', 'img_process_infos', 'fuse_points', 'fuse_label', 'label']:
                    ret[key] = val
                elif key in ["points_2d"]:
                    max_len = max([len(_val) for _val in val])
                    pad_value = 0
                    points = []
                    for _points in val:
                        pad_width = ((0, max_len-len(_points)), (0,0))
                        points_pad = np.pad(_points,
                                pad_width=pad_width,
                                mode='constant',
                                constant_values=pad_value)
                        points.append(points_pad)
                    ret[key] = np.stack(points, axis=0)
                elif key in ['camera_imgs']:
                    ret[key] = torch.stack([torch.stack(imgs,dim=0) for imgs in val],dim=0)
                elif key in ['camera_intrinsics', 'lidar2image', 'camera2lidar', 'lidar2camera', 'img_aug_matrix', 'lidar_aug_matrix']:
                    ret[key] = np.stack(val, axis=0)
                elif type(val[0]) is np.ndarray:
                    if all([ele.shape == val[0].shape for ele in val]):
                        ret[key] = np.stack(val, axis=0)
                    else:
                        ret[key] = val
                else:
                    ret[key] = val
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size * batch_size_ratio
        return ret
