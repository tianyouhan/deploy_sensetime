import copy
import pickle
from pathlib import Path
import os
import json
import random
import numpy as np

from ...utils import pose_utils, ceph_utils, freespace_utils, common_utils
from ..dataset_seg import DatasetTemplateSeg
from PIL import Image
import cv2
import torch
import pcdet.utils.pp_heavy.box_np_ops as box_np_ops
import mmcv
from pcdet.utils.common_utils import save_np
PATH_MAPPING = dict({
                "/mnt/": "aoss-zhc-v2:s3://dcp36_lustre_aoss_v2/",
                "s3://aoss-test-data": "aoss-test-data:s3://aoss-test-data",
                "s3:/aoss-test-data": "aoss-test-data:s3://aoss-test-data",
            })
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
}
class GACBaiduDatasetPVB(DatasetTemplateSeg):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH))
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.use_ceph = False if not self.dataset_cfg.get('CEPH', False) else self.dataset_cfg.CEPH.ENABLE
        if self.use_ceph:
            self.ceph_cfg = self.dataset_cfg.CEPH
            self.ceph_client = ceph_utils.ceph_init()
        else:
            self.ceph_client = None

        if self.dataset_cfg.get('BOX_CONFIG', None) is not None:
            self.box_cfg = self.dataset_cfg.BOX_CONFIG
            self.add_ignore = self.box_cfg.get('ADD_IGNORE', False)

        else:
            self.box_cfg = None

        self.camera_config = self.dataset_cfg.get('CAMERA_CONFIG', None)
        self.lidar_config = self.dataset_cfg.get('LIDAR_CONFIG', None)
        if self.camera_config is not None:
            self.use_camera = self.camera_config.get('USE_CAMERA', True)
            self.camera_image_config = self.camera_config.IMAGE
            self.data_source = self.camera_config.get('DATA_SOURCE', 'gacbaidu')
        else:
            self.use_camera = False
        self.initialized = False
        self.include_data(self.mode)
        if self.training and self.dataset_cfg.get('DATA_RESAMPLE', False):
            self.data_resample(self.dataset_cfg.DATA_RESAMPLE)

        if self.dataset_cfg.get('SEQUENTIAL_DATA', False):
            self.get_sequential_data(self.dataset_cfg.SEQUENTIAL_DATA)
            self.get_interval_data(self.dataset_cfg.SEQUENTIAL_DATA)
            self.seq_data_cfg = self.dataset_cfg.SEQUENTIAL_DATA
            self.seq_data_enable = True
        else:
            self.seq_data_enable = False

        if self.dataset_cfg.get('LABEL_MAPPING', False):
            self.label_mapping_dicts = {int(k):v for k, v in self.dataset_cfg['LABEL_MAPPING'].items()}
            self.label_mapping = np.vectorize(self.label_mapping_dicts.__getitem__)
        else:
            self.label_mapping = None

        if self.dataset_cfg.get('POINT_DIM_ZERO', False):
            self.point_dim_zero_idx = [idx for idx, val in enumerate(self.dataset_cfg.POINT_DIM_ZERO) if val == 0]
        else:
            self.point_dim_zero_idx = None

        if self.dataset_cfg.get('FREESPACE_CONFIG', False) and self.dataset_cfg.FREESPACE_CONFIG.ENABLE:
            self.freespace_cfg = self.dataset_cfg.FREESPACE_CONFIG
        else:
            self.freespace_cfg = None

        self.pp_heavy = self.dataset_cfg.get('PP_HEAVY', False)
        if self.pp_heavy:
            self.process_mode = self.dataset_cfg.PP_HEAVY.get('PROCESS_MODE', 1)
            self.test_limit_range = self.dataset_cfg.PP_HEAVY.get('TEST_LIMIT_RANGE', None)
        if self.pp_heavy:
            self.prepare_pp_heavy()
        self.cnt = 0
    def prepare_pp_heavy(self):
        from pcdet.utils.pp_heavy.target_assigner import AnchorGeneratorRange, AnchorGeneratorStride, TargetAssigner
        import pcdet.utils.pp_heavy.box_coder as box_coder_utils
        voxel_size = self.pp_heavy.VOXEL_SIZE
        pc_range = self.dataset_cfg.POINT_CLOUD_RANGE
        anchor_cfg = self.pp_heavy.TARGET_ASSIGNER.ANCHOR_GENERATOR
        anchor_generators = []
        for a_cfg in anchor_cfg:
            if a_cfg['type'] == "anchor_generator_stride":
                anchor_generator = AnchorGeneratorStride(
                    sizes=a_cfg['sizes'],
                    anchor_strides=a_cfg['strides'],
                    anchor_offsets=a_cfg['offsets'],
                    rotations=a_cfg['rotations'],
                    match_threshold=a_cfg['matched_threshold'],
                    unmatch_threshold=a_cfg['unmatched_threshold'],
                    class_name=a_cfg['class_name'],
                    feature_map_scale_factor=a_cfg['feature_map_scale_factor'],
                )
            elif a_cfg['type'] == "anchor_generator_range":
                anchor_generator = AnchorGeneratorRange(
                    anchor_ranges=a_cfg['anchor_range'],
                    sizes=a_cfg['sizes'],
                    rotations=a_cfg['rotations'],
                    class_name=a_cfg['class_name'],
                    match_threshold=a_cfg['matched_threshold'],
                    unmatch_threshold=a_cfg['unmatched_threshold']
                )
            else:
                raise Exception(f"\"{a_cfg[type]}\" is not a valid generator type!!")
            anchor_generators.append(anchor_generator)
        
        self.box_coder = getattr(box_coder_utils, self.pp_heavy.BOX_CODER)()
        self.target_assigner = TargetAssigner(
            anchor_generators=anchor_generators,
            pos_fraction=self.pp_heavy.TARGET_ASSIGNER.SAMPLE_POS_FRACTION,
            sample_size=self.pp_heavy.TARGET_ASSIGNER.SAMPLE_SIZE,
            region_similarity_fn_name=self.pp_heavy.TARGET_ASSIGNER.REGION_SIMILARITY_FN,
            box_coder=self.box_coder
        )
        out_size_factor = self.pp_heavy.DOWNSAMPLE_FACTOR #cfg.RPN_STAGE.DOWNSAMPLE_FACTOR
        out_size_factor = int(out_size_factor)
        assert out_size_factor > 0
        grid_size = np.array([(pc_range[i+3]-pc_range[i])/voxel_size[i] for i in range(len(voxel_size))])
        feature_map_size = grid_size[:2] // out_size_factor
        feature_map_size = [*feature_map_size, 1][::-1]
        ret = self.target_assigner.generate_anchors(feature_map_size)
        anchors_dict = self.target_assigner.generate_anchors_dict(feature_map_size)
        anchors = ret["anchors"].reshape([-1, 7])
        self.anchor_cache = {
            "anchors": anchors,
            "anchors_dict": anchors_dict,
        }   
        return
    
    def prepare_pp_heavy_gt(self, data_dict):
        anchors = self.anchor_cache["anchors"]
        anchors_dict = self.anchor_cache["anchors_dict"]
        data_dict["anchors"] = anchors
        anchors_mask = None
        if self.training:
            gt_classes = data_dict['gt_boxes'][:,7].astype(np.int32)
            gt_names = np.array(self.class_names)[gt_classes-1]
            targets_dict = self.target_assigner.assign_v2(anchors_dict, data_dict['gt_boxes'][:,0:7], anchors_mask,
                                                        gt_classes=gt_classes, gt_names=gt_names,
                                                        multihead=True)
            data_dict.update({
                'box_labels': targets_dict['labels'],
                'reg_targets': targets_dict['bbox_targets'],
                'reg_weights': targets_dict['bbox_outside_weights'],
            })
            data_dict['backward'] = np.array([0], dtype=np.int64)
        return data_dict

    def include_data(self, mode):
        self.logger.info('Loading dataset')
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            if type(info_path) is not list and os.path.isfile(info_path):
                info_path = [info_path]
            if type(info_path) is list:
                infos = None
                for path in info_path:
                    info = pickle.load(open(path, 'rb'))
                    if infos is None:
                        infos = info.copy()
                    else:
                        if type(info) is list:
                            infos += info
                        elif type(info) is dict:
                            infos.update(info)
                        else:
                            print('Warning! Not support pkl type')
            elif os.path.isdir(info_path):
                infos = [{'velodyne_path':os.path.join(info_path, ele)} for ele in os.listdir(info_path)]
        self.logger.info('Total samples for dataset: %d' % (len(infos)))
        # if 'EXCLUDE_PATH' in self.dataset_cfg:
        #     exclude_path = self.dataset_cfg.EXCLUDE_PATH
        #     exclude_list = json.load(open(exclude_path, 'r'))
        #     # print(exclude_list)
        #     for key in exclude_list:
        #         if key in infos:
        #             infos.pop(key)
        #     self.logger.info('after exclusion, samples for dataset: %d' % (len(infos)))
        self.infos = dict(list(infos.items())[:500])
    
    def data_resample(self, cfg):
        resample_seqs = set()
        for path, top_num in cfg.SEQ_LIST:
            lines = open(path, 'r').readlines()
            for num, line in enumerate(lines):
                if num < int(top_num) and 'AutoCollect' in line:
                    resample_seqs.add(line.split()[0])
        resample_infos_list = []
        for key, val in self.infos.items():
            if key in resample_seqs:
                for num in range(cfg.RESAMPLE_NUMS):
                    resample_infos_list.append([key + f'_resample_{num}', val])
        print('Resample seqs:', len(resample_infos_list), flush=True)
        random.shuffle(resample_infos_list)
        for key, val in resample_infos_list:
            self.infos[key] = val
        
    def get_sequential_data(self, cfg):
        ref_frame = cfg.get('REF_FRAME', None)  # ref_frame should not include key frame
        use_pose = cfg.get('USE_POSE', False)  # pose used to merge cloud

        infos_key, self.infos_seq_id_full, self.infos_ref_frame_full, self.infos_pose_full = [], [], [], []
        seq_list = [key for key in self.infos.keys()]
        # random.seed(1)
        # random.shuffle(seq_list)
        # self.pick_seq = seq_list[0:40]
        for key, val in self.infos.items():
            # if key not in ['2023_10_29_02_14_31_AutoCollect_0']:
            #     continue
            # if key not in self.pick_seq:
            #     continue
            info_id = []
            if use_pose:  # pose txt is same in a sequence
                pose_path = val[0]['lidar_pose_path']
                if self.use_ceph:
                    pose_path = self.mapping_path_to_ceph(pose_path)
                pose_lines = ceph_utils.ceph_read(pose_path, None, self.use_ceph, self.ceph_client)
                pose = pose_utils.get_lidar_pose(pose_lines)

            for i, info in enumerate(val):
                if use_pose:
                    self.infos_pose_full.append(pose)
                infos_key.append(info)
                info_id.append(i)
                if ref_frame != None:
                    info_ref = []
                    for ref_id in ref_frame:
                        info_ref.append(val[min(max(0, ref_id + i), len(val) - 1)])
                    self.infos_ref_frame_full.append(info_ref)
            self.infos_seq_id_full.append(info_id)
        self.infos_full = infos_key
    
    def get_interval_data(self, cfg, cur_epoch=0):
        if 'INTERVAL' in cfg:
            interval = cfg.INTERVAL['train'] if self.training else cfg.INTERVAL['test']
        else:
            interval = 1
        remain = cur_epoch % interval  # dynamic update start point
        self.infos = self.infos_full[remain:][::interval]
        self.infos_ref_frame = self.infos_ref_frame_full[remain:][::interval]
        self.infos_pose = self.infos_pose_full[remain:][::interval]
        self.infos_seq_id = [ele[remain:][::interval] for ele in self.infos_seq_id_full]

    def mapping_path_to_ceph(self, path):
        new_path = path
        if path is None or path == "":
            return None
        for key, val in PATH_MAPPING.items():
            if key in path:
                new_path = path.replace(key, val)
                break
        return new_path

    def get_path(self, info):
        lidar_path, label_path =  info['velodyne_path'], info.get('label_path', None)
        if self.use_ceph:
            lidar_path, label_path = self.mapping_path_to_ceph(lidar_path), self.mapping_path_to_ceph(label_path)
        if label_path is None:
            label_file_exist = False
        else:
            label_file_exist = os.path.exists(label_path)
        if self.use_ceph:
            if not 's3://' in lidar_path:  # if not ceph url
                if self.training:
                    ceph_points_path = self.ceph_cfg.TRAIN_CEPH_POINTS_PATH
                    ceph_labels_path = self.ceph_cfg.TRAIN_CEPH_LABELS_PATH
                else:
                    ceph_points_path = self.ceph_cfg.VAL_CEPH_POINTS_PATH
                    ceph_labels_path = self.ceph_cfg.VAL_CEPH_LABELS_PATH
                points_name, label_name = os.path.split(lidar_path)[1], os.path.split(label_path)[1]
                lidar_path = ceph_utils.ceph_url(prefix=ceph_points_path, filename=points_name)
                label_path = ceph_utils.ceph_url(prefix=ceph_labels_path, filename=label_name)
            if label_path is not None and label_path:
                # print(label_path)
                label_file_exist = self.ceph_client.contains(label_path)
        return lidar_path, label_path, label_file_exist

    def get_points_label_base(self, info, check_points=True):
        def remove_ego_points_label(points, label, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask], label[mask]
        lidar_path, label_path, label_file_exist = self.get_path(info)
        # print(lidar_path)
        points = ceph_utils.ceph_read(lidar_path, np.float32, self.use_ceph, self.ceph_client).reshape((-1, 4))
        if self.lidar_config is not None:
            normalize_calib = self.lidar_config.get('NORM_CALIB', None)
        else:
            normalize_calib = None
            # print('norm calib None')
        if normalize_calib == 'A02-J6M':
            points[:, 2] -= 1.6
        if normalize_calib == 'EPI-MDC':
            points[:, 2] -= 1.6
        if check_points and points.shape[0] < 60000:  # 去除坏点云
            index = np.random.choice(len(self.infos))
            return self.get_points_label_base(self.infos[index], check_points)
        
        if label_file_exist:
            label = ceph_utils.ceph_read(label_path, np.int32, self.use_ceph, self.ceph_client).reshape((-1, 1))
        else:
            label = np.zeros((points.shape[0], 1), dtype=np.int32)
            if label_path != None:
                print(f'Warning! {label_path} not existed')

        if self.label_mapping is not None:
            label = self.label_mapping(label).astype(label.dtype)

        if self.point_dim_zero_idx is not None:
            points[:, self.point_dim_zero_idx] = 0

        points, label = remove_ego_points_label(points, label, center_radius=1.7)
        return points, label, info
    
    def get_points_label_multiframe(self, infos, idx):
        key_frame_name = infos[idx]['frame_name']
        key_pose = self.infos_pose[idx][key_frame_name]
        key_points, key_label, _ = self.get_points_label_base(infos[idx], True)
        ref_points, ref_label, ref_name = [], [], []
        for info_ref in self.infos_ref_frame[idx]:
            ref_frame_name = info_ref['frame_name']
            ref_pose = self.infos_pose[idx][ref_frame_name]
            points, label, _ = self.get_points_label_base(info_ref, False)
            points = pose_utils.warp_cloud(points, src_pose=ref_pose, tar_pose=key_pose)  # src_pose to tar_pose
            ref_points.append(points)
            ref_label.append(label)
            ref_name.append(ref_frame_name)
        fuse_points = np.concatenate(ref_points + [key_points], axis=0)  # key frame idx is -1
        fuse_label = np.concatenate(ref_label + [key_label], axis=0)
        fuse_names = ref_name + [key_frame_name]
        if any([points.shape[0] == 0 for points in fuse_points]):
            print(f'Warning! Broken points in:{fuse_names}')
            index = np.random.choice(len(self.infos))
            return self.get_points_label_multiframe(self.infos[index], index)
        return key_points, key_label, fuse_points, fuse_label, infos[idx]

    def get_points_label_info(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)
        info = copy.deepcopy(self.infos[index])

        if self.seq_data_enable and len(self.infos_ref_frame) > 0 and index < len(self.infos_ref_frame):
            points, label, fuse_points, fuse_label, info = self.get_points_label_multiframe(self.infos, index)
        else:
            points, label, info = self.get_points_label_base(info)
            fuse_points, fuse_label = None, None

        return points, label, fuse_points, fuse_label, info
    
    def get_multi_frame_in_out(self, data_dict):
        mode = self.seq_data_cfg.MODE
        if not (mode['multi_frame_in'] or mode['multi_frame_out']):
            return data_dict
        else:
            ori_dict = data_dict.copy()
            aug_R, aug_T = data_dict['lidar_aug_matrix'][0:3, 0:3], data_dict['lidar_aug_matrix'][0:3, -1]
            data_dict['fuse_points'][:, 0:3] = data_dict['fuse_points'][:, 0:3] @ aug_R.T + aug_T
            data_dict['points'] = data_dict['fuse_points']
            data_dict['label'] = data_dict['fuse_label']
            task_label = {}
            for key in list(data_dict.keys()):
                if '_label' in key and key not in ['fuse_label']:
                    task_label[key] = data_dict.pop(key)
            data_dict = self.point_feature_encoder.forward(data_dict)
            data_dict = self.data_processor.forward(data_dict, disable_keys=['image_calibrate', 'image_normalize'])
            if mode['multi_frame_out']:
                data_dict = self.get_task_label(data_dict)
            else:
                data_dict.update(task_label)
                data_dict.pop('voxel_label')
            if not mode['multi_frame_in']:
                replace_keys = ['points', 'label', 'voxels', 'voxel_coords', 'voxel_num_points']
                for key in replace_keys:
                    if key in data_dict:
                        data_dict[key] = ori_dict[key]
        return data_dict
    
    def get_freespace_label(self, data_dict, info):
        assert len(info['drivable']) > 0, 'Warning! Drivable list not exist'
        drivable_list, non_drivable_list = info['drivable'], info.get('non_drivable', [])
        lidar_aug_matrix = data_dict['lidar_aug_matrix'].astype(np.float32)
        xy_R, xy_T = lidar_aug_matrix[0:2, 0:2], lidar_aug_matrix[None, 0:2, -1]
        ego_flag, voxel_size, pc_range = self.freespace_cfg.EGO_FLAG, self.freespace_cfg.VOXEL_SIZE, self.dataset_cfg.POINT_CLOUD_RANGE

        drivable_list = [points @ xy_R.T + xy_T for points in drivable_list]
        drivable_mask = freespace_utils.create_drivable_mask(drivable_list, voxel_size, pc_range, ego_flag)

        if len(non_drivable_list) > 0:
            non_drivable_list = [points @ xy_R.T + xy_T for points in non_drivable_list]
            non_drivable_mask = freespace_utils.create_drivable_mask(non_drivable_list, voxel_size, pc_range, False)
            drivable_mask[non_drivable_mask > 0] = 0

        if self.freespace_cfg.get('ADJUST_EDGE_WIDTH', None) is not None:  # adjust drivable area
            pixel_width = self.freespace_cfg.ADJUST_EDGE_WIDTH
            if type(pixel_width) in [list, tuple]:
                pixel_width = pixel_width[0] if self.training else pixel_width[1]  # train, val
            kernel = np.ones((abs(pixel_width) * 2 + 1, abs(pixel_width) * 2 + 1), dtype=np.uint8)
            if pixel_width > 0:
                drivable_mask = cv2.dilate(drivable_mask, kernel, iterations=1)
            elif pixel_width < 0:
                drivable_mask = cv2.erode(drivable_mask, kernel, iterations=1)

        data_dict['freespace_label'] = drivable_mask[None, ...]  # (h, w) to (1, h, w)
        return data_dict
    
    def get_gt_box(self, input_dict, info):
        annos = info['annos'].copy()
        data_source = info['data_source'] if 'data_source' in info else 'A02'
        # drop class out of class_names
        if data_source == 'atx':
            l = []
            for cur_cls in annos['name']:
                if cur_cls in CLS_MAPPING:
                    l.append(CLS_MAPPING[cur_cls])
                else:
                    l.append(cur_cls)
                    print(cur_cls)
            annos['name'] = np.array(l)
        keep_indices = [i for i, x in enumerate(annos['name']) if x in self.class_names]
        for key in annos.keys():
            if type(annos[key]) is not np.ndarray:
                if annos[key] is None:
                    annos[key] = np.array([])
                else:
                    annos[key] = np.array(annos[key])
            annos[key] = annos[key][keep_indices]
        # 广汽车yaw角:y负开始顺时针，y==>x，pcdet为:y负开始逆时针，x==>y，这里不做变化，因此需要'random_world_rotation'中做变化
        location, dimensions, rotation_y = annos['location'].reshape(-1, 3), annos['dimensions'].reshape(-1, 3), annos['rotation_y'].reshape(-1, 1)
        dimensions = dimensions[:, [2, 0, 1]]  # lhw > wlh
        # 限制yaw在[0, 2pi]
        rotation_y = common_utils.limit_period(rotation_y, offset=0.5, period=2*np.pi)
        if data_source == 'atx':
            rotation_y *= -1
        gt_boxes_lidar = np.concatenate([location, dimensions, rotation_y], axis=1).astype(np.float32)  # x y z w l h yaw
        
        # filter box
        mask = None
        if self.box_cfg.FILTER_EMPTY_BOXES[self.mode] and gt_boxes_lidar.shape[0] > 0:
            min_filter_points = self.box_cfg.get('MIN_FILTER_POINTS', 0)
            mask = (annos['num_points_in_box'] > min_filter_points)  # filter empty boxes
            
        if 'FILTER_UNNORMAL_SIZE_BOXES' in self.box_cfg and self.box_cfg.FILTER_UNNORMAL_SIZE_BOXES[self.mode] and gt_boxes_lidar.shape[0] > 0:  # filter unnormal size boxes
            cur_W, cur_L, cur_H = gt_boxes_lidar[:,3], gt_boxes_lidar[:,4], gt_boxes_lidar[:,5]
            
            if mask is None:
                mask = np.ones_like(annos['name'], dtype=np.bool)
            if 'barrier' in annos['name']:
                class_mask = annos['name'] == 'barrier'
                cur_mask = (cur_W < 3.0) & (cur_L < 3.0) & class_mask
                mask[class_mask] &= cur_mask[class_mask]

        if mask is not None:
            annos['name'] = annos['name'][mask]
            gt_boxes_lidar = gt_boxes_lidar[mask]
            annos['num_points_in_box'] = annos['num_points_in_box'][mask]
            if 'blocked_label' in annos:
                annos['blocked_label'] = annos['blocked_label'][mask]

        input_dict.update({
            'gt_names': annos['name'],
            'gt_boxes': gt_boxes_lidar,
            'num_points_in_gt': annos['num_points_in_box'],
        })
        if 'blocked_label' in annos:
            input_dict.update({'blocked_label': annos['blocked_label']})
        return input_dict
    
    def crop_image(self, input_dict):
        imgs = input_dict["camera_imgs"]
        img_process_infos = []
        crop_images = []
        crop_coff = self.camera_config.IMAGE.get('CROP_COFF', [0.5 for _ in range(len(imgs))])
        if type(crop_coff) not in [list, tuple]:
            crop_coff = [crop_coff for _ in range(len(imgs))]
        update_resize = self.camera_config.IMAGE.get('UPDATE_RESIZE', None)
        center_crop = self.camera_config.IMAGE.CENTER_CROP['train'] if self.training else self.camera_config.IMAGE.CENTER_CROP['test']
        for idx, img in enumerate(imgs):
            W, H = img.size
            if self.training == True:
                fH, fW = self.camera_image_config.FINAL_DIM
                if update_resize is not None and update_resize[idx] is not None:
                    resize_lim = update_resize[idx]
                else:
                    resize_lim = self.camera_image_config.RESIZE_LIM_TRAIN
                resize = np.random.uniform(*resize_lim)
                resize_dims = (int(W * resize), int(H * resize))
                newW, newH = resize_dims
                crop_h = int(crop_coff[idx] * (newH - fH))
                if center_crop:
                    crop_w = int(max(0, newW - fW) / 2)
                else:
                    crop_w = int(np.random.uniform(0, max(0, newW - fW)))
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            else:
                fH, fW = self.camera_image_config.FINAL_DIM
                if update_resize is not None and update_resize[idx] is not None:
                    resize_lim = update_resize[idx]
                else:
                    resize_lim = self.camera_image_config.RESIZE_LIM_TEST
                resize = np.mean(resize_lim)
                resize_dims = (int(W * resize), int(H * resize))
                newW, newH = resize_dims
                crop_h = int(crop_coff[idx] * (newH - fH))
                crop_w = int(max(0, newW - fW) / 2)
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            
            # reisze and crop image
            img = img.resize(resize_dims)
            img = img.crop(crop)
            crop_images.append(img)
            img_process_infos.append([resize, crop, False, 0])
        
        input_dict['img_process_infos'] = img_process_infos
        input_dict['camera_imgs'] = crop_images
    
        # for i, image in enumerate(crop_images):
        #     image.save(f'{i}.png')
        # from ipdb import set_trace
        # set_trace()
        return input_dict

    def load_intrinsics_calib(self, calib_json_path):
        if self.use_ceph:
            calib_json_path = self.mapping_path_to_ceph(calib_json_path)
        intrinsic_calib = ceph_utils.ceph_read(calib_json_path, None, self.use_ceph, self.ceph_client)
        key = [ele for ele in intrinsic_calib.keys()][0]
        camera_intrinsic_dict = intrinsic_calib[key]

        # camera_dist = np.array(camera_intrinsic_dict['param']['cam_dist']['data'][0])
        if 'cam_K_new' in camera_intrinsic_dict['param']:
            camera_intrinsic = np.array(camera_intrinsic_dict['param']['cam_K_new']['data'])
        elif 'cam_k_new_resize' in camera_intrinsic_dict['param']:
            # print('load intrinsic using cam_k_new_resize')
            camera_intrinsic = np.array(camera_intrinsic_dict['param']['cam_k_new_resize']['data'])
        else:
            camera_intrinsic = np.array(camera_intrinsic_dict['param']['cam_K']['data'])
        return camera_intrinsic
        
    def load_extrinsic_calib(self, calib_json_path, normalize_calib, lidar2camera=True):
        if self.use_ceph:
            calib_json_path = self.mapping_path_to_ceph(calib_json_path)
        extrinsic_calib = ceph_utils.ceph_read(calib_json_path, None, self.use_ceph, self.ceph_client)
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
    
    def GetVirtualIntrinsic(self, src_int, image_size, virtual_image_size,
                         fov, is_change_fov=True, 
                         is_change_cxcy=True):
        virtual_w, virtual_h = virtual_image_size
        dst_int = np.eye(3)
        src_w, src_h = image_size
        resize_x = src_w / virtual_w
        resize_y = src_h / virtual_h
        if is_change_cxcy:
            cx = virtual_w / 2.0
            cy = virtual_h / 2.0
        else:
            cx = src_int[0][2] / resize_x
            cy = src_int[1][2] / resize_y
        if is_change_fov:
            fx = virtual_w / 2.0 / np.tan(fov / 2.0 * np.pi/180)
            fy = src_int[1][1] / src_int[0][0] * fx
        else:
            pass
        dst_int[0][0] = fx
        dst_int[1][1] = fy
        dst_int[0][2] = cx
        dst_int[1][2] = cy
        return dst_int
    
    def load_camera_info(self, input_dict, info):
        input_dict["image_paths"] = []
        input_dict["lidar2camera"] = []
        input_dict["lidar2image"] = []
        input_dict["camera_intrinsics"] = []
        input_dict["camera2lidar"] = []
        self.use_virtual = self.camera_config.get('USE_VIRTUAL', False)
        if self.use_virtual:
            self.fov_map = {
                'center_camera_fov30': 30,
                'center_camera_fov120': 105
            }
            input_dict["src_camera_intrinsics"] = []
        calib_path = info.get('calib_path', None)
        normalize_calib = self.camera_config.get('NORM_CALIB', None)
        if calib_path is None:
            calib_path = '/'.join(info['velodyne_path'].split('/')[:-2]) + '/calib'

        used_cameras = self.camera_config.IMAGE.get('VIEWS', None)
        img_path_dict = {}
        if used_cameras is not None:
            for camera_name in used_cameras:
                img_path_dict[camera_name] = info["img_path"][camera_name]
        else:
            img_path_dict = info["img_path"]

        if self.data_source in ['gacbaidu', 'top_center_lidar']:
            for camera_name, camera_path in img_path_dict.items():
                input_dict["image_paths"].append(camera_path)

                intrinsic_file = f'{camera_name}-intrinsic.json'
                extrinsic_file = f'top_center_lidar-to-{camera_name}-extrinsic.json'
                camera_intrinsics_path = os.path.join(calib_path, camera_name, intrinsic_file)
                camera_extrinsic_path = os.path.join(calib_path, 'top_center_lidar', extrinsic_file)

                # lidar to camera transform
                lidar2camera_rt = self.load_extrinsic_calib(camera_extrinsic_path, normalize_calib, lidar2camera=True)
                input_dict["lidar2camera"].append(lidar2camera_rt)

                # camera intrinsics
                camera_intrinsics = np.eye(4).astype(np.float32)
                camera_intrinsics_r = self.load_intrinsics_calib(camera_intrinsics_path)
                camera_intrinsics[:3, :3] = camera_intrinsics_r
                input_dict["camera_intrinsics"].append(camera_intrinsics)

                # lidar to image transform
                lidar2image = camera_intrinsics @ lidar2camera_rt
                input_dict["lidar2image"].append(lidar2image)

                # camera to lidar transform
                input_dict["camera2lidar"].append(np.linalg.inv(lidar2camera_rt))
        elif self.data_source in ['A02', 'concat_M1_lidar', 'A02-J6M']:
            for camera_name, camera_path in img_path_dict.items():
                input_dict["image_paths"].append(camera_path)

                intrinsic_file = f'{camera_name}-intrinsic.json'
                extrinsic_file = f'{camera_name}-to-car_center-extrinsic.json'
                camera_intrinsics_path = os.path.join(calib_path, camera_name, intrinsic_file)
                camera_extrinsic_path = os.path.join(calib_path, camera_name, extrinsic_file)

                # lidar to camera transform
                lidar2camera_rt = self.load_extrinsic_calib(camera_extrinsic_path, normalize_calib, lidar2camera=False)
                input_dict["lidar2camera"].append(lidar2camera_rt)

                # camera intrinsics
                camera_intrinsics = np.eye(4).astype(np.float32)
                camera_intrinsics_r = self.load_intrinsics_calib(camera_intrinsics_path)
                if self.use_virtual:
                    src_camera_intrinsics = np.eye(4).astype(np.float32)
                    src_camera_intrinsics[:3, :3] = camera_intrinsics_r.copy()
                    input_dict["src_camera_intrinsics"].append(src_camera_intrinsics)
                    image_size = (3840, 2160)
                    virtual_image_size = (1024, 576)
                    fov = self.fov_map[camera_name]
                    virt_int = self.GetVirtualIntrinsic(camera_intrinsics_r.copy(), image_size, virtual_image_size, fov)
                    camera_intrinsics_r = virt_int.copy()
                camera_intrinsics[:3, :3] = camera_intrinsics_r
                input_dict["camera_intrinsics"].append(camera_intrinsics)

                # lidar to image transform
                lidar2image = camera_intrinsics @ lidar2camera_rt
                input_dict["lidar2image"].append(lidar2image)
                if os.getenv("CALIB") == 'True':
                    print("os.getenv(CALIB_p)", os.getenv("CALIB"))
                    CALIB_path = os.getenv("CALIB_PATH")
                    save_dir = os.path.join(CALIB_path, camera_name, 'ints')
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_np(os.path.join(save_dir, "{}".format(self.cnt)), camera_intrinsics)
                    save_dir = os.path.join(CALIB_path, camera_name, 'exts')
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_np(os.path.join(save_dir, "{}".format(self.cnt)), lidar2camera_rt)
                    gt_boxes = input_dict['gt_boxes']
                    save_dir = os.path.join(CALIB_path, 'gt_boxes')
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_np(os.path.join(save_dir, "{}".format(self.cnt)), gt_boxes)
                # camera to lidar transform
                input_dict["camera2lidar"].append(np.linalg.inv(lidar2camera_rt))
        else:
            raise ValueError(f'Not support data_source:{self.data_source}')
        # read image
        image_paths = input_dict["image_paths"]
        images = []
        for k, (camera_name, camera_path) in enumerate(img_path_dict.items()):
        # for path in image_paths:
            path = image_paths[k]
            if self.use_ceph:
                path = self.mapping_path_to_ceph(path)
            img_array = ceph_utils.ceph_read(path, np.uint8, self.use_ceph, self.ceph_client)
            if self.use_virtual:
                distortion = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
                w, h = 1024, 576
                virt_int = input_dict["camera_intrinsics"][k][:3, :3]
                src_int = input_dict["src_camera_intrinsics"][k][:3, :3]
                map1, map2 = cv2.initUndistortRectifyMap(src_int.copy(), distortion, np.eye(3), virt_int.copy(), (w,h), cv2.CV_32FC1)
                undistorted_img = cv2.remap(img_array, map1, map2, interpolation=cv2.INTER_LINEAR)
                img_pil = Image.fromarray(undistorted_img[..., [2, 1, 0]])
                if os.getenv("CALIB") == 'True':
                    CALIB_path = os.getenv("CALIB_PATH")
                    save_dir = os.path.join(CALIB_path, camera_name, 'fig')
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_np(os.path.join(save_dir, "{}".format(self.cnt)), undistorted_img)
            else:
                img_pil = Image.fromarray(img_array[..., [2, 1, 0]])  # bgr2rgb
                if os.getenv("CALIB") == 'True':
                    CALIB_path = os.getenv("CALIB_PATH")
                    save_dir = os.path.join(CALIB_path, camera_name, 'fig')
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_np(os.path.join(save_dir, "{}".format(self.cnt)), img_array)
            images.append(img_pil)
        
        input_dict["camera_imgs"] = images
        input_dict["ori_shape"] = images[0].size

        # resize and crop image
        input_dict = self.crop_image(input_dict)

        return input_dict
    
    def update_gt_box(self, data_dict):
        # cat classes
        gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
        if self.add_ignore and self.training and 'blocked_label' in data_dict:
            maks_blocked_box = (data_dict['blocked_label']=='blocked')&(data_dict['num_points_in_gt']==0)
            gt_classes[maks_blocked_box]=-1
        gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
        data_dict['gt_boxes'] = gt_boxes

        data_dict['metadata']['annos_boxes'] = {
            'gt_names': data_dict['gt_names'],
            'gt_boxes': data_dict['gt_boxes'],
            'num_points_in_gt': data_dict['num_points_in_gt'],
            }
        if 'blocked_label' in data_dict:
            data_dict['metadata']['annos_boxes']['blocked_label'] = data_dict.pop('blocked_label')

        return data_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def __getitem__(self, index):
        points, label, fuse_points, fuse_label, info = self.get_points_label_info(index)
        if os.getenv("CALIB") == 'True':
            print("os.getenv(CALIB_p)", os.getenv("CALIB"))
            CALIB_path = os.getenv("CALIB_PATH")
            save_dir = os.path.join(CALIB_path, 'lidar-points')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_np(os.path.join(save_dir, "{}".format(self.cnt)), points)
        data_source = info['data_source'] if 'data_source' in info else 'A02'
        if data_source == 'atx':
            # print(data_source)
            points[:, 2] -= 1.6
        input_dict = {'points': points, 'label': label, 'metadata': info, 'frame_id': info['frame_name'], \
                      'fuse_points': fuse_points, 'fuse_label': fuse_label}
        
        invalid_points = (points[:, 0] > -20) & (points[:, 0] < 20) & (points[:, 1] > -20) & (points[:, 1] < 20) & (points[:, 2] < -5)  # 地面下的点
        if np.sum(invalid_points) > 100:  # 过滤掉拼接错位的点云
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

        if self.box_cfg is not None:
            input_dict = self.get_gt_box(input_dict, info)
            if self.training and len(input_dict['gt_boxes']) == 0:  # 过滤掉没有box的点云
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)

        if self.use_camera:
            input_dict = self.load_camera_info(input_dict, info)

        data_dict = self.prepare_data(data_dict=input_dict)

        if self.seq_data_enable and self.seq_data_cfg.get('MODE', None) is not None:
            data_dict = self.get_multi_frame_in_out(data_dict)

        if self.freespace_cfg is not None:
            data_dict = self.get_freespace_label(data_dict, info)

        if self.box_cfg is not None:
            data_dict = self.update_gt_box(data_dict)

        if self.pp_heavy:
            data_dict = self.prepare_pp_heavy_gt(data_dict)

        if data_dict['voxels'].shape[0] < 1000:
            input_dict, data_dict = {}, {}
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)
        self.cnt += 1
        return data_dict
    
    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos):
            from ..kitti.kitti_object_eval_python import eval_common as kitti_eval
            from ..kitti import kitti_utils

            map_name_to_kitti = {
                'DontCare': 'DontCare',
                'cone': 'cone',
                'pole': 'pole',
                'isolation_barrel': 'isolation_barrel',
                'triangle_warning': 'triangle_warning',
                'animal': 'animal',
                'gate_rod': 'gate_rod',
                'barrier': 'barrier',
                'construction_sign': 'construction_sign',
            }
            kitti_utils.transform_3dgop_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_3dgop_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]

            # add ignore
            if self.add_ignore:
                for i in range(len(eval_gt_annos)):
                    #gt:'name', 'location', 'dimensions', 'rotation_y', 'num_points_in_gt', 'trackID', 'valid', 'blocked_label', 'gt_boxes_lidar', 'bbox', 'truncated', 'occluded', 'alpha'
                    #det:'name', 'score', 'boxes_lidar', 'pred_labels', 'frame_id', 'metadata', 'bbox', 'truncated', 'occluded', 'location', 'dimensions', 'rotation_y', 'alpha'
                    if 'blocked_label' not in eval_gt_annos[i]:
                        continue
                    assert len(eval_gt_annos[i]['blocked_label'])==len(eval_gt_annos[i]['name'])
                    maks_gt_ignore = ((eval_gt_annos[i]['blocked_label']=='blocked')&
                                      (eval_gt_annos[i]['num_points_in_gt']==0)
                                      )
                    maks_gt_unignore = ((eval_gt_annos[i]['num_points_in_gt']>0) |
                                        ((eval_gt_annos[i]['blocked_label']=='unblocked') &
                                         (eval_gt_annos[i]['num_points_in_gt']==0))
                                        )
                    eval_ignore_gt_annos,eval_unignore_gt_annos = {},{}
                    for key, val in eval_gt_annos[i].items():
                        if key in ['frame_id', 'metadata', 'truncated', 'occluded']:
                            eval_unignore_gt_annos[key] = val
                            eval_ignore_gt_annos[key] = val
                            continue
                        eval_gt_annos[i][key] = val[maks_gt_unignore]
                        eval_unignore_gt_annos[key] = val[maks_gt_unignore]
                        eval_ignore_gt_annos[key] = val[maks_gt_ignore]
                    
                    flag_keep_det = np.array([True]*(len(eval_det_annos[i]['name'])),dtype=np.bool)
                    if len(eval_det_annos[i]['name'])>0 and len(eval_ignore_gt_annos['name'])>0:
                        #calc bev-iou
                        loc_unignore = np.concatenate(
                            [eval_unignore_gt_annos["location"][:, [0, 2]]], 0)
                        dims_unignore = np.concatenate(
                            [eval_unignore_gt_annos["dimensions"][:, [0, 2]]], 0)
                        rots_unignore = np.concatenate([eval_unignore_gt_annos["rotation_y"]], 0)
                        gt_boxes_unignore = np.concatenate(
                            [loc_unignore, dims_unignore, rots_unignore[..., np.newaxis]], axis=1)

                        loc_ignore = np.concatenate(
                            [eval_ignore_gt_annos["location"][:, [0, 2]]], 0)
                        dims_ignore = np.concatenate(
                            [eval_ignore_gt_annos["dimensions"][:, [0, 2]]], 0)
                        rots_ignore = np.concatenate([eval_ignore_gt_annos["rotation_y"]], 0)
                        gt_boxes_ignore = np.concatenate(
                            [loc_ignore, dims_ignore, rots_ignore[..., np.newaxis]], axis=1)

                        loc = np.concatenate(
                            [eval_det_annos[i]["location"][:, [0, 2]]], 0)
                        dims = np.concatenate(
                            [eval_det_annos[i]["dimensions"][:, [0, 2]]], 0)
                        rots = np.concatenate([eval_det_annos[i]["rotation_y"]], 0)
                        dt_boxes = np.concatenate(
                            [loc, dims, rots[..., np.newaxis]], axis=1)

                        overlap = kitti_eval.bev_box_overlap(gt_boxes_ignore, dt_boxes).astype(np.float32)
                        gt_idx = overlap.argmax(axis=0)#len(dt_boxes)
                        gt_det_argmax_iou = overlap[gt_idx,range(dt_boxes.shape[0])]
                        # det_idx_ignore = np.where(maks_gt_ignore[gt_idx]&(gt_det_argmax_iou>0.)==True)[0]
                        det_idx_ignore = np.where((gt_det_argmax_iou>0.))[0]
                        flag_keep_det[det_idx_ignore] = False

                        if len(eval_unignore_gt_annos['name'])>0:
                            overlap_gt_det_ig = kitti_eval.bev_box_overlap(gt_boxes_unignore, dt_boxes[det_idx_ignore]).astype(np.float32)
                            gt_idx_unig = overlap_gt_det_ig.argmax(axis=0)#len(dt_boxes[det_idx_ignore])
                            unig_gt_det_argmax_iou = overlap_gt_det_ig[gt_idx_unig,range(dt_boxes[det_idx_ignore].shape[0])]
                            sub_det_idx_ignore = np.where((unig_gt_det_argmax_iou>0.))[0]
                            flag_keep_det[det_idx_ignore[sub_det_idx_ignore]] = True
                     
                    for key, val in eval_det_annos[i].items():
                        if key in ['frame_id', 'metadata', 'truncated', 'occluded']:
                            eval_det_annos[i][key] = val
                            continue
                        eval_det_annos[i][key] = val[flag_keep_det]

            # 注意gt_boxes中的id已经+1了，所以pred从name映射到id，也要从1开始
            class_to_name = {
                    0: 'DontCare',
                    1: 'cone',
                    2: 'pole',
                    3: 'isolation_barrel',
                    4: 'triangle_warning',
                    5: 'animal',
                    6: 'gate_rod',
                    7: 'barrier',
                    8: 'construction_sign',
                }
            overlap_0_7 = np.array([[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]])
            overlap_0_5 = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])

            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names, 
                overlap_0_7=overlap_0_7, overlap_0_5=overlap_0_5, class_to_name=class_to_name, valid_classes=['cone', 'pole', 'isolation_barrel', 'triangle_warning', 'animal', 'gate_rod', 'barrier', 'construction_sign']
            )
            return ap_result_str, ap_dict

        def waymo_eval(eval_det_annos, eval_gt_annos):
            from ..waymo.waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
            eval = OpenPCDetWaymoDetectionMetricsEstimator()

            ap_dict = eval.waymo_evaluation(
                eval_det_annos, eval_gt_annos, class_name=class_names,
                distance_thresh=1000, fake_gt_infos=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False),
                cam=self.use_cam_metric,
            )
            ap_result_str = '\n'
            for key in ap_dict:
                ap_dict[key] = ap_dict[key][0]
                ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])

            return ap_result_str, ap_dict
        
        def get_range_mask(anno, range1, range2):
            box = anno['boxes_lidar'] if 'boxes_lidar' in anno else anno['gt_boxes_lidar']
            distance = np.sqrt(box[:, 0]**2 + box[:, 1]**2)
            mask = np.logical_and(distance > range1, distance < range2)
            return mask
        
        if not self.dataset_cfg.get('RANGE_EVAL', False):
            eval_det_annos = copy.deepcopy(det_annos)
            eval_gt_annos = [copy.deepcopy(gt_annos['metadata']['annos_boxes']) for gt_annos in det_annos]
        else:
            eval_det_annos = []
            range1, range2 = kwargs['eval_range']
            for anno in det_annos:
                mask = get_range_mask(anno, range1, range2)
                tmp = {}
                for key, val in anno.items():
                    if isinstance(val, np.ndarray):
                        tmp[key] = val[mask] 
                    else:
                        tmp[key] = val
                eval_det_annos.append(copy.deepcopy(tmp))
            
            eval_gt_annos = []
            for gt_annos in det_annos:
                anno = gt_annos['metadata']['annos_boxes']
                mask = get_range_mask(anno, range1, range2)
                tmp = {}
                for key, val in anno.items():
                    if isinstance(val, np.ndarray):
                        tmp[key] = val[mask] 
                    else:
                        tmp[key] = val
                eval_gt_annos.append(copy.deepcopy(tmp))
            
        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos)
        elif kwargs['eval_metric'] == 'waymo':
            ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict
