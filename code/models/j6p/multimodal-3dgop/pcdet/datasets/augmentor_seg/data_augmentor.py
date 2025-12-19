from functools import partial

import numpy as np
from PIL import Image
import cv2
from ...utils import common_utils
from . import augmentor_utils


class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger

        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def disable_augmentation(self, augmentor_configs):
        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        label, points = data_dict['label'], data_dict['points']
        gt_boxes = data_dict.get('gt_boxes', None)
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            probability = config.get('PROBABILITY',None)
            label, points, enable, gt_boxes = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                label, points, return_flip=True, gt_boxes=gt_boxes, probability=probability
            )
            data_dict['flip_%s'%cur_axis] = enable

        data_dict['label'] = label
        data_dict['points'] = points
        data_dict['gt_boxes'] = gt_boxes
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        rot_clockwise = config.get('ROT_CLOCKWISE', False)
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes = data_dict.get('gt_boxes', None)
        label, points, noise_rot, gt_boxes = augmentor_utils.global_rotation(
            data_dict['label'], data_dict['points'], rot_range=rot_range, return_rot=True, gt_boxes=gt_boxes, rot_clockwise=rot_clockwise
        )

        data_dict['label'] = label
        data_dict['points'] = points
        data_dict['noise_rot'] = noise_rot
        data_dict['gt_boxes'] = gt_boxes
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        gt_boxes = data_dict.get('gt_boxes', None)
        label, points, noise_scale, gt_boxes = augmentor_utils.global_scaling(
            data_dict['label'], data_dict['points'], config['WORLD_SCALE_RANGE'], return_scale=True, gt_boxes=gt_boxes
        )

        data_dict['label'] = label
        data_dict['points'] = points
        data_dict['noise_scale'] = noise_scale
        data_dict['gt_boxes'] = gt_boxes
        return data_dict

    def random_world_translation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_translation, config=config)
        noise_translate_std = config['NOISE_TRANSLATE_STD']
        assert len(noise_translate_std) == 3
        noise_translate = np.array([
            np.random.normal(0, noise_translate_std[0], 1),
            np.random.normal(0, noise_translate_std[1], 1),
            np.random.normal(0, noise_translate_std[2], 1),
        ], dtype=np.float32).T

        label, points = data_dict['label'], data_dict['points']
        gt_boxes = data_dict.get('gt_boxes', None)
        points[:, :3] += noise_translate
        if gt_boxes is not None:
            gt_boxes[:, :3] += noise_translate
       
        data_dict['label'] = label
        data_dict['points'] = points
        data_dict['noise_translate'] = noise_translate
        data_dict['gt_boxes'] = gt_boxes
        return data_dict
    
    def random_world_dropout(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_dropout, config=config)
        
        label, points = data_dict['label'], data_dict['points']
        dropout_max_ratio = config['DROPOUT_RATIO']
        if dropout_max_ratio <= 0.0 or dropout_max_ratio >= 1.0:
            return data_dict
        
        dropout_ratio = np.random.uniform(0.0, dropout_max_ratio)
        dropout_prob = np.random.uniform(0.0, 1.0, size=[points.shape[0]])
        dropout_mask = np.where(dropout_prob <= dropout_ratio)[0]
        points = np.delete(points, dropout_mask, axis=0)
        label = np.delete(label, dropout_mask, axis=0)

        data_dict['label'] = label
        data_dict['points'] = points
        data_dict['noise_dropout'] = dropout_ratio
        return data_dict

    def CLAHE(self, img: np.ndarray, 
              clip_limit=40.0, 
              tile_grid_size=(8, 8)) -> np.ndarray:
        """Use CLAHE method to process the image.


        Args:
            clip_limit (float): Threshold for contrast limiting. Default: 40.0.
            tile_grid_size (tuple[int]): Size of grid for histogram equalization.
            Input image will be divided into equally sized rectangular tiles.
            It defines the number of tiles in row and column. Default: (8, 8).
            """
        for i in range(img.shape[2]):
            clahe = cv2.createCLAHE(clip_limit, tile_grid_size)
            img[:,:,i] = clahe.apply(np.array(img[:,:,i], dtype=np.uint8))
        return img

    def AdjustGamma(self, img: np.ndarray,
                    gamma: float = 1.0) -> np.ndarray:
        assert gamma > 0
        assert 0 <= np.min(img) and np.max(img) <= 255
        
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0)**inv_gamma * 255
                            for i in np.arange(256)]).astype('uint8')
        assert table.shape == (256, )
        return cv2.LUT(np.array(img, dtype=np.uint8), table)

    def convert(self,
                img: np.ndarray,
                alpha: int = 1,
                beta: int = 0) -> np.ndarray:
        """Multiple with alpha and add beat with clip.

        Args:
            img (np.ndarray): The input image.
            alpha (int): Image weights, change the contrast/saturation
                of the image. Default: 1
            beta (int): Image bias, change the brightness of the
                image. Default: 0

        Returns:
            np.ndarray: The transformed image.
        """

        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img: np.ndarray, brightness_delta: int = 32) -> np.ndarray:
        """Brightness distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after brightness change.
        """
    
        return self.convert(
                img,
                beta=np.random.uniform(-brightness_delta,
                                    brightness_delta))

    def contrast(self, img: np.ndarray, contrast_range: list = [0.5, 1.5]) -> np.ndarray:
        """Contrast distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after contrast change.
        """
        return self.convert(
                img,
                alpha=np.random.uniform(contrast_range[0], contrast_range[1]))
    
    def Gauss_noise(self, img: np.ndarray, mean: float = 0.0, std: float = 0.1):
        assert std >= 0.0
        rand_std = np.random.uniform(0, std)
        noise = np.random.normal(
                mean, rand_std, size=img.shape)
        noise = noise.astype(img.dtype)
        img = img + noise
        return img 
        
    def imgaug(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.imgaug, config=config)
        imgs = data_dict["camera_imgs"]
        img_process_infos = data_dict['img_process_infos']
        new_imgs = []
        for img, img_process_info in zip(imgs, img_process_infos):
            flip = False
            if config.RAND_FLIP and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*config.ROT_LIM)
            # aug images
            if flip:
                img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
            img = img.rotate(rotate)

            if 'COLOR' in config:
                # color aug images
                def check_color_aug_enable(aug_name, prob=0.25):
                    assert 0 <= prob <= 1.0
                    flag = (np.random.random() < prob) and (aug_name in color_cfg) and color_cfg[aug_name].ENABLE
                    return flag
                color_cfg = config.COLOR
                img = np.array(img, dtype=np.float32)
                if check_color_aug_enable('CONTRAST', prob=color_cfg.CONTRAST.PROB):
                    img = self.contrast(img, color_cfg.CONTRAST.RANGE)
                if check_color_aug_enable('BRIGHTNESS', prob=color_cfg.BRIGHTNESS.PROB):
                    img = self.brightness(img, color_cfg.BRIGHTNESS.DELTA)
                if check_color_aug_enable('CLAHE', prob=color_cfg.CLAHE.PROB):
                    img = self.CLAHE(img, color_cfg.CLAHE.CLIP_LIMIT)
                if check_color_aug_enable('GUASS_NOISE', prob=color_cfg.GUASS_NOISE.PROB):
                    img = self.Gauss_noise(img, std=color_cfg.GUASS_NOISE.STD)
                if check_color_aug_enable('ADJUST_GAMMA', prob=color_cfg.ADJUST_GAMMA.PROB):
                    img = self.AdjustGamma(img, color_cfg.ADJUST_GAMMA.GAMMA)
                img = Image.fromarray(img.astype(np.uint8))

            img_process_info[2] = flip
            img_process_info[3] = rotate
            new_imgs.append(img)

        data_dict["camera_imgs"] = new_imgs
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                label: (N, 1)
                ...

        Returns:
        """
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        return data_dict
