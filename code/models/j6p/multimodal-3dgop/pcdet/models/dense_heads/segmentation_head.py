import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from torch.nn import functional as F
from ...utils import loss_seg_utils, freespace_utils
from pcdet.utils.common_utils import save_np
import os


class BaseHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    nn.BatchNorm2d(input_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        for cur_name in self.sep_head_dict:
            out = self.__getattr__(cur_name)(x)

        return out


class SegHead(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, 
                 voxel_size, predict_boxes_when_training=False):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        self.feature_dwon_stride = self.model_cfg.get('FEATURE_DWON_STRIDE')

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []

        self.use_deconv = self.model_cfg.get('USE_DECONV', False)

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                input_channels, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1,
                bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
            ),
            nn.BatchNorm2d(self.model_cfg.SHARED_CONV_CHANNEL),
            nn.ReLU(),
        )

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
        for idx, cur_name in enumerate(cur_head_dict):
            assert len(self.class_names_each_head[idx]) == cur_head_dict[cur_name]['out_channels'], f'{cur_name} class_names_each_head error'
            self.heads_list.append(
                BaseHead(
                    input_channels=self.model_cfg.SHARED_CONV_CHANNEL,
                    sep_head_dict=cur_head_dict,
                    init_bias=-2.19,
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False))
            )

        if self.use_deconv:
            self.upsample = nn.Sequential(*[
                nn.ConvTranspose2d(self.model_cfg.INPUT_FEATURES, self.model_cfg.INPUT_FEATURES, kernel_size=self.feature_dwon_stride, stride=self.feature_dwon_stride, padding=0, bias=False),
                nn.ReLU(inplace=True)
            ])

        else:
            self.upsample = nn.Upsample(scale_factor=self.feature_dwon_stride, mode='nearest')

        self.forward_ret_dict = {}

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y
    
    def softmax(self, x):
        y = torch.softmax(x, dim=1)
        if not self.training:
            y = self.argmax(y)
        return y
    
    def no(self, x):
        return x
    
    def argmax(self, x):
        y = torch.argmax(x, dim=1, keepdim=True)
        return y

    def get_loss(self, batch_dict):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']

        tb_dict = {}
        loss_weights = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        loss_names = self.model_cfg.LOSS_CONFIG.LOSS_NAMES

        total_loss = 0
        for idx, pred_name in enumerate(pred_dicts):
            task = pred_name.split('_')[0]
            pred = pred_dicts[f'{task}_pred']
            label = target_dicts[f'{task}_label']
            mask = freespace_utils.get_mask(batch_dict, label).float()

            task_loss = 0
            for idx, loss_name_ in enumerate(loss_names[task]):
                if loss_name_ == 'ce':
                    loss = (F.cross_entropy(pred, label.squeeze(1).long(), reduction='none') * mask.squeeze(1)).mean()
                elif loss_name_ == 'bce':
                    loss = F.binary_cross_entropy_with_logits(pred, label.float(), weight=mask)
                else:
                    raise ValueError(f'Not support loss type')
                loss = loss * loss_weights[task]['task_weight'][idx]
                total_loss += loss
                task_loss += loss.item()

            tb_dict[f'loss_{task}'] = task_loss
        return total_loss, tb_dict

    def get_pred_label(self, pred_lists, data_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        act_func = post_process_cfg.ACTIVE_FUNC

        pred_dicts, label_dicts = {}, {}
        for i, name in enumerate(self.separate_head_cfg.HEAD_DICT):
            label_dicts[f'{name}_label'] = data_dict.get(f'{name}_label', None)
            cur_act_func = act_func[name]['train'] if self.training else act_func[name]['test']
            pred_dicts[f'{name}_pred'] = getattr(self, cur_act_func)(pred_lists[i])

        data_dict['label_dicts'] = label_dicts
        data_dict['pred_dicts'] = pred_dicts
        return data_dict

    def forward(self, data_dict):

        spatial_features_2d = data_dict['spatial_features_2d']
        x = self.shared_conv(spatial_features_2d)

        x = self.upsample(x)  # 一定要先上采样再接head

        pred_lists = []
        for head in self.heads_list:
            out = head(x)
            pred_lists.append(out)

        data_dict = self.get_pred_label(pred_lists, data_dict)
        self.forward_ret_dict['target_dicts'] = data_dict['label_dicts']
        self.forward_ret_dict['pred_dicts'] = data_dict['pred_dicts']

        return data_dict
    

class BaseHead_pcseg(nn.Module):
    def __init__(self, input_channels, head_dict, use_bias=False):
        super().__init__()

        output_channels = head_dict['out_channels']
        mid_channels = head_dict.get('mid_channels', 32)
        num_conv = head_dict['num_conv']
        assert num_conv >= 2, 'require num_conv >= 2'
        
        fc_list = []
        fc_list.append(nn.Sequential(
            nn.Conv2d(input_channels, mid_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(mid_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        ))
        for k in range(num_conv - 2):
            fc_list.append(nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(mid_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
        ))
        fc_list.append(nn.Conv2d(mid_channels, output_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False))
        self.layers = nn.Sequential(*fc_list)

    def forward(self, x):
        return self.layers(x)
    

class SegHead_pcseg(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, 
                 voxel_size, predict_boxes_when_training=False):
        super().__init__()
        self.cnt = 0
        self.task = model_cfg.get('TASK', None)
        if self.task is not None:
            self.task = self.task.lower()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        input_channels = self.model_cfg.INPUT_FEATURES

        self.feature_dwon_stride = self.model_cfg.get('FEATURE_DWON_STRIDE')

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []

        self.use_deconv = self.model_cfg.get('USE_DECONV', False)

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
        for idx, cur_name in enumerate(cur_head_dict):
            assert len(self.class_names_each_head[idx]) == cur_head_dict[cur_name]['out_channels'], f'{cur_name} class_names_each_head error'
            self.heads_list.append(
                BaseHead_pcseg(
                    input_channels=input_channels,
                    head_dict=cur_head_dict[cur_name],
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
                )
            )

        if self.use_deconv:
            self.upsample = nn.Sequential(*[
                nn.ConvTranspose2d(input_channels, input_channels, kernel_size=self.feature_dwon_stride, stride=self.feature_dwon_stride, padding=0, bias=False),
                nn.ReLU(inplace=True)
            ])

        else:
            self.upsample = nn.Upsample(scale_factor=self.feature_dwon_stride, mode='nearest')

        self.forward_ret_dict = {}
        self.thresh_list = self.model_cfg.POST_PROCESSING.get('THRESH_LIST', None)

    def sigmoid(self, x, head=None):
        y = torch.sigmoid(x)
        return y
    
    def softmax(self, x, head=None):
        y = torch.softmax(x, dim=1)
        if not self.training:
            y = self.argmax(y)
        return y
    
    def no(self, x, head=None):
        return x
    
    def argmax(self, x, head=None):
        y = torch.argmax(x, dim=1, keepdim=True)
        return y
    
    def get_loss(self, batch_dict):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']

        tb_dict = {}
        loss_weights = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        loss_names = self.model_cfg.LOSS_CONFIG.LOSS_NAMES

        total_loss = 0
        for idx, pred_name in enumerate(pred_dicts):
            task = pred_name.split('_')[0]
            pred = pred_dicts[f'{task}_pred']
            label = target_dicts[f'{task}_label']
            mask = freespace_utils.get_mask(batch_dict, label).float()

            task_loss = 0
            for idx, loss_name_ in enumerate(loss_names[task]):
                class_weight = loss_weights[task]['class_weight']
                act_fn = self.model_cfg.POST_PROCESSING.ACTIVE_FUNC[task]['test']
                if loss_name_ == 'ce':
                    loss = (F.cross_entropy(pred, label.squeeze(1).long(), reduction='none', ignore_index=255) * mask.squeeze(1)).mean()
                elif loss_name_ == 'bce':
                    loss = F.binary_cross_entropy_with_logits(pred, label.float(), weight=mask)
                elif loss_name_ == 'focal':
                    gamma = loss_weights[task].get('gamma', 2.0)
                    loss = loss_seg_utils.FocalLoss(act_fn=act_fn ,alpha=class_weight, gamma=gamma, ignore_index=255)(pred, label, mask=mask)
                elif loss_name_ == 'focalbce':
                    gamma = loss_weights[task].get('gamma', 2.0)
                    alpha = loss_weights[task].get('alpha', 0.25)
                    loss = loss_seg_utils.SigmoidFocalClassificationLoss(alpha=alpha, gamma=gamma, ignore_index=255)(pred, label, mask=mask)
                else:
                    raise ValueError(f'Not support loss type')
                loss = loss * loss_weights[task]['task_weight'][idx]
                total_loss += loss
                task_loss += loss.item()

            tb_dict[f'loss_{task}'] = task_loss
        return total_loss, tb_dict

    def get_pred_label(self, pred_lists, data_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        act_func = post_process_cfg.ACTIVE_FUNC

        pred_dicts, label_dicts = {}, {}
        for i, name in enumerate(self.separate_head_cfg.HEAD_DICT):
            cur_act_func = act_func[name]['train'] if self.training else act_func[name]['test']
            label, pred = data_dict.get(f'{name}_label', None), getattr(self, cur_act_func)(pred_lists[i], head=name)
            label_dicts[f'{name}_label'] = label
            pred_dicts[f'{name}_pred'] = pred
            
        data_dict['label_dicts'] = label_dicts
        data_dict['pred_dicts'] = pred_dicts
        if self.task is not None:
            data_dict[f'seg_pred_dicts_{self.task}'] = pred_dicts
            data_dict[f'seg_label_dicts_{self.task}'] = label_dicts
        else:
            data_dict[f'seg_pred_dicts'] = pred_dicts
            data_dict[f'seg_label_dicts'] = label_dicts
        return data_dict

    def forward(self, data_dict):
        if self.task is not None:
            spatial_features_2d = data_dict[f'spatial_features_2d_{self.task}']
        else:
            spatial_features_2d = data_dict['spatial_features_2d']

        if os.getenv("CALIB") == 'True' and self.task != 'fusion':
            calib_path = os.getenv("CALIB_PATH")
            save_dir = os.path.join(calib_path, f'{self.task}-head')
            save_np(os.path.join(save_dir, "inputs/spatial_features_2d_{}/{}".format(self.task, self.cnt)), spatial_features_2d)

        x = self.upsample(spatial_features_2d)  # 一定要先上采样再接head

        pred_lists = []
        for head in self.heads_list:
            out = head(x)
            pred_lists.append(out)

        data_dict = self.get_pred_label(pred_lists, data_dict)
        self.forward_ret_dict['target_dicts'] = data_dict['label_dicts']
        self.forward_ret_dict['pred_dicts'] = data_dict['pred_dicts']

        if os.getenv("CALIB") == 'True':
            task_folder = f"{self.task}-head" if self.task != 'fusion' else "fuser-fusion-head"
            calib_path = os.getenv("CALIB_PATH")
            save_dir = os.path.join(calib_path, task_folder)
            save_np(os.path.join(save_dir, "outputs/seg_pred_dicts_{}/{}".format(self.task, self.cnt)), data_dict[f'seg_pred_dicts_{self.task}'][f'{self.task}_pred'])
        self.cnt += 1
        return data_dict
    

class SegHead_pcseg_multihead(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, 
                 voxel_size, predict_boxes_when_training=False):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        input_channels = self.model_cfg.INPUT_FEATURES

        self.feature_dwon_stride = self.model_cfg.get('FEATURE_DWON_STRIDE')

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []

        self.use_deconv = self.model_cfg.get('USE_DECONV', False)

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        self.cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
        for idx, cur_name in enumerate(self.cur_head_dict):
            assert len(self.class_names_each_head[idx]) == self.cur_head_dict[cur_name]['out_channels'], f'{cur_name} class_names_each_head error'
            if self.use_deconv:
                head_list = [
                    nn.ConvTranspose2d(input_channels[idx], input_channels[idx], self.feature_dwon_stride[idx], self.feature_dwon_stride[idx], padding=0, bias=False),
                    nn.ReLU(inplace=True)
                ]
            else:
                head_list = [nn.Upsample(scale_factor=self.feature_dwon_stride[idx], mode='nearest')]

            head_list.extend(
                [
                    BaseHead_pcseg(
                    input_channels=input_channels[idx],
                    head_dict=self.cur_head_dict[cur_name],
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False))
                ]
            )
            setattr(self, f'{cur_name}_head', nn.Sequential(*head_list))
            
        self.forward_ret_dict = {}

    def sigmoid(self, x):
        y = torch.sigmoid(x)
        return y
    
    def softmax(self, x):
        y = torch.softmax(x, dim=1)
        if not self.training:
            y = self.argmax(y)
        return y
    
    def no(self, x):
        return x
    
    def argmax(self, x):
        y = torch.argmax(x, dim=1, keepdim=True)
        return y
    
    def get_loss(self, batch_dict):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']

        tb_dict = {}
        loss_weights = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        loss_names = self.model_cfg.LOSS_CONFIG.LOSS_NAMES

        total_loss = 0
        for idx, pred_name in enumerate(pred_dicts):
            task = pred_name.split('_')[0]
            pred = pred_dicts[f'{task}_pred']
            label = target_dicts[f'{task}_label']
            
            mask = freespace_utils.get_mask(batch_dict, label).float()

            task_loss = 0
            for idx, loss_name_ in enumerate(loss_names[task]):
                if loss_name_ == 'ce':
                    loss = (F.cross_entropy(pred, label.squeeze(1).long(), reduction='none') * mask.squeeze(1)).mean()
                elif loss_name_ == 'bce':
                    loss = F.binary_cross_entropy_with_logits(pred, label.float(), weight=mask)
                else:
                    raise ValueError(f'Not support loss type')
                loss = loss * loss_weights[task]['task_weight'][idx]
                total_loss += loss
                task_loss += loss.item()

            tb_dict[f'loss_{task}'] = task_loss
        return total_loss, tb_dict

    def get_pred_label(self, pred_lists, data_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        act_func = post_process_cfg.ACTIVE_FUNC

        pred_dicts, label_dicts = {}, {}
        for i, name in enumerate(self.separate_head_cfg.HEAD_DICT):
            cur_act_func = act_func[name]['train'] if self.training else act_func[name]['test']
            label, pred = data_dict.get(f'{name}_label', None), getattr(self, cur_act_func)(pred_lists[i])
            label_dicts[f'{name}_label'] = label
            pred_dicts[f'{name}_pred'] = pred
            
        data_dict['label_dicts'] = label_dicts
        data_dict['pred_dicts'] = pred_dicts
        return data_dict

    def forward(self, data_dict):
        pred_lists = []
        for name in self.cur_head_dict:
            x = data_dict[f'spatial_features_2d_{name}']
            out = getattr(self, f'{name}_head')(x)
            pred_lists.append(out)

        data_dict = self.get_pred_label(pred_lists, data_dict)
        self.forward_ret_dict['target_dicts'] = data_dict['label_dicts']
        self.forward_ret_dict['pred_dicts'] = data_dict['pred_dicts']

        return data_dict