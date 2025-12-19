import torch
import torch.nn as nn
import torch.nn.functional as F
from pcdet.utils.common_utils import save_np
import os


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]

    Returns:
        [type]: [description]
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    paddings_indicator = actual_num.int() > max_num

    return paddings_indicator


class Empty(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Empty, self).__init__()

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            return args[0]
        elif len(args) == 0:
            return None
        return args


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm,
                 last_layer):
        super().__init__()
        self.name = 'PFNlayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if use_norm:
            self.Conv2d = nn.Conv2d(in_channels, self.units, 1, stride=1, padding=0, bias=False)
            self.norm = nn.BatchNorm2d(self.units, eps=1e-3, momentum=0.01)
        else:
            self.Conv2d = nn.Conv2d(in_channels, self.units, 1, stride=1, padding=0, bias=True)
            self.norm = Empty(self.units)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1).contiguous().unsqueeze(-1)  # (N, 32*, 9) -> (N, 9, 32*, 1)
        x = self.Conv2d(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x)
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x = x.permute(0, 2, 1, 3).contiguous().squeeze()  # (N, 32, 32*, 1) -> (N, 32*, 32)

        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max, x
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated, x
        
    def forward_onnx(self, inputs):
        # inputs: (N, 9, 32*, 1)
        x = self.Conv2d(inputs)
        x = self.norm(x)
        x = F.relu(x)

        x_max = x  # (N, 32, 32*, 1)

        if self.last_vfe:
            return x_max
        else:
            assert False

    def forward_onnx_branch(self, inputs):
        # inputs: (N, 9, 32*, 1)
        x = self.Conv2d(inputs)
        x = self.norm(x)
        x = F.relu(x)
        x = x.permute(0, 2, 1, 3).contiguous()  # (N, 32, 32*, 1) -> (N, 32*, 32, 1)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            assert False


class PillarVFE(nn.Module):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.cnt = 0
        self.model_cfg = model_cfg
        self.name = 'PillarVFE'
        self.select = 0
        num_point_features += 5
        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters)-1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=last_layer)
            )

        self.pfn_layers = nn.ModuleList(pfn_layers)
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]

        if 'INIT_CFG' in self.model_cfg:
            self.init_weights()

    def get_output_feature_dim(self):
        return self.num_filters[-1]
    
    def init_weights(self):
        state_dict = torch.load(self.model_cfg.INIT_CFG.checkpoint, map_location='cpu')
        new_state_dict = {}
        for key, val in state_dict.items():
            key = key.replace('vfe.', '')
            if 'linear' in key:
                key = key.replace('linear', 'Conv2d')
                val = val.view(val.size(0), val.size(1), 1, 1)  # linear to Conv2d
            new_state_dict[key] = val
        self.load_state_dict(new_state_dict)
        print(f'Loaded vfe weight:{self.model_cfg.INIT_CFG.checkpoint}')
        return 

    def forward(self, batch_dict, **kwargs):
        NUM = 20000
        # Forward pass through PFNlayer
        batch_dict = self.vfe_input_prepare(batch_dict)
        features = batch_dict['vfe_input'].permute(0, 2, 1, 3).squeeze(-1)

        if os.getenv("CALIB") == 'True':
            calib_path = os.getenv("CALIB_PATH")
            save_dir = os.path.join(calib_path, 'lidar-vfe')
            save_np(os.path.join(save_dir, "inputs/vfe_input/{}".format(self.cnt)), features.permute(0, 2, 1).unsqueeze(-1))
            if os.getenv("PADDING") != 'True':
                # 原dump方式
                save_dir = os.path.join(calib_path, 'lidar-branch')
                save_np(os.path.join(save_dir, "inputs/vfe_input/{}".format(self.cnt)), features.permute(0, 2, 1).unsqueeze(-1))
            else:
                save_dir = os.path.join(calib_path, 'lidar-branch')
                features_ = features.permute(0, 2, 1).unsqueeze(-1)
                n1 = features_.shape[0]
                if n1 < NUM:
                    pad_f = torch.zeros((NUM-n1, 9, 32, 1)).type_as(features)
                    features_padding = torch.cat((features_, pad_f), dim=0)
                else:
                    features_padding = features_[:NUM,:]
                save_np(os.path.join(save_dir, "inputs/vfe_input/{}".format(self.cnt)), features_padding)
        if os.getenv("PADDING") == 'True':
            n1 = features.shape[0]
            if n1 < NUM:
                pad_f = torch.zeros((NUM-n1, 32, 9)).type_as(features)
                features = torch.cat((features, pad_f), dim=0)
            else:
                features = features[:NUM,:]
        for pfn in self.pfn_layers:
            features, features_nomax = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features

        if os.getenv("CALIB") == 'True':
            calib_path = os.getenv("CALIB_PATH")
            save_dir = os.path.join(calib_path, 'lidar-vfe')
            save_np(os.path.join(save_dir, "outputs/vfe_output/{}".format(self.cnt)), features_nomax)
        self.cnt += 1
        return batch_dict
    
    def forward_onnx(self, batch_dict):
        features = batch_dict['vfe_input']

        # Forward pass through PFNlayer
        for pfn in self.pfn_layers:
            if os.getenv("ONNX_BRANCH") == 'True':
                features = pfn.forward_onnx_branch(features)
                out_name = 'pillar_features'
            else:
                features = pfn.forward_onnx(features)
                out_name = 'vfe_output'
        features = features.squeeze(-1).squeeze(1)
        out_dict = {out_name: features}
        if os.getenv("ONNX_BRANCH") == 'True':
            out_dict.update(batch_dict)
        return out_dict
    
    def vfe_input_prepare(self, batch_dict, **kwargs):
        features, voxel_num_points, coors = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        if self.select:
            points_mean = features[:, :, :3].sum(
                dim=1, keepdim=True) / voxel_num_points.type_as(features).view(-1, 1, 1)
            f_cluster = features[:, :, :3] - points_mean

            f_center = torch.zeros_like(features[:, :, :2])  # initialize f_center
            f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].float().unsqueeze(1) * self.vx + self.x_offset)
            f_center[:, :, 1] = f_center[:, :, 1] - (coors[:, 2].type_as(features).unsqueeze(1) * self.vy + self.y_offset)

            features_ls = [features, f_cluster, f_center]
            if self.with_distance:
                points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
                features_ls.append(points_dist)
            features = torch.cat(features_ls, dim=-1)
        else:
            features_ls = [features]
            points_mean = features[:, :, :3].sum(
                dim=1, keepdim=True) / voxel_num_points.type_as(features).view(-1, 1, 1)
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)

            f_center = features[:, :, :2]
            f_center[:, :, 0] = f_center[:, :, 0] - (coors[:, 3].type_as(features).unsqueeze(1) * self.vx + self.x_offset)
            f_center[:, :, 1] = f_center[:, :, 1] - (coors[:, 2].type_as(features).unsqueeze(1) * self.vy + self.y_offset)
            features_ls.append(f_center)

            features = torch.cat(features_ls, dim=-1)

        voxel_count = features.shape[1]
        mask = get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        NUM = 20000
        if os.getenv("PADDING") == 'True':
            n1 = features.shape[0]
            if n1 < NUM:
                pad_f = torch.zeros((NUM-n1, 32, 9)).type_as(features)
                features = torch.cat((features, pad_f), dim=0)
            else:
                features = features[:NUM,:]

        vfe_input = features.permute(0, 2, 1).unsqueeze(-1)
        out_dict = {"vfe_input": vfe_input}
        out_dict.update(batch_dict)
        return out_dict