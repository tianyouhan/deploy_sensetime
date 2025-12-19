import torch
import torch.nn as nn
import os
from pcdet.utils.common_utils import save_np
from horizon_plugin_pytorch.nn.functional import point_pillars_scatter


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict
    

class PointPillarScatter_Seg(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):

        super().__init__()
        self.cnt = 0
        self.name = 'PillarScatter'
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

        # whether use horizon version of pillar scatter or not
        self.use_horizon_pillar_scatter = False
        if hasattr(self.model_cfg, "USE_HORIZON_PILLAR_SCATTER"):
            self.use_horizon_pillar_scatter = self.model_cfg.USE_HORIZON_PILLAR_SCATTER

        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        NUM = 20000
        batch_dict = self.voxel_coords_prepare(batch_dict)
        if os.getenv("CALIB") == 'True':
            calib_path = os.getenv("CALIB_PATH")
            save_dir = os.path.join(calib_path, 'lidar-branch')
            if os.getenv("PADDING") != 'True':
                # 原dump方式
                save_np(os.path.join(save_dir, "inputs/voxel_coords/{}".format(self.cnt)), batch_dict['voxel_coords'][:, 2:4])
            else:
                print('padding')
                coords_ = batch_dict['voxel_coords'][:, 2:4]
                n1 = coords_.shape[0]
                if n1 < NUM:
                    pad_coord = torch.zeros((NUM-n1, 2)).type_as(coords_)
                    coords_padding = torch.cat((coords_, pad_coord), dim=0)
                else:
                    coords_padding = coords_[:NUM, :]
                save_np(os.path.join(save_dir, "inputs/voxel_coords/{}".format(self.cnt)), coords_padding)
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        if os.getenv("PADDING") == 'True':
            print('in padding mode')
            n1 = coords.shape[0]
            if n1 < NUM:
                # pad_pf = torch.zeros((NUM-n1, 32)).type_as(pillar_features)
                # pillar_features = torch.cat((pillar_features, pad_pf), dim=0)
                pad_coord = torch.zeros((NUM-n1, 4)).type_as(coords)
                coords = torch.cat((coords, pad_coord), dim=0)
            else:
                # pillar_features = pillar_features[:NUM, :]
                coords = coords[:NUM, :]
        batch_size = coords[:, 0].max().int().item() + 1

        batch_spatial_feature = []
        if not self.use_horizon_pillar_scatter:
            for batch_itt in range(batch_size):
                spatial_feature = torch.zeros(
                    self.num_bev_features,
                    self.nx * self.ny,
                    dtype=pillar_features.dtype,
                    device=pillar_features.device)

                batch_mask = coords[:, 0] == batch_itt
                this_coords = coords[batch_mask, :]
                indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
                indices = indices.type(torch.long)
                voxels = pillar_features[batch_mask, :]
                voxels = voxels.t()
                spatial_feature[:, indices] = voxels
                batch_spatial_feature.append(spatial_feature)
        else:
            for batch_itt in range(batch_size):
                out_shape = (1, self.num_bev_features, self.ny, self.nx)
                batch_canvas = point_pillars_scatter(
                    pillar_features, coords, out_shape
                )
                spatial_feature = batch_canvas
                batch_spatial_feature.append(spatial_feature)

        batch_spatial_feature = torch.stack(batch_spatial_feature, 0)
        batch_spatial_feature = batch_spatial_feature.view(
            batch_size, self.num_bev_features, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_feature
        # if os.getenv("CALIB") == 'True':
        #     calib_path = os.getenv("CALIB_PATH")
        #     save_dir = os.path.join(calib_path, 'lidar-branch')
        #     save_np(os.path.join(save_dir, "inputs/spatial_feature/{}".format(self.cnt)), batch_spatial_feature)
        self.cnt += 1
        return batch_dict

    def forward_onnx(self, batch_dict):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']  # N * 2 (x, y)

        if not self.use_horizon_pillar_scatter:
            spatial_feature = torch.zeros(self.num_bev_features, self.nx * self.ny, dtype=pillar_features.dtype, device=pillar_features.device)
            indices = coords[:, 0] * self.nx + coords[:, 1]
            indices = indices.type(torch.long)
            spatial_feature[:, indices] = pillar_features.t()
            spatial_feature = spatial_feature.view(1, self.num_bev_features, self.ny, self.nx)
        else:
            # scatter 优化4
            out_shape = (1, self.num_bev_features, self.ny, self.nx)
            batch_canvas = point_pillars_scatter(
                pillar_features, coords, out_shape
            )
            spatial_feature = batch_canvas

        batch_dict['spatial_features'] = spatial_feature
        return batch_dict

    def voxel_coords_prepare(self, batch_dict, NUM=20000):
        coords = batch_dict['voxel_coords']  # N * 2 (x, y)
        if os.getenv("PADDING") == 'True':
            print('in padding mode')
            n1 = coords.shape[0]
            if n1 < NUM:
                # pad_pf = torch.zeros((NUM-n1, 32)).type_as(pillar_features)
                # pillar_features = torch.cat((pillar_features, pad_pf), dim=0)
                pad_coord = torch.zeros((NUM-n1, 4), dtype=torch.int32).cuda()# .type_as(coords)
                coords = torch.cat((coords, pad_coord), dim=0)
            else:
                # pillar_features = pillar_features[:NUM, :]
                coords = coords[:NUM, :]

        batch_dict['voxel_coords'] = coords
        return batch_dict