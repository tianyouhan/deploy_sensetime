import torch
import torch.nn as nn
import os
from pcdet.utils.common_utils import save_np
from horizon_plugin_pytorch.nn.functional import point_pillars_scatter
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub
class PointPillarScatter_Seg_Qat(nn.Module):

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
        # self.quan_voxel_coords = QuantStub()
        self.spatial_features_2d_dequant = DeQuantStub()

    def forward(self, batch_dict, **kwargs):
        batch_dict = self.voxel_coords_prepare(batch_dict)
        if os.getenv("CALIB") == 'True':
            calib_path = os.getenv("CALIB_PATH")
            save_dir = os.path.join(calib_path, 'lidar-branch')
            save_np(os.path.join(save_dir, "inputs/voxel_coords/{}".format(self.cnt)), batch_dict['voxel_coords'])
            
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_size = coords[:, 0].max().int().item() + 1
        coords = coords.to(torch.int32)

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

        # # scatter 优化1
        # canvas = torch.zeros(1 * self.nx * self.ny, self.num_bev_features, dtype=pillar_features.dtype, device=pillar_features.device)
        # index = coords[:, 0] * self.nx + coords[:, 1]
        # index = index.type(torch.long)
        # canvas[index] = pillar_features
        # canvas = canvas.reshape(1, self.ny, self.nx, self.num_bev_features).permute(0, 3, 1, 2)
        # spatial_feature = canvas
        # print(spatial_feature.size())

        # # scatter 优化2
        # canvas = torch.zeros(1 * self.nx * self.ny, self.num_bev_features, dtype=pillar_features.dtype, device=pillar_features.device)
        # index = coords[:, 0] * self.nx + coords[:, 1]
        # index = index.type(torch.long)
        # index_onehot = torch.nn.functional.one_hot(index, self.ny*self.nx)
        # index_onehot = index_onehot.type(torch.float32)
        # canvas = torch.matmul(pillar_features.t(), index_onehot)

        # # scatter 优化3
        # canvas = torch.zeros(1 * self.nx * self.ny, self.num_bev_features, dtype=pillar_features.dtype, device=pillar_features.device)
        # index = coords[:, 0] * self.nx + coords[:, 1]
        # index = index.type(torch.long)
        # for i in range(index.shape[0]):
        #     canvas[index[i]] = pillar_features[i]
        # canvas = canvas.reshape(1, self.ny, self.nx, self.num_bev_features).permute(0, 3, 1, 2)
        # spatial_feature = canvas

        batch_dict['spatial_features'] = spatial_feature
        return batch_dict

    def voxel_coords_prepare(self, batch_dict, NUM=20000):
        coords = batch_dict['voxel_coords']  # N * 2 (x, y)
        coords = coords.to(torch.float32)
        if os.getenv("PADDING") == 'True':
            print('in padding mode')
            n1 = coords.shape[0]
            if n1 < NUM:
                # pad_pf = torch.zeros((NUM-n1, 32)).type_as(pillar_features)
                # pillar_features = torch.cat((pillar_features, pad_pf), dim=0)
                pad_coord = torch.zeros((NUM-n1, 4), dtype=torch.float32).cuda()# .type_as(coords)
                coords = torch.cat((coords, pad_coord), dim=0)
            else:
                # pillar_features = pillar_features[:NUM, :]
                coords = coords[:NUM, :]

        batch_dict['voxel_coords'] = coords
        return batch_dict

    def forward_qat(self, batch_dict, **kwargs):
        batch_dict = self.voxel_coords_prepare(batch_dict)
        if os.getenv("CALIB") == 'True':
            calib_path = os.getenv("CALIB_PATH")
            save_dir = os.path.join(calib_path, 'lidar-branch')
            save_np(os.path.join(save_dir, "inputs/voxel_coords/{}".format(self.cnt)), batch_dict['voxel_coords'][:, 2:4])
            
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        coords = coords.to(torch.int32)
        batch_size = coords[:, 0].max() + 1
        # coords = self.quan_voxel_coords(coords)

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
        if os.getenv("SPLIT_BKB") == 'True':
            batch_dict['spatial_features'] = self.spatial_features_2d_dequant(batch_spatial_feature)
        else:
            batch_dict['spatial_features'] = batch_spatial_feature
        self.cnt += 1
        return batch_dict

    def forward_qat_onnx(self, batch_dict):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']  # N * 2 (x, y)
        coords = coords.to(torch.int32)
        # coords = self.quan_voxel_coords(coords)

        if not self.use_horizon_pillar_scatter:
            spatial_feature = torch.zeros(self.num_bev_features, self.nx * self.ny, dtype=pillar_features.dtype, device=pillar_features.device)
            indices = coords[:, 0] * self.nx + coords[:, 1]
            indices = indices.type(torch.long)
            spatial_feature[:, indices] = pillar_features.t()
            spatial_feature = spatial_feature.view(1, self.num_bev_features, self.ny, self.nx)
        else:
            out_shape = (1, self.num_bev_features, self.ny, self.nx)
            batch_canvas = point_pillars_scatter(
                pillar_features, coords, out_shape
            )
            spatial_feature = batch_canvas
        if os.getenv("SPLIT_BKB") == 'True':
            batch_dict['spatial_features'] = self.spatial_features_2d_dequant(spatial_feature)
        else:
            batch_dict['spatial_features'] = spatial_feature
        return batch_dict