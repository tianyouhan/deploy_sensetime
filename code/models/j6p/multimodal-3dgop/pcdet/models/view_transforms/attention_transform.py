import torch
from torch import nn
import os
import pickle
import numpy as np
import torch.nn.functional as F
from torch.nn.init import normal_

def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes,
                                        sampling_locations):
    """CPU version of multi-scale deformable attention.

    Args:
        value (torch.Tensor): The value has shape
            (bs, num_keys, mum_heads, embed_dims//num_heads)
        value_spatial_shapes (torch.Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (torch.Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (torch.Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),

    Returns:
        torch.Tensor: has shape (bs, num_queries, embed_dims)
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ =\
        sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes],
                             dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, H_, W_)
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :,
                                          level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        # sampling_value_l_ = GridSampleFuction().apply(        
        #     value_l_,
        #     sampling_grid_l_,
        #     'nearest',
        #     'zeros',
        #     False)
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='nearest',
            padding_mode='zeros',
            )  # torch1.1 without align_corners
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs*num_heads, 1, num_queries, num_levels*num_points)
    # attention_weights = attention_weights.transpose(1, 2).reshape(
    #     bs * num_heads, 1, num_queries, num_levels * num_points)
    # output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
    #           attention_weights).sum(-1).view(bs, num_heads * embed_dims,
    #                                           num_queries)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2)).sum(-1).view(bs, num_heads * embed_dims,
                                              num_queries)
    return output.transpose(1, 2).contiguous()

class AttentionTransform(nn.Module):
    """
        This module implements LSS, which lists images into 3D and then splats onto bev features.
        This code is adapted from https://github.com/mit-han-lab/bevfusion/ with minimal modifications.
    """
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg

        # self.return_intermediate = self.model_cfg.return_intermediate

        self.num_points_in_pillar = self.model_cfg.NUM_POINTS_IN_PILLAR
        self.Zmin = self.model_cfg.get("ZMIN", 0.5)

        self.xbound = self.model_cfg.XBOUND
        self.ybound = self.model_cfg.YBOUND
        self.zbound = self.model_cfg.ZBOUND
        self.out_channel = self.model_cfg.OUT_CHANNEL
        self.in_channel = self.model_cfg.IN_CHANNEL
        self.num_feature_levels = self.model_cfg.NUM_FEATURE_LEVELS
        self.num_views = self.model_cfg.NUM_VIEW
        self.bev_h = int((self.xbound[1]-self.xbound[0])/self.xbound[2])
        self.bev_w = int((self.ybound[1]-self.ybound[0])/self.ybound[2])
        self.bev_z = int((self.zbound[1]-self.zbound[0])/self.zbound[2])
        self.image_size = self.model_cfg.IMAGE_SIZE
        self.feature_size = self.model_cfg.FEATURE_SIZE
        self.use_cams_embeds = self.model_cfg.USE_CAMS_EMBEDS

        self.output_proj_cfg = self.model_cfg.get('OUTPUT_PROJ',None)
        if self.output_proj_cfg is not None:
            self.out_channel = self.output_proj_cfg.MID_CHANNEL
            self.final_out_channel = self.output_proj_cfg.FINAL_OUT_CHANNEL

        self.init_layers()
        self.init_weights()

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.out_channel))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_views, self.out_channel))
        # self.bev_embedding = nn.Embedding(
        #     self.bev_h * self.bev_w, self.out_channel)
        self.feat_proj = nn.Sequential(
                nn.Conv2d(self.in_channel, self.out_channel, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.out_channel),
                nn.ReLU(True),
            )
        if self.output_proj_cfg is not None:
            self.output_proj = nn.Sequential(
                    nn.Conv2d(self.out_channel, self.final_out_channel, 1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(self.final_out_channel),
                    nn.ReLU(True),
                )
    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        for m in self.feat_proj.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if self.output_proj_cfg is not None:
            for m in self.output_proj.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out',nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    if m.weight is not None:
                        nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)    
    def get_reference_points(self,H, W, Z, num_points_in_pillar, Zmin, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        
        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(Zmin, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, W, H) / Z
            xs = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, 1, H).expand(num_points_in_pillar, W, H) / H
            ys = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, W, 1).expand(num_points_in_pillar, W, H) / W                                
            ref_3d = torch.stack((xs, ys, zs), -1)  # (num_points_in_pillar, W, H, 3)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)  # (num_points_in_pillar, W*H, 3)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)  # (bs, num_points_in_pillar, W*H, 3)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_x, ref_y = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_x = ref_x.reshape(-1)[None] / H
            ref_y = ref_y.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    def point_sampling(self, reference_points, lidar2img, img_aug_matrix):     
        # lidar2img (B, N, 4, 4)
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (self.xbound[1] - self.xbound[0]) + self.xbound[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (self.ybound[1] - self.ybound[0]) + self.ybound[0]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (self.zbound[1] - self.zbound[0]) + self.zbound[0]

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
        img_aug_matrix = img_aug_matrix.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5
        reference_points_cam[..., 0:2] /= torch.max(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)  # torch.maximum在torch1.1不支持

        reference_points_cam_ = torch.cat((reference_points_cam[...,:3],torch.ones_like(reference_points_cam[..., 2:3])),-1)
        
        # img_aug
        reference_points_cam_ = torch.matmul(img_aug_matrix.to(torch.float32), 
                                            reference_points_cam_.unsqueeze(-1)).squeeze(-1)
        bev_mask = (reference_points_cam_[..., 2:3] > eps)

        reference_points_cam_ = reference_points_cam_[...,0:2]
        reference_points_cam_[..., 0] /= self.image_size[1]
        reference_points_cam_[..., 1] /= self.image_size[0]

        bev_mask = (bev_mask & (reference_points_cam_[..., 1:2] > 0.0)
                    & (reference_points_cam_[..., 1:2] < 1.0)
                    & (reference_points_cam_[..., 0:1] < 1.0)
                    & (reference_points_cam_[..., 0:1] > 0.0))

        reference_points_cam_ = reference_points_cam_.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam_, bev_mask

    def transform_views_(self,reference_points,value,spatial_shapes):

        # bs, num_query, num_Z_anchors, xy = reference_points.shape
        reference_points = reference_points[:, :, None, None, None, :, :]

        # sampling_locations = reference_points
        bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = reference_points.shape

        # num_all_points = num_points*num_Z_anchors

        sampling_locations = reference_points.view(
            bs, num_query, num_heads, num_levels, num_points*num_Z_anchors, xy)

        value = value.view(bs, value.shape[1], num_heads, -1)

        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations)

        return output

    def forward_export_transform_views_(self,reference_points,value,spatial_shapes):
    
        # bs, num_query, num_Z_anchors, xy = reference_points.shape
        # reference_points = reference_points[:, :, None, None, None, :, :]

        # sampling_locations = reference_points.view(
        #     bs, num_query, 1, 1, num_Z_anchors, xy)
        bs, num_query, _,_,num_Z_anchors, xy = reference_points.shape

        value = value.view(bs, value.shape[1], 1, -1)

        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, reference_points)

        return output
    
    def forward_onnx(self, batch_dict):
        '''
        feat size: Nview, Channel, H, W
        reference_points size: Nview, num_query, num_heads=1, num_levels=1, NUM_POINTS_IN_PILLAR=10, xy
        '''
        feat, reference_points = batch_dict['gridsample_input'], batch_dict['gridsample_ref_points']
        feat_flatten = []
        spatial_shapes = []
        mlvl_feats = [feat]
        bs,num_views = 1, self.num_views
        
        for lvl, feat in enumerate(mlvl_feats):
            feat = self.feat_proj(feat) #in_channel--->out_channel
            bn, c, h, w = feat.size()
            feat = feat.view(int(bn/num_views), num_views, c, h, w)
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)#num_cam,bs,h*w,c
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes_tensor = np.array(spatial_shapes, dtype=np.int64)
        value = feat_flatten.permute(1, 0, 2, 3).reshape(
            bs * num_views, feat_flatten.shape[2], self.out_channel)
        output = self.forward_export_transform_views_(reference_points=reference_points,
                                                                    value=value,
                                                                spatial_shapes=spatial_shapes_tensor)
        batch_dict['gridsample_output'] = output.unsqueeze(0)
        return batch_dict
        
    def forward(self, batch_dict):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        mlvl_feats = batch_dict['image_fpn']
        BN, C, H, W = mlvl_feats[0].size()
        batch_size = BN // self.num_views

        ref_3d = self.get_reference_points(
            self.bev_h, self.bev_w, self.bev_z, self.num_points_in_pillar, Zmin=self.Zmin, dim='3d', bs=batch_size,  device=mlvl_feats[0].device, dtype=mlvl_feats[0].dtype)

        lidar2image = batch_dict['lidar2image']#BN44
        img_aug_matrix = batch_dict['img_aug_matrix']

        reference_points_cam, bev_mask = self.point_sampling(ref_3d, lidar2image, img_aug_matrix)

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            feat = self.feat_proj(feat) #in_channel--->out_channel

            bn, c, h, w = feat.size()
            feat = feat.view(int(bn/self.num_views), self.num_views, c, h, w)
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)#num_cam,bs,h*w,c
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=mlvl_feats[0].device)

        num_cams,bs,num_query,D,xy = reference_points_cam.size()   #  (num_cam, bs, num_query, D, 2) D=num_point_in_pillar
        slots = torch.zeros(bs,num_query,self.out_channel,device=mlvl_feats[0].device)
        indexes = []
        max_len = 0
        for i, mask_per_img in enumerate(bev_mask):  # bev_mask.shape  (num_cam, bs, num_query, D)
            indexes_batch = []
            for j in range(bs):
                index_query_per_img = mask_per_img[j].sum(-1).nonzero().squeeze(-1)  # 可以投影到当前cam图片的query点
                indexes_batch.append(index_query_per_img)
            indexes.append(indexes_batch)
            max_len_i = max([len(each) for each in indexes_batch])
            max_len = max(max_len,max_len_i)
        
        # each camera only interacts with its corresponding BEV queries. This step can greatly save GPU memory.
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, num_cams, max_len, D, 2])
        
        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):   
                index_query_per_img = indexes[i][j]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]

        reference_points=reference_points_rebatch.view(bs*num_cams, max_len, D, 2)
        #N,B,HW,B-->B,N,HW,C-->BN,HW,C
        value = feat_flatten.permute(1, 0, 2, 3).reshape(
            bs * num_cams, feat_flatten.shape[2], self.out_channel)

        output = self.transform_views_(reference_points,
                                       value,
                                       spatial_shapes).view(bs, num_cams, max_len, self.out_channel)
                          
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img[j]] += output[j, i, :len(index_query_per_img[j])]

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None].float()

        slots = slots.view(bs,self.bev_w,self.bev_h,self.out_channel)

        slots = slots.permute(0, 3, 1, 2).contiguous()
        if self.output_proj_cfg is not None:
            slots = self.output_proj(slots)
        batch_dict['spatial_features_img'] = slots
        torch.cuda.empty_cache()
        return batch_dict
