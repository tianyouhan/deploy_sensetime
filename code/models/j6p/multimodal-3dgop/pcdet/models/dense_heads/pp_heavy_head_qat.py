import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
# from ...utils import loss_utils
import pcdet.utils.pp_heavy.loss_utils as loss_utils
import pcdet.ops.iou3d_nms.iou3d_nms_utils as iou3d_utils
import pcdet.utils.pp_heavy.box_torch_ops as box_torch_ops
from pcdet.models.backbones_3d.pfe.rpn_head import ConvHead
from pcdet.utils.common_utils import save_np
import os

class pp_heavy_head_qat(nn.Module):
    def __init__(self,
                 cfg,
                 dataset,
                 num_class=2,
                 num_anchor_per_cls=2,
                 num_direction_bins=2,
                 drop_ratio=0.):
        """
            upsample_strides support float: [0.25, 0.5, 1]
            if upsample_strides < 1, conv2d will be used instead of convtranspose2d.
        """
        super(pp_heavy_head_qat, self).__init__()
        self.cnt = 0
        self.task = cfg.get('TASK', None)
        if self.task is not None:
            self.task = self.task.lower()
        self.cfg = cfg
        rpn_head_cfgs = cfg.RPN_STAGE.RPN_HEAD.RPN_HEAD_ARGS
        rpn_base_args = cfg.RPN_STAGE.RPN_HEAD.RPN_BASE_ARGS
        encode_background_as_zeros = cfg.RPN_STAGE.ENCODE_BG_AS_ZEROS
        use_direction_classifier = cfg.RPN_STAGE.RPN_HEAD.USE_DIRECTION_CLASSIFIER
        box_code_size = dataset.target_assigner.box_coder.code_size if hasattr(dataset, 'target_assigner') else 7
        self.box_coder = dataset.target_assigner.box_coder if hasattr(dataset, 'target_assigner') else None

        self._num_direction_bins = num_direction_bins
        self._num_class = num_class
        self.num_class = num_class
        self._drop_ratio = drop_ratio
        self._use_direction_classifier = use_direction_classifier
        self._box_code_size = box_code_size
        self._split_fpn = rpn_base_args['split_fpn']
        if 'input_channels' in rpn_head_cfgs[0]:
            final_num_filters = rpn_head_cfgs[0]['input_channels']
        else:
            final_num_filters = sum(rpn_base_args['num_upsample_filters'])

        self._make_head(
                        rpn_head_cfgs,
                        final_num_filters,
                        num_anchor_per_cls,
                        encode_background_as_zeros,
                        use_direction_classifier,
                        box_code_size,
                        num_direction_bins,
                        rpn_base_args["use_norm"])
        self.forward_ret_dict = {}

        if cfg.get('USE_PPHEAVY_LOSS', True):
            self.build_losses()
        else:
            self.build_losses_pcdet()

    def build_losses(self):
        cfg = self.cfg
        self.rpn_cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(cfg, alpha=0.25, gamma=2.0)
        self.rpn_reg_loss_func = loss_utils.WeightedSmoothL1LocalizationLoss(cfg,sigma=cfg.RPN_STAGE.RPN_HEAD.LOC_SIGMA)
        self.rpn_dir_loss_func = loss_utils.WeightedSoftmaxClassificationLoss()
        # self.add_module('hm_loss_func', loss_utils.FocalLossCenterNet())
        # self.add_module('reg_loss_func', loss_utils.RegLossCenterNet())

    def build_losses_pcdet(self):
        from pcdet.utils import loss_utils as loss_utils_ori
        self.rpn_cls_loss_func = loss_utils_ori.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        self.rpn_reg_loss_func = loss_utils_ori.WeightedSmoothL1Loss(code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.rpn_dir_loss_func = loss_utils_ori.WeightedCrossEntropyLoss()


    def _make_head(self,
                   rpn_head_cfgs,
                   final_num_filters,
                   num_anchor_per_cls,
                   encode_background_as_zeros,
                   use_direction_classifier,
                   box_code_size,
                   num_direction_bins,
                   use_norm):
        cfg = self.cfg
        rpn_heads = []
        self.rpn_head_input_key = []
        for i, rpn_head_cfg in enumerate(rpn_head_cfgs):
            self.rpn_head_input_key.append('out')
            in_num_filters = final_num_filters
            stride = rpn_head_cfg.downsample_level // 2
            if i == 1 and self._split_fpn == 1:
                in_num_filters = final_num_filters * 2 // 3
                stride = 1  # this is a hard code here
            elif i == 1 and self._split_fpn == 2:
                in_num_filters = final_num_filters // 3
                stride = 1  # this is a hard code here
            elif i == 1 and (cfg.RPN_STAGE.BACKBONE.FPN.FPN_OUT2 and cfg.RPN_STAGE.BACKBONE.FPN.FPN_SUM):
                stride = 1

            rpn_heads.append(
                ConvHead(
                    cfg=cfg,
                    in_filters=in_num_filters,
                    num_class=self._num_class,
                    num_anchor_per_cls=num_anchor_per_cls,
                    encode_background_as_zeros=encode_background_as_zeros,
                    use_direction_classifier=use_direction_classifier,
                    box_code_size=box_code_size,
                    num_direction_bins=num_direction_bins,
                    num_filters=rpn_head_cfg.num_filters,
                    head_cls_num=len(rpn_head_cfg.class_name),
                    conv_nums=rpn_head_cfg.conv_nums,
                    dilations=rpn_head_cfg.dilation,
                    stride=stride,
                    use_se=rpn_head_cfg.use_se,
                    use_res=rpn_head_cfg.use_res,
                    use_glore=rpn_head_cfg.use_glore,
                    use_shuffle=rpn_head_cfg.use_shuffle,
                    ratio=rpn_head_cfg.ratio,
                    se_ratio=rpn_head_cfg.se_ratio,
                    num_res=rpn_head_cfg.num_res,
                    ds_keep_ratio=rpn_head_cfg.ds_keep_ratio,
                    use_norm=use_norm,
                    split_class=rpn_head_cfg.split_class if 'split_class' in rpn_head_cfg else False,
                    split_class_branch=rpn_head_cfg.split_class_branch if 'split_class_branch' in rpn_head_cfg else False,
                    split_class_branch_convnum=rpn_head_cfg.split_class_branch_convnum if 'split_class_branch_convnum' in rpn_head_cfg else 1,
                    split_class_branch_convfeats=rpn_head_cfg.split_class_branch_convfeats if 'split_class_branch_convfeats' in rpn_head_cfg else 32,
                    num_anchor_per_cls_list=rpn_head_cfg.num_anchor_per_cls_list if 'num_anchor_per_cls_list' in rpn_head_cfg else [num_anchor_per_cls]*len(rpn_head_cfg.class_name),
                    ks1=rpn_head_cfg.ks1 if 'ks1' in rpn_head_cfg else 1,
                    firstBN=rpn_head_cfg.firstBN if 'firstBN' in rpn_head_cfg else False,
                    DH_scale=rpn_head_cfg.DH_scale if 'DH_scale' in rpn_head_cfg else False,
                    ssratio=rpn_head_cfg.ssratio if 'ssratio' in rpn_head_cfg else None,
                    decoupled_head=rpn_head_cfg.decoupled_head if 'decoupled_head' in rpn_head_cfg else False,
                    decoupled_ch=rpn_head_cfg.decoupled_ch if 'decoupled_ch' in rpn_head_cfg else None
                )
            )
        
        self.rpn_heads = nn.ModuleList(rpn_heads)

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, 7)
        rot_gt = reg_targets[..., -1] + anchors[..., -1]
        dir_cls_targets = (rot_gt > 0).long()
        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), 2, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets
    
    @staticmethod
    def add_sin_difference(boxes1, boxes2):
        rad_pred_encoding = torch.sin(boxes1[..., -1:]) * torch.cos(
            boxes2[..., -1:])
        rad_tg_encoding = torch.cos(boxes1[..., -1:]) * torch.sin(boxes2[..., -1:])
        boxes1 = torch.cat([boxes1[..., :-1], rad_pred_encoding], dim=-1)
        boxes2 = torch.cat([boxes2[..., :-1], rad_tg_encoding], dim=-1)
        return boxes1, boxes2

    def get_loss(self):
        cfg = self.cfg
        tb_dict = {}
        # rpn head losses
        cls_preds = self.ret_dict['rpn_cls_preds'].float()
        box_preds = self.ret_dict['rpn_box_preds'].float() 
        dir_cls_preds = self.ret_dict['rpn_dir_cls_preds'].float() 
        iou_preds = self.ret_dict['rpn_iou_preds'] 
        var_preds = self.ret_dict['rpn_var_preds'].float() 
        
        labels, reg_targets = self.labels, self.reg_targets

        cared = labels >= 0  # [N, num_anchors]
        positives = labels > 0
        negatives = labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()

        if cfg.RPN_STAGE.RPN_HEAD.FPGA_IOU_LOSS.USE and torch.sum(positives).item() > 0:
            batch_size = self.anchors.shape[0]
            batch_anchors = self.anchors.view(batch_size, -1, 7)
            batch_box_preds = self.box_coder.decode_torch(box_preds, batch_anchors).float()
            batch_box_targets = reg_targets.view(batch_size, -1, 7)
            batch_box_targets = self.box_coder.decode_torch(batch_box_targets, batch_anchors).float()
            index = torch.nonzero(positives.view(-1)).view(-1)

            batch_box_preds_bev = torch.index_select(batch_box_preds.view(-1, 7), 0, index)
            batch_box_targets_bev = torch.index_select(batch_box_targets.view(-1, 7), 0, index)
            positive_iou = iou3d_utils.boxes_iou_bev(batch_box_preds_bev, batch_box_targets_bev)

            positive_iou = torch.diagonal(positive_iou)

            if cfg.RPN_STAGE.RPN_HEAD.FPGA_IOU_LOSS.CLS_IOU_LOSS:
                cls_weights = cls_weights.view(-1)
                iou_weight = 0.6 * torch.exp(positive_iou)
                cls_weights[index] = iou_weight
                cls_weights = cls_weights.view(batch_size, -1)
            if cfg.RPN_STAGE.RPN_HEAD.FPGA_IOU_LOSS.LOC_IOU_LOSS:
                reg_weights = reg_weights.view(-1)
                iou_weight = 0.8 * torch.exp(1 - positive_iou)
                reg_weights[index] = iou_weight
                reg_weights = reg_weights.view(batch_size, -1)

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        cls_targets = labels * cared.type_as(labels)
        cls_targets = cls_targets.unsqueeze(-1)

        box_code_size = self.box_coder.code_size
        batch_size = int(box_preds.shape[0])
        num_class = self.num_class
        anchors = self.anchors.view(batch_size, -1, 7)[0]

        box_preds = box_preds.view(batch_size, -1, box_code_size)
        cls_targets = cls_targets.squeeze(-1)
        one_hot_targets = torch.zeros((*list(cls_targets.shape), num_class + 1), dtype=torch.float,
                                      device=cls_targets.device)
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)

        cls_preds = cls_preds.view(batch_size, -1, num_class)
        one_hot_targets_bg = one_hot_targets[..., 0].cpu().detach().numpy()
        one_hot_targets = one_hot_targets[..., 1:]
       
        masks = [one_hot_targets[:, :, i] == 1 for i in range(self.num_class)]

        
        for cls_id in range(self.num_class):
            cur_mask = masks[cls_id]
            reg_weights[cur_mask] = reg_weights[cur_mask] * cfg.TRAIN.LS_WEIGHTS[cls_id]

        if cfg.RPN_STAGE.RPN_HEAD.ENCODE_RAD_ERROR_BY_SIN:
            # sin(a - b) = sinacosb-cosasinb
            box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, reg_targets)
            loc_loss = self.rpn_reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights, 
                NRE=self.NormRegEncod if cfg.RPN_STAGE.NORM_REG_ENCOD else None, var_preds=var_preds)  # [N, M]
        else:
            loc_loss = self.rpn_reg_loss_func(box_preds, reg_targets, weights=reg_weights)  # [N, M]

       
        cls_loss = self.rpn_cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]

        loss_weights_dict = cfg.RPN_STAGE.RPN_HEAD.LOSS_WEIGHTS
        loc_loss_reduced = loc_loss.sum() / batch_size
        loc_loss_reduced_total = loc_loss_reduced * loss_weights_dict['loc_weight']
        cls_loss_reduced = cls_loss.sum() / batch_size
        cls_loss_reduced_total = cls_loss_reduced * loss_weights_dict['cls_weight']
        rpn_loss = loc_loss_reduced_total + cls_loss_reduced_total

        tb_dict['rpn_loss_loc'] = loc_loss_reduced_total.item()
        tb_dict['rpn_loss_cls'] = cls_loss_reduced_total.item()

        if cfg.RPN_STAGE.RPN_HEAD.USE_DIRECTION_CLASSIFIER:
            dir_targets = self.get_direction_target(self.anchors, reg_targets)
            dir_logits = dir_cls_preds.view(batch_size, -1, 2)
            weights = (labels > 0).type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            for cls_id in range(self.num_class):
                cur_mask = masks[cls_id]
                weights[cur_mask] = weights[cur_mask] * cfg.TRAIN.LS_WEIGHTS[cls_id]
            dir_loss = self.rpn_dir_loss_func(dir_logits, dir_targets, weights=weights)

            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * loss_weights_dict['dir_weight']
            rpn_loss += dir_loss
            tb_dict["rpn_dir_loss_reduced"] = dir_loss.item()

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict
    
    def scale_nms(self, cfg, box_preds):
        boxes_for_nms = box_preds.clone()
        if cfg.get('SCALE_NMS', None) is not None and cfg.SCALE_NMS.ENABLE:
            x1, y1, x2, y2 = cfg.SCALE_NMS.RANGE  # [x1, y1, x2, y2]
            boxes_x, boxes_y = boxes_for_nms[:, 0], boxes_for_nms[:, 1]
            scale_mask = (x1 <= boxes_x) & (boxes_x <= x2) & (y1 <= boxes_y) & (boxes_y <= y2)
            boxes_for_nms[:, 3][scale_mask] *= cfg.SCALE_NMS.RATIO
            boxes_for_nms[:, 4][scale_mask] *= cfg.SCALE_NMS.RATIO
        return boxes_for_nms

    def predict_rpn(self, input_dict, preds_dict):
        cfg = self.cfg
        batch_size = input_dict['anchors'].shape[0]
        batch_anchors = input_dict["anchors"].view(batch_size, -1, 7)
        if "anchors_mask" not in input_dict:
            batch_anchors_mask = [None] * batch_size
        else:
            batch_anchors_mask = input_dict["anchors_mask"].view(batch_size, -1)
        # TODO
        self.anchors_mask = [None] * batch_size
        if self.anchors_mask[0] is not None:
            batch_anchors_mask = self.anchors_mask

        num_class_with_bg = self.num_class + 1 if not cfg.RPN_STAGE.ENCODE_BG_AS_ZEROS else self.num_class
        batch_box_preds = preds_dict["rpn_box_preds"].view(batch_size, -1, self.box_coder.code_size)
        batch_cls_preds = preds_dict["rpn_cls_preds"].view(batch_size, -1, num_class_with_bg).float()
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors, NRE=self.NormRegEncod if cfg.RPN_STAGE.NORM_REG_ENCOD else None).float()

        if cfg.RPN_STAGE.RPN_HEAD.USE_DIRECTION_CLASSIFIER:
            batch_dir_preds = preds_dict["rpn_dir_cls_preds"].view(batch_size, -1, 2)
        else:
            batch_dir_preds = [None] * batch_size

        batch_iou_preds = [None] * batch_size

        if cfg.RPN_STAGE.RPN_HEAD.REG_VARIANCE.OUTPUT:
            batch_var_preds = preds_dict['rpn_var_preds'].view(batch_size, -1, self.box_coder.code_size)
            batch_var_preds = torch.sigmoid(batch_var_preds)
        else:
            batch_var_preds = [None] * batch_size

        predictions_dicts = []
        num = 0
        for box_preds, cls_preds, dir_preds, iou_preds, var_preds, a_mask in zip(
                batch_box_preds, batch_cls_preds, batch_dir_preds, batch_iou_preds, batch_var_preds, batch_anchors_mask):
            if a_mask is not None:
                if a_mask.cpu().sum()==0:
                    a_mask=torch.from_numpy(np.array([0],dtype=np.long)).cuda()
                box_preds = box_preds[a_mask]
                cls_preds = cls_preds[a_mask]
                if cfg.RPN_STAGE.RPN_HEAD.REG_VARIANCE.OUTPUT:
                    var_preds = var_preds[a_mask]

            if cfg.RPN_STAGE.RPN_HEAD.USE_DIRECTION_CLASSIFIER:
                if a_mask is not None:
                    if a_mask.cpu().sum() == 0:
                        a_mask = torch.from_numpy(np.array([0], dtype=np.long)).cuda()
                    dir_preds = dir_preds[a_mask]
                # dir_preds[..., 1::2] = 0 # 0.5 = torch.sigmoid(0)
                dir_labels = torch.max(dir_preds, dim=-1)[1]         
            rank_scores = torch.sigmoid(cls_preds)

            if num_class_with_bg == 1:
                top_scores = rank_scores.squeeze(-1)
                top_labels = torch.zeros(rank_scores.shape[0], device=rank_scores.device, dtype=torch.long)
            else:
                top_scores, top_labels = torch.max(rank_scores, dim=-1)
            # thresh = torch.tensor([cfg.TEST.SCORE_THRESH], device=rank_scores.device).type_as(rank_scores)
            # top_scores_keep = (top_scores >= thresh)
            # top_scores = top_scores.masked_select(top_scores_keep)
            thresh_list = [torch.tensor([cfg.TEST.SCORE_THRESH_LIST[k]], device=rank_scores.device, dtype=torch.float) for k in range(len(cfg.TEST.SCORE_THRESH_LIST))]
            N = top_scores.shape[0]
            top_scores_keep = torch.zeros((N, ), dtype=torch.uint8, device=top_scores.device)
            for k in range(len(cfg.CLASS_NAMES)):
                top_scores_tmp = (top_scores>=thresh_list[k]) & (top_labels==k)
                top_scores_keep = top_scores_keep | top_scores_tmp
            
            if cfg.FPGA_NMS:
                # apply NMS in BEV view
                nms_func = iou3d_utils.nms_gpu if cfg.RPN_STAGE.USE_ROTATED_NMS else iou3d_utils.nms_normal_gpu
                
                selected_boxes_ls = []
                selected_labels_ls = []
                selected_scores_ls = []
                selected_dir_ls = []
                if cfg.RPN_STAGE.RPN_HEAD.REG_VARIANCE.OUTPUT:
                    selected_var_ls = []
                for k in range(len(cfg.CLASS_NAMES)):
                    top_scores_keep_tmp = (top_scores>=thresh_list[k]) & (top_labels==k)
                    car_box_preds = box_preds[top_scores_keep_tmp]
                    boxes_for_nms = self.scale_nms(cfg, car_box_preds)
                    if car_box_preds.shape[0] > 0:
                        keep_idx, _ = nms_func(
                            boxes_for_nms,
                            top_scores[top_scores_keep_tmp], cfg.TEST.NMS_THRESH_LIST[k])
                        selected_boxes_ls.append(car_box_preds[keep_idx])
                        selected_labels_ls.append(top_labels[top_scores_keep_tmp][keep_idx])
                        selected_scores_ls.append(top_scores[top_scores_keep_tmp][keep_idx])
                        if cfg.RPN_STAGE.RPN_HEAD.USE_DIRECTION_CLASSIFIER:
                            selected_dir_ls.append(dir_labels[top_scores_keep_tmp][keep_idx])
                        if cfg.RPN_STAGE.RPN_HEAD.REG_VARIANCE.OUTPUT:
                            selected_var_ls.append(var_preds[top_scores_keep_tmp][keep_idx])

                if selected_boxes_ls and len(selected_boxes_ls[0])>0:
                    selected_boxes = torch.cat(selected_boxes_ls, dim=0)
                    selected_labels = torch.cat(selected_labels_ls, dim=0)
                    selected_scores = torch.cat(selected_scores_ls, dim=0)
                    if cfg.RPN_STAGE.RPN_HEAD.USE_DIRECTION_CLASSIFIER:
                        selected_dir_labels = torch.cat(selected_dir_ls, dim=0)
                    if cfg.RPN_STAGE.RPN_HEAD.REG_VARIANCE.OUTPUT:
                        selected_var = torch.cat(selected_var_ls, dim=0)
                else:
                    selected = []
                    selected_boxes = box_preds[selected]
                    selected_labels = top_labels[selected]
                    selected_scores = top_scores[selected]
                    if cfg.RPN_STAGE.RPN_HEAD.USE_DIRECTION_CLASSIFIER:
                        selected_dir_labels = dir_labels[selected]
                    if cfg.RPN_STAGE.RPN_HEAD.REG_VARIANCE.OUTPUT:
                        selected_var = var_preds[selected]
            else:
                top_scores = top_scores.masked_select(top_scores_keep)
                # apply NMS in BEV view
                nms_func = iou3d_utils.nms_gpu if cfg.RPN_STAGE.USE_ROTATED_NMS else iou3d_utils.nms_normal_gpu
                if top_scores.shape[0] != 0:
                    box_preds = box_preds[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]

                    # top_scores = top_scores[:cfg.TEST.NMS_PRE_MAXSIZE]
                    # box_preds = box_preds[:cfg.TEST.NMS_PRE_MAXSIZE]
                    # top_labels = top_labels[:cfg.TEST.NMS_PRE_MAXSIZE]
                    if cfg.RPN_STAGE.RPN_HEAD.USE_DIRECTION_CLASSIFIER:
                        dir_labels = dir_labels[top_scores_keep]  # [:cfg.TEST.NMS_PRE_MAXSIZE]
                    if cfg.RPN_STAGE.RPN_HEAD.REG_VARIANCE.OUTPUT:
                        var_preds = var_preds[top_scores_keep]
                    boxes_for_nms = self.scale_nms(cfg, box_preds)
                    if cfg.get('DEVICE', 'gpu') == 'cpu':
                        keep_idx, _  = nms_func(boxes_for_nms.cuda(), top_scores.cuda(), cfg.TEST.NMS_THRESH)
                        selected = keep_idx.cpu()
                    else:
                        keep_idx, _  = nms_func(boxes_for_nms, top_scores, cfg.TEST.NMS_THRESH)
                        selected = keep_idx
                    
                    # selected = keep_idx[:cfg.TEST.NMS_POST_MAXSIZE]
                else:
                    selected = []

                selected_boxes = box_preds[selected]
                selected_labels = top_labels[selected]
                selected_scores = top_scores[selected]

                if cfg.RPN_STAGE.RPN_HEAD.USE_DIRECTION_CLASSIFIER:
                    selected_dir_labels = dir_labels[selected]
                if cfg.RPN_STAGE.RPN_HEAD.REG_VARIANCE.OUTPUT:
                    selected_var = var_preds[selected]

            # finally generate predictions.
            if selected_boxes.shape[0] != 0:
                if cfg.RPN_STAGE.RPN_HEAD.USE_DIRECTION_CLASSIFIER:
                    opp_labels = (selected_boxes[..., -1] > 0) ^ selected_dir_labels.byte()
                    selected_boxes[..., -1] += torch.where(opp_labels, torch.tensor(np.pi).type_as(selected_boxes),
                                                           torch.tensor(0.0).type_as(selected_boxes))
                final_box_preds = selected_boxes
                final_scores = selected_scores
                final_labels = selected_labels

                # predictions
                predictions_dict = {
                    "pred_boxes": final_box_preds,
                    "pred_scores": final_scores,
                    "pred_labels": final_labels + 1,  # pcdet格式要求预测结果从1开始
                }
                if cfg.RPN_STAGE.RPN_HEAD.REG_VARIANCE.OUTPUT:
                    predictions_dict["var"] = selected_var
            else:
                dtype = batch_box_preds.dtype
                device = batch_box_preds.device
                predictions_dict = {
                    "pred_boxes": torch.zeros([0, 7], dtype=dtype, device=device),
                    "pred_scores": torch.zeros([0], dtype=dtype, device=device),
                    "pred_labels": torch.zeros([0, 4], dtype=top_labels.dtype, device=device),
                }
                if cfg.RPN_STAGE.RPN_HEAD.REG_VARIANCE.OUTPUT:
                    predictions_dict["var"] = torch.zeros([0, 7], dtype=dtype, device=device)
            predictions_dicts.append(predictions_dict)
            num += 1
        return predictions_dicts

    def forward(self, data_dict):
        cfg = self.cfg
        if self.task is not None:
            x = data_dict[f'spatial_features_2d_{self.task}']
        else:
            x = data_dict['spatial_features_2d']
        image_idx = -1
        sparse_masks = None
        ret_dicts = []
        for i, rpn_head in enumerate(self.rpn_heads):
            head_input = x
            ret_dict, _ = rpn_head(head_input, sparse_masks=sparse_masks, head_idx=i, image_idx=image_idx)
            ret_dicts.append(ret_dict)

        ret = {
            "box_preds": torch.cat([ret_dict["box_preds"] for ret_dict in ret_dicts], dim=1),
            "cls_preds": torch.cat([ret_dict["cls_preds"] for ret_dict in ret_dicts], dim=1),
        }
        if self._use_direction_classifier:
            ret["dir_cls_preds"] = torch.cat([ret_dict["dir_cls_preds"] for ret_dict in ret_dicts], dim=1)
        if cfg.RPN_STAGE.RPN_HEAD.REG_VARIANCE.OUTPUT:
            ret["var_preds"] = torch.cat([ret_dict["var_preds"] for ret_dict in ret_dicts], dim=1)

        # return ret, distiller_res
        rpn_box_preds = ret['box_preds']
        rpn_cls_preds = ret['cls_preds']
        rpn_dir_cls_preds = ret['dir_cls_preds'] if 'dir_cls_preds' in ret else None
        rpn_iou_preds = ret['iou_preds'] if 'iou_preds' in ret else None
        rpn_var_preds = ret['var_preds'] if 'var_preds' in ret else None
        self.ret_dict = {}
        self.ret_dict['rpn_cls_preds'] = rpn_cls_preds
        self.ret_dict['rpn_box_preds'] = rpn_box_preds
        self.ret_dict['rpn_dir_cls_preds'] = rpn_dir_cls_preds
        self.ret_dict['rpn_iou_preds'] = rpn_iou_preds
        self.ret_dict['rpn_var_preds'] = rpn_var_preds

        if os.getenv("CALIB") == 'True':
            task_folder = f"{self.task}-head/" if self.task != 'fusion' else "fuser-fusion-head"
            calib_path = os.getenv("CALIB_PATH")
            save_dir = os.path.join(calib_path, task_folder)
            if os.getenv("det_cls_with_sigmoid") == 'True':
                save_np(os.path.join(save_dir, "outputs/det_pred_dicts_{}_cls/{}".format(self.task, self.cnt)), torch.sigmoid(rpn_cls_preds.unsqueeze(0)))
            else:
                save_np(os.path.join(save_dir, "outputs/det_pred_dicts_{}_cls/{}".format(self.task, self.cnt)), rpn_cls_preds.unsqueeze(0))
            save_np(os.path.join(save_dir, "outputs/det_pred_dicts_{}_box/{}".format(self.task, self.cnt)), rpn_box_preds.unsqueeze(0))
            save_np(os.path.join(save_dir, "outputs/det_pred_dicts_{}_dir_cls/{}".format(self.task, self.cnt)), rpn_dir_cls_preds.unsqueeze(0))

        #
        if self.training:
            self.labels = data_dict['box_labels']  # -1为背景，类别从0开始
            self.reg_targets = data_dict['reg_targets']
            self.anchors = data_dict['anchors']
        else:
            if self.cfg.get('ONNX_FORWARD', False):
                self.ret_dict.pop('rpn_iou_preds')
                self.ret_dict.pop('rpn_var_preds')
                self.ret_dict['rpn_cls_preds'] = rpn_cls_preds.view(1, rpn_cls_preds.shape)
                self.ret_dict['rpn_box_preds'] = rpn_box_preds
                self.ret_dict['rpn_dir_cls_preds'] = rpn_dir_cls_preds
                data_dict[f'det_pred_dicts_{self.task}'] = self.ret_dict
            else:
                data_dict_task = {}
                pred_dicts = self.predict_rpn(data_dict, self.ret_dict)
                data_dict_task['final_box_dicts'] = pred_dicts
                if self.task is not None:
                    data_dict[f'det_pred_dicts_{self.task}'] = data_dict_task
                else:
                    data_dict.update(data_dict_task)
        self.cnt += 1
        return data_dict
    
    def forward_onnx(self, data_dict):
        cfg = self.cfg
        if self.task is not None:
            x = data_dict[f'spatial_features_2d_{self.task}']
        else:
            x = data_dict['spatial_features_2d']
        image_idx = -1
        sparse_masks = None
        ret_dicts = []
        for i, rpn_head in enumerate(self.rpn_heads):
            head_input = x
            ret_dict, _ = rpn_head(head_input, sparse_masks=sparse_masks, head_idx=i, image_idx=image_idx)
            ret_dicts.append(ret_dict)

        ret = {
            "box_preds": torch.cat([ret_dict["box_preds"] for ret_dict in ret_dicts], dim=1),
            "cls_preds": torch.cat([ret_dict["cls_preds"] for ret_dict in ret_dicts], dim=1),
        }
        if self._use_direction_classifier:
            ret["dir_cls_preds"] = torch.cat([ret_dict["dir_cls_preds"] for ret_dict in ret_dicts], dim=1)
        if cfg.RPN_STAGE.RPN_HEAD.REG_VARIANCE.OUTPUT:
            ret["var_preds"] = torch.cat([ret_dict["var_preds"] for ret_dict in ret_dicts], dim=1)

        # return ret, distiller_res
        rpn_box_preds = ret['box_preds']
        rpn_cls_preds = ret['cls_preds']
        rpn_dir_cls_preds = ret['dir_cls_preds'] if 'dir_cls_preds' in ret else None
        # rpn_iou_preds = ret['iou_preds'] if 'iou_preds' in ret else None
        # rpn_var_preds = ret['var_preds'] if 'var_preds' in ret else None
        self.ret_dict = {}
        self.ret_dict['rpn_cls_preds'] = rpn_cls_preds.unsqueeze(0)
        # self.ret_dict['rpn_cls_preds'] = torch.sigmoid(rpn_cls_preds.unsqueeze(0))
        self.ret_dict['rpn_box_preds'] = rpn_box_preds.unsqueeze(0)
        self.ret_dict['rpn_dir_cls_preds'] = rpn_dir_cls_preds.unsqueeze(0)
        # self.ret_dict['rpn_iou_preds'] = rpn_iou_preds
        # self.ret_dict['rpn_var_preds'] = rpn_var_preds
        data_dict[f'det_pred_dicts_{self.task}'] = self.ret_dict
        return data_dict
