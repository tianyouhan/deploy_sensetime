from abc import ABCMeta, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import box_torch_ops
from . import box_np_ops
import pcdet.ops.iou3d_nms.iou3d_nms_utils as iou3d_utils

def get_bin_based_reg_loss_lidar(pred_reg, reg_label, loc_scope, loc_bin_size, num_head_bin, anchor_size,
                                 get_xz_fine=True, get_y_by_bin=False, loc_y_scope=0.5, loc_y_bin_size=0.25,
                                 get_ry_fine=False):
    """
    :param pred_reg: (N, C)
    :param reg_label: (N, 7) [dx, dy, dz, h, w, l, ry]
    :param loc_scope: constant
    :param loc_bin_size: constant
    :param num_head_bin: constant
    :param anchor_size: (N, 3) or (3)
    :param get_xz_fine:
    :param get_y_by_bin:
    :param loc_y_scope:
    :param loc_y_bin_size:
    :param get_ry_fine:
    :return:
    """
    per_loc_bin_num = int(loc_scope / loc_bin_size) * 2
    loc_y_bin_num = int(loc_y_scope / loc_y_bin_size) * 2

    reg_loss_dict = {}
    loc_loss = 0

    # xz localization loss
    x_offset_label, y_offset_label, z_offset_label = reg_label[:, 0], reg_label[:, 2], reg_label[:, 1]
    x_shift = torch.clamp(x_offset_label + loc_scope, 0, loc_scope * 2 - 1e-3)
    z_shift = torch.clamp(z_offset_label + loc_scope, 0, loc_scope * 2 - 1e-3)
    x_bin_label = (x_shift / loc_bin_size).floor().long()
    z_bin_label = (z_shift / loc_bin_size).floor().long()

    x_bin_l, x_bin_r = 0, per_loc_bin_num
    z_bin_l, z_bin_r = per_loc_bin_num, per_loc_bin_num * 2
    start_offset = z_bin_r

    loss_x_bin = F.cross_entropy(pred_reg[:, x_bin_l: x_bin_r], x_bin_label)
    loss_z_bin = F.cross_entropy(pred_reg[:, z_bin_l: z_bin_r], z_bin_label)
    reg_loss_dict['loss_x_bin'] = loss_x_bin.item()
    reg_loss_dict['loss_z_bin'] = loss_z_bin.item()
    loc_loss += loss_x_bin + loss_z_bin

    if get_xz_fine:
        x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
        z_res_l, z_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
        start_offset = z_res_r

        x_res_label = x_shift - (x_bin_label.float() * loc_bin_size + loc_bin_size / 2)
        z_res_label = z_shift - (z_bin_label.float() * loc_bin_size + loc_bin_size / 2)
        x_res_norm_label = x_res_label / loc_bin_size
        z_res_norm_label = z_res_label / loc_bin_size

        x_bin_onehot = torch.cuda.FloatTensor(x_bin_label.size(0), per_loc_bin_num).zero_()
        x_bin_onehot.scatter_(1, x_bin_label.view(-1, 1).long(), 1)
        z_bin_onehot = torch.cuda.FloatTensor(z_bin_label.size(0), per_loc_bin_num).zero_()
        z_bin_onehot.scatter_(1, z_bin_label.view(-1, 1).long(), 1)

        loss_x_res = F.smooth_l1_loss((pred_reg[:, x_res_l: x_res_r] * x_bin_onehot).sum(dim=1), x_res_norm_label)
        loss_z_res = F.smooth_l1_loss((pred_reg[:, z_res_l: z_res_r] * z_bin_onehot).sum(dim=1), z_res_norm_label)
        reg_loss_dict['loss_x_res'] = loss_x_res.item()
        reg_loss_dict['loss_z_res'] = loss_z_res.item()
        loc_loss += loss_x_res + loss_z_res

    # y localization loss
    if get_y_by_bin:
        y_bin_l, y_bin_r = start_offset, start_offset + loc_y_bin_num
        y_res_l, y_res_r = y_bin_r, y_bin_r + loc_y_bin_num
        start_offset = y_res_r

        y_shift = torch.clamp(y_offset_label + loc_y_scope, 0, loc_y_scope * 2 - 1e-3)
        y_bin_label = (y_shift / loc_y_bin_size).floor().long()
        y_res_label = y_shift - (y_bin_label.float() * loc_y_bin_size + loc_y_bin_size / 2)
        y_res_norm_label = y_res_label / loc_y_bin_size

        y_bin_onehot = torch.cuda.FloatTensor(y_bin_label.size(0), loc_y_bin_num).zero_()
        y_bin_onehot.scatter_(1, y_bin_label.view(-1, 1).long(), 1)

        loss_y_bin = F.cross_entropy(pred_reg[:, y_bin_l: y_bin_r], y_bin_label)
        loss_y_res = F.smooth_l1_loss((pred_reg[:, y_res_l: y_res_r] * y_bin_onehot).sum(dim=1), y_res_norm_label)

        reg_loss_dict['loss_y_bin'] = loss_y_bin.item()
        reg_loss_dict['loss_y_res'] = loss_y_res.item()

        loc_loss += loss_y_bin + loss_y_res
    else:
        y_offset_l, y_offset_r = start_offset, start_offset + 1
        start_offset = y_offset_r

        loss_y_offset = F.smooth_l1_loss(pred_reg[:, y_offset_l: y_offset_r].sum(dim=1), y_offset_label)
        reg_loss_dict['loss_y_offset'] = loss_y_offset.item()
        loc_loss += loss_y_offset

    # angle loss
    ry_bin_l, ry_bin_r = start_offset, start_offset + num_head_bin
    ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + num_head_bin

    ry_label = reg_label[:, 6]

    if get_ry_fine:
        # divide pi/2 into several bins
        angle_per_class = (np.pi / 2) / num_head_bin

        ry_label = ry_label % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (ry_label > np.pi * 0.5) & (ry_label < np.pi * 1.5)
        ry_label[opposite_flag] = (ry_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        shift_angle = (ry_label + np.pi * 0.5) % (2 * np.pi)  # (0 ~ pi)

        shift_angle = torch.clamp(shift_angle - np.pi * 0.25, min=1e-3, max=np.pi * 0.5 - 1e-3)  # (0, pi/2)

        # bin center is (5, 10, 15, ..., 85)
        ry_bin_label = (shift_angle / angle_per_class).floor().long()
        ry_res_label = shift_angle - (ry_bin_label.float() * angle_per_class + angle_per_class / 2)
        ry_res_norm_label = ry_res_label / (angle_per_class / 2)

    else:
        # divide 2pi into several bins
        angle_per_class = (2 * np.pi) / num_head_bin
        heading_angle = ry_label % (2 * np.pi)  # 0 ~ 2pi

        shift_angle = (heading_angle + angle_per_class / 2) % (2 * np.pi)
        ry_bin_label = (shift_angle / angle_per_class).floor().long()
        ry_res_label = shift_angle - (ry_bin_label.float() * angle_per_class + angle_per_class / 2)
        ry_res_norm_label = ry_res_label / (angle_per_class / 2)

    ry_bin_onehot = torch.cuda.FloatTensor(ry_bin_label.size(0), num_head_bin).zero_()
    ry_bin_onehot.scatter_(1, ry_bin_label.view(-1, 1).long(), 1)
    loss_ry_bin = F.cross_entropy(pred_reg[:, ry_bin_l:ry_bin_r], ry_bin_label)
    loss_ry_res = F.smooth_l1_loss((pred_reg[:, ry_res_l: ry_res_r] * ry_bin_onehot).sum(dim=1), ry_res_norm_label)

    reg_loss_dict['loss_ry_bin'] = loss_ry_bin.item()
    reg_loss_dict['loss_ry_res'] = loss_ry_res.item()
    angle_loss = loss_ry_bin + loss_ry_res

    # size loss
    size_res_l, size_res_r = ry_res_r, ry_res_r + 3
    assert pred_reg.shape[1] == size_res_r, '%d vs %d' % (pred_reg.shape[1], size_res_r)

    size_res_norm_label = (reg_label[:, 3:6] - anchor_size) / anchor_size
    size_res_norm = pred_reg[:, size_res_l:size_res_r]
    size_loss = F.smooth_l1_loss(size_res_norm, size_res_norm_label)

    # Total regression loss
    reg_loss_dict['loss_loc'] = loc_loss
    reg_loss_dict['loss_angle'] = angle_loss
    reg_loss_dict['loss_size'] = size_loss

    return loc_loss, angle_loss, size_loss, reg_loss_dict


def huber_loss(error, delta):
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic ** 2 + delta * linear
    return losses


def get_corner_loss_lidar(pred_bbox3d, gt_bbox3d):
    """
    :param pred_bbox3d: (N, 7)
    :param gt_bbox3d: (N, 7)
    :return: corner_loss: (N)
    """
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

    pred_box_corners = box_torch_ops.center_to_corner_box3d(
        pred_bbox3d[:, 0:3], pred_bbox3d[:, 3:6], pred_bbox3d[:, 6], [0.5, 0.5, 0], axis=2)
    gt_box_corners = box_torch_ops.center_to_corner_box3d(
        gt_bbox3d[:, 0:3], gt_bbox3d[:, 3:6], gt_bbox3d[:, 6], [0.5, 0.5, 0], axis=2)

    gt_bbox3d_flip = gt_bbox3d.clone()
    gt_bbox3d_flip[:, 6] += np.pi
    gt_box_corners_flip = box_torch_ops.center_to_corner_box3d(
        gt_bbox3d_flip[:, 0:3], gt_bbox3d_flip[:, 3:6], gt_bbox3d_flip[:, 6], [0.5, 0.5, 0], axis=2)

    corner_dist = torch.min(torch.norm(pred_box_corners - gt_box_corners, dim=2),
                            torch.norm(pred_box_corners - gt_box_corners_flip, dim=2))  # (N, 8)
    corner_loss = huber_loss(corner_dist, delta=1.0)  # (N, 8)

    return corner_loss.mean(dim=1)


class Loss(object):
    """Abstract base class for loss functions."""
    __metaclass__ = ABCMeta

    def __call__(self,
                 prediction_tensor,
                 target_tensor,
                 ignore_nan_targets=False,
                 scope=None,
                 **params):
        """Call the loss function.

    Args:
      prediction_tensor: an N-d tensor of shape [batch, anchors, ...]
        representing predicted quantities.
      target_tensor: an N-d tensor of shape [batch, anchors, ...] representing
        regression or classification targets.
      ignore_nan_targets: whether to ignore nan targets in the loss computation.
        E.g. can be used if the target tensor is missing groundtruth data that
        shouldn't be factored into the loss.
      scope: Op scope name. Defaults to 'Loss' if None.
      **params: Additional keyword arguments for specific implementations of
              the Loss.

    Returns:
      loss: a tensor representing the value of the loss function.
    """
        if ignore_nan_targets:
            target_tensor = torch.where(torch.isnan(target_tensor),
                                        prediction_tensor,
                                        target_tensor)
        return self._compute_loss(prediction_tensor, target_tensor, **params)

    @abstractmethod
    def _compute_loss(self, prediction_tensor, target_tensor, **params):
        """Method to be overridden by implementations.

    Args:
      prediction_tensor: a tensor representing predicted quantities
      target_tensor: a tensor representing regression or classification targets
      **params: Additional keyword arguments for specific implementations of
              the Loss.

    Returns:
      loss: an N-d tensor of shape [batch, anchors, ...] containing the loss per
        anchor
    """
        pass


class WeightedSmoothL1LocalizationLoss(Loss):
    """Smooth L1 localization loss function.
    The smooth L1_loss is defined elementwise as .5 x^2 if |x|<1 and |x|-.5
    otherwise, where x is the difference between predictions and target.

    See also Equation (3) in the Fast R-CNN paper by Ross Girshick (ICCV 2015)
    """

    def __init__(self, cfg, sigma=3.0, code_weights=None, codewise=True):
        super().__init__()
        self.cfg = cfg
        self._sigma = sigma
        if code_weights is not None:
            self._code_weights = np.array(code_weights, dtype=np.float32)
            self._code_weights = torch.from_numpy(self._code_weights).cuda()
        else:
            self._code_weights = None
        self._codewise = codewise

    def _compute_loss(self, prediction_tensor, target_tensor, weights=None, NRE=None, var_preds=None):
        """Compute loss function.
        Args:
          prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            code_size] representing the (encoded) predicted locations of objects.
          target_tensor: A float tensor of shape [batch_size, num_anchors,
            code_size] representing the regression targets
          weights: a float tensor of shape [batch_size, num_anchors]

        Returns:
          loss: a float tensor of shape [batch_size, num_anchors] tensor
            representing the value of the loss function.
        """
        cfg = self.cfg
        diff = prediction_tensor - target_tensor
        if cfg.RPN_STAGE.NORM_REG_ENCOD:
            for i in range(0,len(cfg.RPN_STAGE.CIL)):
                diff[:,cfg.RPN_STAGE.CIL[i][0]:cfg.RPN_STAGE.CIL[i][1],3:-1] = \
                    NRE.weight[i]*prediction_tensor[:,cfg.RPN_STAGE.CIL[i][0]:cfg.RPN_STAGE.CIL[i][1],3:-1] - \
                    target_tensor[:,cfg.RPN_STAGE.CIL[i][0]:cfg.RPN_STAGE.CIL[i][1],3:-1]
        if self._code_weights is not None:
            code_weights = self._code_weights.type_as(prediction_tensor)
            diff = code_weights.view(1, 1, -1) * diff
        abs_diff = torch.abs(diff)
        abs_diff_lt_1 = torch.le(abs_diff, 1 / (self._sigma ** 2)).type_as(abs_diff)
        loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * self._sigma, 2) \
               + (abs_diff - 0.5 / (self._sigma ** 2)) * (1. - abs_diff_lt_1)
        if cfg.RPN_STAGE.RPN_HEAD.REG_VARIANCE.OUTPUT:
            if cfg.RPN_STAGE.RPN_HEAD.REG_VARIANCE.KL_LOSS:
                # var_preds = 1.5 * torch.sigmoid(var_preds)
                # loss = torch.exp(-var_preds) * loss + var_preds * 0.5

                var_preds = torch.sigmoid(var_preds)
                loss = torch.exp(-2.0 * (var_preds - 0.5)) * loss + 0.25 * var_preds * 0.5

                # var_preds = torch.sigmoid(var_preds)
                # loss_weighted = loss.clone()
                # loss_weighted[:,0:458752,:] = torch.exp(-2.0 * (var_preds[:,0:458752,:] - 0.5)) * loss[:,0:458752,:] + 0.3 * var_preds[:,0:458752,:] * 0.5
                # loss_weighted[:,917504:,:] = torch.exp(-2.0 * (var_preds[:,917504:,:] - 0.5)) * loss[:,917504:,:] + 0.3 * var_preds[:,917504:,:] * 0.5

                # var_preds = torch.sigmoid(var_preds)
                # loss_weighted = loss.clone()
                # loss_weighted[:,0:458752,:] = torch.exp(-2.0 * (var_preds[:,0:458752,:] - 0.5)) * loss[:,0:458752,:] + 0.3 * var_preds[:,0:458752,:] * 0.5
                # loss_weighted[:,1376256:,:] = torch.exp(-2.0 * (var_preds[:,1376256:,:] - 0.5)) * loss[:,1376256:,:] + 0.3 * var_preds[:,1376256:,:] * 0.5

                # var_preds = torch.sigmoid(var_preds)
                # loss_weighted = loss.clone()
                # loss_weighted[:,0:458752,:] = torch.exp(-2.75 * (var_preds[:,0:458752,:] - 0.5)) * loss[:,0:458752,:] + 0.2 * var_preds[:,0:458752,:] * 0.5
                # loss_weighted[:,458752:917504,:] = torch.exp(-1.5 * (var_preds[:,458752:917504,:] - 0.5)) * loss[:,458752:917504,:] + 0.2 * var_preds[:,458752:917504,:] * 0.5
                # loss_weighted[:,917504:1376256,:] = torch.exp(-1.5 * (var_preds[:,917504:1376256,:] - 0.5)) * loss[:,917504:1376256,:] + 0.3 * var_preds[:,917504:1376256,:] * 0.5
                # loss_weighted[:,1376256:,:] = torch.exp(-2.0 * (var_preds[:,1376256:,:] - 0.5)) * loss[:,1376256:,:] + 0.2 * var_preds[:,1376256:,:] * 0.5

            # elif cfg.RPN_STAGE.RPN_HEAD.REG_VARIANCE.KLD_LOSS:
            #     import math
            #     label_val = float(cfg.RPN_STAGE.RPN_HEAD.REG_VARIANCE.KLD_LOSS_LABEL_VAR)
            #     var_preds = torch.sigmoid(var_preds)
            #     loss = torch.exp(-2.0 * (var_preds - 0.5)) * loss + \
            #         0.25 * (var_preds - math.log(label_val) + label_val * torch.exp(-2.0 * var_preds)) * 0.5
        if self._codewise:
            anchorwise_smooth_l1norm = loss
            # if cfg.RPN_STAGE.RPN_HEAD.REG_ATTRIBUTES_WEIGHTS.USE:
            #     anchorwise_smooth_l1norm[:,:,0:2] *= cfg.RPN_STAGE.RPN_HEAD.REG_ATTRIBUTES_WEIGHTS.BEV
            #     anchorwise_smooth_l1norm[:,:,2:3] *= cfg.RPN_STAGE.RPN_HEAD.REG_ATTRIBUTES_WEIGHTS.H3D
            #     anchorwise_smooth_l1norm[:,:,3:5] *= cfg.RPN_STAGE.RPN_HEAD.REG_ATTRIBUTES_WEIGHTS.BEV
            #     anchorwise_smooth_l1norm[:,:,5:6] *= cfg.RPN_STAGE.RPN_HEAD.REG_ATTRIBUTES_WEIGHTS.H3D
            #     anchorwise_smooth_l1norm[:,:,6:7] *= cfg.RPN_STAGE.RPN_HEAD.REG_ATTRIBUTES_WEIGHTS.BEV
            if weights is not None:
                anchorwise_smooth_l1norm *= weights.unsqueeze(-1)
        else:
            anchorwise_smooth_l1norm = torch.sum(loss, 2)  # * weights
            if weights is not None:
                anchorwise_smooth_l1norm *= weights
        return anchorwise_smooth_l1norm


class AuthenticSmoothL1LocalizationLoss(Loss):
    def __init__(self, cfg):
        super(AuthenticSmoothL1LocalizationLoss, self).__init__()
        self.mu = 0.02
        self.bins = cfg.RPN_STAGE.GHM_REG_BINS
        self.edges = [float(x) / self.bins for x in range(self.bins+1)]
        self.edges[-1] = 1e3
        self.momentum = cfg.RPN_STAGE.GHM_REG_MOM
        if self.momentum > 0:
            self.acc_sum = [0.0 for _ in range(self.bins)]

    def _compute_loss(self, prediction_tensor, target_tensor, weights=None, NRE=None):
        mu = self.mu
        edges = self.edges
        mmt = self.momentum
        weights = weights.unsqueeze(-1)
        weights = torch.cat([weights]*7,dim=-1)
        diff = prediction_tensor - target_tensor
        loss = torch.sqrt(diff * diff + mu * mu) - mu

        # g = torch.abs(diff / torch.sqrt(mu * mu + diff * diff)).detach()
        # weights_ghm = torch.zeros_like(g)
        # valid = weights > 0
        # tot = max(valid.float().sum().item(), 1.0)
        # n = 0
        # for i in range(self.bins):
        #     inds = (g >= edges[i]) & (g < edges[i+1]) & valid
        #     num_in_bin = inds.sum().item()
        #     if num_in_bin > 0:
        #         n += 1
        #         if mmt > 0:
        #             self.acc_sum[i] = mmt * self.acc_sum[i] \
        #                 + (1 - mmt) * num_in_bin
        #             weights_ghm[inds] = tot / self.acc_sum[i]
        #         else:
        #             weights_ghm[inds] = tot / num_in_bin
        # if n > 0:
        #     weights_ghm /= n
        # loss = loss * weights_ghm

        return loss * weights


class WeightedSoftmaxClassificationLoss(Loss):
    """Softmax loss function."""

    def __init__(self, logit_scale=1.0):
        """Constructor.

    Args:
      logit_scale: When this value is high, the prediction is "diffused" and
                   when this value is low, the prediction is made peakier.
                   (default 1.0)

    """
        self._logit_scale = logit_scale

    def _compute_loss(self, prediction_tensor, target_tensor, weights):
        """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
      weights: a float tensor of shape [batch_size, num_anchors]

    Returns:
      loss: a float tensor of shape [batch_size, num_anchors]
        representing the value of the loss function.
    """
        num_classes = prediction_tensor.shape[-1]
        prediction_tensor = torch.div(
            prediction_tensor, self._logit_scale)
        per_row_cross_ent = (_softmax_cross_entropy_with_logits(
            labels=target_tensor.view(-1, num_classes),
            logits=prediction_tensor.view(-1, num_classes)))
        return per_row_cross_ent.view(weights.shape) * weights


def indices_to_dense_vector(indices,
                            size,
                            indices_value=1.,
                            default_value=0,
                            dtype=np.float32):
    """Creates dense vector with indices set to specific value and rest to zeros.

    This function exists because it is unclear if it is safe to use
    tf.sparse_to_dense(indices, [size], 1, validate_indices=False)
    with indices which are not ordered.
    This function accepts a dynamic size (e.g. tf.shape(tensor)[0])

    Args:
    indices: 1d Tensor with integer indices which are to be set to
        indices_values.
    size: scalar with size (integer) of output Tensor.
    indices_value: values of elements specified by indices in the output vector
    default_value: values of other elements in the output vector.
    dtype: data type.

    Returns:
    dense 1D Tensor of shape [size] with indices set to indices_values and the
        rest set to default_value.
    """
    dense = torch.zeros(size).fill_(default_value)
    dense[indices] = indices_value

    return dense


class QualityFocalLoss(Loss):
    """Generalized Focal Loss.
    """
    def __init__(self, cfg, gamma=2.0):
        self._gamma = gamma
        self.cfg = cfg

    def NearestIOULabelNoGrad(self, anchors, box_preds, reg_targets, target_tensor):
        from .box_coder import ResidualCoder
        with torch.no_grad():
            anchors_np = anchors.cpu().detach().numpy()
            box_preds_np = box_preds.cpu().detach().numpy()
            reg_targets_np = reg_targets.cpu().detach().numpy()
            for b_i in range(box_preds_np.shape[0]):
                pos_inds, pos_gt_inds = np.where(target_tensor[b_i].cpu() > 0)
                if pos_inds.shape[0] > 0:
                    box_preds_np_pos = ResidualCoder.decode_np(box_preds_np[b_i][pos_inds], anchors_np[pos_inds])
                    reg_targets_np_pos = ResidualCoder.decode_np(reg_targets_np[b_i][pos_inds], anchors_np[pos_inds])
                    box_preds_np_pos = box_np_ops.rbbox2d_to_near_bbox(box_preds_np_pos[:, [0, 1, 3, 4, 6]])
                    reg_targets_np_pos = box_np_ops.rbbox2d_to_near_bbox(reg_targets_np_pos[:, [0, 1, 3, 4, 6]])
                    for j in range(pos_inds.shape[0]):
                        iou = box_np_ops.iou_jit(box_preds_np_pos[j:j+1], reg_targets_np_pos[j:j+1], eps=0.0)
                        target_tensor[b_i,pos_inds[j],pos_gt_inds[j]] = float(iou[0,0])

    def rbbox2d_to_near_bbox_torch(self, boxes):
        rots = boxes[..., -1]
        rots_limited = rots - torch.floor(rots / np.pi + 0.5) * np.pi
        rots_0_pi_div_2 = torch.abs(rots_limited)
        # cond = (rots_0_pi_div_2 > (np.pi/4))
        # bboxes_center = boxes[:, :4].clone()
        # bboxes_center[cond, :] = bboxes_center[cond, [0, 1, 3, 2]]
        cond = (rots_0_pi_div_2 > (np.pi/4)).unsqueeze(-1)
        bboxes_center = torch.where(cond, boxes[:, [0, 1, 3, 2]], boxes[:, :4])
        centers = bboxes_center[:, :2]
        dims = bboxes_center[:, 2:]
        bboxes = torch.cat([centers - dims / 2, centers + dims / 2], dim=-1)
        return bboxes

    def nearest_iou_single_ops_torch(self, boxes, query_boxes, eps=0.0):
        box_area = ((query_boxes[2] - query_boxes[0] + eps) *
                    (query_boxes[3] - query_boxes[1] + eps))
        iw = (min(boxes[2], query_boxes[2]) -
              max(boxes[0], query_boxes[0]) + eps)
        iou = 0.0
        if iw > 0:
            ih = (min(boxes[3], query_boxes[3]) -
                  max(boxes[1], query_boxes[1]) + eps)
            if ih > 0:
                ua = (
                    (boxes[2] - boxes[0] + eps) *
                    (boxes[3] - boxes[1] + eps) + box_area - iw * ih)
                iou = iw * ih / ua
        return iou

    def NearestIOULabel(self, anchors, box_preds, reg_targets, target_tensor):
        from .box_coder import ResidualCoder
        for b_i in range(box_preds.shape[0]):
            pos_inds, pos_gt_inds = np.where(target_tensor[b_i].cpu() > 0)
            if pos_inds.shape[0] > 0:
                box_preds_pos = ResidualCoder.decode_torch(box_preds[b_i][pos_inds], anchors[pos_inds])
                reg_targets_pos = ResidualCoder.decode_torch(reg_targets[b_i][pos_inds], anchors[pos_inds])
                box_preds_pos_bv = self.rbbox2d_to_near_bbox_torch(box_preds_pos[:, [0, 1, 3, 4, 6]])
                reg_targets_pos_bv = self.rbbox2d_to_near_bbox_torch(reg_targets_pos[:, [0, 1, 3, 4, 6]])
                for j in range(pos_inds.shape[0]):
                    iou = self.nearest_iou_single_ops_torch(box_preds_pos_bv[j], reg_targets_pos_bv[j], eps=0.0)
                    target_tensor[b_i,pos_inds[j],pos_gt_inds[j]] = iou

    def BevRotatedIOULabelNoGrad(self, anchors, box_preds, reg_targets, target_tensor):
        from .box_coder import ResidualCoder
        for b_i in range(box_preds.shape[0]):
            pos_inds, pos_gt_inds = np.where(target_tensor[b_i].cpu() > 0)
            if pos_inds.shape[0] > 0:
                box_preds_pos = ResidualCoder.decode_torch(box_preds[b_i][pos_inds], anchors[pos_inds])
                reg_targets_pos = ResidualCoder.decode_torch(reg_targets[b_i][pos_inds], anchors[pos_inds])
                for j in range(pos_inds.shape[0]):
                    iou = iou3d_utils.boxes_iou_bev_lidar(box_preds_pos[j:j+1], reg_targets_pos[j:j+1])
                    target_tensor[b_i,pos_inds[j],pos_gt_inds[j]] = iou[0][0]

    def Rotated3dIOULabelNoGrad(self, anchors, box_preds, reg_targets, target_tensor):
        from .box_coder import ResidualCoder
        for b_i in range(box_preds.shape[0]):
            pos_inds, pos_gt_inds = np.where(target_tensor[b_i].cpu() > 0)
            if pos_inds.shape[0] > 0:
                box_preds_pos = ResidualCoder.decode_torch(box_preds[b_i][pos_inds], anchors[pos_inds])
                reg_targets_pos = ResidualCoder.decode_torch(reg_targets[b_i][pos_inds], anchors[pos_inds])
                for j in range(pos_inds.shape[0]):
                    iou = iou3d_utils.boxes_iou3d_gpu_lidar(box_preds_pos[j:j+1], reg_targets_pos[j:j+1])
                    target_tensor[b_i,pos_inds[j],pos_gt_inds[j]] = iou[0][0]

    def _compute_loss(self,
                      prediction_tensor,
                      target_tensor,
                      weights,
                      anchors=None,
                      box_preds=None,
                      reg_targets=None,
                      learnedweight=None,
                      class_indices=None
                      ):
        cfg = self.cfg
        if weights.shape.__len__() == 2:
            weights = weights.unsqueeze(2)
            weights = torch.cat([weights]*4,dim=-1)
            if not cfg.RPN_STAGE.AUTOWEIGHT:
                weights[:,:,0] = weights[:,:,0] * cfg.TRAIN.CAR_LS
                weights[:,:,1] = weights[:,:,1] * cfg.TRAIN.PED_LS
                weights[:,:,2] = weights[:,:,2] * cfg.TRAIN.CYC_LS
                weights[:,:,3] = weights[:,:,3] * cfg.TRAIN.TRUCK_LS
            else:
                weights_auto = weights.clone()
                if (not cfg.RPN_STAGE.AUTO_ONLY_PED_CYC) and (not cfg.RPN_STAGE.AUTO_ONLY_PED_CYC_TRUCK):
                    weights_auto[:,:,0] = weights[:,:,0] * torch.exp(-1 * learnedweight.cls_weight[0])
                    weights_auto[:,:,1] = weights[:,:,1] * torch.exp(-1 * learnedweight.cls_weight[1])
                    weights_auto[:,:,2] = weights[:,:,2] * torch.exp(-1 * learnedweight.cls_weight[2])
                    weights_auto[:,:,3] = weights[:,:,3] * torch.exp(-1 * learnedweight.cls_weight[3])
                elif cfg.RPN_STAGE.AUTO_ONLY_PED_CYC:
                    weights_auto[:,:,0] = weights[:,:,0] * cfg.TRAIN.CAR_LS
                    weights_auto[:,:,1] = weights[:,:,1] * torch.exp(-1 * learnedweight.cls_weight[0])
                    weights_auto[:,:,2] = weights[:,:,2] * torch.exp(-1 * learnedweight.cls_weight[1])
                    weights_auto[:,:,3] = weights[:,:,3] * cfg.TRAIN.TRUCK_LS
                else:
                    weights_auto[:,:,0] = weights[:,:,0] * cfg.TRAIN.CAR_LS
                    weights_auto[:,:,1] = weights[:,:,1] * torch.exp(-1 * learnedweight.cls_weight[0])
                    weights_auto[:,:,2] = weights[:,:,2] * torch.exp(-1 * learnedweight.cls_weight[1])
                    weights_auto[:,:,3] = weights[:,:,3] * torch.exp(-1 * learnedweight.cls_weight[2])

            if cfg.RPN_STAGE.AUTOWEIGHT and cfg.RPN_STAGE.AUTO_POS_CLS:
                weights_auto_pos = weights_auto.clone()
                weights_auto_pos[target_tensor[:,:,:]==1] = weights_auto[target_tensor[:,:,:]==1] * torch.exp(-1 * learnedweight.cls_weight[-1])
            else:
                if cfg.RPN_STAGE.AUTOWEIGHT:
                    weights_auto_final = weights_auto.clone()
                    weights_auto_final[target_tensor[:,:,:]==1] = weights_auto[target_tensor[:,:,:]==1] * cfg.TRAIN.POS_LS
                else:
                    weights[target_tensor[:,:,:]==1] *= cfg.TRAIN.POS_LS

        if cfg.RPN_STAGE.GFL.QFL_NEAREST_IOU:
            if cfg.RPN_STAGE.GFL.QFL_NO_GRAD:
                self.NearestIOULabelNoGrad(anchors, box_preds, reg_targets, target_tensor)
            else:
                self.NearestIOULabel(anchors, box_preds, reg_targets, target_tensor)

        if cfg.RPN_STAGE.GFL.QFL_BEV_ROTATED_IOU:
            if cfg.RPN_STAGE.GFL.QFL_NO_GRAD:
                self.BevRotatedIOULabelNoGrad(anchors, box_preds, reg_targets, target_tensor)
            else:
                assert False, "BEV_ROTATED_IOU_GRAD not supported"

        if cfg.RPN_STAGE.GFL.QFL_3D_ROTATED_IOU:
            if cfg.RPN_STAGE.GFL.QFL_NO_GRAD:
                self.Rotated3dIOULabelNoGrad(anchors, box_preds, reg_targets, target_tensor)
            else:
                assert False, "3D_ROTATED_IOU_GRAD not supported"

        prediction_probabilities = torch.sigmoid(prediction_tensor)
        # p_t = -((target_tensor * torch.log(prediction_probabilities)) +
        #        ((1 - target_tensor) * torch.log(1 - prediction_probabilities)))
        # p_t = (_sigmoid_cross_entropy_with_logits(
        #     labels=target_tensor, logits=prediction_tensor))
        p_t = F.binary_cross_entropy_with_logits(prediction_probabilities, target_tensor, reduction='none')
        modulating_factor = torch.pow(torch.abs(target_tensor-prediction_probabilities), self._gamma)
        focal_cross_entropy_loss = (modulating_factor * p_t)
        if cfg.RPN_STAGE.GFL.QFL_ALPHA > 0:
            alpha = cfg.RPN_STAGE.GFL.QFL_ALPHA
            alpha_weight_factor = target_tensor * alpha + (1 - target_tensor) * (1 - alpha)
            focal_cross_entropy_loss = focal_cross_entropy_loss * alpha_weight_factor

        if cfg.RPN_STAGE.AUTOWEIGHT and cfg.RPN_STAGE.AUTO_POS_CLS: 
            return focal_cross_entropy_loss * weights_auto_pos
        else:
            if cfg.RPN_STAGE.AUTOWEIGHT:
                return focal_cross_entropy_loss * weights_auto_final
            else:
                return focal_cross_entropy_loss * weights


class IoUPredLoss(nn.Module):
    """IoUPredLoss.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def BevRotatedIOULabelNoGrad(self, anchors, box_preds, reg_targets, target_tensor, one_hot_targets):
        from .box_coder import ResidualCoder
        for b_i in range(box_preds.shape[0]):
            pos_inds = np.where(one_hot_targets[b_i].cpu() > 0)[0]
            if pos_inds.shape[0] > 0:
                box_preds_pos = ResidualCoder.decode_torch(box_preds[b_i][pos_inds], anchors[pos_inds])
                reg_targets_pos = ResidualCoder.decode_torch(reg_targets[b_i][pos_inds], anchors[pos_inds])
                for j in range(pos_inds.shape[0]):
                    iou = iou3d_utils.boxes_iou_bev_lidar(box_preds_pos[j:j+1], reg_targets_pos[j:j+1])
                    target_tensor[b_i,pos_inds[j]] = iou[0][0]

    def Rotated3dIOULabelNoGrad(self, anchors, box_preds, reg_targets, target_tensor, one_hot_targets):
        from .box_coder import ResidualCoder
        for b_i in range(box_preds.shape[0]):
            pos_inds = np.where(one_hot_targets[b_i].cpu() > 0)[0]
            if pos_inds.shape[0] > 0:
                box_preds_pos = ResidualCoder.decode_torch(box_preds[b_i][pos_inds], anchors[pos_inds])
                reg_targets_pos = ResidualCoder.decode_torch(reg_targets[b_i][pos_inds], anchors[pos_inds])
                for j in range(pos_inds.shape[0]):
                    iou = iou3d_utils.boxes_iou3d_gpu_lidar(box_preds_pos[j:j+1], reg_targets_pos[j:j+1])
                    target_tensor[b_i,pos_inds[j]] = iou[0][0]

    def forward(self,
                iou_preds,
                one_hot_targets=None,
                weights=None,
                anchors=None,
                box_preds=None,
                reg_targets=None
                ):
        cfg = self.cfg
        target_tensor = torch.cuda.FloatTensor(torch.Size((iou_preds.shape[0], iou_preds.shape[1]))).zero_()
        if cfg.RPN_STAGE.IOU_HEAD.ROTATED_BEV_IOU:
            if cfg.RPN_STAGE.IOU_HEAD.LABEL_NO_GRAD:
                self.BevRotatedIOULabelNoGrad(anchors, box_preds, reg_targets, target_tensor, one_hot_targets)
            else:
                assert False, "BEV_ROTATED_IOU_GRAD not supported"

        if cfg.RPN_STAGE.IOU_HEAD.ROTATED_3D_IOU:
            if cfg.RPN_STAGE.IOU_HEAD.LABEL_NO_GRAD:
                self.Rotated3dIOULabelNoGrad(anchors, box_preds, reg_targets, target_tensor, one_hot_targets)
            else:
                assert False, "3D_ROTATED_IOU_GRAD not supported"

        prediction_probabilities = torch.sigmoid(iou_preds)
        p_t = F.binary_cross_entropy_with_logits(prediction_probabilities, target_tensor, reduction='none')
        return p_t * weights


class SigmoidFocalClassificationLoss(Loss):
    """Sigmoid focal cross entropy loss.

    Focal loss down-weights well classified examples and focusses on the hard
    examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
    """

    def __init__(self, cfg, gamma=2.0, alpha=0.25):
        """Constructor.
        Args:
          gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
          alpha: optional alpha weighting factor to balance positives vs negatives.
          all_zero_negative: bool. if True, will treat all zero as background.
                else, will treat first label as background. only affect alpha.
        """
        self.cfg = cfg
        self._alpha = alpha
        self._gamma = gamma

        if cfg.RPN_STAGE.GHM_CLS:
            self.bins = cfg.RPN_STAGE.GHM_CLS_BINS
            self.momentum = cfg.RPN_STAGE.GHM_CLS_MOM
            self.edges = [float(x) / self.bins for x in range(self.bins+1)]
            self.edges[-1] += 1e-6
            if self.momentum > 0:
                self.acc_sum = [0.0 for _ in range(self.bins)]

    def _EQL_weights(self, weights, target_tensor):
        cfg = self.cfg
        weights_exclude = weights.clone()
        weights_exclude.zero_()
        rare_class = cfg.RPN_STAGE.EQL_RARE_CLASS
        assert len(rare_class) > 0
        for b_i in range(target_tensor.shape[0]):
            pos_inds, pos_gt_inds = np.where(target_tensor[b_i].cpu() > 0)
            weights_exclude[b_i][pos_inds] = 1.0
        weights_thres = weights_exclude.clone()
        for b_i in range(target_tensor.shape[0]):
            pos_inds, pos_gt_inds = np.where(target_tensor[b_i].cpu() > 0)
            for i in range(pos_inds.shape[0]):
                for c in range(target_tensor.shape[2]):
                    if c not in rare_class:
                        weights_thres[b_i,pos_inds[i],c] = 0.0

        return 1.0 - weights_exclude * weights_thres * ((1 - target_tensor).float())

    def _compute_loss(self,
                      prediction_tensor,
                      target_tensor,
                      weights,
                      learnedweight=None,
                      class_indices=None):
        """Compute loss function.

        Args:
          prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
          target_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
          weights: a float tensor of shape [batch_size, num_anchors]
          class_indices: (Optional) A 1-D integer tensor of class indices.
            If provided, computes loss only for the specified class indices.

        Returns:
          loss: a float tensor of shape [batch_size, num_anchors, num_classes]
            representing the value of the loss function.
        """
        cfg = self.cfg
        if weights.shape.__len__() == 2:
            num_class = len(cfg.TRAIN.LS_WEIGHTS)
            weights = weights.unsqueeze(2)
            weights = torch.cat([weights]*num_class,dim=-1)
            weights_ori = weights.clone()
            if not cfg.RPN_STAGE.AUTOWEIGHT:
                if cfg.TRAIN.get('LS_WEIGHTS', False):
                    for cls_id in range(num_class):
                        weights[:,:,cls_id] = weights[:,:,cls_id] * cfg.TRAIN.LS_WEIGHTS[cls_id]
                else:
                    weights[:,:,0] = weights[:,:,0] * cfg.TRAIN.CAR_LS
                    weights[:,:,1] = weights[:,:,1] * cfg.TRAIN.PED_LS
                    weights[:,:,2] = weights[:,:,2] * cfg.TRAIN.CYC_LS
                    weights[:,:,3] = weights[:,:,3] * cfg.TRAIN.TRUCK_LS
            else:
                weights_auto = weights.clone()
                if (not cfg.RPN_STAGE.AUTO_ONLY_PED_CYC) and (not cfg.RPN_STAGE.AUTO_ONLY_PED_CYC_TRUCK):
                    weights_auto[:,:,0] = weights[:,:,0] * torch.exp(-1 * learnedweight.cls_weight[0])
                    weights_auto[:,:,1] = weights[:,:,1] * torch.exp(-1 * learnedweight.cls_weight[1])
                    weights_auto[:,:,2] = weights[:,:,2] * torch.exp(-1 * learnedweight.cls_weight[2])
                    weights_auto[:,:,3] = weights[:,:,3] * torch.exp(-1 * learnedweight.cls_weight[3])
                elif cfg.RPN_STAGE.AUTO_ONLY_PED_CYC:
                    weights_auto[:,:,0] = weights[:,:,0] * cfg.TRAIN.CAR_LS
                    weights_auto[:,:,1] = weights[:,:,1] * torch.exp(-1 * learnedweight.cls_weight[0])
                    weights_auto[:,:,2] = weights[:,:,2] * torch.exp(-1 * learnedweight.cls_weight[1])
                    weights_auto[:,:,3] = weights[:,:,3] * cfg.TRAIN.TRUCK_LS
                else:
                    weights_auto[:,:,0] = weights[:,:,0] * cfg.TRAIN.CAR_LS
                    weights_auto[:,:,1] = weights[:,:,1] * torch.exp(-1 * learnedweight.cls_weight[0])
                    weights_auto[:,:,2] = weights[:,:,2] * torch.exp(-1 * learnedweight.cls_weight[1])
                    weights_auto[:,:,3] = weights[:,:,3] * torch.exp(-1 * learnedweight.cls_weight[2])

            if cfg.RPN_STAGE.AUTOWEIGHT and cfg.RPN_STAGE.AUTO_POS_CLS:
                weights_auto_pos = weights_auto.clone()
                weights_auto_pos[target_tensor[:,:,:]==1] = weights_auto[target_tensor[:,:,:]==1] * torch.exp(-1 * learnedweight.cls_weight[-1])
            else:
                if cfg.RPN_STAGE.AUTOWEIGHT:
                    weights_auto_final = weights_auto.clone()
                    if not cfg.TRAIN.POS_LS_INDEPENDENT:
                        weights_auto_final[target_tensor[:,:,:]==1] = weights_auto[target_tensor[:,:,:]==1] * cfg.TRAIN.POS_LS
                    else:
                        for b_i in range(target_tensor.shape[0]):
                            pos_inds, pos_gt_inds = np.where(target_tensor[b_i].cpu() > 0)
                            for i in range(pos_inds.shape[0]):
                                for c in range(target_tensor.shape[2]):
                                    if c == 0 and target_tensor[b_i,pos_inds[i],c] == 1:
                                        weights_auto_final[b_i,pos_inds[i],c] = weights_auto[b_i,pos_inds[i],c] * cfg.TRAIN.POS_LS_CAR
                                    if c == 1 and target_tensor[b_i,pos_inds[i],c] == 1:
                                        weights_auto_final[b_i,pos_inds[i],c] = weights_auto[b_i,pos_inds[i],c] * cfg.TRAIN.POS_LS_PED
                                    if c == 2 and target_tensor[b_i,pos_inds[i],c] == 1:
                                        weights_auto_final[b_i,pos_inds[i],c] = weights_auto[b_i,pos_inds[i],c] * cfg.TRAIN.POS_LS_CYC
                                    if c == 3 and target_tensor[b_i,pos_inds[i],c] == 1:
                                        weights_auto_final[b_i,pos_inds[i],c] = weights_auto[b_i,pos_inds[i],c] * cfg.TRAIN.POS_LS_TRUCK
                else:
                    weights[target_tensor[:,:,:]==1] *= cfg.TRAIN.POS_LS

        per_entry_cross_ent = (_sigmoid_cross_entropy_with_logits(
            labels=target_tensor, logits=prediction_tensor))
        if cfg.RPN_STAGE.FOCAL_LOSS:
            prediction_probabilities = torch.sigmoid(prediction_tensor)
            p_t = ((target_tensor * prediction_probabilities) +
                   ((1 - target_tensor) * (1 - prediction_probabilities)))
            modulating_factor = 1.0
            if self._gamma:
                modulating_factor = torch.pow(1.0 - p_t, self._gamma)
            alpha_weight_factor = 1.0
            if self._alpha is not None:
                alpha_weight_factor = (target_tensor * self._alpha +
                                       (1 - target_tensor) * (1 - self._alpha))

            focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                        per_entry_cross_ent)
        elif cfg.RPN_STAGE.GHM_CLS:
            bins = self.bins
            edges = self.edges
            mmt = self.momentum
            g = torch.abs(prediction_tensor.sigmoid().detach() - target_tensor)
            n = 0
            valid = weights_ori > 0
            num_labels = max(valid.float().sum().item(), 1.0)
            for i in range(bins):
                inds = (g >= edges[i]) & (g < edges[i+1]) & valid
                num_in_bin = inds.sum().item()
                if num_in_bin > 0:
                    if mmt > 0:
                        self.acc_sum[i] = mmt * self.acc_sum[i] \
                            + (1 - mmt) * num_in_bin
                        weights_ori[inds] = num_labels / self.acc_sum[i]
                    else:
                        weights_ori[inds] = num_labels / num_in_bin
                    n += 1
            if n > 0:
                weights_ori = weights_ori / n
            focal_cross_entropy_loss = per_entry_cross_ent * weights_ori
        else:
            focal_cross_entropy_loss = per_entry_cross_ent

        if cfg.RPN_STAGE.EQL_CLS:
            EQL_weights = self._EQL_weights(weights_ori, target_tensor)
            focal_cross_entropy_loss = focal_cross_entropy_loss * EQL_weights

        if cfg.RPN_STAGE.AUTOWEIGHT and cfg.RPN_STAGE.AUTO_POS_CLS: 
            return focal_cross_entropy_loss * weights_auto_pos
        else:
            if cfg.RPN_STAGE.AUTOWEIGHT:
                return focal_cross_entropy_loss * weights_auto_final
            else:
                return focal_cross_entropy_loss * weights


class SigmoidFocalClassificationLoss2(nn.Module):
    """Sigmoid focal cross entropy loss.
      Focal loss down-weights well classified examples and focusses on the hard
      examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        """Constructor.
        Args:
            gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
            alpha: optional alpha weighting factor to balance positives vs negatives.
            all_zero_negative: bool. if True, will treat all zero as background.
            else, will treat first label as background. only affect alpha.
        """
        super().__init__()
        self._alpha = alpha
        self._gamma = gamma

    def forward(self,
                prediction_tensor,
                target_tensor,
                weights):
        """Compute loss function.

        Args:
            prediction_tensor: A float tensor of shape [batch_size, num_anchors,
              num_classes] representing the predicted logits for each class
            target_tensor: A float tensor of shape [batch_size, num_anchors,
              num_classes] representing one-hot encoded classification targets
            weights: a float tensor of shape [batch_size, num_anchors]
            class_indices: (Optional) A 1-D integer tensor of class indices.
              If provided, computes loss only for the specified class indices.

        Returns:
          loss: a float tensor of shape [batch_size, num_anchors, num_classes]
            representing the value of the loss function.
        """
        per_entry_cross_ent = (_sigmoid_cross_entropy_with_logits(
            labels=target_tensor, logits=prediction_tensor))
        prediction_probabilities = torch.sigmoid(prediction_tensor)
        p_t = ((target_tensor * prediction_probabilities) +
               ((1 - target_tensor) * (1 - prediction_probabilities)))
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = torch.pow(1.0 - p_t, self._gamma)
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = (target_tensor * self._alpha + (1 - target_tensor) * (1 - self._alpha))

        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor * per_entry_cross_ent)
        return focal_cross_entropy_loss * weights


def _sigmoid_cross_entropy_with_logits(logits, labels):
    # to be compatible with tensorflow, we don't use ignore_idx
    loss = torch.clamp(logits, min=0) - logits * labels.type_as(logits)
    loss += torch.log1p(torch.exp(-torch.abs(logits)))
    # transpose_param = [0] + [param[-1]] + param[1:-1]
    # logits = logits.permute(*transpose_param)
    # loss_ftor = nn.NLLLoss(reduce=False)
    # loss = loss_ftor(F.logsigmoid(logits), labels)
    return loss


def _softmax_cross_entropy_with_logits(logits, labels):
    param = list(range(len(logits.shape)))
    transpose_param = [0] + [param[-1]] + param[1:-1]
    logits = logits.permute(*transpose_param) # [N, ..., C] -> [N, C, ...]
    loss_ftor = nn.CrossEntropyLoss(reduction='none')
    loss = loss_ftor(logits, labels.max(dim=-1)[1])
    return loss
