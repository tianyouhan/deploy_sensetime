import logging
import os
import pickle
import random
import shutil
import subprocess
import SharedArray
import json

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from numba import jit
import cv2

def save_np(path, tensor, format="npy"):
    # save_list = ['spetr_head']

    # keep_flag = False
    # for keep_path in save_list:
    #     if keep_path in path:
    #         keep_flag = True
    # if not keep_flag:
    #     return

    format='bin'
    if os.getenv('saving_format') == 'npy':
        format = 'npy'
    idx_saving = int(path.split('/')[-1])
    if idx_saving % int(os.getenv("SR")) == 0:
        path = '/'.join(path.split('/')[:-1])+ '/' + str(int(idx_saving // int(os.getenv("SR"))))
        os.system("mkdir -p " + os.path.dirname(path))
        if not isinstance(tensor, np.ndarray):
            tm=tensor.cpu().detach().numpy()
            tm = tm.astype(np.float32)
            print(path, tm.shape)
            if format == 'bin':
                tm.tofile(path+".bin")
            elif format == 'npy':
                np.save(path+".npy", np.array(tensor.cpu()))
            else:
                tm.tofile(path+".bin")
                np.save(path+".npy", np.array(tensor.cpu()))
        else:
            np.save(path+".npy", tensor)
    else:
        pass

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def limit_period(val, offset=0.5, period=2*np.pi):
    val, is_numpy = check_numpy_to_torch(val)
    ans = val - torch.floor(val / period + offset) * period
    return ans.numpy() if is_numpy else ans


def drop_info_with_name(info, name):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['name']) if x != name]
    for key in info.keys():
        ret_info[key] = info[key][keep_indices]
    return ret_info


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def angle2matrix(angle):
    """
    Args:
        angle: angle along z-axis, angle increases x ==> y
    Returns:
        rot_matrix: (3x3 Tensor) rotation matrix
    """

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    rot_matrix = torch.tensor([
        [cosa, -sina, 0],
        [sina, cosa,  0],
        [   0,    0,  1]
    ])
    return rot_matrix


def mask_points_by_range(points, limit_range):
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    return mask


@jit(nopython=True)
def mask_points_by_fov(img_size, points, ins_mat, ext_mat, img_aug_mat=None, gt_depth=None):
    img_w, img_h = img_size
    # ascontiguousarray
    ext_mat_R_transpose, ext_mat_T = np.ascontiguousarray(ext_mat[:3, :3].T), ext_mat[:3, -1]
    ins_mat_R_transpose, ins_mat_T = np.ascontiguousarray(ins_mat[:3, :3].T), ins_mat[:3, -1]
    img_aug_mat_R_transpose, img_aug_mat_T = np.ascontiguousarray(img_aug_mat[:3, :3].T), img_aug_mat[:3, -1]
    # lidar to camera
    points_in_cam = points @ ext_mat_R_transpose + ext_mat_T  # p1 = R.p0 + T
    points_z = points_in_cam[:, 2]  # the depth of points
    # camera to image
    pixels_in_img = points_in_cam @ ins_mat_R_transpose + ins_mat_T
    pixels_in_img[:, :2] /= pixels_in_img[:, 2:3]
    # apply image aug
    if img_aug_mat is not None:
        pixels_in_img = pixels_in_img @ img_aug_mat_R_transpose + img_aug_mat_T
    # remove out of fov points
    fov_mask = (0 <= pixels_in_img[:, 0]) & (pixels_in_img[:, 0] < img_w) & \
                (0 <= pixels_in_img[:, 1]) & (pixels_in_img[:, 1] < img_h) & (points_z > 0)
    
    if gt_depth is not None:
        coor = (pixels_in_img[fov_mask][:, :2]).astype(np.int32)
        depth = points_z[fov_mask]
        ranks = coor[:, 0] + coor[:, 1] * img_w
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]  # 把所有fov点按depth排序
        min_depth_mask = np.ones(coor.shape[0], dtype=np.bool_)
        min_depth_mask[1:] = (ranks[1:] != ranks[:-1])  # 保留同一个图像坐标位置对应的点
        coor, depth = coor[min_depth_mask], depth[min_depth_mask]
        for idx in range(coor.shape[0]):
            gt_depth[..., coor[idx, 1], coor[idx, 0]] = depth[idx]
    return fov_mask


def dilate_erode_base(input, pixel_width):
    kernel = np.ones((abs(pixel_width), abs(pixel_width)), dtype=np.uint8)
    if pixel_width >= 0:
        output = cv2.dilate(input, kernel, iterations=1)
    else:
        output = cv2.erode(input, kernel, iterations=1)
    return output


def fill_box_in_bev_label(label, boxes, cfg):
    for idx in range(boxes.shape[0]):
        mask = np.zeros_like(label)
        cv2.fillConvexPoly(mask, boxes[idx], color=(255))
        if np.sum(mask) == 0:
            continue
        pos_mask = mask > 0
        label[pos_mask] = mask[pos_mask]
    return label


def get_convexHull(label, boxes, cfg):
    mode = cfg.MODE
    pixel_width = cfg.get('PIXEL_WIDTH', None)
    maks_range = cfg.get('RANGE', None)
    if maks_range is not None:
        x1, y1, x2, y2 = maks_range
    
    for idx in range(boxes.shape[0]):
        mask = np.zeros_like(label)
        cv2.fillConvexPoly(mask, boxes[idx], color=(1))
        if np.sum(label * mask) == 0 and mode in ['convexHull', 'fillBox']:
            continue
        if np.sum(mask) == 0 and mode in ['zerobox']:
            continue
        if mode == 'convexHull':
            # 得到box内的正样本点，然后根据这些点计算凸包，再填充凸包
            mask_conv = np.zeros_like(label)
            points_in_box = np.argwhere((label * mask) > 0)[:, ::-1]  # y, x to x, y
            convexHull_points = cv2.convexHull(points_in_box, returnPoints=True)
            cv2.fillConvexPoly(mask_conv, convexHull_points, color=(1))
            if pixel_width is not None:
                if maks_range is not None:
                    mask_conv[y1:y2, x1:x2] = dilate_erode_base(mask_conv[y1:y2, x1:x2], pixel_width)
                else:
                    mask_conv = dilate_erode_base(mask_conv, pixel_width)
            pos_mask = mask_conv > 0
            label[pos_mask] = mask_conv[pos_mask]
        elif mode in ['fillBox', 'zerobox']:
            # 填充box对应的bev区域，然后腐蚀一部分
            if pixel_width is not None:
                if maks_range is not None:
                    mask[y1:y2, x1:x2] = dilate_erode_base(mask[y1:y2, x1:x2], pixel_width)
                else:
                    mask = dilate_erode_base(mask, pixel_width)
            pos_mask = mask > 0
            label[pos_mask] = mask[pos_mask]
        else:
            raise ValueError(f'Not support mode:{mode}')
    return label


def get_voxel_centers(voxel_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        downsample_times:
        voxel_size:
        point_cloud_range:

    Returns:

    """
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id, seed=666):
    if seed is not None:
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)
        torch.cuda.manual_seed(seed + worker_id)
        torch.cuda.manual_seed_all(seed + worker_id)


def get_pad_params(desired_size, cur_size):
    """
    Get padding parameters for np.pad function
    Args:
        desired_size: int, Desired padded output size
        cur_size: int, Current size. Should always be less than or equal to cur_size
    Returns:
        pad_params: tuple(int), Number of values padded to the edges (before, after)
    """
    assert desired_size >= cur_size

    # Calculate amount to pad
    diff = desired_size - cur_size
    pad_params = (0, diff)

    return pad_params


def keep_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


def init_dist_slurm(tcp_port, local_rank, backend='nccl'):
    """
    modified from https://github.com/open-mmlab/mmdetection
    Args:
        tcp_port:
        backend:

    Returns:

    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    rank = dist.get_rank()
    return total_gpus, rank


def init_dist_pytorch(tcp_port, local_rank, backend='nccl'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    # os.environ['MASTER_PORT'] = str(tcp_port)
    # os.environ['MASTER_ADDR'] = 'localhost'
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        # init_method='tcp://127.0.0.1:%d' % tcp_port,
        # rank=local_rank,
        # world_size=num_gpus
    )
    rank = dist.get_rank()
    return num_gpus, rank


def get_dist_info(return_gpu_per_machine=False):
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if return_gpu_per_machine:
        gpu_per_machine = torch.cuda.device_count()
        return rank, world_size, gpu_per_machine

    return rank, world_size


def merge_results_dist(result_part, size, tmpdir):
    rank, world_size = get_dist_info()
    os.makedirs(tmpdir, exist_ok=True)

    dist.barrier()
    pickle.dump(result_part, open(os.path.join(tmpdir, 'result_part_{}.pkl'.format(rank)), 'wb'))
    dist.barrier()

    if rank != 0:
        return None

    part_list = []
    for i in range(world_size):
        part_file = os.path.join(tmpdir, 'result_part_{}.pkl'.format(i))
        part_list.append(pickle.load(open(part_file, 'rb')))

    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    ordered_results = ordered_results[:size]
    shutil.rmtree(tmpdir)
    return ordered_results


def merge_dict_results_dist(result_part, tmpdir, size=None, data_type='pkl'):
    rank, world_size = get_dist_info()
    os.makedirs(tmpdir, exist_ok=True)

    dist.barrier()
    if data_type == 'pkl':
        pickle.dump(result_part, open(os.path.join(tmpdir, 'result_part_{}.pkl'.format(rank)), 'wb'))
    elif data_type == 'json':
        json.dump(result_part, open(os.path.join(tmpdir, 'result_part_{}.json'.format(rank)), 'w', encoding='utf-8'))
    dist.barrier()

    if rank != 0:
        return None

    if data_type == 'pkl':
        merged_dict = {t: [] for t in result_part.keys()}
        for i in range(world_size):
            part_file = os.path.join(tmpdir, 'result_part_{}.pkl'.format(i))
            part = pickle.load(open(part_file, 'rb'))
            for t in part.keys():
                merged_dict[t].append(part[t])
        
        ordered_results = {t: [] for t in result_part.keys()}
        for t in result_part.keys():
            for res in zip(*merged_dict[t]):
                ordered_results[t].extend(list(res))
            ordered_results[t] = ordered_results[t][:size]
    elif data_type == 'json':
        ordered_results = {}
        for i in range(world_size):
            part_file = os.path.join(tmpdir, 'result_part_{}.json'.format(i))
            part = json.load(open(part_file, 'r', encoding='utf-8'))
            ordered_results.update(part)

    shutil.rmtree(tmpdir)
    return ordered_results


def scatter_point_inds(indices, point_inds, shape):
    ret = -1 * torch.ones(*shape, dtype=point_inds.dtype, device=point_inds.device)
    ndim = indices.shape[-1]
    flattened_indices = indices.view(-1, ndim)
    slices = [flattened_indices[:, i] for i in range(ndim)]
    ret[slices] = point_inds
    return ret


def generate_voxel2pinds(sparse_tensor):
    device = sparse_tensor.indices.device
    batch_size = sparse_tensor.batch_size
    spatial_shape = sparse_tensor.spatial_shape
    indices = sparse_tensor.indices.long()
    point_indices = torch.arange(indices.shape[0], device=device, dtype=torch.int32)
    output_shape = [batch_size] + list(spatial_shape)
    v2pinds_tensor = scatter_point_inds(indices, point_indices, output_shape)
    return v2pinds_tensor


def sa_create(name, var):
    x = SharedArray.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def nested_getattr(obj, attrs):
    result = obj
    if type(attrs) is not list:
        if '.' in attrs:
            attrs = [attr for attr in attrs.split('.')]
        else:
            attrs = [attrs]
    for attr in attrs:
        result = getattr(result, attr)
    return result


def fix_modules(model, fix_cfg):
    modules = fix_cfg.get('FIX_MODULES', None)
    ignore_keys = fix_cfg.get('IGNORE_KEYs', None)
    grad_back_flag = fix_cfg.get('GRAD_BACK_FLAG', False)

    if modules is not None:
        for name in modules:
            try:
                nested_getattr(model.module, name).eval()
                for key, param in nested_getattr(model.module, name).named_parameters():
                    param.requires_grad = grad_back_flag
            except:
                nested_getattr(model, name).eval()
                for key, param in nested_getattr(model, name).named_parameters():
                    param.requires_grad = grad_back_flag                
    if ignore_keys is not None:
        for key in ignore_keys:
            try:
                nested_getattr(model.module, key).train()
            except:
                nested_getattr(model, key).train()
    return
