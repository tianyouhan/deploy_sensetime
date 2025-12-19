import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path
from easydict import EasyDict
import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils, eval_seg_det_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

def recursive_easydict(d):
    if isinstance(d, dict):
        return EasyDict({k: recursive_easydict(v) for k, v in d.items()})
    return d

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--infer_time', action='store_true', default=False, help='calculate inference latency')

    parser.add_argument('--eval_type', type=str, default='det', choices=['det', 'seg_det'])
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--iou_thresh', type=float, default=0.3, help='iou_thresh to save predict')
    parser.add_argument('--vis_dir', type=str, default=None, help='pred label vis')
    parser.add_argument('--output_folder', type=str, default='output', help='output folder')

    args = parser.parse_args()
    global cfg
    cfg_from_yaml_file(args.cfg_file, cfg)
    # import ipdb; ipdb.set_trace()
    cfg = recursive_easydict(cfg)
    # import ipdb; ipdb.set_trace()
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    # load checkpoint
    model.eval()
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test, 
                                pre_trained_path=args.pretrained_model)
    model.cuda()

    if dist_test:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])

    # start evaluation
    if args.eval_type == 'seg_det':
        eval_seg_det_utils.eval_one_epoch(
            cfg, args, model, test_loader, epoch_id, logger, dist_test=False,
            result_dir=eval_output_dir
        )
    else:
        eval_utils.eval_one_epoch(
            cfg, args, model, test_loader, epoch_id, logger, dist_test=True,
            result_dir=eval_output_dir
        )


def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args):
    ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()]

    for cur_ckpt in ckpt_list:
        num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)
        if num_list.__len__() == 0:
            continue

        epoch_id = num_list[-1]
        if 'optim' in epoch_id:
            continue
        if float(epoch_id) not in evaluated_ckpt_list and int(float(epoch_id)) >= args.start_epoch:
            return epoch_id, cur_ckpt
    return -1, None


def repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=False):
    # evaluated ckpt record
    ckpt_record_file = eval_output_dir / ('eval_list_%s.txt' % cfg.DATA_CONFIG.DATA_SPLIT['test'])
    with open(ckpt_record_file, 'a'):
        pass

    # tensorboard log
    if cfg.LOCAL_RANK == 0:
        tb_log = SummaryWriter(log_dir=str(eval_output_dir / ('tensorboard_%s' % cfg.DATA_CONFIG.DATA_SPLIT['test'])))
    total_time = 0
    first_eval = True

    while True:
        # check whether there is checkpoint which is not evaluated
        cur_epoch_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args)
        if cur_epoch_id == -1 or int(float(cur_epoch_id)) < args.start_epoch:
            wait_second = 30
            if cfg.LOCAL_RANK == 0:
                print('Wait %s seconds for next check (progress: %.1f / %d minutes): %s \r'
                      % (wait_second, total_time * 1.0 / 60, args.max_waiting_mins, ckpt_dir), end='', flush=True)
            time.sleep(wait_second)
            total_time += 30
            if total_time > args.max_waiting_mins * 60 and (first_eval is False):
                break
            continue

        total_time = 0
        first_eval = False

        model.eval()
        model.load_params_from_file(filename=cur_ckpt, logger=logger, to_cpu=dist_test)
        model.cuda()

        if dist_test:
            model_parallel = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
        else:
            model_parallel = model

        # start evaluation
        cur_result_dir = eval_output_dir / ('epoch_%s' % cur_epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']

        if args.eval_type == 'seg_det':
            tb_dict = eval_seg_det_utils.eval_one_epoch(
                cfg, args, model_parallel, test_loader, cur_epoch_id, logger, dist_test=True,
                result_dir=cur_result_dir
            )
        else:
            tb_dict = eval_utils.eval_one_epoch(
                cfg, args, model_parallel, test_loader, cur_epoch_id, logger, dist_test=dist_test,
                result_dir=cur_result_dir
            )

        if cfg.LOCAL_RANK == 0:
            for key, val in tb_dict.items():
                tb_log.add_scalar(key, val, cur_epoch_id)

        # record this epoch which has been evaluated
        with open(ckpt_record_file, 'a') as f:
            print('%s' % cur_epoch_id, file=f)
        logger.info('Epoch %s has been evaluated' % cur_epoch_id)


def main():
    args, cfg = parse_config()
    # import ipdb; ipdb.set_trace()
    if args.infer_time:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / args.output_folder / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'
    qat_output_dir = output_dir / 'qat'

    num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
    epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
    eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    qat_output_dir = qat_output_dir / ('epoch_%s' % epoch_id)

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    qat_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = qat_output_dir / ('log_qat_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'
    eval_data = os.getenv('eval_pkl', False)
    if eval_data:
        cfg.DATA_CONFIG.INFO_PATH['test'] = [eval_data]
    eval_gop = os.getenv('eval_gop', 'False')
    print(eval_gop)
    if eval_gop != 'False':
        cfg.MODEL['EVAL_GOP'] = True
        print('eval_gop', cfg.MODEL.get("EVAL_GOP", False))
    print(cfg.DATA_CONFIG.INFO_PATH['test']) 
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=False,
        total_epochs=1,
        seed=None
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    #calib
    from horizon_plugin_pytorch.march import March, set_march
    from horizon_plugin_pytorch.quantization.qconfig import (
        default_calib_8bit_weight_16bit_act_fake_quant_qconfig,
        get_qconfig,
    ) 
    from horizon_plugin_pytorch.quantization.observer_v2 import MinMaxObserver
    from horizon_plugin_pytorch.dtype import qint8,qint16
    from horizon_plugin_pytorch.quantization.qconfig_template import (
        calibration_8bit_weight_16bit_act_qconfig_setter,
        ModuleNameQconfigSetter, 
    )
    from horizon_plugin_pytorch.quantization import prepare,PrepareMethod,FakeQuantState
    from horizon_plugin_pytorch.quantization.fake_quantize import set_fake_quantize


    from pcdet.models import load_data_to_gpu
    from mmcv.runner import save_checkpoint

    march = March.NASH_E
    set_march(march)
    data_val = {}
    for i, data in enumerate(train_loader):
        if i == 0:
            '''
            data dict_keys(['points', 'label', 'metadata', 'frame_id', 'fuse_points', 'fuse_label', 
            'gt_names', 'gt_boxes', 'num_points_in_gt', 'lidar_aug_matrix', 'use_lead_xyz', 
            'voxels', 'voxel_coords', 'voxel_num_points', 'lidar_label', 'anchors', 'batch_size'])
            '''
            #print("data: ",data)
            # data["return_loss"] = False
            # data["rescale"] = True
            load_data_to_gpu(data)
            data_val = data
        else:
            break
    #print(input_data)
    example_inputs_dataloader = (data_val)
    # model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test, 
    #                             pre_trained_path=args.pretrained_model)
    model.cuda()
    model.forward = model.forward_qat
    
    module_name_to_qconfig = {}
    qat_model = prepare(model,
                        example_inputs_dataloader,
                        qconfig_setter=(
                            ModuleNameQconfigSetter(module_name_to_qconfig),
                            calibration_8bit_weight_16bit_act_qconfig_setter,), # 全int16设定
                        method=PrepareMethod.JIT_STRIP)
    
    qat_model.eval()
    # set_fake_quantize(qat_model, FakeQuantState.CALIBRATION)
    #calib
    if dist_test:
        qat_model = torch.nn.parallel.DistributedDataParallel(qat_model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    max_iter= 2000  # 700
    # print('length of calib dataset', len(test_loader))
    # for i, data in enumerate(test_loader):
    #     if i==max_iter:break
    #     load_data_to_gpu(data)
    #     with torch.no_grad():
    #         print("itera:*******",i)
    #         pred_dicts, ret_dict = qat_model(data)
    qat_model.eval()
    # save_checkpoint(qat_model, os.path.join(qat_output_dir, "calib-max_iter_{}.pth".format(max_iter)))
    # import ipdb; ipdb.set_trace()
    ckpt_path = args.ckpt
    checkpoint_qat = torch.load(ckpt_path, map_location=None)
    # import ipdb; ipdb.set_trace()
    qat_model.load_state_dict(checkpoint_qat)
    # import ipdb; ipdb.set_trace()
    set_fake_quantize(qat_model, FakeQuantState.VALIDATION)

    with torch.no_grad():
        # start evaluation
        if args.eval_type == 'seg_det':
            eval_seg_det_utils.eval_one_epoch(
                cfg, args, qat_model, test_loader, epoch_id, logger, dist_test=False,
                result_dir=eval_output_dir
            )
        else:
            eval_utils.eval_one_epoch(
                cfg, args, qat_model, test_loader, epoch_id, logger, dist_test=True,
                result_dir=eval_output_dir
            )

if __name__ == '__main__':
    main()
