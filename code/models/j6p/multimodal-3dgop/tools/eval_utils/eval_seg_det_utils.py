import pickle
import time
import os

import numpy as np
import torch
import tqdm
import cv2
import json

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils, freespace_utils
from eval_utils.metric import *



def statistics_info(cfg, ret_dict, metric, disp_dict):
    
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])
    
    
def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=True, result_dir=None, best_miou=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataloader.dataset
    class_names = dataset.class_names
    tasks = cfg.DATA_CONFIG.TASKS

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)

    num_classes, hist_classes, eval_types = {}, {}, {}
    for module in ['DENSE_HEAD', 'DENSE_HEAD_SEG', 'DENSE_HEAD_SEG_FUSION', 'DENSE_HEAD_SEG_LIDAR', 'DENSE_HEAD_SEG_CAM']:
        if cfg.MODEL.get(module, None) is not None:
            num_classes.update(getattr(cfg.MODEL, module).SEPARATE_HEAD_CFG.HEAD_DICT)
            hist_classes.update(getattr(cfg.MODEL, module).LOSS_CONFIG.HIST_CLASS)
            eval_types.update(getattr(cfg.MODEL, module).POST_PROCESSING.ACTIVE_FUNC)
    # 
    eval_dict = []
    for idx, task in enumerate(tasks):
        if task in num_classes and num_classes[task]['out_channels'] > 1:
            eval_dict.append({float(1.0): []})
        else:
            eval_dict.append({float(t)/100: [] for t in range(30, 70, 1)})

    metric = {'gt_num': 0,}
    if 'POST_PROCESSING' in cfg.MODEL and 'RECALL_THRESH_LIST' in cfg.MODEL.POST_PROCESSING:
        for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
            metric['recall_roi_%s' % str(cur_thresh)] = 0
            metric['recall_rcnn_%s' % str(cur_thresh)] = 0
    metric = [metric.copy() for _ in tasks]
    det_annos = [[] for _ in tasks]

    if args.vis_dir is not None:
        folder_list = ['image', 'cloud_bin', 'gt', 'pred']
        save_paths = []
        for folder in folder_list:
            save_paths.append(creat_dir(args.vis_dir, folder))
        img_path = {}

    disp_dict = {'task': tasks}
    vis_flag = False

    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        torch.cuda.empty_cache()
        with torch.no_grad():
            pred_dicts, label_dicts = model(batch_dict)
            if cfg.MODEL.get("EVAL_GOP", False):
                
                seg_enable, det_enable = (f'seg_pred_dicts_{tasks[0]}' in pred_dicts), (f'det_pred_dicts_{tasks[0]}_gop' in pred_dicts)
            else:
                seg_enable, det_enable = (f'seg_pred_dicts_{tasks[0]}' in pred_dicts), (f'det_pred_dicts_{tasks[0]}' in pred_dicts)
        
        for idx, task in enumerate(tasks):
            ## seg eval
            if seg_enable:
                num_class, hist_class, eval_type = num_classes[task]['out_channels'], hist_classes[task], eval_types[task]['test']
                score_thresh_list = eval_types[task].get('score', None)
                pred = pred_dicts[f'seg_pred_dicts_{task}'][f'{task}_pred']
                label = label_dicts[f'seg_label_dicts_{task}'][f'{task}_label'].to(torch.uint8)
                mask = freespace_utils.get_mask(batch_dict, label).to(torch.uint8)
                if mask is not None:
                    pred, label = pred.float() * mask.float(), label * mask
                for t in eval_dict[idx].keys():
                    pred_t = get_pred_t(pred, t, num_class, eval_type, score_thresh_list)
                    for j in range(pred_t.shape[0]):
                        eval_dict[idx][t].append(gpu_hist_crop_numba(pred_t[j], label[j], torch.tensor([c for c in range(hist_class)]).cuda(), fg_mask=True))
                        if args.vis_dir is not None and (t == args.iou_thresh or len(eval_dict[idx].keys()) == 1):
                            # img_path = batch_dict['metadata'][j]['img_path']['center_camera_fov120']
                            # os.system(f'cp {img_path} {save_paths[0]}')
                            frame_name = batch_dict['metadata'][j]['frame_name']
                            img_path[frame_name] = batch_dict['metadata'][j]['img_path']
                            os.makedirs(os.path.join(save_paths[1], f'{task}'), exist_ok=True)
                            os.makedirs(os.path.join(save_paths[2], f'{task}'), exist_ok=True)
                            os.makedirs(os.path.join(save_paths[3], f'{task}'), exist_ok=True)
                            batch_dict['points'][:, 1:].cpu().numpy().tofile(os.path.join(save_paths[1], f'{task}', f'{frame_name}.bin'))
                            label[j].squeeze().cpu().numpy().astype(np.uint8).tofile(os.path.join(save_paths[2], f'{task}', f'{frame_name}.label'))
                            pred_t[j].squeeze().cpu().numpy().astype(np.uint8).tofile(os.path.join(save_paths[3], f'{task}', f'{frame_name}.label'))
                            vis_flag = True
            if det_enable:
                ## det eval
                if cfg.MODEL.get("EVAL_GOP", False):
                    
                    det_pred_dicts = pred_dicts[f'det_pred_dicts_{task}_gop']
                    det_ret_dict = label_dicts[f'det_recall_dicts_{task}_gop']
                    statistics_info(cfg, det_ret_dict, metric[idx], disp_dict)
                    annos = dataset.generate_prediction_dicts(
                        batch_dict, det_pred_dicts, class_names,
                        output_path=final_output_dir if args.save_to_file else None,
                        head_mode='_gop',
                    )
                else:
                    det_pred_dicts = pred_dicts[f'det_pred_dicts_{task}']
                    det_ret_dict = label_dicts[f'det_recall_dicts_{task}']
                    statistics_info(cfg, det_ret_dict, metric[idx], disp_dict)
                    annos = dataset.generate_prediction_dicts(
                        batch_dict, det_pred_dicts, class_names,
                        output_path=final_output_dir if args.save_to_file else None
                    )
                det_annos[idx] += annos

                if (args.vis_dir is not None) and (not vis_flag):
                    for j in range(len(det_pred_dicts)):
                        frame_name = batch_dict['metadata'][j]['frame_name']
                        os.makedirs(os.path.join(save_paths[1], f'{task}'), exist_ok=True)
                        os.makedirs(os.path.join(save_paths[2], f'{task}'), exist_ok=True)
                        os.makedirs(os.path.join(save_paths[3], f'{task}'), exist_ok=True)
                        batch_dict['points'][:, 1:].cpu().numpy().tofile(os.path.join(save_paths[1], f'{task}', f'{frame_name}.bin'))

        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()
            
    if args.vis_dir is not None:
        img_path = common_utils.merge_dict_results_dist(img_path, tmpdir=result_dir / 'tmpdir', data_type='json')
        if cfg.LOCAL_RANK == 0:
            with open(os.path.join(args.vis_dir, 'image_path.json'), 'w', encoding='utf-8') as f:
                json.dump(img_path, f, indent=4)

    for idx, task in enumerate(tasks):
        if seg_enable:
            ## seg eval
            hist_merge = common_utils.merge_dict_results_dist(eval_dict[idx], size=len(dataset), tmpdir=result_dir / 'tmpdir')
            if cfg.LOCAL_RANK == 0:
                best_miou = seg_metric_display(idx, hist_merge, dataset, model, best_miou, logger, task, num_class, epoch_id, result_dir)
        
        if det_enable:
            ## det eval
            rank, world_size = common_utils.get_dist_info()
            det_annos[idx] = common_utils.merge_results_dist(det_annos[idx], len(dataset), tmpdir=result_dir / 'tmpdir')
            metric[idx] = common_utils.merge_results_dist([metric[idx]], world_size, tmpdir=result_dir / 'tmpdir')

    if cfg.LOCAL_RANK != 0:
        return {}
    
    ret_dict = {}
    if det_enable:
        for idx, task in enumerate(tasks):
            logger.info(f'*************** Task:{task} detection evaluation *****************')
            ret_dict = det_metric_display(cfg, metric[idx], det_annos[idx], dataset, logger, task, world_size, result_dir, final_output_dir)
            if args.vis_dir is not None:
                if cfg.MODEL.get("EVAL_GOP", False):
                    with open(os.path.join(args.vis_dir, f'result_{task}gop.pkl'), 'wb') as f:
                        pickle.dump(det_annos[idx], f)
                else:
                    with open(os.path.join(args.vis_dir, f'result_{task}.pkl'), 'wb') as f:
                        pickle.dump(det_annos[idx], f)

    progress_bar.close()
    logger.info('*************** Epoch %s evaluation done. *****************' % epoch_id)

    return ret_dict


def seg_metric_display(idx, hist_merge, dataset, model, best_miou, logger, task, num_class, epoch_id, result_dir):
    assert len(list(hist_merge.values())[0]) == len(dataset), \
        'Merged result length is not equal to validatation set'

    val_miou, val_biou = 0, 0
    per_class_metric_miou, per_class_metric_biou = {}, {}
    for k, v in hist_merge.items():
        assert len(v) > 0
        sum_v = sum(v)
        val_miou_t_ = per_class_iu(sum_v) * 100
        val_biou_t_ = per_class_biou(sum_v) * 100
        val_recall_t_ = per_class_recall(sum_v) * 100
        val_prec_t_ = per_class_precision(sum_v) * 100

        val_miou_t = np.nanmean(val_miou_t_[1:])
        val_biou_t = val_biou_t_[1]  # foreground
        val_recall_t = np.nanmean(val_recall_t_[1:])
        val_prec_t = np.nanmean(val_prec_t_[1:])
        if k == 0.4:
            FP_list = [ele[0, 1] for ele in v]
            FP_ratio = sum(ele > 0 for ele in FP_list) / len(FP_list)
            FP_pixel = sum(FP_list) / len(FP_list)
            logger.info(f'FP ratio:{FP_ratio}, FP pixel per frame:{FP_pixel}')

        if val_miou < val_miou_t:
            val_miou = val_miou_t
            per_class_metric_miou = {'iou':val_miou_t_, 'recall':val_recall_t_, 'prec':val_prec_t_, 'biou':val_biou_t_}
        
        if val_biou < val_biou_t:
            val_biou = val_biou_t
            per_class_metric_biou = {'iou':val_miou_t_, 'recall':val_recall_t_, 'prec':val_prec_t_, 'biou':val_biou_t_}

        logger.info('Task:{}; Current mIoU is {:.3f}; IoU thresh {:.2f}; Current biou is {:.3f}; Current recall is {:.3f}; Current precision is {:.3f}'\
            .format(task, val_miou_t, k, val_biou_t, val_recall_t, val_prec_t))
    
    if best_miou is not None:
        if best_miou['miou'][idx] < val_miou:
            best_miou['miou'][idx] = val_miou
            torch.save(model.state_dict(), result_dir / "{}_checkpoint_epoch{}_{:.2f}.pt".format(task, epoch_id, val_miou))

        if best_miou['biou'][idx] < val_biou:
            best_miou['biou'][idx] = val_biou
            if num_class > 1:
                torch.save(model.state_dict(), result_dir / "{}_checkpoint_epoch{}_biou_{:.2f}.pt".format(task, epoch_id, val_biou))

    logger.info('*************** best miou results *****************')
    metric_display(per_class_metric_miou, logger)
    if num_class > 1:
        logger.info('*************** best biou results *****************')
        metric_display(per_class_metric_biou, logger)
        
    if best_miou is not None:
        logger.info('Task:{}; Current mIoU is {:.3f}; Best mIoU is {:.3f}; Current bIoU is {:.3f}; Best bIoU is {:.3f}'\
            .format(task, val_miou, best_miou['miou'][idx], val_biou, best_miou['biou'][idx]))
    return best_miou
        

def det_metric_display(cfg, metric, det_annos, dataset, logger, task, world_size, result_dir, final_output_dir):
    ret_dict = {}
    for key, val in metric[0].items():
        for k in range(1, world_size):
            metric[0][key] += metric[k][key]
    metric = metric[0]

    gt_num_cnt = metric['gt_num']
    logger.info('gt_num: %s', gt_num_cnt)
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    if cfg.MODEL.get("EVAL_GOP", False):
        print('gop evaluation')
        with open(result_dir / f'result_{task}gop.pkl', 'wb') as f:
            pickle.dump(det_annos, f)
        result_str, result_dict = dataset.evaluation_gop(
            det_annos, dataset.class_names,
            eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
            output_path=final_output_dir
        )
    else:
        with open(result_dir / f'result_{task}.pkl', 'wb') as f:
            pickle.dump(det_annos, f)
        result_str, result_dict = dataset.evaluation(
            det_annos, dataset.class_names,
            eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
            output_path=final_output_dir
        )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


def get_pred_t(predict, thresh, num_class, eval_type, score_thresh_list=None):
    if eval_type == 'softmax':
        pred_t = predict.to(torch.uint8)
    elif eval_type == 'sigmoid':
        if num_class <= 1:
            pred_t = (predict > thresh).to(torch.uint8)
        else:
            num_classes = predict.shape[1] - 1
            # predict: [b,c,h,w]，第一类是背景
            if score_thresh_list is None:
                fg_thresh_map = {key + 1:0.3 for key in range(num_classes)}
            else:
                fg_thresh_map = {key + 1:val for key, val in enumerate(score_thresh_list)}
            # 得到前景最大的预测概率和对应的类别index
            fg_pred_max, fg_pred_index = torch.max(predict[:, 1:, ...], dim=1, keepdim=True)
            # 因为背景的index=0,所以index+1
            fg_pred_index += 1
            fg_pred_thresh = torch.zeros_like(fg_pred_index).float()
            # 把每个pixel的预测概率和对应类别的设定阈值比较，小于阈值的当作背景
            for key, val in fg_thresh_map.items():
                fg_pred_thresh[fg_pred_index == key] = val
            fg_pred_index[fg_pred_max < fg_pred_thresh] = 0
            pred_t = fg_pred_index.to(torch.uint8)
    else:
        raise ValueError(f'Not support eval_type:{eval_type}')
    return pred_t


def metric_display(per_class_metric, logger):
    for title, metric in per_class_metric.items():
        logger.info(f'{title.ljust(10)}:{[round(ele, 2) for ele in metric]}')


def creat_dir(root_dir, folder):
    save_path = os.path.join(root_dir, folder)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    return save_path


if __name__ == '__main__':
    pass
