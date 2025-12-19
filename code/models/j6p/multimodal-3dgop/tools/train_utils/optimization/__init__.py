from functools import partial
import copy

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper
from .learning_schedules_fastai import CosineWarmupLR, OneCycle, CosineAnnealing


def build_optimizer(model, optim_cfg, filter_grad=True):
    if filter_grad:
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    else:
        trainable_params = model.parameters()

    if optim_cfg.get('LAYER_LR', None) is not None:
        def get_nested_attribute(obj, attr):
            attrs = attr.split('.')
            if len(attrs) == 1:
                return getattr(obj, attrs[0])
            else:
                return get_nested_attribute(getattr(obj, attrs[0]), '.'.join(attrs[1:]))
        layer_params, layer_id = copy.deepcopy(optim_cfg.LAYER_LR), []
        for idx in range(len(layer_params)):
            now_params = get_nested_attribute(model, layer_params[idx]['params']).parameters()
            layer_id += list(map(id, now_params))
            layer_params[idx]['params'] = get_nested_attribute(model, layer_params[idx]['params']).parameters()
        trainable_params = [{'params': filter(lambda p: id(p) not in layer_id, trainable_params)}] + layer_params
        
    if optim_cfg.OPTIMIZER == 'adam':
        optimizer = optim.Adam(trainable_params, lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY)
    elif optim_cfg.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            trainable_params, lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY,
            momentum=optim_cfg.MOMENTUM
        )
    elif optim_cfg.OPTIMIZER in ['adam_onecycle','adam_cosineanneal']:
        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]
        betas = optim_cfg.get('BETAS', (0.9, 0.99))
        betas = tuple(betas)
        optimizer_func = partial(optim.Adam, betas=betas)
        if optim_cfg.get('LAYER_LR', None) is not None:
            params_lr = copy.deepcopy(optim_cfg.LAYER_LR)
            layer_groups_n = [[] for _ in range(len(params_lr) + 1)]
            used_module = []
            for name, module in model.named_modules():
                for idx, params_lr_dict in enumerate(params_lr):
                    params = params_lr_dict['params']
                    if params == name:
                        tmp_module = flatten_model(module)
                        layer_groups_n[idx].extend(tmp_module)
                        used_module.extend(tmp_module)
            layer_groups_n[-1] = [module for module in flatten_model(model) if module not in used_module]
            layer_groups = [nn.Sequential(*layer_groups_n[idx]) for idx in range(len(layer_groups_n))]

            layer_lr = [ele['lr'] for ele in params_lr]
            layer_lr.append(optim_cfg.LR)
            fix_layer_lr = optim_cfg.FIX_LAYER_LR
            optimizer = OptimWrapper.create(
                optimizer_func, 3e-3, layer_groups, wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True, layer_lr=layer_lr, fix_layer_lr=fix_layer_lr
            )
        else:
            layer_groups = get_layer_groups(model)
            optimizer = OptimWrapper.create(
                optimizer_func, 3e-3, layer_groups, wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
            )
    else:
        raise NotImplementedError

    return optimizer


def build_scheduler(optimizer, total_iters_each_epoch, total_epochs, last_epoch, optim_cfg):
    decay_steps = [x * total_iters_each_epoch for x in optim_cfg.DECAY_STEP_LIST]
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in decay_steps:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * optim_cfg.LR_DECAY
        return max(cur_decay, optim_cfg.LR_CLIP / optim_cfg.LR)

    lr_warmup_scheduler = None
    total_steps = total_iters_each_epoch * total_epochs
    if optim_cfg.OPTIMIZER == 'adam_onecycle':
        lr_scheduler = OneCycle(
            optimizer, total_steps, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
        )
    elif optim_cfg.OPTIMIZER == 'adam_cosineanneal':
        lr_scheduler = CosineAnnealing(
            optimizer, total_steps, total_epochs, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.PCT_START, optim_cfg.WARMUP_ITER
        )
    elif optim_cfg.OPTIMIZER == 'adam':
        lr_scheduler = lr_sched.MultiStepLR(optimizer, milestones=optim_cfg.DECAY_STEP_LIST, gamma=0.1)
        if optim_cfg.LR_WARMUP:
            lr_warmup_scheduler = CosineWarmupLR(
                optimizer, T_max=optim_cfg.WARMUP_EPOCH * total_iters_each_epoch,
                eta_min=optim_cfg.LR / optim_cfg.DIV_FACTOR
            )
    else:
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

        if optim_cfg.LR_WARMUP:
            lr_warmup_scheduler = CosineWarmupLR(
                optimizer, T_max=optim_cfg.WARMUP_EPOCH * total_iters_each_epoch,
                eta_min=optim_cfg.LR / optim_cfg.DIV_FACTOR
            )

    return lr_scheduler, lr_warmup_scheduler
