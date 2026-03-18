import copy
import json
import math
import os
import pathlib
import sys

from typing import Callable, Dict

import torch
import torch.nn as nn
from termcolor import colored

from mappolicy.helper.common import Logger


def get_optimizer_groups(model, default_wd):
    param_group_names, param_group_vars = dict(), dict()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "token" in n:
            name_apx = "t"
            wd_val = 0.0
        elif "pos_embed" in n:
            name_apx = "p"
            wd_val = 0.0
        elif "bn" in n or "ln" in n or "norm" in n:
            name_apx = "n"
            wd_val = 0.0
        elif "bias" in n:
            name_apx = "b"
            wd_val = 0.0
        else:
            name_apx = "w"
            wd_val = default_wd

        param_group = f"wd:{name_apx}"
        if param_group not in param_group_names:
            item = {"params": [], "weight_decay": wd_val}
            param_group_names[param_group] = copy.deepcopy(item)
            param_group_vars[param_group] = copy.deepcopy(item)
        param_group_names[param_group]["params"].append(n)
        param_group_vars[param_group]["params"].append(p)

    param_list = list(param_group_vars.values())

    param_group_str = colored(
        json.dumps(param_group_names, sort_keys=True, indent=2), "blue"
    )
    print("Parameter groups:\n" + param_group_str)

    return param_list


class Optimizers(object):
    @staticmethod
    def get_constant_scheduler(optimizer: torch.optim.Optimizer, lr: float):
        lambda_func = lambda _: 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_func)
        return scheduler

    @staticmethod
    def get_warmup_cosine_annealing_scheduler(
        optimizer: torch.optim.Optimizer,
        num_warmup_epochs: int,
        num_epochs: int,
        warmup_factor: float = 0.1,
        eta_min: float = 1e-6,  # [推荐] 增加这个参数，防止LR降到0
    ):
        # 1. Warmup 阶段: LR 从 (lr * warmup_factor) 线性增加到 lr
        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=warmup_factor,
            end_factor=1.0, 
            total_iters=num_warmup_epochs,
        )
        
        # 2. Cosine 阶段: LR 从 lr 余弦衰减到 eta_min
        scheduler_train = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=(num_epochs - num_warmup_epochs),
            eta_min=eta_min,
        )
        
        # 3. 串联: 前 num_warmup_epochs 个 epoch 用 scheduler_warmup，之后用 scheduler_train
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[scheduler_warmup, scheduler_train],
            milestones=[num_warmup_epochs],
        )
        
        return scheduler

    @staticmethod
    def get_step_cosine_warmup_scheduler(
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int,
        warmup_start_factor: float = 0.01,
        min_lr_ratio: float = 0.01,
    ):
        """Step-based cosine scheduler with linear warmup.

        Designed for training loops where `scheduler.step()` is called on every
        optimizer update step.
        """
        total_steps = max(1, int(total_steps))
        warmup_steps = max(0, min(int(warmup_steps), total_steps - 1))
        warmup_start_factor = float(max(1e-6, min(1.0, warmup_start_factor)))
        min_lr_ratio = float(max(0.0, min(1.0, min_lr_ratio)))

        def lr_lambda(current_step: int) -> float:
            current_step = int(max(0, current_step))
            if warmup_steps > 0 and current_step < warmup_steps:
                alpha = current_step / max(1, warmup_steps)
                return warmup_start_factor + (1.0 - warmup_start_factor) * alpha

            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lr_lambda,
        )
        return scheduler


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count


def log_params_to_file(model, filename, requires_grad):
    save_dir = pathlib.Path(filename).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    trainable_params = []
    for name, p in model.named_parameters():
        if p.requires_grad == requires_grad:
            trainable_params.append(name)

    with open(filename, "w") as f:
        for param in trainable_params:
            f.write(f"{param}\n")

    Logger.log_info(
        f"{'Trainable' if requires_grad else 'Freezed'} parameters saved to {filename}"
    )


def safe_normalize(v, eps=1e-8):
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result

def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module