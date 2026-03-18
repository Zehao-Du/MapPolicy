import functools
import json
import os
import pathlib
import copy
import math
from datetime import datetime

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from termcolor import colored
import tqdm

import os
import sys
if __package__ is None or __package__ == "":
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.abspath(os.path.join(_current_dir, ".."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

from mappolicy.envs.evaluator import Evaluator
from mappolicy.helper.common import set_seed
from mappolicy.helper.logger import Logger, WandBLogger
from mappolicy.helper.pytorch import log_params_to_file, Optimizers

from mappolicy.models.diffusion_policy.diffusion.ema_model import EMAModel
from mappolicy.helper.pytorch import dict_apply

os.environ['HYDRA_FULL_ERROR'] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def _format_postfix(epoch, global_step, loss, lr):
    return {
        'epoch': int(epoch),
        'step': int(global_step),
        'loss': f"{_to_float(loss):.4f}",
        'lr': f"{_to_float(lr):.2e}",
    }


def _append_epoch_log(log_file: pathlib.Path, epoch: int, train_loss: float, lr: float, val_loss: float = None):
    val_str = f"{val_loss:.6f}" if val_loss is not None else "NA"
    line = (
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"epoch={epoch} | train_loss={train_loss:.6f} | lr={lr:.8e} | val_loss={val_str}\n"
    )
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line)


@hydra.main(version_base=None, config_path="config", config_name="traindp_maniskill")
def main(config):
    # torch.autograd.set_detect_anomaly(True)
    #############################
    # log important information #
    #############################
    
    Logger.log_info(
        f'Running {colored(pathlib.Path(__file__).absolute(), "red")} with following config:'
    )
    Logger.log_info(f'Task: {colored(config.task_name, "green")}')
    Logger.log_info(f'Dataset directory: {colored(config.dataset_dir, "green")}')
    Logger.log_info(f'Image size: {colored(config.image_size, "green")}')
    Logger.log_info(
        f'WandB: Project {colored(config.wandb.project, "green")}; '
        f'Group {colored(config.wandb.group, "green")}; '
        f'Name {colored(config.wandb.name, "green")}; '
        f'Notes {colored(config.wandb.notes, "green")}; '
        f'Mode {colored(config.wandb.mode, "green")}'
    )
    Logger.log_info(
        f'Agent: {colored(config.agent.name, color="green")}\n{json.dumps(OmegaConf.to_container(config.agent, resolve=True), indent=4)}'
    )
    Logger.log_info(
        f'Benchmark: {colored(config.benchmark.name, color="green")}\n{json.dumps(OmegaConf.to_container(config.benchmark, resolve=True), indent=4)}'
    )
    Logger.print_seperator()
    
    ############
    # set seed #
    ############
    set_seed(config.seed)
    
    ################
    # wandb logger #
    ################
    wandb_logger = WandBLogger(
        config=config,
        hyperparameters=OmegaConf.to_container(config, resolve=True),
    )
    wandb_logger.run.define_metric("train_interation/*", step_metric="iteration_step")
    wandb_logger.run.define_metric("train_epoch/*", step_metric="epoch_step")
    wandb_logger.run.define_metric("validation/*", step_metric="epoch_step")

    ##########################
    # datasets and dataloaders #
    ##########################
    dataset = instantiate(config.benchmark.dataset_instantiate_config)
    val_dataset = dataset.get_validation_dataset()
    normalizer = dataset.get_normalizer()
    DataLoaderConstuctor = functools.partial(
        torch.utils.data.DataLoader,
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.num_workers,
        shuffle=config.dataloader.shuffle,
        pin_memory=config.dataloader.pin_memory,
        drop_last=config.dataloader.drop_last,
    )
    train_loader = DataLoaderConstuctor(dataset)
    val_loader = DataLoaderConstuctor(val_dataset)
    
    # image, depth, pcd, pcd_no_robot, robot_state, action
    sample_batch = next(iter(train_loader))
    # sample_image = sample_batch['image']
    # sample_depth = sample_batch['depth']
    # sample_pcd = sample_batch['point_cloud']
    # sample_pcd_no_robot = sample_batch['point_cloud_no_robot']
    sample_robot_state = sample_batch['robot_state']
    sample_action = sample_batch['action']
    robot_state_dim = sample_robot_state.size(-1)
    action_dim = sample_action.size(-1)
    Logger.log_info(f'Robot state dim: {colored(robot_state_dim, "red")}')
    Logger.log_info(f'Action dim: {colored(action_dim, "red")}')
    Logger.log_info(f"dataloader instantiated successfully with sample batch keys: {list(sample_batch.keys())} and shapes: { {k: v.shape for k, v in sample_batch.items()} }")
    
    ############
    # evaluator #
    ############
    evaluator: Evaluator = instantiate(config=config.benchmark.evaluator_instantiate_config)
    
    #########
    # Model #
    #########
    model = instantiate( config=config.agent.instantiate_config )
    model.set_normalizer(normalizer)
    
    ema_model = None
    ema: EMAModel = None
    if config.train.use_ema:
        ema_model = copy.deepcopy(model)
        ema_model.set_normalizer(normalizer)
        ema = instantiate(config.ema, model=ema_model)
        
    # device transfer
    model = model.to(config.device)
    if config.train.use_ema:
        ema_model = ema_model.to(config.device)

    #############
    # Optimizer #
    #############
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=1e-4  
    )
    local_run_output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    log_params_to_file(model, local_run_output_dir / "train_model_params.txt", True)
    log_params_to_file(
        model, local_run_output_dir / "train_model_params_freeze.txt", False
    )

    grad_accum_steps = config.train.gradient_accumulate_every

    ###########################
    # Learning rate scheduler #
    ###########################
    effective_train_batches = len(train_loader)
    if config.train.max_train_steps is not None:
        effective_train_batches = min(effective_train_batches, int(config.train.max_train_steps))
    updates_per_epoch = math.ceil(effective_train_batches / max(1, grad_accum_steps))
    total_update_steps = max(1, updates_per_epoch * int(config.train.num_epochs))

    if "lr_scheduler" in config.train and config.train.lr_scheduler is not None:
        warmup_ratio = float(config.train.lr_scheduler.get("warmup_steps_ratio", 0.05))
        warmup_steps = int(total_update_steps * warmup_ratio)
        warmup_start_factor = float(config.train.lr_scheduler.get("warmup_start_factor", 0.01))
        min_lr_ratio = float(config.train.lr_scheduler.get("min_lr_ratio", 0.01))
        scheduler: torch.optim.lr_scheduler.LRScheduler = Optimizers.get_step_cosine_warmup_scheduler(
            optimizer=optimizer,
            total_steps=total_update_steps,
            warmup_steps=warmup_steps,
            warmup_start_factor=warmup_start_factor,
            min_lr_ratio=min_lr_ratio,
        )
        Logger.log_info(
            f"Using step-based cosine warmup scheduler: total_update_steps={total_update_steps}, "
            f"warmup_steps={warmup_steps}, warmup_start_factor={warmup_start_factor}, "
            f"min_lr_ratio={min_lr_ratio}"
        )
    else:
        # Backward-compatible fallback
        scheduler: torch.optim.lr_scheduler.LRScheduler = instantiate(
            config=config.train.scheduler_instantiate_config,
            optimizer=optimizer,
        )
        Logger.log_warning("Using legacy scheduler_instantiate_config. Consider migrating to train.lr_scheduler.")
    
    ############
    # Training #
    ############
    
    epoch_log_file = local_run_output_dir / "train_epoch.log"
    with open(epoch_log_file, "w", encoding="utf-8") as f:
        f.write("# timestamp | epoch | train_loss | lr | val_loss\n")
    
    epoch = 0
    global_step = 0
    train_sampling_batch = None
    optimizer.zero_grad(set_to_none=True)
    
    max_success, max_rewards, best_success = 0.0, 0.0, -1.0
    
    tqdm_interval = config.train.get('tqdm_interval_sec', 1.0)
    for _ in range(config.train.num_epochs):
        step_log = {}
        avg_success, avg_rewards = 0.0, 0.0
        
        # ========= train for this epoch ==========
        train_losses = []
        with tqdm.tqdm(train_loader, desc=f"Training epoch {epoch}",
            leave=False, mininterval=tqdm_interval) as tepoch: 
            for batch_idx, batch in enumerate(tepoch):
                # device transfer
                batch = dict_apply(batch, lambda x: x.to(config.device, non_blocking=True))
                if train_sampling_batch is None:
                    train_sampling_batch = batch
                    
                # compute loss
                raw_loss = model.compute_loss(batch)
                loss = raw_loss / grad_accum_steps
                loss.backward()

                is_last_batch = (batch_idx == (len(train_loader) - 1))
                should_step = ((batch_idx + 1) % grad_accum_steps == 0) or is_last_batch
                
                # step optimizer
                if should_step:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    # update ema on optimizer step
                    if config.train.use_ema:
                        ema.step(model)
                    
                # logging
                raw_loss_cpu = raw_loss.item()
                train_losses.append(raw_loss_cpu)
                current_lr = scheduler.get_last_lr()[0]
                step_log = {
                    'train_loss': raw_loss_cpu,
                    'global_step': global_step,
                    'epoch': epoch,
                    'lr': current_lr
                }
                tepoch.set_postfix(_format_postfix(epoch, global_step, raw_loss_cpu, current_lr), refresh=False)
            
            
                if not is_last_batch:
                    # log of last step is combined with validation and rollout
                    wandb_logger.log(step_log, step=global_step)
                    # Logger.log_info(step_log)
                    global_step += 1

                if (config.train.max_train_steps is not None) \
                    and batch_idx >= (config.train.max_train_steps-1):
                    break
        
        # at the end of each epoch
        # replace train_loss with epoch average
        train_loss = float(np.mean(train_losses)) if len(train_losses) > 0 else 0.0
        step_log['train_loss'] = train_loss
        
        # ========= eval for this epoch ==========
        policy = model
        if config.train.use_ema:
            policy = ema_model
        policy.eval()
        
        # run validation
        val_loss = 0.0
        if (epoch % config.train.val_every) == 0:
            with torch.no_grad():
                val_losses = []
                with tqdm.tqdm(val_loader, desc=f"Validation epoch {epoch}", 
                        leave=False, mininterval=config.train.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(config.device, non_blocking=True))
                        loss = model.compute_loss(batch)
                        val_losses.append(loss)
                        tepoch.set_postfix(_format_postfix(epoch, global_step, loss.item(), scheduler.get_last_lr()[0]), refresh=False)
                        if (config.train.max_val_steps is not None) \
                            and batch_idx >= (config.train.max_val_steps-1):
                            break
                if len(val_losses) > 0:
                    val_loss = torch.mean(torch.tensor(val_losses)).item()
                    step_log['val_loss'] = val_loss
                    
        # run rollout
        if (epoch >= config.evaluation.num_skip_epochs and (epoch - config.evaluation.num_skip_epochs) % config.evaluation.validation_frequency_epochs == 0):
            avg_success, avg_rewards = evaluator.evaluate(
                config.evaluation.validation_trajs_num, model, device=config.device
            )
            max_success, max_rewards = max(max_success, avg_success), max(
                max_rewards, avg_rewards
            )
            Logger.log_info(
                f"[validation] epoch={epoch}, "
                f"validation_loss={val_loss}, "
                f"avg_success={avg_success}, "
                f"avg_rewards={avg_rewards}, "
                f"max_success={max_success}, "
                f"max_rewards={max_rewards}"
            )
                    
        # # run diffusion sampling on a training batch
        # if (epoch % config.train.sample_every) == 0:
        #     with torch.no_grad():
        #         # sample trajectory from training set, and evaluate difference
        #         batch = dict_apply(train_sampling_batch, lambda x: x.to(config.device, non_blocking=True))
        #         obs_dict = batch['obs']
        #         gt_action = batch['action']
                
        #         result = policy.predict_action(obs_dict)
        #         pred_action = result['action_pred']
        #         mse = torch.nn.functional.mse_loss(pred_action, gt_action)
        #         step_log['train_action_mse_error'] = mse.item()
        #         del batch
        #         del obs_dict
        #         del gt_action
        #         del result
        #         del pred_action
        #         del mse
                
        # save best model
        if config.evaluation.save_best_model and avg_success > best_success:
            best_success = avg_success
            model_path = os.path.join(
                hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"],
                "best_model.pth",
            )
            torch.save(model.state_dict(), model_path)
            Logger.log_info(f'Save best model to {colored(model_path, "red")}')
            with open(
                os.path.join(
                    hydra.core.hydra_config.HydraConfig.get()["runtime"][
                        "output_dir"
                    ],
                    "best_model.json",
                ),
                "w",
            ) as f:
                model_info = {
                    "epoch": epoch,
                    "loss": val_loss,
                    "avg_success": avg_success,
                    "avg_rewards": float(avg_rewards),
                }
                json.dump(model_info, f, indent=4)
        if avg_success >= 1.0:
            Logger.log_info(colored(f"Success Rate reached 1.0 at epoch {epoch}, stopping training.", "green"))
            break
                
                
        # ========= eval end for this epoch ==========
        policy.train()

        # end of epoch
        # log of last step is combined with validation and rollout
        wandb_logger.log(step_log, step=global_step)
        # Logger.log_info(step_log)
        _append_epoch_log(
            epoch_log_file,
            epoch=epoch,
            train_loss=train_loss,
            lr=step_log['lr'],
            val_loss=step_log.get('val_loss', None)
        )
        global_step += 1
        epoch += 1

if __name__ == "__main__":
    main()