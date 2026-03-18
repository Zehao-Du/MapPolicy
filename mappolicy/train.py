import functools
import json
import os
import pathlib
import sys
from datetime import datetime

import hydra
import numpy as np
import torch
import wandb
from hydra.utils import call, instantiate
from omegaconf import OmegaConf
from termcolor import colored
import tqdm

from mappolicy.envs.evaluator import Evaluator
from mappolicy.helper.common import set_seed
from mappolicy.helper.logger import Logger, WandBLogger
from mappolicy.helper.pytorch import log_params_to_file

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

@hydra.main(version_base=None, config_path="config", config_name="train_maniskill")
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
    # datasets and evaluator #
    ##########################
    train_dataset = instantiate(
        config=config.benchmark.dataset_instantiate_config,
        data_dir=config.dataset_dir,
        split="train",
    )
    valid_dataset = instantiate(
        config=config.benchmark.dataset_instantiate_config,
        data_dir=config.dataset_dir,
        split="validation",
    )
    evaluator: Evaluator = instantiate(
        config=config.benchmark.evaluator_instantiate_config,
        task_name=config.task_name,
    )

    ###############
    # dataloaders #
    ###############
    DataLoaderConstuctor = functools.partial(
        torch.utils.data.DataLoader,
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.num_workers,
        shuffle=config.dataloader.shuffle,
        pin_memory=config.dataloader.pin_memory,
        drop_last=config.dataloader.drop_last,
    )
    train_loader = DataLoaderConstuctor(train_dataset)
    valid_loader = DataLoaderConstuctor(valid_dataset)
    _, _, _, sample_robot_state, _, sample_action, _ = next(iter(train_loader))
    robot_state_dim = sample_robot_state.size(-1)
    action_dim = sample_action.size(-1)
    Logger.log_info(f'Robot state dim: {colored(robot_state_dim, "red")}')
    Logger.log_info(f'Action dim: {colored(action_dim, "red")}')
    Logger.log_info(f"train data: {len(train_dataset)}")
    Logger.log_info(f"validation data: {len(valid_dataset)}")

    #########
    # Model #
    #########
    model = instantiate(
        config=config.agent.instantiate_config,
        robot_state_dim=robot_state_dim,
        action_dim=action_dim,
    )
    model = model.to(config.device)

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

    ###########################
    # Learning rate scheduler #
    ###########################
    scheduler: torch.optim.lr_scheduler.LRScheduler = instantiate(
        config=config.train.scheduler_instantiate_config,
        optimizer=optimizer,
    )

    ############
    # Training #
    ############
    epoch_log_file = local_run_output_dir / "train_epoch.log"
    with open(epoch_log_file, "w", encoding="utf-8") as f:
        f.write("# timestamp | epoch | train_loss | lr | val_loss\n")

    grad_accum_steps = config.train.get('gradient_accumulate_every', 1)
    max_train_steps = config.train.get('max_train_steps', None)
    max_val_steps = config.train.get('max_val_steps', None)
    tqdm_interval = config.train.get('tqdm_interval_sec', 1.0)

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    max_success, max_rewards, best_success = 0.0, 0.0, -1.0

    for cur_epoch in range(config.train.num_epochs):
        step_log = {
            "epoch": cur_epoch,
            "epoch_step": cur_epoch + 1,
        }

        train_losses = []
        model.train()
        with tqdm.tqdm(
            train_loader,
            desc=f"Training epoch {cur_epoch}",
            leave=False,
            mininterval=tqdm_interval,
        ) as tepoch:
            for cur_iter, (
                images,
                depths,
                point_clouds,
                point_cloud_no_robot,
                robot_states,
                raw_states,
                actions,
                texts,
            ) in enumerate(tepoch):
                # training iteration
                images = images.to(config.device, non_blocking=True)
                point_clouds = point_clouds.to(config.device, non_blocking=True)
                point_cloud_no_robot = point_cloud_no_robot.to(config.device, non_blocking=True)
                robot_states = robot_states.to(config.device, non_blocking=True)
                actions = actions.to(config.device, non_blocking=True)

                preds = model(images, point_clouds, point_cloud_no_robot, robot_states, texts)
                loss_result = call(config.benchmark.loss_func, preds, actions)

                # loss verbose
                if isinstance(loss_result, tuple):
                    raw_loss = loss_result[0]
                    loss_dict = loss_result[1]
                else:
                    raw_loss = loss_result
                    loss_dict = {}

                loss = raw_loss / grad_accum_steps
                loss.backward()

                is_last_batch = (cur_iter == (len(train_loader) - 1))
                should_step = ((cur_iter + 1) % grad_accum_steps == 0) or is_last_batch

                if should_step:
                    # clip gradient
                    if config.train.clip_grad_value > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=config.train.clip_grad_value
                        )

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                raw_loss_cpu = raw_loss.item()
                train_losses.append(raw_loss_cpu)
                current_lr = scheduler.get_last_lr()[0]
                tepoch.set_postfix(
                    _format_postfix(cur_epoch, global_step, raw_loss_cpu, current_lr),
                    refresh=False,
                )

                iteration_info = {
                    "iteration_step": global_step,
                    "train_interation/epoch": cur_epoch,
                    "train_interation/loss": raw_loss_cpu,
                    "train_interation/learning_rate": current_lr,
                }
                for key, value in loss_dict.items():
                    iteration_info[f"train_interation/{key}"] = _to_float(value)

                # keep last step for end-of-epoch merged logging
                if not is_last_batch:
                    wandb_logger.log(iteration_info, step=global_step)
                    global_step += 1

                if (max_train_steps is not None) and cur_iter >= (max_train_steps - 1):
                    break

        # training epoch log
        train_loss = float(np.mean(train_losses)) if len(train_losses) > 0 else 0.0
        step_log.update({
            "train_epoch/epoch_loss": train_loss,
            "train_loss": train_loss,
            "lr": scheduler.get_last_lr()[0],
            "global_step": global_step,
        })
        Logger.log_info(f"[train] epoch={cur_epoch}, loss={train_loss}")

        # Validation
        periodic_validation = (cur_epoch + 1 > config.evaluation.num_skip_epochs) and (
            (cur_epoch + 1) % config.evaluation.validation_frequency_epochs == 0
        )
        last_epoch = (cur_epoch + 1) == config.train.num_epochs
        if periodic_validation or last_epoch:
            model.eval()

            # validation loss
            val_loss = 0.0
            with torch.no_grad():
                val_losses = []
                with tqdm.tqdm(
                    valid_loader,
                    desc=f"Validation epoch {cur_epoch}",
                    leave=False,
                    mininterval=tqdm_interval,
                ) as tepoch:
                    for cur_iter, (
                        images,
                        point_clouds,
                        point_cloud_no_robot,
                        robot_states,
                        raw_states,
                        actions,
                        texts,
                    ) in enumerate(tepoch):
                        images = images.to(config.device, non_blocking=True)
                        point_clouds = point_clouds.to(config.device, non_blocking=True)
                        point_cloud_no_robot = point_cloud_no_robot.to(config.device, non_blocking=True)
                        robot_states = robot_states.to(config.device, non_blocking=True)
                        actions = actions.to(config.device, non_blocking=True)

                        preds = model(images, point_clouds, point_cloud_no_robot, robot_states, texts)
                        loss_result = call(config.benchmark.loss_func, preds, actions)

                        if isinstance(loss_result, tuple):
                            loss = loss_result[0]
                            loss_dict = loss_result[1]
                            for key, value in loss_dict.items():
                                step_log[f"validation/{key}"] = _to_float(value)
                        else:
                            loss = loss_result

                        val_losses.append(loss.item())
                        tepoch.set_postfix(
                            _format_postfix(cur_epoch, global_step, loss.item(), scheduler.get_last_lr()[0]),
                            refresh=False,
                        )

                        if (max_val_steps is not None) and cur_iter >= (max_val_steps - 1):
                            break

                if len(val_losses) > 0:
                    val_loss = float(np.mean(val_losses))
                    step_log["validation/loss"] = val_loss
                    step_log["val_loss"] = val_loss

            # validation success and rewards
            avg_success, avg_rewards = evaluator.evaluate(
                config.evaluation.validation_trajs_num, model
            )
            max_success, max_rewards = max(max_success, avg_success), max(
                max_rewards, avg_rewards
            )
            step_log.update(
                {
                    "validation/epoch": cur_epoch,
                    "validation/success": avg_success,
                    "validation/rewards": avg_rewards,
                    "validation/max_success": max_success,
                    "validation/max_rewards": max_rewards,
                    "validation/video_steps": wandb.Video(
                        evaluator.env.get_frames().transpose(0, 3, 1, 2), fps=30
                    ),
                }
            )
            evaluator.callback(step_log)
            Logger.log_info(
                f"[validation] epoch={cur_epoch}, "
                f"validation_loss={val_loss}, "
                f"avg_success={avg_success}, "
                f"avg_rewards={avg_rewards}, "
                f"max_success={max_success}, "
                f"max_rewards={max_rewards}"
            )

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
                        "epoch": cur_epoch,
                        "loss": val_loss,
                        "avg_success": avg_success,
                        "avg_rewards": float(avg_rewards),
                    }
                    json.dump(model_info, f, indent=4)
            if avg_success >= 1.0:
                Logger.log_info(colored(f"Success Rate reached 1.0 at epoch {cur_epoch}, stopping training.", "green"))
                break

        # log epoch info
        wandb_logger.log(step_log, step=global_step)
        _append_epoch_log(
            epoch_log_file,
            epoch=cur_epoch,
            train_loss=train_loss,
            lr=step_log['lr'],
            val_loss=step_log.get('val_loss', None),
        )
        global_step += 1

    Logger.log_ok("Training Finished!")


if __name__ == "__main__":
    main()
