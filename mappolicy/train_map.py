import functools
import os
import pathlib
from datetime import datetime
from typing import Dict

import hydra
import numpy as np
import torch
from hydra.utils import call, instantiate
from termcolor import colored
import tqdm

from mappolicy.helper.common import set_seed
from mappolicy.helper.logger import Logger

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


def _to_xyz(pc: torch.Tensor) -> torch.Tensor:
    if not isinstance(pc, torch.Tensor):
        raise TypeError(f"Point cloud must be torch.Tensor, got {type(pc)}")
    if pc.ndim != 3 or pc.size(-1) < 3:
        raise ValueError(f"Expected point cloud shape [B, N, C>=3], got {tuple(pc.shape)}")
    return pc[..., :3]


def _save_point_cloud_pairs(
    save_root: pathlib.Path,
    epoch: int,
    batch_idx: int,
    pred_xyz: torch.Tensor,
    gt_xyz: torch.Tensor,
    max_items: int,
):
    """Save prediction and GT point clouds as .npy files for side-by-side comparison."""
    batch_dir = save_root / f"epoch_{epoch:04d}" / f"batch_{batch_idx:04d}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    num_items = min(max_items, pred_xyz.shape[0], gt_xyz.shape[0])
    pred_np = pred_xyz.detach().cpu().numpy()
    gt_np = gt_xyz.detach().cpu().numpy()
    for i in range(num_items):
        np.save(batch_dir / f"sample_{i:02d}_pred.npy", pred_np[i])
        np.save(batch_dir / f"sample_{i:02d}_gt.npy", gt_np[i])

    # Save one bundled file for convenience
    np.savez_compressed(
        batch_dir / "comparison.npz",
        pred=pred_np[:num_items],
        gt=gt_np[:num_items],
    )

@hydra.main(version_base=None, config_path="config", config_name="train_map")
def main(config):
    # torch.autograd.set_detect_anomaly(True)
    Logger.log_info(f'Task: {colored(config.task_name, "green")}')
    Logger.log_info(f'Dataset directory: {colored(config.dataset_dir, "green")}')
    Logger.log_info(f'Model: {colored(config.agent.name, "green")}')
    Logger.log_info(f'benchmark: {colored(config.benchmark.name, "green")}')
    Logger.log_info(f'Image size: {colored(config.image_size, "green")}')

    ############
    # set seed #
    ############
    set_seed(config.seed)

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

    model = instantiate(
        config=config.agent.instantiate_config,
    )
    model = model.to(config.device)

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=1e-4  
    )
    local_run_output_dir = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )

    scheduler: torch.optim.lr_scheduler.LRScheduler = instantiate(
        config=config.train.scheduler_instantiate_config,
        optimizer=optimizer,
    )

    epoch_log_file = local_run_output_dir / "train_epoch.log"
    with open(epoch_log_file, "w", encoding="utf-8") as f:
        f.write("# timestamp | epoch | train_loss | lr | val_loss\n")

    sample_save_root = local_run_output_dir / "pointcloud_samples"
    sample_save_root.mkdir(parents=True, exist_ok=True)

    grad_accum_steps = config.train.get('gradient_accumulate_every', 1)
    max_train_steps = config.train.get('max_train_steps', None)
    max_val_steps = config.train.get('max_val_steps', None)
    tqdm_interval = config.train.get('tqdm_interval_sec', 1.0)

    eval_cfg = config.get("evaluation", {})
    num_skip_epochs = eval_cfg.get("num_skip_epochs", 0)
    validation_frequency_epochs = eval_cfg.get("validation_frequency_epochs", 10)
    save_best_model = eval_cfg.get("save_best_model", True)
    save_point_clouds = eval_cfg.get("save_point_clouds", True)
    save_num_batches = eval_cfg.get("save_num_batches", 1)
    save_num_samples_per_batch = eval_cfg.get("save_num_samples_per_batch", 4)

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    best_val_loss = float("inf")

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
                actions,
            ) in enumerate(tepoch):
                # training iteration
                images = images.to(config.device, non_blocking=True)
                depths = depths.to(config.device, non_blocking=True)
                point_clouds = point_clouds.to(config.device, non_blocking=True)
                point_cloud_no_robot = point_cloud_no_robot.to(config.device, non_blocking=True)

                sence_map = model(point_cloud_no_robot)
                preds = sence_map.complete_point_cloud()
                pred_pc = _to_xyz(preds)
                gt_pc = _to_xyz(point_cloud_no_robot)

                loss_result = call(config.benchmark.loss_func, pred_pc, gt_pc)

                loss = loss_result / grad_accum_steps
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

                raw_loss_cpu = loss_result.item()
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

                # keep last step for end-of-epoch merged logging
                if not is_last_batch:
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

        # Validation
        periodic_validation = (cur_epoch + 1 > num_skip_epochs) and (
            (cur_epoch + 1) % validation_frequency_epochs == 0
        )
        last_epoch = (cur_epoch + 1) == config.train.num_epochs
        if periodic_validation or last_epoch:
            model.eval()

            # validation loss
            val_loss = 0.0
            with torch.no_grad():
                val_losses = []
                saved_batches = 0
                with tqdm.tqdm(
                    valid_loader,
                    desc=f"Validation epoch {cur_epoch}",
                    leave=False,
                    mininterval=tqdm_interval,
                ) as tepoch:
                    for cur_iter, (
                        images,
                        depths,
                        point_clouds,
                        point_cloud_no_robot,
                        robot_states,
                        actions,
                    ) in enumerate(tepoch):
                        images = images.to(config.device, non_blocking=True)
                        depths = depths.to(config.device, non_blocking=True)
                        point_clouds = point_clouds.to(config.device, non_blocking=True)
                        point_cloud_no_robot = point_cloud_no_robot.to(config.device, non_blocking=True)

                        sence_map = model(point_cloud_no_robot)
                        preds = sence_map.complete_point_cloud()
                        pred_pc = _to_xyz(preds)
                        gt_pc = _to_xyz(point_cloud_no_robot)

                        loss_result = call(config.benchmark.loss_func, pred_pc, gt_pc)

                        val_losses.append(loss_result.item())
                        tepoch.set_postfix(
                            _format_postfix(cur_epoch, global_step, loss_result.item(), scheduler.get_last_lr()[0]),
                            refresh=False,
                        )

                        # Save predicted and GT point clouds for comparison
                        if save_point_clouds and saved_batches < save_num_batches:
                            _save_point_cloud_pairs(
                                save_root=sample_save_root,
                                epoch=cur_epoch,
                                batch_idx=cur_iter,
                                pred_xyz=pred_pc,
                                gt_xyz=gt_pc,
                                max_items=save_num_samples_per_batch,
                            )
                            saved_batches += 1

                        if (max_val_steps is not None) and cur_iter >= (max_val_steps - 1):
                            break

                if len(val_losses) > 0:
                    val_loss = float(np.mean(val_losses))
                    step_log["validation/loss"] = val_loss
                    step_log["val_loss"] = val_loss

                    Logger.log_info(
                        f"[validation] epoch={cur_epoch}, val_loss={val_loss:.6f}, "
                        f"saved_batches={saved_batches}"
                    )

                    if save_best_model and val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_path = local_run_output_dir / "best_model.pth"
                        torch.save(model.state_dict(), best_model_path)
                        Logger.log_ok(f"New best model saved: {best_model_path} (val_loss={best_val_loss:.6f})")

        # log epoch info
        _append_epoch_log(
            epoch_log_file,
            epoch=cur_epoch,
            train_loss=train_loss,
            lr=step_log['lr'],
            val_loss=step_log.get('val_loss', None),
        )

        # epoch-based LR scheduling: step once per epoch
        scheduler.step()
        global_step += 1

    Logger.log_ok("Training Finished!")


if __name__ == "__main__":
    main()
