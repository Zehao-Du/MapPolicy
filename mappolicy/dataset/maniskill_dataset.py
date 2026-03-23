import torch
import zarr
from termcolor import colored
import numpy as np
import os

from mappolicy.helper.logger import Logger

class ManiskillDataset(torch.utils.data.Dataset):
    """
    Dataset for Maniskill Benchmark.

    Images range: [0, 255]
    Robot states range: [-1.0, 1.0]
    Raw states range: [-1.0, 1.0]
    Actions range: [-7.0, 13.0]
    """

    SPLIT_SIZE = {"train": 900, "validation": 100, "custom": None}

    def __init__(self, data_dir, split: str = None, custom_split_size: int = None):
        zarr_root = zarr.open_group(data_dir, mode="r")
        self._episode_ends = zarr_root["meta"]["episode_ends"][:]

        if split not in self.SPLIT_SIZE:
            raise ValueError(f"Invalid split: {split}")

        if split == "custom" and custom_split_size is None:
            raise ValueError(f"custom_split_size must be provided for split: {split}")

        begin_index, end_index = (
            (0, self._episode_ends[self.SPLIT_SIZE["train"] - 1])
            if split == "train"
            else (
                (
                    self._episode_ends[self.SPLIT_SIZE["train"] - 1],
                    self._episode_ends[
                        self.SPLIT_SIZE["train"] + self.SPLIT_SIZE["validation"] - 1
                    ],
                )
                if split == "validation"
                else (0, self._episode_ends[custom_split_size - 1])
            )
        )

        self._images = zarr_root["data"]["images"][begin_index:end_index].transpose(
            0, 3, 1, 2
        )
        assert self._images.shape[1] == 3
        self._depth = zarr_root["data"]["depths"][begin_index:end_index]
        self._point_clouds = zarr_root["data"]["point_clouds"][begin_index:end_index]
        self._point_clouds_no_robot = zarr_root["data"]["point_clouds_no_robot"][begin_index:end_index]
        self._robot_states = zarr_root["data"]["robot_states"][begin_index:end_index]
        self._actions = zarr_root["data"]["actions"][begin_index:end_index]
        assert len(self._images) == len(self._robot_states) == len(self._actions)
        self._dataset_size = len(self._actions)

        # report split sizes for debugging
        train_steps = int(self._episode_ends[self.SPLIT_SIZE["train"] - 1])
        val_steps = int(
            self._episode_ends[self.SPLIT_SIZE["train"] + self.SPLIT_SIZE["validation"] - 1]
            - train_steps
        )
        Logger.log_info(f"train steps: {train_steps}, validation steps: {val_steps}")

    def __getitem__(self, idx):
        image = torch.from_numpy(self._images[idx]).float()
        depth = torch.from_numpy(self._depth[idx]).float()
        depth = depth.unsqueeze(0)  # (1, H, W)
        depth = depth / depth.max()  # normalize depth to [0, 1]
        point_cloud = torch.from_numpy(self._point_clouds[idx]).float()
        point_cloud_no_robot = torch.from_numpy(self._point_clouds_no_robot[idx]).float()
        robot_state = torch.from_numpy(self._robot_states[idx]).float()
        action = torch.from_numpy(self._actions[idx]).float()
        return image, depth, point_cloud, point_cloud_no_robot, robot_state, action

    def __len__(self):
        return self._dataset_size

    def print_info(self):
        Logger.log_info(f"Maniskill Dataset Info:")
        Logger.log_info(
            f'point_cloud ({colored(self._point_clouds.dtype, "red")}): {colored(self._point_clouds.shape, "red")}, '
            f'xyz_range: [{colored(self._point_clouds[..., 0:3].min(), "red")}, {colored(self._point_clouds[..., 0:3].max(), "red")}], '
            f'rgb_range: [{colored(self._point_clouds[..., 3:6].min(), "red")}, {colored(self._point_clouds[..., 3:6].max(), "red")}]'
        )
        Logger.log_info(
            f'point_cloud_no_robot ({colored(self._point_clouds_no_robot.dtype, "red")}): {colored(self._point_clouds_no_robot.shape, "red")}, '
            f'xyz_range: [{colored(self._point_clouds_no_robot[..., 0:3].min(), "red")}, {colored(self._point_clouds_no_robot[..., 0:3].max(), "red")}], '
            f'rgb_range: [{colored(self._point_clouds_no_robot[..., 3:6].min(), "red")}, {colored(self._point_clouds_no_robot[..., 3:6].max(), "red")}]'
        )
        Logger.log_info(
            f'robot_state ({colored(self._robot_states.dtype, "red")}): {colored(self._robot_states.shape, "red")}, range: [{colored(self._robot_states.min(), "red")}, {colored(self._robot_states.max(), "red")}]'
        )
        Logger.log_info(
            f'action ({colored(self._actions.dtype, "red")}): {colored(self._actions.shape, "red")}, range: [{colored(self._actions.min(), "red")}, {colored(self._actions.max(), "red")}]'
        )
        Logger.log_info(
            f'episode_ends ({colored(self._episode_ends.dtype, "red")}): {colored(self._episode_ends.shape, "red")}, range: [{colored(self._episode_ends.min(), "red")}, {colored(self._episode_ends.max(), "red")}]'
        )
        Logger.print_seperator()

import copy
from typing import Dict

from mappolicy.helper.pytorch import dict_apply
from mappolicy.helper.replay_buffer import ReplayBuffer
from mappolicy.helper.sampler import SequenceSampler, get_val_mask, downsample_mask
from mappolicy.models.diffusion_policy.common.normalizer import LinearNormalizer, get_image_range_normalizer

class ManiSkillDataset_DP(torch.utils.data.Dataset):
    """
    Dataset for Maniskill Benchmark.

    Images range: [0, 255]
    Robot states range: [-1.0, 1.0]
    Raw states range: [-1.0, 1.0]
    Actions range: [-7.0, 13.0]
    """
    
    def __init__(self, data_dir, horizon=1, pad_before=0, pad_after=0, seed=42, val_ratio=0.1, max_train_episodes=None):
        # replay_buffer
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path=data_dir, keys=['images', 'depths', 'point_clouds', 'point_clouds_no_robot', 'robot_states', 'actions'])

        # create train/val split
        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        # sampler
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon, 
            pad_before=pad_before, 
            pad_after=pad_after, 
            episode_mask=train_mask
        )
        
        # save for later use
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon, 
            pad_before=self.pad_before, 
            pad_after=self.pad_after, 
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set
        
    def get_normalizer(self, mode='limits', **kwargs):
        '''
        Now: action, robot_state, image
        TODO: depth, point_cloud
        '''
        data = {
            'action': self.replay_buffer['actions'],
            'robot_state': self.replay_buffer['robot_states']
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer
    
    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        image = np.moveaxis(sample['images'], -1, 1)/255.0
        depth = np.moveaxis(sample['depths'], -1, 1)
        point_cloud = sample['point_clouds']
        point_cloud_no_robot = sample['point_clouds_no_robot']
        robot_state = sample['robot_states']
        action = sample['actions']
        data = {
            'image': image,
            'depth': depth, 
            'point_cloud': point_cloud,
            'point_cloud_no_robot': point_cloud_no_robot,
            'robot_state': robot_state,
            'action': action
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
        
    
def test_dataset(data_dir, mode="dp"):
    """
    测试数据集函数
    :param data_dir: zarr 数据路径
    :param mode: "base" 测试 ManiskillDataset, "dp" 测试 ManiSkillDataset_DP
    """
    Logger.log_info(f"Starting dataset tests in [ {mode.upper()} ] mode...")

    if mode == "base":
        # --- 测试基础版本 ManiskillDataset ---
        ds = ManiskillDataset(data_dir, split="train")
        ds.print_info()
        Logger.log_info(f"ManiskillDataset loaded. Total size (steps): {len(ds)}")

        dataloader = DataLoader(ds, batch_size=4, shuffle=True)
        # ManiskillDataset 返回的是 tuple
        image, depth, pc, pc_no_robot, robot_state, action = next(iter(dataloader))

        Logger.log_info("Checking batch shapes (Base Mode):")
        print(f"  image:       {image.shape}")       # [B, 3, H, W]
        print(f"  depth:       {depth.shape}")       # [B, 1, H, W]
        print(f"  point_cloud: {pc.shape}")          # [B, N, 6]
        print(f"  robot_state: {robot_state.shape}") # [B, D_robot]
        print(f"  action:      {action.shape}")      # [B, D_action]

        assert image.max() <= 255.0 and image.min() >= 0.0
        Logger.log_info("ManiskillDataset test passed.\n")

    elif mode == "dp":
        # --- 测试 Diffusion Policy 版本 ManiSkillDataset_DP ---
        horizon = 16
        ds_dp = ManiSkillDataset_DP(data_dir, horizon=horizon)
        Logger.log_info(f"ManiSkillDataset_DP loaded. Total size (sequences): {len(ds_dp)}")
        print(f"  horizon: {ds_dp.horizon}")

        dataloader = DataLoader(ds_dp, batch_size=2, shuffle=True)
        # ManiSkillDataset_DP 返回的是 dict
        batch = next(iter(dataloader))

        # 验证输出的 keys
        expected_keys = {'image', 'depth', 'point_cloud', 'point_cloud_no_robot', 'robot_state', 'action'}
        Logger.log_info("Checking batch shapes (DP Mode - Sequence):")
        for key in expected_keys:
            assert key in batch, f"Key {key} missing in batch"
            # DP 版本的维度通常是 [B, T, ...]
            print(f"  {key:20s}: {batch[key].shape}")

        # 验证 horizon 维度
        assert batch['action'].shape[1] == horizon, f"Expected horizon {horizon}, got {batch['action'].shape[1]}"

        # 测试 Validation Dataset
        val_ds = ds_dp.get_validation_dataset()
        Logger.log_info(f"Validation dataset created. Size: {len(val_ds)}")

        # 测试 Normalizer
        normalizer = ds_dp.get_normalizer()
        Logger.log_info("Normalizer initialized successfully.")
        
        Logger.log_info("ManiSkillDataset_DP test passed.\n")
    else:
        Logger.log_error(f"Unknown mode: {mode}. Please choose 'base' or 'dp'.")

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    DATA_PATH = f"{os.getenv('MAPPOLICY_ROOT', 'your_path_to_project_root')}/data/maniskill/StackCube-v1_base_camera.zarr"
    
    test_dataset(DATA_PATH, mode="base")
    # test_dataset(DATA_PATH, mode="dp")