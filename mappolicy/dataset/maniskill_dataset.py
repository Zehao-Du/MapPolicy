import torch
import zarr
from termcolor import colored
import numpy as np
import os
import copy
from typing import Dict
from torch.utils.data import DataLoader

from mappolicy.helper.logger import Logger
from mappolicy.helper.pytorch import dict_apply
from mappolicy.helper.replay_buffer import ReplayBuffer
from mappolicy.helper.sampler import SequenceSampler, get_val_mask, downsample_mask
from mappolicy.models.diffusion_policy.common.normalizer import LinearNormalizer, get_image_range_normalizer

# =========================================================================
# 1. Base Dataset (Optimized for Large Data / Lazy Loading)
# =========================================================================
class ManiskillDataset(torch.utils.data.Dataset):
    """
    Dataset for Maniskill Benchmark. (Optimized for 25GB+ Zarr datasets)

    Images range: [0, 255] -> Tensor: [0.0, 255.0]
    Robot states range: [-1.0, 1.0]
    Raw states range: [-1.0, 1.0]
    Actions range: [-7.0, 13.0]
    """

    SPLIT_SIZE = {"train": 900, "validation": 100, "custom": None}

    def __init__(self, data_dir, split: str = None, custom_split_size: int = None):
        # 1. 打开 Zarr 根组，保持引用，而不立即将数据读入内存
        self.zarr_root = zarr.open_group(data_dir, mode="r")
        
        # metadata 通常非常小，直接加载到内存没问题
        self._episode_ends = self.zarr_root["meta"]["episode_ends"][:]

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

        # 2. 记录偏移量和长度，不再进行切片加载数据以防止 OOM！
        self.begin_index = int(begin_index)
        self.end_index = int(end_index)
        self._dataset_size = self.end_index - self.begin_index

        # 3. 仅保存 Zarr Array 的代理对象 (Lazy Mode)
        self._images = self.zarr_root["data"]["images"]
        self._depth = self.zarr_root["data"]["depths"]
        self._point_clouds = self.zarr_root["data"]["point_clouds"]
        self._point_clouds_no_robot = self.zarr_root["data"]["point_clouds_no_robot"]
        self._robot_states = self.zarr_root["data"]["robot_states"]
        self._actions = self.zarr_root["data"]["actions"]

        # 确保数据维度对齐
        assert self._images.shape[3] == 3 
        assert self._images.shape[0] == self._robot_states.shape[0] == self._actions.shape[0]

        train_steps = int(self._episode_ends[self.SPLIT_SIZE["train"] - 1])
        val_steps = int(
            self._episode_ends[self.SPLIT_SIZE["train"] + self.SPLIT_SIZE["validation"] - 1]
            - train_steps
        )
        Logger.log_info(f"train steps: {train_steps}, validation steps: {val_steps}")

    def __getitem__(self, idx):
        # 计算在整个 Zarr 数据集中的真实全局索引
        actual_idx = self.begin_index + idx

        # 4. 按需读取磁盘中的单条数据
        image_np = self._images[actual_idx].transpose(2, 0, 1) # (H, W, 3) -> (3, H, W)
        image = torch.from_numpy(image_np).float()

        depth = torch.from_numpy(self._depth[actual_idx]).float()
        depth = depth.unsqueeze(0)  # (H, W) -> (1, H, W)
        depth_max = depth.max()
        # 防 NaN 保护：当画面没有深度信息（全黑）时防止除零
        depth = depth / (depth_max if depth_max > 0 else 1.0) 
        
        point_cloud = torch.from_numpy(self._point_clouds[actual_idx]).float()
        point_cloud_no_robot = torch.from_numpy(self._point_clouds_no_robot[actual_idx]).float()
        robot_state = torch.from_numpy(self._robot_states[actual_idx]).float()
        action = torch.from_numpy(self._actions[actual_idx]).float()
        
        return image, depth, point_cloud, point_cloud_no_robot, robot_state, action

    def __len__(self):
        return self._dataset_size

    def print_info(self):
        Logger.log_info(f"Maniskill Dataset Info:")
        
        # 5. 安全警告：仅提取前 100 条数据作为样本估算 range，防止扫描全盘 25GB 导致程序假死
        sample_size = min(100, self._dataset_size)
        sample_pc = self._point_clouds[self.begin_index : self.begin_index + sample_size]
        sample_pc_nr = self._point_clouds_no_robot[self.begin_index : self.begin_index + sample_size]
        sample_rs = self._robot_states[self.begin_index : self.begin_index + sample_size]
        sample_act = self._actions[self.begin_index : self.begin_index + sample_size]

        Logger.log_info(f"Note: min/max ranges are estimated from the first {sample_size} samples for performance.")

        Logger.log_info(
            f'point_cloud ({colored(self._point_clouds.dtype, "red")}): {colored(self._point_clouds.shape, "red")}, '
            f'xyz_range: [{colored(sample_pc[..., 0:3].min(), "red")}, {colored(sample_pc[..., 0:3].max(), "red")}], '
            f'rgb_range: [{colored(sample_pc[..., 3:6].min(), "red")}, {colored(sample_pc[..., 3:6].max(), "red")}]'
        )
        Logger.log_info(
            f'point_cloud_no_robot ({colored(self._point_clouds_no_robot.dtype, "red")}): {colored(self._point_clouds_no_robot.shape, "red")}, '
            f'xyz_range: [{colored(sample_pc_nr[..., 0:3].min(), "red")}, {colored(sample_pc_nr[..., 0:3].max(), "red")}], '
            f'rgb_range: [{colored(sample_pc_nr[..., 3:6].min(), "red")}, {colored(sample_pc_nr[..., 3:6].max(), "red")}]'
        )
        Logger.log_info(
            f'robot_state ({colored(self._robot_states.dtype, "red")}): {colored(self._robot_states.shape, "red")}, range: [{colored(sample_rs.min(), "red")}, {colored(sample_rs.max(), "red")}]'
        )
        Logger.log_info(
            f'action ({colored(self._actions.dtype, "red")}): {colored(self._actions.shape, "red")}, range: [{colored(sample_act.min(), "red")}, {colored(sample_act.max(), "red")}]'
        )
        Logger.log_info(
            f'episode_ends ({colored(self._episode_ends.dtype, "red")}): {colored(self._episode_ends.shape, "red")}, range: [{colored(self._episode_ends.min(), "red")}, {colored(self._episode_ends.max(), "red")}]'
        )
        Logger.print_seperator()


# =========================================================================
# 2. Diffusion Policy Dataset
# =========================================================================
class ManiSkillDataset_DP(torch.utils.data.Dataset):
    """
    Dataset for Maniskill Benchmark Diffusion Policy Training.
    """
    
    def __init__(self, data_dir, horizon=1, pad_before=0, pad_after=0, seed=42, val_ratio=0.1, max_train_episodes=None):
        # 注意: 确保你的 ReplayBuffer 内部实现没有把整个 Zarr 文件加载进内存，
        # 如果 ReplayBuffer 支持 lazy loading，这里就不会导致 OOM。
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path=data_dir, 
            keys=['images', 'depths', 'point_clouds', 'point_clouds_no_robot', 'robot_states', 'actions']
        )

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
        # 仅使用低维数据进行 fit，避免对图片等大数据进行计算引起 OOM
        data = {
            'action': self.replay_buffer['actions'][:],
            'robot_state': self.replay_buffer['robot_states'][:]
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer
    
    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # image 原始 shape: (T, H, W, 3) -> moveaxis后: (T, 3, H, W)
        image = np.moveaxis(sample['images'], -1, 1) / 255.0
        
        # [修复 Bug]: depth 在 zarr 中通常是 (H, W)，sampler 拿出来是 (T, H, W)
        # 用 np.moveaxis(..., -1, 1) 会导致变成 (T, W, H)，这是错误的！应该用 expand_dims 变成 (T, 1, H, W)
        depth = np.expand_dims(sample['depths'], axis=1)
        
        # 增加和 Base 一致的安全深度归一化防 NaN
        depth_max = depth.max()
        depth = depth / (depth_max if depth_max > 0 else 1.0)

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
        torch_data = dict_apply(data, lambda x: torch.from_numpy(x).float())
        return torch_data


# =========================================================================
# 3. Test Script
# =========================================================================
def test_dataset(data_dir, mode="base"):
    """
    测试数据集函数
    :param data_dir: zarr 数据路径
    :param mode: "base" 测试 ManiskillDataset, "dp" 测试 ManiSkillDataset_DP
    """
    Logger.log_info(f"Starting dataset tests in [ {mode.upper()} ] mode...")

    if mode == "base":
        ds = ManiskillDataset(data_dir, split="train")
        ds.print_info()
        Logger.log_info(f"ManiskillDataset loaded. Total size (steps): {len(ds)}")

        dataloader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
        image, depth, pc, pc_no_robot, robot_state, action = next(iter(dataloader))

        Logger.log_info("Checking batch shapes (Base Mode):")
        print(f"  image:       {image.shape}")       # Expected: [B, 3, H, W]
        print(f"  depth:       {depth.shape}")       # Expected: [B, 1, H, W]
        print(f"  point_cloud: {pc.shape}")          # Expected: [B, N, 6]
        print(f"  robot_state: {robot_state.shape}") # Expected: [B, D_robot]
        print(f"  action:      {action.shape}")      # Expected: [B, D_action]

        assert image.max() <= 255.0 and image.min() >= 0.0
        # 判断 Depth 是否不含 NaN 且正确归一化
        assert not torch.isnan(depth).any(), "Depth contains NaN values!"
        Logger.log_info("ManiskillDataset test passed.\n")

    elif mode == "dp":
        horizon = 16
        ds_dp = ManiSkillDataset_DP(data_dir, horizon=horizon)
        Logger.log_info(f"ManiSkillDataset_DP loaded. Total sequences: {len(ds_dp)}")
        print(f"  horizon: {ds_dp.horizon}")

        dataloader = DataLoader(ds_dp, batch_size=2, shuffle=True, num_workers=0)
        batch = next(iter(dataloader))

        expected_keys = {'image', 'depth', 'point_cloud', 'point_cloud_no_robot', 'robot_state', 'action'}
        Logger.log_info("Checking batch shapes (DP Mode - Sequence):")
        for key in expected_keys:
            assert key in batch, f"Key {key} missing in batch"
            print(f"  {key:20s}: {batch[key].shape}")

        # 验证 horizon 维度 (B, T, ...)
        assert batch['action'].shape[1] == horizon, f"Expected horizon {horizon}, got {batch['action'].shape[1]}"
        # 验证修复后的 depth 维度
        assert batch['depth'].dim() == 5, f"Expected depth to be 5D [B, T, C, H, W], got {batch['depth'].shape}"
        assert batch['depth'].shape[2] == 1, "Depth channel dimension should be 1"

        # 测试 Validation Dataset 和 Normalizer
        val_ds = ds_dp.get_validation_dataset()
        Logger.log_info(f"Validation dataset created. Size: {len(val_ds)}")

        normalizer = ds_dp.get_normalizer()
        Logger.log_info("Normalizer initialized successfully.")
        
        Logger.log_info("ManiSkillDataset_DP test passed.\n")
    else:
        Logger.log_error(f"Unknown mode: {mode}. Please choose 'base' or 'dp'.")


if __name__ == "__main__":
    # 配置正确的 Zarr 数据路径
    DATA_PATH = f"{os.getenv('MAPPOLICY_ROOT', '.')}/data/maniskill/StackCube-v1_base_camera.zarr"
    
    if os.path.exists(DATA_PATH):
        test_dataset(DATA_PATH, mode="base")
        # print("="*60)
        # test_dataset(DATA_PATH, mode="dp")
    else:
        print(f"Dataset path not found: {DATA_PATH}\nPlease verify your MAPPOLICY_ROOT or dataset path.")