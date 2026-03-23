from __future__ import annotations
import argparse
import os
import sys
import pathlib
import json
import re
import time
import shutil
from contextlib import contextmanager
from natsort import natsorted
import numpy as np
import h5py
import zarr
from numcodecs import Blosc
import tqdm
from termcolor import cprint
import multiprocessing as mp

# --- GPU 加速所需库 ---
import torch
import torch.nn.functional as F
try:
    from pytorch3d.ops import sample_farthest_points
except ImportError:
    cprint("❌ 无法导入 pytorch3d！请先安装: pip install 'git+https://github.com/facebookresearch/pytorch3d.git'", "red")
    sys.exit(1)

DEFAULT_COMPRESSOR = Blosc(cname="zstd", clevel=1, shuffle=1)

# ==========================================
# ⏱ 性能分析工具 (Profiler)
# ==========================================
class Profiler:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.stats = {}

    @contextmanager
    def profile(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        yield
        t1 = time.perf_counter()
        self.stats[name] = self.stats.get(name, 0.0) + (t1 - t0)

    def reset(self):
        self.stats = {}

    def print_stats(self, traj_name: str, worker_id: int = 0):
        if not self.enabled: return
        cprint(f"\n📊 [Worker {worker_id}] 性能分析 [{traj_name}]", "cyan")
        total_time = sum(self.stats.values())
        for k, v in self.stats.items():
            pct = (v / total_time) * 100 if total_time > 0 else 0
            cprint(f"  ├─ {k:<25}: {v:.3f} 秒 ({pct:>5.1f}%)", "yellow")
        cprint(f"  └─ {'总计':<25}: {total_time:.3f} 秒\n", "green")


# ==========================================
# 🚀 神级优化：FPS 任务批处理管理器
# ==========================================
class FPSTaskManager:
    def __init__(self, device: torch.device):
        self.device = device
        self.tasks = {}       
        self.results = {}     
        self.counter = 0

    def add(self, pc: np.ndarray, K: int) -> int:
        tid = self.counter
        self.counter += 1
        if K not in self.tasks: self.tasks[K] = []
        self.tasks[K].append((tid, pc))
        return tid

    def execute(self):
        for K, items in self.tasks.items():
            if not items: continue
            task_ids = [it[0] for it in items]
            pcs = [it[1] for it in items]

            B = len(pcs)
            N_real_max = max(pc.shape[0] for pc in pcs)
            N_max = max(N_real_max, K)

            if N_real_max == 0:
                for tid in task_ids:
                    self.results[tid] = np.zeros((K, 6), dtype=np.float32)
                continue

            # 直接在 GPU 上初始化张量，消除 CPU 拷贝
            padded_t = torch.zeros((B, N_max, 6), dtype=torch.float32, device=self.device)
            for i, pc in enumerate(pcs):
                n = pc.shape[0]
                if n > 0:
                    padded_t[i, :n] = torch.from_numpy(pc).to(self.device)
                    if n < N_max:
                        idx = torch.randint(0, n, (N_max - n,), device=self.device)
                        padded_t[i, n:] = padded_t[i, idx]

            xyz = padded_t[..., :3]

            with torch.no_grad():
                _, indices = sample_farthest_points(xyz, K=K)
                idx_expanded = indices.unsqueeze(-1).expand(-1, -1, 6)
                sampled_t = torch.gather(padded_t, 1, idx_expanded)

            sampled_np = sampled_t.cpu().numpy()
            for i, tid in enumerate(task_ids):
                self.results[tid] = sampled_np[i]

        self.tasks.clear()

    def get(self, tid: int) -> np.ndarray:
        return self.results[tid]


# ==========================================
# 辅助函数
# ==========================================
def parse_name_keywords(spec: str | None) -> list[str]:
    if spec is None: return ["robot", "panda", "ground", "desk", "table"]
    return [x.strip().lower() for x in spec.split(",") if x.strip() != ""]

def parse_seg_id_list_with_auto(spec: str | None) -> set[int] | None:
    if spec is None: return None
    spec = spec.strip().lower()
    if spec in ("auto", ""): return set() if spec == "" else None
    return {int(x.strip()) for x in spec.split(",") if x.strip() != ""}

def convert_depth_to_meters(depth_raw: np.ndarray) -> np.ndarray:
    depth = depth_raw.astype(np.float32)
    if np.nanmax(depth) > 20.0: depth = depth / 1000.0
    return depth

def infer_auto_remove_seg_ids(seg: np.ndarray, topk: int = 2) -> set[int]:
    flat = seg.reshape(-1)
    vals, cnts = np.unique(flat, return_counts=True)
    order = np.argsort(cnts)[::-1]
    top_ids = vals[order[:topk]].tolist()
    return {int(v) for v in top_ids}

def resize_images(images: np.ndarray, target_size: int, device: torch.device, is_seg: bool = False) -> np.ndarray:
    if images.shape[1] == target_size and images.shape[2] == target_size: return images
    if images.ndim == 3: t = torch.from_numpy(images).unsqueeze(1).float().to(device)
    else: t = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(device)

    mode = "nearest" if is_seg else "bilinear"
    t_resized = F.interpolate(t, size=(target_size, target_size), mode=mode, align_corners=False if mode=="bilinear" else None)

    if images.ndim == 3: out = t_resized.squeeze(1).cpu().numpy()
    else: out = t_resized.permute(0, 2, 3, 1).cpu().numpy()

    if is_seg: out = out.astype(images.dtype)
    return out

def depth_to_pcd(depth: np.ndarray, rgb: np.ndarray, intrinsic: np.ndarray, grid_y: np.ndarray, grid_x: np.ndarray, return_valid_mask: bool = False):
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    z = depth
    x_world = (grid_x - cx) * z / fx
    y_world = (grid_y - cy) * z / fy
    points = np.stack([x_world, y_world, z], axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3)
    mask = np.isfinite(z).reshape(-1) & (z.reshape(-1) > 0)
    pc = np.concatenate([points, colors], axis=-1)[mask]
    if return_valid_mask: return pc, mask
    return pc

def split_by_foreground_ids(point_cloud: np.ndarray, seg_ids: np.ndarray, foreground_seg_ids: set[int]) -> tuple[np.ndarray, np.ndarray]:
    seg_ids = seg_ids.reshape(-1)
    fg_mask = np.isin(seg_ids, list(foreground_seg_ids))
    fg = point_cloud[fg_mask]
    rest = point_cloud[~fg_mask]
    return fg, rest

# ==========================================
# 数据提取主逻辑
# ==========================================
def extract_traj_data(
    traj_group, profiler: Profiler, device: torch.device, camera_name: str = "base_camera",
    ground_seg_ids: set[int] | None = None, robot_seg_ids: set[int] | None = None, non_foreground_seg_ids: set[int] | None = None,
    pc_foreground_quota: int = 128, pc_no_robot_foreground_quota: int = 256,
    auto_remove_topk: int = 2, num_points: int = 1024, image_size: int = 128,
):
    with profiler.profile("1. 磁盘读取 H5 (I/O)"):
        obs_group = traj_group["obs"]
        sensor_data = obs_group["sensor_data"][camera_name]
        rgb = sensor_data["rgb"][:]        
        depth = sensor_data["depth"][:]    
        seg = sensor_data["segmentation"][:] 
        intrinsic = obs_group["sensor_param"][camera_name]["intrinsic_cv"][:]
        actions = traj_group["actions"][:]  
        T = actions.shape[0]  
        
        rgb = rgb[:T].astype(np.float32)
        depth = convert_depth_to_meters(depth[:T].squeeze(-1))
        seg = seg[:T].squeeze(-1)
        intrinsic = intrinsic[:T]

    with profiler.profile("2. 图像 Resize (GPU)"):
        orig_H, orig_W = rgb.shape[1], rgb.shape[2]
        if orig_H != image_size or orig_W != image_size:
            scale_y = image_size / orig_H
            scale_x = image_size / orig_W
            rgb = resize_images(rgb, image_size, device, is_seg=False)
            depth = resize_images(depth, image_size, device, is_seg=False)
            seg = resize_images(seg, image_size, device, is_seg=True)
            
            intrinsic_resized = intrinsic.copy()
            intrinsic_resized[:, 0, 0] *= scale_x
            intrinsic_resized[:, 0, 2] *= scale_x
            intrinsic_resized[:, 1, 1] *= scale_y
            intrinsic_resized[:, 1, 2] *= scale_y
            intrinsic = intrinsic_resized

    if ground_seg_ids is None:
        seg_probe = seg[: min(10, T)]
        ground_seg_ids = infer_auto_remove_seg_ids(seg_probe, topk=1)
    if robot_seg_ids is None: robot_seg_ids = set()
    if non_foreground_seg_ids is None: non_foreground_seg_ids = set()

    foreground_exclude_ids = set(ground_seg_ids) | set(robot_seg_ids) | set(non_foreground_seg_ids)
    remove_ids_no_robot = set(ground_seg_ids) | set(robot_seg_ids)
    
    manager = FPSTaskManager(device)
    stage1_tasks = []
    grid_y, grid_x = np.meshgrid(np.arange(image_size), np.arange(image_size), indexing='ij')

    for i in range(T):
        with profiler.profile("3. 深度反投影点云"):
            pc_full, valid_mask = depth_to_pcd(depth[i], rgb[i], intrinsic[i], grid_y, grid_x, return_valid_mask=True)
            seg_valid = seg[i].reshape(-1)[valid_mask]

        with profiler.profile("4. 语义 Mask 过滤"):
            keep_pc_mask = ~np.isin(seg_valid, list(ground_seg_ids))
            pc_candidates = pc_full[keep_pc_mask]
            seg_pc = seg_valid[keep_pc_mask]
            pc_fg_ids = set(np.unique(seg_pc).tolist()) - foreground_exclude_ids
            fg_pc, rest_pc = split_by_foreground_ids(pc_candidates, seg_pc, pc_fg_ids)

            keep_nr_mask = ~np.isin(seg_valid, list(remove_ids_no_robot))
            pc_nr_candidates = pc_full[keep_nr_mask]
            seg_nr = seg_valid[keep_nr_mask]
            pc_nr_fg_ids = set(np.unique(seg_nr).tolist()) - foreground_exclude_ids
            fg_nr, rest_nr = split_by_foreground_ids(pc_nr_candidates, seg_nr, pc_nr_fg_ids)

        with profiler.profile("5. 组织 FPS 批处理任务"):
            fg_task1 = manager.add(fg_pc, pc_foreground_quota) if fg_pc.shape[0] > pc_foreground_quota else fg_pc
            remain1 = max(0, num_points - (pc_foreground_quota if fg_pc.shape[0] > pc_foreground_quota else fg_pc.shape[0]))
            bg_task1 = manager.add(rest_pc, remain1) if remain1 > 0 and rest_pc.shape[0] > remain1 else (rest_pc if remain1 > 0 and rest_pc.shape[0] > 0 else np.zeros((remain1, 6), dtype=np.float32))

            fg_task2 = manager.add(fg_nr, pc_no_robot_foreground_quota) if fg_nr.shape[0] > pc_no_robot_foreground_quota else fg_nr
            remain2 = max(0, num_points - (pc_no_robot_foreground_quota if fg_nr.shape[0] > pc_no_robot_foreground_quota else fg_nr.shape[0]))
            bg_task2 = manager.add(rest_nr, remain2) if remain2 > 0 and rest_nr.shape[0] > remain2 else (rest_nr if remain2 > 0 and rest_nr.shape[0] > 0 else np.zeros((remain2, 6), dtype=np.float32))

            stage1_tasks.append({'fg1': fg_task1, 'bg1': bg_task1, 'fg2': fg_task2, 'bg2': bg_task2})

    with profiler.profile("6. 批量 FPS 计算 (GPU 满载)"):
        manager.execute()

    stage2_tasks = []
    with profiler.profile("5. 组织 FPS 批处理任务"):
        for i in range(T):
            t1 = stage1_tasks[i]
            
            fg1 = manager.get(t1['fg1']) if isinstance(t1['fg1'], int) else t1['fg1']
            bg1 = manager.get(t1['bg1']) if isinstance(t1['bg1'], int) else t1['bg1']
            mixed1 = np.concatenate([fg1, bg1], axis=0) if len(bg1) > 0 else fg1
            m1_task = manager.add(mixed1, num_points) if mixed1.shape[0] != num_points else mixed1

            fg2 = manager.get(t1['fg2']) if isinstance(t1['fg2'], int) else t1['fg2']
            bg2 = manager.get(t1['bg2']) if isinstance(t1['bg2'], int) else t1['bg2']
            mixed2 = np.concatenate([fg2, bg2], axis=0) if len(bg2) > 0 else fg2
            m2_task = manager.add(mixed2, num_points) if mixed2.shape[0] != num_points else mixed2

            stage2_tasks.append({'m1': m1_task, 'm2': m2_task})

    with profiler.profile("6. 批量 FPS 计算 (GPU 满载)"):
        manager.execute()

    pcs = []
    pcs_nr = []
    for i in range(T):
        t2 = stage2_tasks[i]
        pcs.append(manager.get(t2['m1']) if isinstance(t2['m1'], int) else t2['m1'])
        pcs_nr.append(manager.get(t2['m2']) if isinstance(t2['m2'], int) else t2['m2'])

    tcp_pose = obs_group["extra"]["tcp_pose"][:T]
    qpos = obs_group["agent"]["qpos"][:T]
    gripper_width = qpos[..., -1:] + qpos[..., -2:-1]
    robot_state = np.concatenate([tcp_pose, gripper_width], axis=-1)

    return {
        "images": rgb,
        "depths": depth,
        "point_clouds": np.stack(pcs),
        "point_clouds_no_robot": np.stack(pcs_nr),
        "robot_states": robot_state,
        "actions": actions,
    }


# ==========================================
# Worker 进程 (独立处理 H5 分片)
# ==========================================
def worker_process(args):
    h5_path, zarr_save_path, traj_keys, worker_id, kwargs = args
    
    # 每个进程内部初始化设备，避免 PyTorch CUDA 死锁
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    profiler = Profiler(enabled=kwargs.get("enable_profile", False))
    
    store = zarr.DirectoryStore(str(zarr_save_path))
    zarr_root = zarr.group(store=store, overwrite=True)
    zarr_data = zarr_root.create_group("data")
    
    episode_ends = []
    total_step = 0
    datasets = {} 
    
    with h5py.File(h5_path, "r", libver='latest', swmr=True) as f:
        traj_iter = tqdm.tqdm(traj_keys, desc=f"Worker-{worker_id}", position=worker_id, leave=False, disable=kwargs.get("quiet", False))
        
        for traj_key in traj_iter:
            profiler.reset()
            data = extract_traj_data(
                f[traj_key], profiler=profiler, device=device, camera_name=kwargs["camera_name"],
                ground_seg_ids=kwargs["ground_seg_ids"], robot_seg_ids=kwargs["robot_seg_ids"], non_foreground_seg_ids=kwargs["non_foreground_seg_ids"],
                pc_foreground_quota=kwargs["pc_foreground_quota"], pc_no_robot_foreground_quota=kwargs["pc_no_robot_foreground_quota"],
                auto_remove_topk=kwargs["auto_remove_topk"], num_points=kwargs["num_points"], image_size=kwargs["image_size"],      
            )
            
            with profiler.profile("7. Zarr 落盘写入 (I/O)"):
                for k, val in data.items():
                    if k not in datasets:
                        datasets[k] = zarr_data.create_dataset(
                            name=k, shape=(0, *val.shape[1:]), chunks=(32, *val.shape[1:]),
                            compressor=DEFAULT_COMPRESSOR, dtype=val.dtype
                        )
                    datasets[k].append(val)
                total_step += len(data["actions"])
                episode_ends.append(total_step)
            
            profiler.print_stats(traj_key, worker_id)

    zarr_root.create_group("meta").create_dataset("episode_ends", data=np.array(episode_ends))
    return zarr_save_path


# ==========================================
# 主协调器 (合并分块)
# ==========================================
def merge_temp_zarrs(temp_paths: list[str], final_path: str):
    cprint("\n🔄 开始合并分块 Zarr 文件...", "blue")
    store = zarr.DirectoryStore(str(final_path))
    final_root = zarr.group(store=store, overwrite=True)
    final_data = final_root.create_group("data")
    
    episode_ends = []
    current_step = 0
    datasets = {}

    for p in tqdm.tqdm(temp_paths, desc="合并进度"):
        part_root = zarr.open(p, mode='r')
        part_data = part_root['data']
        part_ends = part_root['meta']['episode_ends'][:]
        
        for k in part_data.keys():
            arr = part_data[k]
            if k not in datasets:
                datasets[k] = final_data.create_dataset(
                    name=k, shape=(0, *arr.shape[1:]), chunks=arr.chunks,
                    compressor=DEFAULT_COMPRESSOR, dtype=arr.dtype
                )
            
            # 流式安全 Append，防止合并时内存爆炸
            chunk_size = 1000
            for i in range(0, arr.shape[0], chunk_size):
                datasets[k].append(arr[i:i+chunk_size])
        
        # 调整 episode_ends 的累加偏移量
        if len(part_ends) > 0:
            adjusted_ends = part_ends + current_step
            episode_ends.extend(adjusted_ends.tolist())
            current_step = adjusted_ends[-1]

    final_root.create_group("meta").create_dataset("episode_ends", data=np.array(episode_ends))
    cprint(f"✅ 合并完成！最终文件：{final_path}", "green")
    
    # 清理临时文件
    for p in temp_paths:
        shutil.rmtree(p)


def main(args):
    # 必须设置 spawn 模式以确保多进程安全调用 CUDA
    mp.set_start_method('spawn', force=True)
    
    pathlib.Path(args.zarr_save_dir).mkdir(parents=True, exist_ok=True)
    cprint(f"📌 目标Zarr保存目录：{args.zarr_save_dir}", "green")
    cprint(f"⚙️  配置: 并行进程 {args.num_workers} | 采样点 {args.num_points} | 图像 {args.image_size}x{args.image_size}", "cyan")

    # 预解析参数
    ground_seg_ids = parse_seg_id_list_with_auto(args.ground_seg_ids)
    robot_seg_ids = parse_seg_id_list_with_auto(args.robot_seg_ids)
    non_foreground_seg_ids = parse_seg_id_list_with_auto(args.non_foreground_seg_ids)

    # 提取轨迹总数
    with h5py.File(args.input_path, "r", libver='latest', swmr=True) as f:
        traj_keys = natsorted([k for k in f.keys() if k.startswith("traj_")])
        if args.max_episode is not None:
            traj_keys = traj_keys[:args.max_episode]

    if len(traj_keys) == 0:
        cprint("❌ 找不到任何轨迹！", "red")
        return

    # 分割任务块
    num_workers = min(args.num_workers, len(traj_keys))
    chunk_size = int(np.ceil(len(traj_keys) / num_workers))
    
    h5_path_stem = pathlib.Path(args.input_path).stem
    final_zarr_path = pathlib.Path(args.zarr_save_dir) / f"{h5_path_stem}_{args.camera_name}.zarr"
    
    kwargs = {
        "camera_name": args.camera_name,
        "ground_seg_ids": ground_seg_ids,
        "robot_seg_ids": robot_seg_ids,
        "non_foreground_seg_ids": non_foreground_seg_ids,
        "pc_foreground_quota": args.pc_foreground_quota,
        "pc_no_robot_foreground_quota": args.pc_no_robot_foreground_quota,
        "auto_remove_topk": args.auto_remove_topk,
        "num_points": args.num_points,
        "image_size": args.image_size,
        "enable_profile": args.profile,
        "quiet": args.quiet
    }

    worker_args = []
    temp_paths = []
    for i in range(num_workers):
        sub_keys = traj_keys[i*chunk_size : (i+1)*chunk_size]
        if not sub_keys: continue
        temp_dir = pathlib.Path(args.zarr_save_dir) / f"_temp_part_{i}.zarr"
        temp_paths.append(str(temp_dir))
        worker_args.append((args.input_path, str(temp_dir), sub_keys, i, kwargs))

    # 并行执行
    cprint(f"🚀 开始分配任务至 {num_workers} 个进程...", "magenta")
    with mp.Pool(processes=num_workers) as pool:
        pool.map(worker_process, worker_args)
    
    # 自动合并所有临时 Zarr 文件
    merge_temp_zarrs(temp_paths, final_zarr_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--zarr-save-dir", type=str, default=str(pathlib.Path(__file__).resolve().parent / "data_new" / "maniskill_zarr"))
    parser.add_argument("--camera-name", type=str, default="base_camera")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max-episode", type=int, default=None)
    parser.add_argument("--ground-seg-ids", type=str, default="auto")
    parser.add_argument("--robot-seg-ids", type=str, default="auto")
    parser.add_argument("--non-foreground-seg-ids", type=str, default="auto")
    parser.add_argument("--auto-remove-topk", type=int, default=2)
    parser.add_argument("--pc-foreground-quota", type=int, default=128)
    parser.add_argument("--pc-no-robot-foreground-quota", type=int, default=256)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--profile", action="store_true")
    
    # 🚀 新增并行控制参数
    parser.add_argument("--num-workers", type=int, default=4, help="启动多少个进程同时转换 (推荐设置为云服务器 CPU 核心数一半)")
    
    args = parser.parse_args()
    main(args)