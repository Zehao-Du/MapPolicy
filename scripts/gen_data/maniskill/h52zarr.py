from __future__ import annotations
import argparse
import os
import sys
import pathlib
import json
import re
from natsort import natsorted
import numpy as np
import h5py
import zarr
from numcodecs import Blosc
import tqdm
from termcolor import cprint
    
from mappolicy.helper.graphics import PointCloud

DEFAULT_COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=1)
DEFAULT_POINT_CLOUD_NPOINTS = 1024
DEFAULT_ACTION_DIM = 4


def parse_name_keywords(spec: str | None) -> list[str]:
    if spec is None:
        return ["robot", "panda", "ground", "desk", "table"]
    return [x.strip().lower() for x in spec.split(",") if x.strip() != ""]


def parse_seg_id_list_with_auto(spec: str | None) -> set[int] | None:
    """
    解析分割ID配置字符串：
    - "auto" / None -> None（表示自动推断）
    - "" -> 空集合
    - "1,2,3" -> {1,2,3}
    """
    if spec is None:
        return None
    spec = spec.strip().lower()
    if spec == "auto":
        return None
    if spec == "":
        return set()
    return {int(x.strip()) for x in spec.split(",") if x.strip() != ""}

def parse_seg_id_list(spec: str | None) -> set[int] | None:
    """
    解析分割ID配置字符串：
    - "auto" / None -> None（表示自动推断）
    - "" -> 空集合
    - "1,2,3" -> {1,2,3}
    """
    if spec is None:
        return None
    spec = spec.strip().lower()
    if spec == "auto":
        return None
    if spec == "":
        return set()
    return {int(x.strip()) for x in spec.split(",") if x.strip() != ""}


def convert_depth_to_meters(depth_raw: np.ndarray) -> np.ndarray:
    """
    ManiSkill 回放数据中 depth 可能是米或毫米。
    经验规则：若最大值 > 20，通常是毫米，转换到米。
    """
    depth = depth_raw.astype(np.float32)
    if np.nanmax(depth) > 20.0:
        depth = depth / 1000.0
    return depth


def infer_auto_remove_seg_ids(seg: np.ndarray, topk: int = 2) -> set[int]:
    """
    自动推断需要移除的 segmentation id（用于 point_clouds_no_robot）。
    默认移除像素占比最高的 top-k 类，通常对应桌面/地面与机器人。
    """
    flat = seg.reshape(-1)
    vals, cnts = np.unique(flat, return_counts=True)
    order = np.argsort(cnts)[::-1]
    top_ids = vals[order[:topk]].tolist()
    return {int(v) for v in top_ids}


def _extract_entity_name(entity) -> str:
    """
    从 ManiSkill segmentation_id_map 的 value 中提取名称。
    常见 repr: <panda_link0: struct of type ...>
    """
    name = getattr(entity, "name", None)
    if isinstance(name, str) and name:
        return name
    text = str(entity)
    m = re.match(r"^<([^:>]+)", text)
    if m:
        return m.group(1)
    return text


def infer_seg_ids_from_name_keywords(
    h5_path: pathlib.Path,
    name_keywords: list[str],
) -> set[int] | None:
    """
    根据 ManiSkill segmentation_id_map 的实体名字来确定 seg id。
    读取同名 json 中的 env_info（env_id / control_mode），然后创建 env 查询映射。
    """
    json_path = h5_path.with_suffix(".json")
    if not json_path.exists():
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    env_info = meta.get("env_info", {})
    env_id = env_info.get("env_id", None)
    env_kwargs = env_info.get("env_kwargs", {})
    control_mode = env_kwargs.get("control_mode", "pd_ee_delta_pose")
    if env_id is None:
        return None

    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
    except Exception:
        return None

    env = gym.make(
        env_id,
        obs_mode="rgb+depth+segmentation",
        control_mode=control_mode,
        render_mode="rgb_array",
    )
    try:
        env.reset(seed=0)
        seg_id_map = getattr(env.unwrapped, "segmentation_id_map", None)
        if not isinstance(seg_id_map, dict):
            return None

        remove_ids: set[int] = set()
        for sid, entity in seg_id_map.items():
            entity_name = _extract_entity_name(entity).lower()
            if any(kw in entity_name for kw in name_keywords):
                remove_ids.add(int(sid))
        return remove_ids
    finally:
        env.close()


def split_by_foreground_ids(
    point_cloud: np.ndarray,
    seg_ids: np.ndarray,
    foreground_seg_ids: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    seg_ids = seg_ids.reshape(-1)
    if seg_ids.shape[0] != point_cloud.shape[0]:
        raise ValueError(
            f"seg_ids长度({seg_ids.shape[0]})与point_cloud点数({point_cloud.shape[0]})不一致，"
            "请检查 depth 有效点掩码与 segmentation 对齐。"
        )
    fg_mask = np.isin(seg_ids, list(foreground_seg_ids))
    fg = point_cloud[fg_mask]
    rest = point_cloud[~fg_mask]
    return fg, rest


def split_foreground_background(
    point_cloud: np.ndarray,
    seg_ids: np.ndarray,
    background_seg_ids: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    seg_ids = seg_ids.reshape(-1)
    if seg_ids.shape[0] != point_cloud.shape[0]:
        raise ValueError(
            f"seg_ids长度({seg_ids.shape[0]})与point_cloud点数({point_cloud.shape[0]})不一致，"
            "请检查 depth 有效点掩码与 segmentation 对齐。"
        )
    bg_mask = np.isin(seg_ids, list(background_seg_ids))
    bg = point_cloud[bg_mask]
    fg = point_cloud[~bg_mask]
    return fg, bg


def sample_with_foreground_quota(
    foreground_pc: np.ndarray,
    background_pc: np.ndarray,
    total_num_points: int,
    foreground_quota: int,
) -> np.ndarray:
    """
    前景/背景混合采样：
    - 若前景点数 > foreground_quota，则前景采样到 foreground_quota
    - 若前景点数 <= foreground_quota，则前景全保留（不降采样）
    - 背景补足到 total_num_points
    """
    if foreground_pc.shape[0] == 0 and background_pc.shape[0] == 0:
        return np.zeros((total_num_points, 6), dtype=np.float32)

    if foreground_pc.shape[0] > foreground_quota:
        fg_keep = sample_point_cloud(foreground_pc, foreground_quota)
    else:
        fg_keep = foreground_pc.astype(np.float32)

    remain = max(0, total_num_points - fg_keep.shape[0])
    if remain > 0:
        if background_pc.shape[0] > 0:
            bg_keep = sample_point_cloud(background_pc, remain)
        else:
            bg_keep = np.zeros((remain, 6), dtype=np.float32)
        mixed = np.concatenate([fg_keep, bg_keep], axis=0)
    else:
        mixed = fg_keep

    if mixed.shape[0] != total_num_points:
        mixed = sample_point_cloud(mixed, total_num_points)
    return mixed.astype(np.float32)


def depth_to_pcd(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsic: np.ndarray,
    return_valid_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    将深度图转换为点云 (N, 6)
    depth: (H, W), float32
    rgb: (H, W, 3), float32 (0-255)
    intrinsic: (3, 3) 相机内参
    """
    H, W = depth.shape
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    # 生成网格坐标
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # 反投影：x = (u - cx) * z / fx, y = (v - cy) * z / fy
    z = depth
    x_world = (x - cx) * z / fx
    y_world = (y - cy) * z / fy
    
    points = np.stack([x_world, y_world, z], axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3)
    
    # 过滤掉无效深度点
    mask = np.isfinite(z).reshape(-1) & (z.reshape(-1) > 0)
    pc = np.concatenate([points, colors], axis=-1)[mask]
    if return_valid_mask:
        return pc, mask
    return pc


def sample_point_cloud(point_cloud: np.ndarray, num_points: int = 1024) -> np.ndarray:
    if point_cloud.shape[0] == 0:
        return np.zeros((num_points, 6), dtype=np.float32)
    return PointCloud.point_cloud_sampling(
        point_cloud, num_points, "fps"
    ).astype(np.float32)

def extract_traj_data(
    traj_group,
    camera_name: str = "base_camera",
    ground_seg_ids: set[int] | None = None,
    robot_seg_ids: set[int] | None = None,
    non_foreground_seg_ids: set[int] | None = None,
    pc_foreground_quota: int = 128,
    pc_no_robot_foreground_quota: int = 256,
    auto_remove_topk: int = 2,
):
    obs_group = traj_group["obs"]
    
    # 1. 修正传感器数据路径
    # 路径：traj_x/obs/sensor_data/base_camera/
    sensor_data = obs_group["sensor_data"][camera_name]
    rgb = sensor_data["rgb"][:]        # (85, 128, 128, 3)
    depth = sensor_data["depth"][:]    # (85, 128, 128, 1) -> 需要 squeeze
    seg = sensor_data["segmentation"][:] # (85, 128, 128, 1) -> 需要 squeeze
    
    # 修正内参路径：traj_x/obs/sensor_param/base_camera/intrinsic_cv
    # 注意：intrinsic_cv 是随时间变化的 (85, 3, 3)
    intrinsic = obs_group["sensor_param"][camera_name]["intrinsic_cv"][:]
    
    actions = traj_group["actions"][:]  # (84, 7)
    T = actions.shape[0]  # 84
    
    # 预处理：切片到 T 步
    rgb = rgb[:T].astype(np.float32)
    depth = convert_depth_to_meters(depth[:T].squeeze(-1))
    seg = seg[:T].squeeze(-1)
    intrinsic = intrinsic[:T]

    if ground_seg_ids is None:
        # 回退策略：自动把像素占比最高类作为地面/桌面
        seg_probe = seg[: min(10, T)]
        ground_seg_ids = infer_auto_remove_seg_ids(seg_probe, topk=1)
    if robot_seg_ids is None:
        robot_seg_ids = set()
    if non_foreground_seg_ids is None:
        non_foreground_seg_ids = set()

    foreground_exclude_ids = set(ground_seg_ids) | set(robot_seg_ids) | set(non_foreground_seg_ids)
    remove_ids_no_robot = set(ground_seg_ids) | set(robot_seg_ids)
    
    pcs = []
    pcs_nr = []
    
    for i in range(T):
        # 深度转点云 (使用该时间步的内参)
        pc_full, valid_mask = depth_to_pcd(depth[i], rgb[i], intrinsic[i], return_valid_mask=True)
        seg_valid = seg[i].reshape(-1)[valid_mask]

        # 1) point_clouds: 过滤地面，保留其余（机器人/桌子/物体）
        keep_pc_mask = ~np.isin(seg_valid, list(ground_seg_ids))
        pc_candidates = pc_full[keep_pc_mask]
        seg_pc = seg_valid[keep_pc_mask]
        pc_foreground_ids = set(np.unique(seg_pc).tolist()) - foreground_exclude_ids
        fg_pc, rest_pc = split_by_foreground_ids(pc_candidates, seg_pc, pc_foreground_ids)
        pc = sample_with_foreground_quota(
            fg_pc,
            rest_pc,
            DEFAULT_POINT_CLOUD_NPOINTS,
            foreground_quota=pc_foreground_quota,
        )

        # 2) point_clouds_no_robot: 过滤地面+机器人，保留其余
        keep_nr_mask = ~np.isin(seg_valid, list(remove_ids_no_robot))
        pc_nr_candidates = pc_full[keep_nr_mask]
        seg_nr = seg_valid[keep_nr_mask]
        pc_nr_foreground_ids = set(np.unique(seg_nr).tolist()) - foreground_exclude_ids
        fg_nr, rest_nr = split_by_foreground_ids(
            pc_nr_candidates, seg_nr, pc_nr_foreground_ids
        )
        pc_nr = sample_with_foreground_quota(
            fg_nr,
            rest_nr,
            DEFAULT_POINT_CLOUD_NPOINTS,
            foreground_quota=pc_no_robot_foreground_quota,
        )
        
        pcs.append(pc)
        pcs_nr.append(pc_nr)
    
    # 机器人状态提取
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

def h52zarr_single(
    h5_path,
    zarr_save_dir,
    camera_name="base_camera",
    quiet=False,
    max_episode=None,
    ground_seg_ids: set[int] | None = None,
    robot_seg_ids: set[int] | None = None,
    non_foreground_seg_ids: set[int] | None = None,
    pc_foreground_quota: int = 128,
    pc_no_robot_foreground_quota: int = 256,
    auto_remove_topk: int = 2,
):
    h5_path = pathlib.Path(h5_path)
    zarr_path = pathlib.Path(zarr_save_dir) / f"{h5_path.stem}_{camera_name}.zarr"
    
    all_data = {"images": [], "depths": [], "point_clouds": [], "point_clouds_no_robot": [], "robot_states": [], "actions": [], "episode_ends": []}
    
    with h5py.File(h5_path, "r") as f:
        traj_keys = natsorted([k for k in f.keys() if k.startswith("traj_")])
        if max_episode is not None:
            traj_keys = traj_keys[:max_episode]
        total_step = 0
        
        # 使用 tqdm 包装轨迹遍历
        # 如果 quiet 为 True，则不显示进度条
        disable_tqdm = quiet
        traj_iter = tqdm.tqdm(traj_keys, desc="转换轨迹", disable=disable_tqdm)
        
        for traj_key in traj_iter:
            # 在进度条上方显示当前处理的轨迹名称
            if not quiet:
                traj_iter.set_postfix(traj=traj_key)
            
            data = extract_traj_data(
                f[traj_key],
                camera_name,
                ground_seg_ids=ground_seg_ids,
                robot_seg_ids=robot_seg_ids,
                non_foreground_seg_ids=non_foreground_seg_ids,
                pc_foreground_quota=pc_foreground_quota,
                pc_no_robot_foreground_quota=pc_no_robot_foreground_quota,
                auto_remove_topk=auto_remove_topk,
            )
            for k in data:
                all_data[k].append(data[k])
            total_step += len(data["actions"])
            all_data["episode_ends"].append(total_step)

    # 合并并保存
    cprint(f"📥 正在写入 Zarr 文件: {zarr_path}", "blue")
    zarr_root = zarr.group(zarr_path, overwrite=True)
    zarr_data = zarr_root.create_group("data", overwrite=True)
    
    # 写入数据
    for k in ["images", "depths", "point_clouds", "point_clouds_no_robot", "robot_states", "actions"]:
        arr = np.concatenate(all_data[k], axis=0)
        
        zarr_data.create_dataset(
            name=k,
            data=arr,
            chunks=(100, *arr.shape[1:]),
            compressor=DEFAULT_COMPRESSOR, # 这里必须是单数 compressor
            dtype=arr.dtype,
            overwrite=True
        )
    
    zarr_root.create_group("meta").create_dataset("episode_ends", data=np.array(all_data["episode_ends"]))
    cprint(f"✅ 转换完成，总步数: {total_step}", "green")
    

def main(args):
    pathlib.Path(args.zarr_save_dir).mkdir(parents=True, exist_ok=True)
    cprint(f"📌 目标Zarr保存目录：{args.zarr_save_dir}", "green")

    # 1) ground seg ids
    ground_seg_ids = parse_seg_id_list_with_auto(args.ground_seg_ids)
    if ground_seg_ids is None:
        ground_keywords = parse_name_keywords(args.ground_name_keywords)
        ground_seg_ids = infer_seg_ids_from_name_keywords(
            pathlib.Path(args.input_path),
            ground_keywords,
        )
    if ground_seg_ids is not None:
        cprint(f"🧠 ground seg ids: {sorted(ground_seg_ids)}", "yellow")
    else:
        cprint("⚠️ 未能解析 ground seg ids，将在轨迹内自动估计", "yellow")

    # 2) robot seg ids
    robot_seg_ids = parse_seg_id_list_with_auto(args.robot_seg_ids)
    if robot_seg_ids is None:
        robot_keywords = parse_name_keywords(args.robot_name_keywords)
        robot_seg_ids = infer_seg_ids_from_name_keywords(
            pathlib.Path(args.input_path),
            robot_keywords,
        )
    if robot_seg_ids is not None:
        cprint(f"🧠 robot seg ids: {sorted(robot_seg_ids)}", "yellow")
    else:
        cprint("⚠️ 未能解析 robot seg ids，默认空集合", "yellow")

    # 3) non-foreground seg ids（如 desk/table）
    non_foreground_seg_ids = parse_seg_id_list_with_auto(args.non_foreground_seg_ids)
    if non_foreground_seg_ids is None:
        non_fg_keywords = parse_name_keywords(args.non_foreground_name_keywords)
        non_foreground_seg_ids = infer_seg_ids_from_name_keywords(
            pathlib.Path(args.input_path),
            non_fg_keywords,
        )
    if non_foreground_seg_ids is not None:
        cprint(f"🧠 non-foreground seg ids: {sorted(non_foreground_seg_ids)}", "yellow")
    else:
        cprint("⚠️ 未能解析 non-foreground seg ids，默认空集合", "yellow")

    if os.path.isfile(args.input_path):
        # 传入 args.quiet 参数
        h52zarr_single(
            args.input_path,
            args.zarr_save_dir,
            args.camera_name,
            args.quiet,
            args.max_episode,
            ground_seg_ids=ground_seg_ids,
            robot_seg_ids=robot_seg_ids,
            non_foreground_seg_ids=non_foreground_seg_ids,
            pc_foreground_quota=args.pc_foreground_quota,
            pc_no_robot_foreground_quota=args.pc_no_robot_foreground_quota,
            auto_remove_topk=args.auto_remove_topk,
        )
    else:
        cprint(f"❌ 输入路径无效：{args.input_path}", "red")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ManiSkill H5转Zarr脚本（适配多轨迹H5结构）")
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="H5文件路径 或 存放H5文件的目录（支持批量转换）"
    )
    parser.add_argument(
        "--zarr-save-dir",
        type=str,
        default=str(pathlib.Path(__file__).resolve().parent / "data_new" / "maniskill_zarr"),
        help="Zarr文件保存根目录（默认与你原有代码data_new同级）"
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default="base_camera",
        help="相机名称（匹配你H5中的sensor_param/base_camera）"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式（关闭进度条和详细打印）"
    )
    parser.add_argument(
        "--max-episode",
        type=int,
        default=None,
        help="最多处理的轨迹条数（默认处理全部）"
    )
    parser.add_argument(
        "--ground-seg-ids",
        type=str,
        default="auto",
        help="地面 segmentation id；格式如 '17'，默认 auto（按名称或自动估计）",
    )
    parser.add_argument(
        "--robot-seg-ids",
        type=str,
        default="auto",
        help="机器人 segmentation id；格式如 '1,2,3'，默认 auto（按名称推断）",
    )
    parser.add_argument(
        "--non-foreground-seg-ids",
        type=str,
        default="auto",
        help="非前景（如桌子）segmentation id；格式如 '16'，默认 auto（按名称推断）",
    )
    parser.add_argument(
        "--ground-name-keywords",
        type=str,
        default="ground,floor",
        help="当 --ground-seg-ids=auto 时，按名称匹配地面的关键词，逗号分隔",
    )
    parser.add_argument(
        "--robot-name-keywords",
        type=str,
        default="robot,panda",
        help="当 --robot-seg-ids=auto 时，按名称匹配机器人的关键词，逗号分隔",
    )
    parser.add_argument(
        "--non-foreground-name-keywords",
        type=str,
        default="desk,table,workspace",
        help="当 --non-foreground-seg-ids=auto 时，按名称匹配非前景(背景物体)关键词",
    )
    parser.add_argument(
        "--auto-remove-topk",
        type=int,
        default=2,
        help="仅当 ground 无法自动匹配时，用像素占比回退估计（保留兼容参数）",
    )
    parser.add_argument(
        "--pc-foreground-quota",
        type=int,
        default=128,
        help="point_clouds 前景点保底数量（前景点不足则全部保留）",
    )
    parser.add_argument(
        "--pc-no-robot-foreground-quota",
        type=int,
        default=256,
        help="point_clouds_no_robot 前景点保底数量（前景点不足则全部保留）",
    )
    args = parser.parse_args()
    main(args)