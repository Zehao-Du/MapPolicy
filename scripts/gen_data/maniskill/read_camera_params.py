import h5py
import numpy as np

# ===================== 替换为你的 .h5 文件路径 =====================
traj_h5_path = "/data2/zehao/MapPolicy/Data/maniskill/StackCube-v1/motionplanning/StackCube.rgb+depth+segmentation.pd_ee_delta_pose.physx_cpu.h5"
# ==================================================================

# 这些关键字用于在 h5 中定位相机参数
CAMERA_HINTS = (
    # "camera",
    # "cam",
    # "sensor_param",
    "intrinsic",
    "extrinsic",
    "cam2world",
    "world2cam",
)

INTRINSIC_HINTS = ("intrinsic", "k", "camera_matrix")
EXTRINSIC_HINTS = ("extrinsic", "cam2world", "world2cam", "pose", "rt")


def _sorted_traj_keys(keys):
    """把 traj_0, traj_1 ... 做自然排序。"""

    def _key_fn(x):
        if x.startswith("traj_"):
            suffix = x.split("traj_")[-1]
            if suffix.isdigit():
                return (0, int(suffix))
        return (1, x)

    return sorted(keys, key=_key_fn)


def _is_camera_related(path):
    p = path.lower()
    return any(h in p for h in CAMERA_HINTS)


def _classify_param(path):
    p = path.lower()
    if any(h in p for h in INTRINSIC_HINTS):
        return "intrinsic"
    if any(h in p for h in EXTRINSIC_HINTS):
        return "extrinsic"
    return "other"


def _preview_array(arr):
    """尽量把高维数组压到 2D/1D，便于打印预览。"""
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    sample = arr
    while sample.ndim > 2:
        sample = sample[0]

    if sample.ndim == 0:
        return sample
    if sample.ndim == 1:
        return sample[: min(10, sample.shape[0])]
    return sample


def _collect_camera_datasets(group, base_path=""):
    """递归收集组内与相机参数相关的数据集。"""
    results = []
    for key, item in group.items():
        path = f"{base_path}/{key}" if base_path else key
        if isinstance(item, h5py.Dataset):
            if _is_camera_related(path):
                results.append((path, item))
        elif isinstance(item, h5py.Group):
            results.extend(_collect_camera_datasets(item, path))
    return results


def read_camera_params_from_h5(file_path):
    with h5py.File(file_path, "r") as f:
        traj_keys = _sorted_traj_keys(list(f.keys()))
        print(f"发现 {len(traj_keys)} 条轨迹")
        print("=" * 70)

        for traj_key in traj_keys:
            traj_group = f[traj_key]
            obs_key = "observations" if "observations" in traj_group else ("obs" if "obs" in traj_group else None)

            print(f"\n📌 轨迹: {traj_key}")
            if obs_key is None:
                print("  ❌ 未找到 observations/obs，跳过")
                continue

            obs_group = traj_group[obs_key]
            cam_datasets = _collect_camera_datasets(obs_group, base_path=obs_key)

            if not cam_datasets:
                print("  ❌ 未找到相机相关参数数据集")
                continue

            print(f"  ✅ 找到 {len(cam_datasets)} 个相机相关数据集")
            for ds_path, ds in cam_datasets:
                arr = ds[()]
                kind = _classify_param(ds_path)
                preview = _preview_array(arr)

                print(f"\n  [{kind}] {ds_path}")
                print(f"    shape={arr.shape}, dtype={arr.dtype}")
                print("    preview:")
                print(preview)
                # break

            print("-" * 70)
            break


if __name__ == "__main__":
    read_camera_params_from_h5(traj_h5_path)
