import h5py
import numpy as np

# ===================== 替换为你的.h5文件路径 =====================
traj_h5_path = "/data2/zehao/MapPolicy/Data/maniskill/StackCube-v1/motionplanning/StackCube.rgb+depth+segmentation.pd_ee_delta_pose.physx_cpu.h5"
# =================================================================

def print_h5_structure(h5obj, indent=0):
    """递归打印h5对象的完整结构，包括组和数据集的形状/类型。"""
    for key, item in h5obj.items():
        prefix = "  " * indent
        if isinstance(item, h5py.Dataset):
            print(f"{prefix}{key}  [dataset] shape={item.shape} dtype={item.dtype}")
        else:
            print(f"{prefix}{key}  [group]")
            print_h5_structure(item, indent + 1)


def read_maniskill_multi_traj_h5(file_path):
    # 用with语句打开h5文件，自动管理文件句柄
    with h5py.File(file_path, "r") as f:
        # 打印整个文件结构
        print("完整的 h5 文件结构如下：")
        print_h5_structure(f)
        print("="*60)
        # 1. 查看所有顶层轨迹子组（traj_0 ~ traj_9）
        traj_keys = sorted(f.keys())  # 按轨迹序号排序
        print(f"h5文件包含 {len(traj_keys)} 条轨迹：{traj_keys}")
        print("="*60)

        # 2. 遍历每条轨迹，读取核心数据
        for traj_key in traj_keys:
            print(f"\n📌 正在读取轨迹：{traj_key}")
            traj_group = f[traj_key]  # 获取当前轨迹的子组

            # 3. 读取【动作数据】（pd_ee_delta_pos，维度应为3）
            if "actions" in traj_group:
                actions = traj_group["actions"][:]  # shape=(当前轨迹步数, 动作维度)
                print(f"  ✅ 动作数据shape：{actions.shape} (步数: {actions.shape[0]}, 维度: {actions.shape[1]})")
                print(f"  前2步动作：\n{actions[:2]}")
            else:
                print(f"  ❌ 该轨迹无动作数据")

            # 4. 读取【点云观测数据】（在observations子组中）
            if "observations" in traj_group:
                obs_group = traj_group["observations"]
                print(f"  观测数据包含子键：{list(obs_group.keys())}")
                
                # 读取点云核心数据（xyzw=坐标+齐次项，rgb=颜色）
                if "pointcloud" in obs_group:
                    pcd_group = obs_group["pointcloud"]
                    # 点云坐标（x,y,z,w）：shape=(当前轨迹步数, 点云点数, 4)
                    pcd_xyzw = pcd_group["xyzw"][:]
                    # 点云颜色（r,g,b）：shape=(当前轨迹步数, 点云点数, 3)
                    pcd_rgb = pcd_group["rgb"][:] if "rgb" in pcd_group else None
                    
                    print(f"  ✅ 点云xyzw shape：{pcd_xyzw.shape}")
                    if pcd_rgb is not None:
                        print(f"  ✅ 点云rgb shape：{pcd_rgb.shape}")
                    # 打印第1步的前3个点坐标（只看x,y,z，忽略w）
                    print(f"  第1步前3个点坐标：\n{pcd_xyzw[0, :3, :3]}")
            else:
                print(f"  ❌ 该轨迹无观测数据")

            # 5. 读取【环境状态数据】（可选）
            if "env_states" in traj_group:
                env_states = traj_group["env_states"][:]
                print(f"  ✅ 环境状态shape：{env_states.shape}")
            else:
                print(f"  ❌ 该轨迹无环境状态数据")

            # 6. 读取【奖励数据】（若开启--record-rewards则存在）
            if "rewards" in traj_group:
                rewards = traj_group["rewards"][:]
                print(f"  ✅ 奖励数据shape：{rewards.shape}，前2步奖励：{rewards[:2]}")
            else:
                print(f"  ❌ 该轨迹无奖励数据")

            print("-"*50)  # 分隔每条轨迹的输出

# 执行读取函数
if __name__ == "__main__":
    read_maniskill_multi_traj_h5(traj_h5_path)