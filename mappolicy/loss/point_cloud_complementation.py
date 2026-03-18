import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer
)

# coordinate frame in SceneMap is x(right), y(upward), z(backward)

# metaworld, camera extrinsic matrix
Metaworld_extrinsic_matrix = {
    "corner": np.array(
        [
            [-0.7071, 0.7071, 0.0, -0.495],
            [0.1925, 0.1925, 0.9623, -0.2887],
            [0.6804, 0.6804, -0.2722, 1.1839],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "corner2": np.array(
        [
            [-0.5499, -0.8332, 0.0584, 0.484],
            [-0.3762, 0.3095, 0.8733, -0.4096],
            [-0.7457, 0.4582, -0.4837, 1.5931],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
}

# metaworld, camera intrinsic matrix
Metaworld_intrinsic_matrix = {
    "corner": np.array(
        [
            [270.3919, 0.0, 112.0],
            [0.0, 270.3919, 112.0],
            [0.0, 0.0, 1.0],
        ]
    ),
    "corner2": np.array(
        [
            [193.9897, 0.0, 112.0],
            [0.0, 193.9897, 112.0],
            [0.0, 0.0, 1.0],
        ]
    ),
}

# The coordinate frame in Sapien is: x(forward), y(left), z(upward)
# rlbench, camera extrinsic matrix
RLBench_extrinsic_matrix = {
    "front": np.array(
        [
            [ 1.16849501e-07, -1.00000000e+00, -6.03176614e-07,  8.32426571e-07,],
            [-4.22617918e-01, -5.63571533e-07,  9.06307948e-01, -8.61432102e-01,],
            [-9.06307948e-01,  1.31264604e-07, -4.22617948e-01,  1.89125107e+00,],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00,],
        ]
    ),
}

# rlbench, camera intrinsic matrix
RLBench_intrinsic_matrix = {
    "front": np.array(
        [
            [-307.7174807,0.0,112.0,],
            [0.0,-307.7174807,112.0,],
            [0.0,0.0,1.0,],
        ]
    )
}

# The coordinate frame in ManiSkill2 is: x(forward), y(right), z(downward)
# maniskill, camera extrinsic matrix
ManiSkill_extrinsic_matrix = {
    "base_camera": np.array(
        [
            [ 0.0,        1.0,        0.0,          0.0      ],
            [ 0.78086877,  0.0,      -0.62469506,  0.14055645],
            [-0.62469506,  0.0,      -0.78086877,  0.6559298 ],
            [ 0.0,        0.0,        0.0,          1.0      ],
        ]
    )
}

# maniskill, camera intrinsic matrix
ManiSkill_intrinsic_matrix = {
    "base_camera": np.array(
        [
            [112.0,   0.0,  112.0],
            [  0.0, 112.0, 112.0],
            [  0.0,   0.0,   1.0]
        ]
    )
}

def chamfer_loss(pc_a, pc_b):
    # pc_a: [B, N, 3], pc_b: [B, M, 3]
    # loss 直接返回平均距离，默认已经处理了双向
    loss, _ = chamfer_distance(pc_a[:, :, :3], pc_b[:, :, :3])
    return loss

def unidirectional_chamfer_loss(pc_a, pc_b):
    # point_reduction="mean" 对应外层的 .mean()
    # batch_reduction="mean" 对应 batch 维度的 .mean()
    loss, _ = chamfer_distance(
        pc_a[:, :, :3], 
        pc_b[:, :, :3], 
        single_directional=True # 设为 True 变为单向
    )
    return loss

def get_visible_points_pytorch3d(full_points, K, extrinsic, image_size=[224, 224]):
    """
    使用 PyTorch3D 提取可见点云
    
    参数:
    - full_points: torch.Tensor, (N, 3), 完整物体的 3D 点云
    - K: torch.Tensor, (3, 3), 相机内参
    - extrinsic: torch.Tensor, (4, 4), 相机外参矩阵 (World-to-Camera)
    - image_size: tuple 或 list, (H, W)
    
    返回:
    - visible_points: torch.Tensor, (M, 3), 过滤后的可见点云
    """
    device = full_points.device
    H, W = image_size

    # 1. 从 4x4 外参矩阵提取 R 和 T
    # OpenCV 格式: P_cam = R_std @ P_world + T_std
    # PyTorch3D 格式要求: R 是转置的 (因为它是行向量乘法 P_world @ R + T)
    R_std = extrinsic[:3, :3]
    T_std = extrinsic[:3, 3]
    
    R_pytorch3d = R_std.transpose(0, 1).unsqueeze(0) # (1, 3, 3)
    T_pytorch3d = T_std.unsqueeze(0)                # (1, 3)

    # 2. 从 3x3 内参矩阵提取参数
    focal_length = torch.tensor([[K[0, 0], K[1, 1]]], device=device)
    principal_point = torch.tensor([[K[0, 2], K[1, 2]]], device=device)
    
    # 3. 设置相机
    # in_ndc=False 表示使用像素坐标系定义
    cameras = PerspectiveCameras(
        focal_length=focal_length,
        principal_point=principal_point,
        in_ndc=False,
        image_size=((H, W),),
        R=R_pytorch3d,
        T=T_pytorch3d,
        device=device
    )

    # 4. 光栅化配置
    # radius 的大小决定了点的“厚度”。
    # 对于 1024-2048 个点的物体，0.01-0.03 通常比较合适
    raster_settings = PointsRasterizationSettings(
        image_size=(H, W), 
        radius=0.015,         
        points_per_pixel=1   # Z-Buffer: 每个像素只选最近的点
    )

    # 5. 封装点云
    point_cloud = Pointclouds(points=[full_points])

    # 6. 执行光栅化 (深度测试)
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(point_cloud)

    # fragments.idx: (1, H, W, 1) 存储可见点的索引
    # 获取唯一的点索引，并去掉背景值 -1
    visible_indices = fragments.idx.unique()
    visible_indices = visible_indices[visible_indices >= 0]
    
    # 7. 提取可见点
    visible_points = full_points[visible_indices]

    return visible_points

def point_cloud_complementation_loss_metaworld(preds, gts, camera_name):
    device = preds.device
    K = torch.from_numpy(Metaworld_intrinsic_matrix[camera_name]).float().to(device)
    extrinsic = torch.from_numpy(Metaworld_extrinsic_matrix[camera_name]).float().to(device)
    pred_visible = get_visible_points_pytorch3d(preds, K, extrinsic)
    loss = chamfer_loss(pred_visible, gts)
    return loss

def point_cloud_complementation_loss_rlbench(preds, gts, camera_name):
    # preds: [B, N, 3] in world frame
    # gts:   [B, M, 3] in camera frame (will be transformed to world frame)
    device = preds.device
    if preds.ndim != 3 or gts.ndim != 3:
        raise ValueError(f"Expected preds/gts to be 3D tensors [B, N, 3]/[B, M, 3], got {preds.shape=} {gts.shape=}")
    if preds.size(0) != gts.size(0):
        raise ValueError(f"Batch size mismatch: preds batch={preds.size(0)} vs gts batch={gts.size(0)}")

    K = torch.from_numpy(RLBench_intrinsic_matrix[camera_name]).to(device=device, dtype=preds.dtype)
    extrinsic = torch.from_numpy(RLBench_extrinsic_matrix[camera_name]).to(device=device, dtype=preds.dtype)

    # gts camera -> world
    # X_cam_row = X_world_row @ R^T + t  =>  X_world_row = (X_cam_row - t) @ R
    R_std = extrinsic[:3, :3]
    t_std = extrinsic[:3, 3]
    gts_world = (gts - t_std.view(1, 1, 3)) @ R_std

    losses = []
    for b in range(preds.size(0)):
        pred_visible_b = get_visible_points_pytorch3d(preds[b], K, extrinsic)

        if pred_visible_b.numel() == 0:
            pred_visible_b = preds[b, :1, :]

        loss_b = chamfer_loss(pred_visible_b.unsqueeze(0), gts_world[b:b + 1])
        losses.append(loss_b)

    loss = torch.stack(losses).mean()
    return loss

def point_cloud_complementation_loss_maniskill(preds, gts, camera_name):
    # preds: [B, N, 3] in SceneMap world frame
    # gts:   [B, M, 3] in camera frame (will be transformed to ManiSkill world frame)
    device = preds.device
    if preds.ndim != 3 or gts.ndim != 3:
        raise ValueError(f"Expected preds/gts to be 3D tensors [B, N, 3]/[B, M, 3], got {preds.shape=} {gts.shape=}")
    if preds.size(0) != gts.size(0):
        raise ValueError(f"Batch size mismatch: preds batch={preds.size(0)} vs gts batch={gts.size(0)}")
    
    # transform preds from SceneMap coordinate frame (x right, y up, z backward)
    # to ManiSkill2 world coordinate frame (x forward, y right, z downward)
    # R 矩阵解释：
    # 第一列：SceneMap 的 x(右) 对应 ManiSkill 的 y(右) -> [0, 1, 0]
    # 第二列：SceneMap 的 y(上) 对应 ManiSkill 的 z(下) 的反方向 -> [0, 0, -1]
    # 第三列：SceneMap 的 z(后) 对应 ManiSkill 的 x(前) 的反方向 -> [-1, 0, 0]
    R = torch.tensor([
        [ 0,  0, -1],
        [ 1,  0,  0],
        [ 0, -1,  0]
    ], device=device, dtype=preds.dtype) # (3, 3)
    preds = preds @ R.T 
    
    # 获取相机参数并计算可见点云
    K = torch.from_numpy(ManiSkill_intrinsic_matrix[camera_name]).to(device=device, dtype=preds.dtype)
    extrinsic = torch.from_numpy(ManiSkill_extrinsic_matrix[camera_name]).to(device=device, dtype=preds.dtype)

    # gts is in camera frame -> transform to ManiSkill world frame
    # extrinsic follows OpenCV convention (column vectors): X_cam = R * X_world + t
    # for row vectors used here: X_cam_row = X_world_row @ R^T + t
    # therefore inverse is:     X_world_row = (X_cam_row - t) @ R
    R_std = extrinsic[:3, :3]
    t_std = extrinsic[:3, 3]
    gts_world = (gts - t_std.view(1, 1, 3)) @ R_std

    # get_visible_points_pytorch3d 只接受单个点云 (N, 3)，这里按 batch 逐个处理
    losses = []
    for b in range(preds.size(0)):
        pred_visible_b = get_visible_points_pytorch3d(preds[b], K, extrinsic)

        # 极端情况下如果没有可见点，使用一个退化点避免 chamfer_distance 崩溃
        if pred_visible_b.numel() == 0:
            pred_visible_b = preds[b, :1, :]

        loss_b = chamfer_loss(pred_visible_b.unsqueeze(0), gts_world[b:b + 1])
        losses.append(loss_b)
    
    # 计算 Chamfer 距离
    loss = torch.stack(losses).mean()
    return loss


def _example_main():
    """Constructed-point-cloud validation for computed loss vs theoretical loss."""
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[example] device={device}")

    # 1) 构造 ManiSkill 坐标系下的“预测点”（每个 batch 只有 1 个点，便于推导理论值）
    #    这些点靠近相机视野中心，通常可见。
    preds_maniskill = torch.tensor(
        [
            [[0.00, 0.00, 0.00]],
            [[0.05, 0.02, 0.00]],
        ],
        device=device,
        dtype=torch.float32,
    )  # [B=2, N=1, 3]

    # 2) 将输入转换为 SceneMap 坐标（因为函数内部会再转回 ManiSkill）
    #    已知: scene @ R^T = maniskill  ->  scene = maniskill @ R
    R = torch.tensor(
        [
            [0, 0, -1],
            [1, 0, 0],
            [0, -1, 0],
        ],
        device=device,
        dtype=torch.float32,
    )
    preds_scene = preds_maniskill @ R

    # 3) 构造 GT（先在 ManiSkill 世界坐标系下）：与预测点有固定偏移 delta
    #    对于单点对单点，双向 Chamfer(默认) 的理论值 = 2 * ||delta||^2
    delta = torch.tensor(
        [
            [[0.10, 0.00, 0.00]],
            [[-0.05, 0.00, 0.00]],
        ],
        device=device,
        dtype=torch.float32,
    )
    gts_maniskill = preds_maniskill + delta

    # 4) 将 GT 从 ManiSkill 世界坐标系转换到相机坐标系（以匹配函数输入定义）
    extrinsic = torch.from_numpy(ManiSkill_extrinsic_matrix["base_camera"]).to(device=device, dtype=torch.float32)
    R_std = extrinsic[:3, :3]
    t_std = extrinsic[:3, 3]
    gts_camera = gts_maniskill @ R_std.T + t_std.view(1, 1, 3)

    # 5) 计算函数输出 loss
    computed_loss = point_cloud_complementation_loss_maniskill(preds_scene, gts_camera, "base_camera")

    # 6) 理论 loss（逐 batch，再取均值）
    #    每个 batch: L_b = 2 * ||delta_b||^2
    per_batch_theoretical = 2.0 * (delta.squeeze(1) ** 2).sum(dim=-1)
    theoretical_loss = per_batch_theoretical.mean()

    # 7) 对比
    print(f"[example] per_batch_theoretical={per_batch_theoretical.tolist()}")
    print(f"[example] computed_loss   ={computed_loss.item():.8f}")
    print(f"[example] theoretical_loss={theoretical_loss.item():.8f}")
    print(f"[example] equal(allclose)={torch.allclose(computed_loss, theoretical_loss, atol=1e-6, rtol=1e-6)}")

    # 8) 附加：梯度可回传检查
    preds_scene_grad = preds_scene.clone().requires_grad_(True)
    loss_grad = point_cloud_complementation_loss_maniskill(preds_scene_grad, gts_camera, "base_camera")
    loss_grad.backward()
    grad_ok = preds_scene_grad.grad is not None and torch.isfinite(preds_scene_grad.grad).all().item()
    print(f"[example] grad_ok={grad_ok}")


if __name__ == "__main__":
    _example_main()