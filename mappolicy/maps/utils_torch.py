import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_euler_angles, rotation_6d_to_matrix
import clip
from torchvision.transforms import Normalize

class CLIPEncoder(nn.Module):
    def __init__(self, model_name, freeze=True):
        super(CLIPEncoder, self).__init__()
        self.model, self.preprocess = clip.load(model_name)
        # see: https://github.com/openai/CLIP/blob/main/clip/clip.py line 79
        self.preprocess = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        self.feature_dim = {
            "ViT-B/32": 512,
        }[model_name]
        if freeze:
            self.freeze()

    def forward(self, x):
        # x = self.preprocess(x)
        return self.model.encode_text(x)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

def normalize(v):
    """可求导的向量归一化"""
    norm = torch.linalg.norm(v, dim=-1, keepdim=True)
    # 避免除以零，添加一个小 epsilon
    return v / (norm + 1e-6)

def get_rodrigues_matrix(axis: torch.Tensor, angle: torch.Tensor, device=None) -> torch.Tensor:
    """
    计算罗德里格斯旋转矩阵 (Rodrigues' Rotation Matrix)，支持批处理。

    Args:
        axis: 旋转轴向量，形状应为 [B, 3]。
        angle: 旋转角度（弧度），形状应为 [B] 或 [B, 1]。
        device: Tensor 所在的设备。

    Returns:
        3x3 旋转矩阵，形状为 [B, 3, 3]。
    """
    
    if not isinstance(axis, torch.Tensor):
        axis = torch.as_tensor(axis, dtype=torch.float32, device=device)
    if not isinstance(angle, torch.Tensor):
        angle = torch.as_tensor(angle, dtype=torch.float32, device=device)

    if device is None:
        device = axis.device
    
    B = angle.shape[0]

    if axis.shape[-1] != 3 or axis.ndim != 2:
        raise ValueError(f"Axis vector must have shape [B, 3], got {axis.shape}")
    
    if angle.ndim == 1:
        angle = angle.unsqueeze(-1)
    
    axis = F.normalize(axis, p=2, dim=-1)

    identity = torch.eye(3, dtype=axis.dtype, device=device).unsqueeze(0).repeat(B, 1, 1)
    
    ax = axis[:, 0].unsqueeze(-1)
    ay = axis[:, 1].unsqueeze(-1)
    az = axis[:, 2].unsqueeze(-1)
    
    # S1: 批次化的叉积矩阵 [B, 3, 3]
    # S1 的每个元素都是一个张量，需要堆叠成 [B, 3, 3]
    s1 = torch.zeros((B, 3, 3), dtype=axis.dtype, device=device)
    s1[:, 0, 1] = -az.squeeze(-1)
    s1[:, 0, 2] = ay.squeeze(-1)
    s1[:, 1, 0] = az.squeeze(-1)
    s1[:, 1, 2] = -ax.squeeze(-1)
    s1[:, 2, 0] = -ay.squeeze(-1)
    s1[:, 2, 1] = ax.squeeze(-1)
    
    # S2: 批次化的外积矩阵 [B, 3, 3]
    # axis.unsqueeze(-1) -> [B, 3, 1]
    # axis.unsqueeze(1)  -> [B, 1, 3]
    s2 = torch.matmul(axis.unsqueeze(-1), axis.unsqueeze(1))
    
    # 3. 三角函数计算 (形状 [B, 1])
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)

    # 4. 罗德里格斯公式组合
    # cos_angle, sin_angle, (1.0 - cos_angle) 都是 [B, 1]，需要扩展到 [B, 1, 1] 来广播
    cos_angle_b = cos_angle.unsqueeze(-1) # [B, 1, 1]
    sin_angle_b = sin_angle.unsqueeze(-1) # [B, 1, 1]
    one_minus_cos_angle_b = (1.0 - cos_angle).unsqueeze(-1) # [B, 1, 1]

    # R = cos(a) * I + sin(a) * S1 + (1 - cos(a)) * S2
    rodrigues_matrix = (cos_angle_b * identity) + (sin_angle_b * s1) + (one_minus_cos_angle_b * s2)
    
    return rodrigues_matrix

def euler_to_matrix(rotation: torch.Tensor, rotation_order: str = "XYZ") -> torch.Tensor:
    """
    将欧拉角 [B, 3] 转换为旋转矩阵 [B, 3, 3]。
    
    Args:
        rotation: 欧拉角 (alpha, beta, gamma)，形状 [B, 3]。
        rotation_order: 旋转顺序，如 "XYZ"。
        
    Returns:
        旋转矩阵 R，形状 [B, 3, 3]。
    """
    B = rotation.shape[0]
    device = rotation.device
    dtype = rotation.dtype
    
    # 定义标准轴向量 [B, 3]
    standard_axis = {
        "X": torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device).unsqueeze(0).repeat(B, 1),
        "Y": torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device).unsqueeze(0).repeat(B, 1),
        "Z": torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device).unsqueeze(0).repeat(B, 1),
    }

    # 计算三个基础旋转矩阵
    angles = {"X": rotation[:, 0], "Y": rotation[:, 1], "Z": rotation[:, 2]}
    axis_map = {"X": standard_axis["X"], "Y": standard_axis["Y"], "Z": standard_axis["Z"]}
    
    # 计算三个罗德里格斯矩阵
    R_mats = {}
    for i, axis_char in enumerate("XYZ"):
        R_mats[axis_char] = get_rodrigues_matrix(axis_map[axis_char], rotation[:, i])
    
    # 组合旋转矩阵 R_total = R_final * ... * R_first
    # 在欧拉角旋转中，点 P 绕 Z 轴旋转 (Rz)，然后绕 Y 轴旋转 (Ry)，然后绕 X 轴旋转 (Rx)
    # R_total = Rx @ Ry @ Rz (如果使用左乘/列向量惯例)
    # 对于行向量 P' = P @ R^T，则 R_total^T = Rz^T @ Ry^T @ Rx^T
    
    # 初始化总旋转矩阵为单位矩阵 [B, 3, 3]
    R_total = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)

    # 按照 rotation_order 组合，顺序从左到右 (对应于列向量右乘的 Rx@Ry@Rz)
    for axis_char in rotation_order:
        R_total = torch.matmul(R_mats[axis_char], R_total)
        
    # 注意: 罗德里格斯公式返回的 R 是 R_total，其转置 R_total^T 用于行向量的 P' = P @ R_total^T
    return R_total

def relative_pose_euler(
    pos_A: torch.Tensor, rot_A: torch.Tensor, 
    pos_B: torch.Tensor, rot_B: torch.Tensor, 
    rotation_order: str = "XYZ"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    计算坐标系 B 相对于坐标系 A 的相对位姿 (T_A_inv * T_B)，输入为位置和欧拉角。

    Args:
        pos_A: 坐标系 A 的位置向量 t_A，形状 [B, 3]。
        rot_A: 坐标系 A 的欧拉角，形状 [B, 3]。
        pos_B: 坐标系 B 的位置向量 t_B，形状 [B, 3]。
        rot_B: 坐标系 B 的欧拉角，形状 [B, 3]。
        rotation_order: 欧拉角旋转顺序，如 "XYZ"。

    Returns:
        tuple (R_rel, t_rel):
            R_rel: 相对旋转矩阵 R_A_to_B，形状 [B, 3, 3]。
            t_rel: 相对平移向量 t_A_to_B，形状 [B, 3]。
    """
    
    # --- 1. 欧拉角转旋转矩阵 ---
    
    R_A = euler_to_matrix(rot_A, rotation_order)  # [B, 3, 3]
    R_B = euler_to_matrix(rot_B, rotation_order)  # [B, 3, 3]
    
    # --- 2. 核心计算 ---
    
    # 统一平移向量形状 [B, 3, 1]
    t_A = pos_A.unsqueeze(-1)
    t_B = pos_B.unsqueeze(-1)
    
    # 相对旋转矩阵 R_rel = R_A^T @ R_B
    R_A_T = R_A.transpose(-1, -2) 
    R_rel = torch.matmul(R_A_T, R_B)  # [B, 3, 3]

    # 相对平移向量 t_rel = R_A^T @ (t_B - t_A)
    t_diff = t_B - t_A  # [B, 3, 1]
    t_rel_3x1 = torch.matmul(R_A_T, t_diff)  # [B, 3, 1]
    
    # 恢复平移向量形状 [B, 3]
    t_rel = t_rel_3x1.squeeze(-1)
    rot_rel = matrix_to_euler_angles(R_rel, rotation_order)
    
    return t_rel, rot_rel

def rotate(points: torch.Tensor, rotation: torch.Tensor, rotation_order="XYZ", device=None):
    """
    旋转点集 (points) 的位置坐标部分，支持批处理和欧拉角旋转。
    
    :param points: 输入点集。形状为 [B, N, D] 或 [B, D] (其中 D >= 3，前 3 维为位置坐标)
    :type points: torch.Tensor
    :param rotation: 欧拉角 [B, 3]
    :type rotation: torch.Tensor
    :param rotation_order: 旋转顺序，如 "XYZ"
    :param device: cuda device
    
    :returns: 旋转后的点集，形状与输入相同。
    :rtype: torch.Tensor
    """
    dtype = points.dtype
    if device == None:
        device = points.device
    
    original_ndim = points.ndim
    if original_ndim == 2:
        points_reshaped = points.unsqueeze(1)
    elif original_ndim == 3:
        points_reshaped = points
    else:
        raise ValueError(f"Unsupported points tensor rank: {original_ndim}. Expected 2 or 3.")
    B, N, D = points_reshaped.shape
    
    if B > 10000: 
        raise ValueError(f"CRITICAL ERROR: Detected abnormally large Batch Size B={B}. "
                         f"Points shape: {points_reshaped.shape}. "
                         "Check your data loader for corrupt samples!")
    
    if D < 3:
        raise ValueError(f"Points dimension D must be at least 3 (for X, Y, Z), got {D}.")
     
    position = points_reshaped[:, :, 0:3] 
    feature = None
    if D > 3:
        feature = points_reshaped[:, :, 3:]
     
    standard_axis = {
        "X": torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device).unsqueeze(0).repeat(B, 1), # [B, 3]
        "Y": torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device).unsqueeze(0).repeat(B, 1), # [B, 3]
        "Z": torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device).unsqueeze(0).repeat(B, 1), # [B, 3]
    }
    
    rot_mat = {}
    rot_mat["X"] = get_rodrigues_matrix(standard_axis["X"], rotation[:, 0], device=device) # [B, 3, 3]
    rot_mat["Y"] = get_rodrigues_matrix(standard_axis["Y"], rotation[:, 1], device=device) # [B, 3, 3]
    rot_mat["Z"] = get_rodrigues_matrix(standard_axis["Z"], rotation[:, 2], device=device) # [B, 3, 3]
    
    for s in rotation_order:
        R_T = rot_mat[s].transpose(-1, -2) # [B, 3, 3]
        position = torch.matmul(position, R_T)
    
    if feature is not None:
        rotated_points_reshaped = torch.cat([position, feature], dim=-1)
    else:
        rotated_points_reshaped = position
        
    if original_ndim == 2:
        return rotated_points_reshaped.squeeze(1) # [B, D]
    
    return rotated_points_reshaped

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    辅助函数：将 3x3 旋转矩阵转换为 6D 向量
    (直接提取矩阵的前两列)
    Input: [B, 3, 3]
    Output: [B, 6]
    """
    # 取第一列 (x轴) 和 第二列 (y轴)
    # matrix[..., 0] shape is [B, 3]
    return torch.cat([matrix[..., 0], matrix[..., 1]], dim=-1)

def rotate_6D(points: torch.Tensor, rotation: torch.Tensor, device=None):
    """
    旋转点集 (points) 的位置坐标部分，支持批处理和 6D 旋转表示。
    
    :param points: 输入点集。形状为 [B, N, D] 或 [B, D] (其中 D >= 3，前 3 维为位置坐标)
    :type points: torch.Tensor
    :param rotation: 6D 旋转表示 [B, 6] (前3维为向量a1，后3维为向量a2)
    :type rotation: torch.Tensor
    :param device: cuda device (可选)
    
    :returns: 旋转后的点集，形状与输入相同。
    :rtype: torch.Tensor
    """
    # ---------------- 数据准备与检查 ----------------
    if device is None:
        device = points.device
    
    original_ndim = points.ndim
    if original_ndim == 2:
        points_reshaped = points.unsqueeze(1) # [B, 1, D]
    elif original_ndim == 3:
        points_reshaped = points              # [B, N, D]
    else:
        raise ValueError(f"Unsupported points tensor rank: {original_ndim}. Expected 2 or 3.")
        
    B, N, D = points_reshaped.shape
    
    if B > 10000: 
        raise ValueError(f"CRITICAL ERROR: Detected abnormally large Batch Size B={B}. "
                         f"Points shape: {points_reshaped.shape}. "
                         "Check your data loader for corrupt samples!")
    
    if D < 3:
        raise ValueError(f"Points dimension D must be at least 3 (for X, Y, Z), got {D}.")
    
    # ---------------- 核心逻辑 ----------------
    
    # 1. 获取位置和特征
    position = points_reshaped[:, :, 0:3] # [B, N, 3]
    feature = None
    if D > 3:
        feature = points_reshaped[:, :, 3:]
     
    # 2. 计算旋转矩阵 R [B, 3, 3]
    # 这里不需要 rotation_order 了，6D 直接定矩阵
    R = rotation_6d_to_matrix(rotation) 
    
    # 3. 应用旋转
    # 点云通常是行向量 [x, y, z]，公式为 p_new = p * R^T
    # R 的形状是 [B, 3, 3]，我们需要它的转置来右乘
    R_T = R.transpose(-1, -2) # [B, 3, 3]
    
    position = torch.matmul(position, R_T) # [B, N, 3]
    
    # ---------------- 结果重组 ----------------
    
    if feature is not None:
        rotated_points_reshaped = torch.cat([position, feature], dim=-1)
    else:
        rotated_points_reshaped = position
        
    if original_ndim == 2:
        return rotated_points_reshaped.squeeze(1) # [B, D]
    
    return rotated_points_reshaped

def relative_pose_6d(
    pos_A: torch.Tensor, rot_A: torch.Tensor, 
    pos_B: torch.Tensor, rot_B: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    计算坐标系 B 相对于坐标系 A 的相对位姿 (T_A_inv * T_B)。
    输入输出均为位置和 6D 旋转表示。

    Args:
        pos_A: 坐标系 A 的绝对位置，形状 [B, 3]。
        rot_A: 坐标系 A 的 6D 绝对旋转，形状 [B, 6]。
        pos_B: 坐标系 B 的绝对位置，形状 [B, 3]。
        rot_B: 坐标系 B 的 6D 绝对旋转，形状 [B, 6]。

    Returns:
        tuple (t_rel, rot_rel_6d):
            t_rel: 相对平移向量，形状 [B, 3]。
            rot_rel_6d: 相对 6D 旋转向量，形状 [B, 6]。
    """
    
    # --- 1. 6D 转 旋转矩阵 ---
    # 这一步是为了进行物理上正确的几何运算
    R_A = rotation_6d_to_matrix(rot_A)  # [B, 3, 3]
    R_B = rotation_6d_to_matrix(rot_B)  # [B, 3, 3]
    
    # --- 2. 核心计算 (在矩阵层面进行) ---
    
    # 准备转置矩阵 R_A^T
    R_A_T = R_A.transpose(-1, -2) 
    
    # 计算相对旋转矩阵 R_rel = R_A^T @ R_B
    R_rel_matrix = torch.matmul(R_A_T, R_B)  # [B, 3, 3]

    # 计算相对平移向量 t_rel = R_A^T @ (t_B - t_A)
    # 扩展维度以进行矩阵乘法: [B, 3] -> [B, 3, 1]
    t_diff = (pos_B - pos_A).unsqueeze(-1) 
    t_rel_3x1 = torch.matmul(R_A_T, t_diff)  # [B, 3, 1]
    
    # 恢复平移向量形状 [B, 3]
    t_rel = t_rel_3x1.squeeze(-1)
    
    # --- 3. 结果转回 6D ---
    # 将计算得到的相对旋转矩阵转回 6D 表示
    rot_rel_6d = matrix_to_rotation_6d(R_rel_matrix)
    
    return t_rel, rot_rel_6d