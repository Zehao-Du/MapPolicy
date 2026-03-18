import torch
import numpy as np
from torch_geometric.data import Data

def pad_anchor(anchor_data, target_dim=12):
    """
    将 6, 9 维度的 Anchor 补零填充到 12 维
    anchor_data: list or np.array
    """
    arr = np.array(anchor_data, dtype=np.float32).flatten()
    current_dim = arr.shape[0]
    if current_dim < target_dim:
        # 在末尾补零
        return np.pad(arr, (0, target_dim - current_dim), 'constant')
    return arr[:target_dim]

def compute_relative_pose(node_i, node_j):
    """
    计算 Node_i 到 Node_j 的相对位姿
    输入是 StructureNode 对象
    """
    # 假设 position 是 [x, y, z]
    pos_i = np.array(node_i.position)
    pos_j = np.array(node_j.position)
    
    # 1. 相对位置 (在全局坐标系下)
    # 或者更高级一点：转到 Node_i 的局部坐标系: R_i.T * (P_j - P_i)
    # 这里为了演示简单，使用全局差值
    rel_pos = pos_j - pos_i 
    
    # 2. 相对旋转
    # 假设 rotation 是欧拉角或四元数。这里需要根据你具体的数据格式处理。
    # 简单起见，这里做差值，实际上应该做四元数乘法 q_i^{-1} * q_j
    rel_rot = np.array(node_j.rotation) - np.array(node_i.rotation)
    
    return np.concatenate([rel_pos, rel_rot])
