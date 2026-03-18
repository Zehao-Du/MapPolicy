# import os
# import sys
# os.environ["WAYLAND_DISPLAY"] = ""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_torch import normalize, rotate_6D
from base_template import StructureNode

import open3d as o3d

eps = 1e-8 

# ================================================================
# 全文件统一坐标轴约定（Local Primitive Coordinate System）:
#   +X: Right  (向右)
#   +Y: Up     (向上, 高度方向)
#   +Z: Front  (向前)
# 对应地:
#   -X: Left, -Y: Down, -Z: Back
#
# 说明:
# 1) 本文件所有 primitive 默认都遵循该约定（除非某段代码有显式特别说明）。
# 2) position = [x, y, z] 采用同一坐标系。
# 3) rotation 作用于该局部坐标系后再平移到世界坐标。
# ================================================================

class Cuboid(StructureNode):
    def __init__(self, height, top_length, top_width = None, bottom_length = None, bottom_width = None, top_offset = [0, 0], back_height = None, position = [0, 0, 0], rotation = [1, 0, 0, 0, 1, 0], rotation_order = "XYZ", Semantic = None, Affordance = None):
        """
        :param height: front-side height in +Y direction (Y is up)
        :param top_length: length of the top surface of the cuboid in the X-axis direction
        :param top_width: width of the top surface of the cuboid in the Z-axis direction
        :param bottom_length: length of the bottom surface of the cuboid in the X-axis direction
        :param bottom_width: width of the bottom surface of the cuboid in the Z-axis direction
        :param top_offset: offset between the upper and lower surface of the cuboid in the X-axis and Z-axis directions
        :param back_height: back-side height in +Y direction (Front is +Z, Back is -Z)
        :param position: position (x, y, z) of the cuboid
        :param rotation: rotation of the cuboid, represented via Euler angles (x, y, z)
        :param rotation_order: rotation order of the three rotation axes of the cuboid
        """
        
        self.dtype = height.dtype
        self.device = height.device
        self.B = height.shape[0]
        
        # Filling Missing Values
        if top_width == None:
            top_width = top_length
        if bottom_length == None:
            bottom_length = top_length
        if bottom_width == None:
            bottom_width = top_width
        if back_height == None:
            back_height = height
            
        if isinstance(top_offset, (list, tuple)):
            top_offset = torch.as_tensor(top_offset, dtype=self.dtype, device=self.device).unsqueeze(0).repeat(self.B, 1)
        if isinstance(position, (list, tuple)):
            position = torch.as_tensor(position, dtype=self.dtype, device=self.device).unsqueeze(0).repeat(self.B, 1)
        if isinstance(rotation, (list, tuple)):
            rotation = torch.as_tensor(rotation, dtype=self.dtype, device=self.device).unsqueeze(0).repeat(self.B, 1)

        # Record Parameters
        self.height = height
        self.top_length = top_length
        self.top_width = top_width
        self.bottom_length = bottom_length
        self.bottom_width = bottom_width
        self.top_offset = top_offset
        self.back_height = back_height
        self.position = position
        self.rotation = rotation
        self.rotation_order = rotation_order
            
        # Manually Defined Default Template Instance 
        self.vertices = torch.tensor([
            [-1 / 2, 1 / 2, 1 / 2],
            [1 / 2, 1 / 2, 1 / 2],
            [-1 / 2, 1 / 2, -1 / 2],
            [1 / 2, 1 / 2, -1 / 2],
            [-1 / 2, -1 / 2, 1 / 2],
            [1 / 2, -1 / 2, 1 / 2],
            [-1 / 2, -1 / 2, -1 / 2],
            [1 / 2, -1 / 2, -1 / 2]
        ], dtype=self.dtype, device=self.device).unsqueeze(0).repeat(self.B, 1, 1)

        self.faces = torch.tensor([
            [0, 1, 2], [1, 3, 2],   # node_face 1
            [4, 6, 5], [5, 6, 7],   # node_face 2
            [0, 4, 5], [0, 5, 1],   # node_face 3
            [2, 7, 6], [2, 3, 7],   # node_face 4
            [0, 6, 4], [0, 2, 6],   # node_face 5
            [1, 5, 7], [1, 7, 3]    # node_face 6
        ], dtype=torch.int16, device=self.device).unsqueeze(0).repeat(self.B, 1, 1)

        # Differentiable Deformation
        vertices_resize = torch.stack([
            torch.stack([self.top_length, self.height, self.top_width], dim=-1), # [B, 3]
            torch.stack([self.top_length, self.height, self.top_width], dim=-1), 
            torch.stack([self.top_length, self.back_height, self.top_width], dim=-1),
            torch.stack([self.top_length, self.back_height, self.top_width], dim=-1),
            torch.stack([self.bottom_length, self.height, self.bottom_width], dim=-1),
            torch.stack([self.bottom_length, self.height, self.bottom_width], dim=-1),
            torch.stack([self.bottom_length, self.back_height, self.bottom_width], dim=-1),
            torch.stack([self.bottom_length, self.back_height, self.bottom_width], dim=-1)
        ], dim=0).transpose(0, 1) # [B, 8, 3]
        self.vertices = self.vertices * vertices_resize

        y_offset_term = (self.back_height - self.height) / 2 # [B]
        zero = torch.tensor(0.0, dtype=self.dtype, device=self.device) 
        top_offset_x = self.top_offset[:, 0] # [B]
        top_offset_z = self.top_offset[:, 1] # [B]
        
        zero = torch.tensor(0.0, dtype=torch.float32, device=self.device) 
        vertices_offset = torch.stack([
            torch.stack([top_offset_x, zero.repeat(self.B), top_offset_z], dim=-1), 
            torch.stack([top_offset_x, zero.repeat(self.B), top_offset_z], dim=-1), 
            torch.stack([top_offset_x, y_offset_term, top_offset_z], dim=-1),
            torch.stack([top_offset_x, y_offset_term, top_offset_z], dim=-1),
            torch.stack([zero.repeat(self.B), zero.repeat(self.B), zero.repeat(self.B)], dim=-1),
            torch.stack([zero.repeat(self.B), zero.repeat(self.B), zero.repeat(self.B)], dim=-1),
            torch.stack([zero.repeat(self.B), y_offset_term, zero.repeat(self.B)], dim=-1),
            torch.stack([zero.repeat(self.B), y_offset_term, zero.repeat(self.B)], dim=-1)
        ], dim=0).transpose(0, 1) # [B, 8, 3]
        self.vertices = self.vertices + vertices_offset

        # Global Transformation
        self.vertices = rotate_6D(self.vertices, rotation) + position.unsqueeze(1)
        

        # Node Interface <Face, Axis>
        self.Node_Face = {}
        Face_Indices = [
            [0, 1, 3, 2],   # Back
            [4, 5, 7, 6],   # Front
            [0, 1, 5, 4],   # Top
            [3, 2, 6, 7],   # Bottom
            [0, 2, 6, 4],   # Left
            [1, 3, 7, 5]    # Right
        ]
        for i, index in enumerate(Face_Indices):
            v_face = self.vertices[:, index, :]
            p = torch.mean(v_face, dim=1)
            t = normalize(v_face[:, 1] - v_face[:, 0])
            edge_side = v_face[:, 2] - v_face[:, 1]
            n = normalize(torch.cross(edge_side, t, dim=-1))
            b = torch.cross(n, t, dim=-1)
            self.Node_Face[i] = {'p': p, 'n': n, 't': t, 'b': b}
            
        self.Node_Axis = {}
        Axis_Indices = [
            [0, 1], [1, 3], [3, 2], [2, 0],
            [4, 5], [5, 7], [7, 6], [6, 4],
            [0, 4], [1, 5],
            [2, 6], [3, 7]
        ]
        for i, index in enumerate(Axis_Indices):
            v0 = self.vertices[:, index[0], :] # [B, 3]
            v1 = self.vertices[:, index[1], :] # [B, 3]
            p = v0
            d = normalize(v1 - v0)
            self.Node_Axis[i] = {'p': p, 'd': d}

        # === 添加 Main Axis (主轴) ===
        # Index: 12
        # 定义: 从下表面中心 (Bottom Center) 指向上表面中心 (Top Center)
        center_top = torch.mean(self.vertices[:, 0:4, :], dim=1) # Top vertices 0,1,2,3
        center_bot = torch.mean(self.vertices[:, 4:8, :], dim=1) # Bottom vertices 4,5,6,7
        d_main = normalize(center_top - center_bot)
        
        self.Node_Axis[len(Axis_Indices)] = {'p': center_bot, 'd': d_main}
        
        super().__init__(position, rotation, rotation_order,
                         Node_Position=self.vertices, Node_Face=self.Node_Face, Node_Axis=self.Node_Axis,
                         Node_Semantic=Semantic, Node_Affordance=Affordance)

    def get_surface_points(self, total_points=600):
        if total_points <= 0:
            return torch.zeros((self.vertices.shape[0], 0, 3), device=self.device, dtype=self.dtype)

        # 1. 定义面的顶点索引
        Face_Indices = [
            [0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
            [3, 2, 6, 7], [0, 2, 6, 4], [1, 3, 7, 5]
        ]
        
        # 2. 计算每个面的面积
        face_areas = []
        for idx in Face_Indices:
            v = self.vertices[:, idx, :] 
            diag1 = v[:, 2, :] - v[:, 0, :]
            diag2 = v[:, 3, :] - v[:, 1, :]
            # cross product 可能产生 NaN，如果 vertices 已经是 NaN
            area = 0.5 * torch.norm(torch.cross(diag1, diag2, dim=-1), dim=-1)
            face_areas.append(area)
        
        areas = torch.stack(face_areas, dim=1) # [B, 6]
        
        # --- 【修复重点 1】: 清洗数据，将 NaN/Inf 变为 0 ---
        # 这样即使前面的 vertices 坏了，这里也会变成 0，进入均匀分配逻辑
        areas = torch.nan_to_num(areas, nan=0.0, posinf=0.0, neginf=0.0)

        # 计算平均面积比例
        mean_areas = areas.mean(dim=0)
        total_mean_area = mean_areas.sum()
        
        # 防止除以 0 (total_mean_area 可能因为 nan_to_num 变成了 0)
        if total_mean_area < 1e-6:
            # 如果所有面面积无效，则平均分配
            num_per_face = torch.full((6,), total_points // 6, dtype=torch.long, device=self.device)
        else:
            probs = mean_areas / total_mean_area
            # 双重保险：防止 probs 里依然有 weird values
            probs = torch.clamp(probs, min=0.0, max=1.0) 
            num_per_face = (probs * total_points).long()
        
        # --- 【修复重点 2】: 强制非负 ---
        # 防止因转换精度问题出现负数
        num_per_face = torch.clamp(num_per_face, min=0)

        # 补齐因取整丢掉的点
        current_sum = num_per_face.sum()
        diff = total_points - current_sum
        
        # --- 【修复重点 3】: 安全的 diff 分配 ---
        # 只有 diff 在合理范围内才进行修正，防止逻辑溢出
        if diff > 0 and diff < total_points:
            # 加给概率最大的那个面，或者默认第一个面
            target_idx = torch.argmax(num_per_face)
            num_per_face[target_idx] += diff
        elif diff < 0:
            # 这种情况极少见，除非 clamp 截断了很大的数
            # 简单粗暴重置为平均分配，防止崩溃
            num_per_face = torch.full((6,), total_points // 6, dtype=torch.long, device=self.device)
            num_per_face[-1] += total_points - num_per_face.sum()

        # 3. 开始采样
        all_surface_points = []
        batch_size = self.vertices.shape[0]
        
        for i, idx in enumerate(Face_Indices):
            N = num_per_face[i].item()
            
            # 再次检查 N 是否合理 (防止爆炸)
            if N <= 0:
                continue
            if N > total_points * 2: # 熔断机制
                N = total_points // 6
            
            v = self.vertices[:, idx, :] 
            v0 = v[:, 0, :].unsqueeze(1) # [B, 1, 3]
            v1 = v[:, 1, :].unsqueeze(1) # [B, 1, 3]
            v3 = v[:, 2, :].unsqueeze(1) # [B, 1, 3]
            v2 = v[:, 3, :].unsqueeze(1) # [B, 1, 3]

            u = torch.rand(1, N, 1, device=self.device, dtype=self.dtype)
            vc = torch.rand(1, N, 1, device=self.device, dtype=self.dtype)

            points = (1 - u) * (1 - vc) * v0 + \
                     u * (1 - vc) * v1 + \
                     (1 - u) * vc * v2 + \
                     u * vc * v3
            
            all_surface_points.append(points)

        # --- 修复点 ---
        if not all_surface_points:
            return torch.zeros((batch_size, 0, 3), device=self.device, dtype=self.dtype)

        return torch.cat(all_surface_points, dim=1)

class Cylinder(StructureNode):
    def __init__(self, height, top_radius, bottom_radius = None, top_radius_z = None, bottom_radius_z = None, is_half = False, is_quarter = False, position = [0, 0, 0], rotation = [1, 0, 0, 0, 1, 0], rotation_order = "XYZ", num_of_segment = 16, Semantic=None, Affordance=None):
        """
        :param height: height of the cylinder in the Y-axis direction (Y-up)
        :param top_radius: radius of the top surface of the cylinder in the X-axis direction
        :param bottom_radius: radius of the bottom surface of the cylinder in the X-axis direction
        :param top_radius_z: radius of the top surface of the cylinder in the Z-axis direction
        :param bottom_radius_z: radius of the bottom surface of the cylinder in the Z-axis direction
        :param is_half: whether the cylinder is half
        :param is_quarter: whether the cylinder is quarter
        :param position: position (x, y, z) of the cylinder
        :param rotation: rotation of the cylinder, represented via Euler angles (x, y, z)
        :param rotation_order: rotation order of the three rotation axes of the cylinder
        """
        dtype = height.dtype
        device = height.device
        B = height.shape[0]

        # Filling Missing Values
        if bottom_radius == None:
            bottom_radius = top_radius
        if top_radius_z == None:
            top_radius_z = top_radius
        if bottom_radius_z == None:
            bottom_radius_z = bottom_radius
            
        if isinstance(position, (list, tuple)):
            position = torch.as_tensor(position, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1)
        if isinstance(rotation, (list, tuple)):
            rotation = torch.as_tensor(rotation, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1)

        # Record Parameters
        self.height = height
        self.top_radius = top_radius
        self.bottom_radius = bottom_radius
        self.top_radius_z = top_radius_z
        self.bottom_radius_z = bottom_radius_z
        self.is_half = is_half
        self.is_quarter = is_quarter
        self.position = position
        self.rotation = rotation
        self.rotation_order = rotation_order
            
        # Manually Defined Default Template Instance 
        angles = torch.arange(0, num_of_segment, dtype=dtype, device=device) * (2 * math.pi / num_of_segment) # [num_of_segment]
        if is_half:
            angles = angles / 2
        elif is_quarter:
            angles = angles / 4
        x_coords = torch.cos(angles) # [num_of_segment]
        z_coords = torch.sin(angles) # [num_of_segment]
        top_center = torch.tensor([0, 0.5, 0], dtype=dtype, device=device).unsqueeze(0) # [1, 3]
        top_template = torch.stack([x_coords, 0.5 * torch.ones_like(x_coords), z_coords], dim=-1) # [num_of_segment, 3]
        bottom_center = torch.tensor([0, -0.5, 0], dtype=dtype, device=device).unsqueeze(0) # [1, 3]
        bottom_template = torch.stack([x_coords, -0.5 * torch.ones_like(x_coords), z_coords], dim=-1) # [num_of_segment, 3]
        self.vertices = torch.cat([top_center, top_template, bottom_center, bottom_template], dim=0) # [2+2*num_of_segment, 3]
        self.vertices = self.vertices.unsqueeze(0).repeat(B, 1, 1) 


        # Differentiable Deformation
        scale_factors_top = torch.stack([self.top_radius, self.height, self.top_radius_z], dim=-1).unsqueeze(1).repeat(1, num_of_segment + 1, 1) # [B, N/2, 3]
        scale_factors_bottom = torch.stack([self.bottom_radius, self.height, self.bottom_radius_z], dim=-1).unsqueeze(1).repeat(1, num_of_segment + 1, 1) # [B, N/2, 3]
        scale_factors = torch.cat([scale_factors_top, scale_factors_bottom], dim=1)
        self.vertices = self.vertices * scale_factors

        # Global Transformation
        self.vertices = rotate_6D(self.vertices, self.rotation) + position.unsqueeze(1)
        
        # TODO: adding is_half and is_quarter
        # just considering full cylinder for now
        
        # Node Interface
        self.Node_Face = {}
        p = []
        n = []
        t = []
        b = []
        
        # 0. Top Face (顶面)
        Top_Face_Center = self.vertices[:, 0]
        p.append(Top_Face_Center)
        t.append(normalize(self.vertices[:, 1] - Top_Face_Center))
        edge_side = self.vertices[:, 2] - Top_Face_Center
        n.append(normalize(torch.cross(t[0], edge_side, dim=-1)))
        b.append(normalize(torch.cross(t[0], n[0], dim=-1)))

        # 1. Bottom Face (底面)
        Bottom_Face_Center = self.vertices[:, num_of_segment + 1]
        p.append(Bottom_Face_Center)
        t.append(normalize(self.vertices[:, num_of_segment + 2] - Bottom_Face_Center))
        edge_side = self.vertices[:, num_of_segment + 3] - Bottom_Face_Center
        n.append(normalize(torch.cross(t[1], edge_side, dim=-1)))
        b.append(normalize(torch.cross(t[1], n[1], dim=-1)))

        # 2. Side Face
        Side_Face_Center_Base = (Top_Face_Center + Bottom_Face_Center) * 0.5
        side_n = normalize(self.vertices[:, 1] - Top_Face_Center)
        avg_radius = (self.top_radius + self.bottom_radius) * 0.5
        side_t = normalize(Top_Face_Center - Bottom_Face_Center) # 切线：沿柱体轴向向上
        side_b = normalize(torch.cross(side_n, side_t, dim=-1))   # 副切线：绕柱体表面的切向
        p.append(Side_Face_Center_Base + side_n * avg_radius.unsqueeze(-1))
        n.append(side_n)
        t.append(side_t)
        b.append(side_b)
        
        for i in range(len(p)):
            self.Node_Face[i] = {'p': p[i], 'n': n[i], 't': t[i], 'b': b[i]}
        
        self.Node_Axis = {}
        p.clear()
        d = []
        
        p.append(Side_Face_Center_Base)
        d.append(normalize(Top_Face_Center - Bottom_Face_Center))
        
        for i in range(len(p)):
            self.Node_Axis[i] = {'p': p[i], 'd': d[i]}
        
        super().__init__(position, rotation, rotation_order,
                         Node_Position=self.vertices, Node_Face=self.Node_Face, Node_Axis=self.Node_Axis,
                         Node_Semantic=Semantic, Node_Affordance=Affordance)
        
    
    def get_surface_points(self, total_points=600):
        """
        在世界坐标系下，根据圆柱/半圆柱/四分之一圆柱的组成部分面积比例均匀采样
        """

        B = self.height.shape[0]
        device = self.height.device
        dtype = self.height.dtype

        # --- 1. 分类计算面积与确定角度 ---
        # 基础面积分量
        circ_top_avg = (self.top_radius + self.top_radius_z) / 2
        circ_bot_avg = (self.bottom_radius + self.bottom_radius_z) / 2
        
        # 初始化变量，防止未定义
        areas = None
        parts_labels = []
        phi = 2 * math.pi # Default

        if self.is_quarter:
            phi = 0.5 * math.pi
            area_side = (phi * (circ_top_avg + circ_bot_avg) / 2) * self.height
            area_top = (phi / 2.0) * self.top_radius * self.top_radius_z
            area_bot = (phi / 2.0) * self.bottom_radius * self.bottom_radius_z
            # 两个半径截面 (X轴一个，Z轴一个)
            area_cut1 = ((self.top_radius + self.bottom_radius) / 2) * self.height
            area_cut2 = ((self.top_radius_z + self.bottom_radius_z) / 2) * self.height
            areas = torch.stack([area_side, area_top, area_bot, area_cut1, area_cut2], dim=1)
            parts_labels = ['side', 'top', 'bot', 'cut_x', 'cut_z']
            
        elif self.is_half:
            phi = math.pi
            area_side = (phi * (circ_top_avg + circ_bot_avg) / 2) * self.height
            area_top = (phi / 2.0) * self.top_radius * self.top_radius_z
            area_bot = (phi / 2.0) * self.bottom_radius * self.bottom_radius_z
            # 一个跨越直径的截面
            area_cut1 = (self.top_radius + self.bottom_radius) * self.height
            areas = torch.stack([area_side, area_top, area_bot, area_cut1], dim=1)
            parts_labels = ['side', 'top', 'bot', 'cut_x']
            
        else: # Full Cylinder
            phi = 2 * math.pi
            area_side = (phi * (circ_top_avg + circ_bot_avg) / 2) * self.height
            area_top = (phi / 2.0) * self.top_radius * self.top_radius_z
            area_bot = (phi / 2.0) * self.bottom_radius * self.bottom_radius_z
            areas = torch.stack([area_side, area_top, area_bot], dim=1)
            parts_labels = ['side', 'top', 'bot']

        # --- 2. 统一分配点数 (关键修复部分) ---
        
        # [安全修复 1]: 清洗 NaN/Inf。如果模型参数崩坏，这里强制变成 0，防止 NaN 传播
        mean_areas = areas.mean(dim=0)
        mean_areas = torch.nan_to_num(mean_areas, nan=0.0, posinf=0.0, neginf=0.0)
        
        total_mean_area = mean_areas.sum()
        
        # [安全修复 2]: 防止除以 0
        if total_mean_area < 1e-6:
            # 如果面积无效，均匀分配
            probs = torch.ones_like(mean_areas) / len(mean_areas)
        else:
            probs = mean_areas / total_mean_area
            # 再次 clamp 确保概率在 [0, 1] 之间
            probs = torch.clamp(probs, min=0.0, max=1.0)
            
        # [安全修复 3]: 计算点数并强制非负
        # NaN 转 Long 会变成巨大负数，clamp(min=0) 是防止 OOM 的关键
        num_parts = (probs * total_points).long()
        num_parts = torch.clamp(num_parts, min=0) 
        
        # 弥补舍入误差 (安全的 Diff 计算)
        current_sum = num_parts.sum()
        diff = total_points - current_sum
        
        # 只有 diff 正常才分配，如果 diff 异常大（说明 num_parts 出了问题），则不分配或重置
        if diff > 0 and diff < total_points:
            num_parts[torch.argmax(probs)] += diff
        elif diff < 0 or diff >= total_points:
            # 熔断机制：如果分配逻辑崩了，回退到均匀分配
            num_parts = torch.full((len(mean_areas),), total_points // len(mean_areas), device=device, dtype=torch.long)
            num_parts[-1] += total_points - num_parts.sum()

        # --- 3. 采样逻辑 ---
        all_points = []
        # 将 num_parts 映射回具体组件
        points_dict = {label: num.item() for label, num in zip(parts_labels, num_parts)}

        # (A) 侧面采样
        N = points_dict.get('side', 0)
        if N > 0:
            # [安全修复 4]: 熔断机制，防止 N 依然过大
            if N > total_points * 2: N = total_points
            
            u, v = torch.rand(1, N, 1, device=device, dtype=dtype), torch.rand(1, N, 1, device=device, dtype=dtype)
            theta = u * phi
            curr_rx = (1 - v) * self.bottom_radius.view(B, 1, 1) + v * self.top_radius.view(B, 1, 1)
            curr_rz = (1 - v) * self.bottom_radius_z.view(B, 1, 1) + v * self.top_radius_z.view(B, 1, 1)
            px = curr_rx * torch.cos(theta)
            py = (v - 0.5) * self.height.view(B, 1, 1)
            pz = curr_rz * torch.sin(theta)
            all_points.append(torch.cat([px, py, pz], dim=-1))

        # (B) 上底面采样
        N = points_dict.get('top', 0)
        if N > 0:
            if N > total_points * 2: N = total_points
            r = torch.sqrt(torch.rand(1, N, 1, device=device, dtype=dtype))
            theta = torch.rand(1, N, 1, device=device, dtype=dtype) * phi
            px = r * self.top_radius.view(B, 1, 1) * torch.cos(theta)
            py = 0.5 * self.height.view(B, 1, 1).expand(B, N, 1)
            pz = r * self.top_radius_z.view(B, 1, 1) * torch.sin(theta)
            all_points.append(torch.cat([px, py, pz], dim=-1))

        # (C) 下底面采样
        N = points_dict.get('bot', 0)
        if N > 0:
            if N > total_points * 2: N = total_points
            r = torch.sqrt(torch.rand(1, N, 1, device=device, dtype=dtype))
            theta = torch.rand(1, N, 1, device=device, dtype=dtype) * phi
            px = r * self.bottom_radius.view(B, 1, 1) * torch.cos(theta)
            py = -0.5 * self.height.view(B, 1, 1).expand(B, N, 1)
            pz = r * self.bottom_radius_z.view(B, 1, 1) * torch.sin(theta)
            all_points.append(torch.cat([px, py, pz], dim=-1))

        # (D) 截面 X (theta=0 平面)
        N = points_dict.get('cut_x', 0)
        if N > 0:
            if N > total_points * 2: N = total_points
            u, v = torch.rand(1, N, 1, device=device, dtype=dtype), torch.rand(1, N, 1, device=device, dtype=dtype)
            curr_rx = (1 - v) * self.bottom_radius.view(B, 1, 1) + v * self.top_radius.view(B, 1, 1)
            if self.is_half:
                # Half cylinder cut: diameter from -R to R
                px = (u * 2 - 1) * curr_rx 
            else: 
                # Quarter cylinder cut: radius from 0 to R
                px = u * curr_rx
            py = (v - 0.5) * self.height.view(B, 1, 1)
            pz = torch.zeros(B, N, 1, device=device, dtype=dtype)
            all_points.append(torch.cat([px, py, pz], dim=-1))

        # (E) 截面 Z (仅 Quarter 的 theta=phi 平面)
        N = points_dict.get('cut_z', 0)
        if N > 0:
            if N > total_points * 2: N = total_points
            u, v = torch.rand(1, N, 1, device=device, dtype=dtype), torch.rand(1, N, 1, device=device, dtype=dtype)
            curr_rz = (1 - v) * self.bottom_radius_z.view(B, 1, 1) + v * self.top_radius_z.view(B, 1, 1)
            px = torch.zeros(B, N, 1, device=device, dtype=dtype)
            py = (v - 0.5) * self.height.view(B, 1, 1)
            pz = u * curr_rz
            all_points.append(torch.cat([px, py, pz], dim=-1))

        # --- 4. 汇总与世界坐标变换 ---
        if not all_points:
            # 防止没有任何点生成的情况
            return torch.zeros((B, 0, 3), device=device, dtype=dtype)

        local_points = torch.cat(all_points, dim=1) # [B, total_points, 3]
        
        # 旋转和平移
        world_points = rotate_6D(local_points, self.rotation) + self.position.unsqueeze(1)
        
        return world_points
    
class Sphere(StructureNode):
    def __init__(self, radius, top_angle = 0, bottom_angle = math.pi, radius_y = None, radius_z = None, longitude_angle = math.pi*2, position = [0, 0, 0], rotation = [1, 0, 0, 0, 1, 0], rotation_order = "XYZ", Semantic=None, Affordance=None):
        """
        :param radius: length of the half-axis of the sphere in the X-axis direction
        :param top_angle: latitude starting angle of the sphere
        :param bottom_angle: latitude ending angle of the sphere
        :param radius_y: length of the half-axis of the sphere in the Y-axis direction
        :param radius_z: length of the half-axis of the sphere in the Z-axis direction
        :param longitude_angle: longitude covered angle of the sphere
        :param position: position (x, y, z) of the sphere
        :param rotation: rotation of the sphere, represented via Euler angles (x, y, z)
        :param rotation_order: rotation order of the three rotation axes of the sphere
        """

        self.dtype = radius.dtype
        self.device = radius.device
        self.B = radius.shape[0]

        # Filling Missing Values
        if radius_y == None:
            radius_y = radius
        if radius_z == None:
            radius_z = radius
        if isinstance(top_angle, (int, float)):
            top_angle = torch.as_tensor(top_angle, dtype=self.dtype, device=self.device).unsqueeze(0).repeat(self.B, 1)
        if isinstance(bottom_angle, (int, float)):
            bottom_angle = torch.as_tensor(bottom_angle, dtype=self.dtype, device=self.device).unsqueeze(0).repeat(self.B, 1)
        if isinstance(longitude_angle, (int, float)):
            longitude_angle = torch.as_tensor(longitude_angle, dtype=self.dtype, device=self.device).unsqueeze(0).repeat(self.B, 1)
        if isinstance(position, (list, tuple)):
            position = torch.as_tensor(position, dtype=self.dtype, device=self.device).unsqueeze(0).repeat(self.B, 1)
        if isinstance(rotation, (list, tuple)):
            rotation = torch.as_tensor(rotation, dtype=self.dtype, device=self.device).unsqueeze(0).repeat(self.B, 1)

        # Record Parameters
        self.radius = radius
        self.top_angle = top_angle
        self.bottom_angle = bottom_angle
        self.radius_y = radius_y
        self.radius_z = radius_z
        self.longitude_angle = longitude_angle
        self.position = position
        self.rotation = rotation
        self.rotation_order = rotation_order

        n_lat = 8
        n_lon = 8
        # 1. 创建参数网格 (u, v) -> (0..1)
        # u 对应纬度 (top -> bottom), v 对应经度 (0 -> longitude_angle)
        u = torch.linspace(0, 1, n_lat + 1, device=self.device)
        v = torch.linspace(0, 1, n_lon + 1, device=self.device)

        # 2. 扩展维度以支持广播 (1, N_lat, 1) 和 (1, 1, N_lon)
        u = u.view(1, -1, 1)
        v = v.view(1, 1, -1)
        
        # 3. 计算实际角度 (B, N_lat+1, N_lon+1)
        # theta (latitude angle): shape (B, N_lat+1, 1)
        theta_start = self.top_angle.view(self.B, 1, 1)
        theta_end = self.bottom_angle.view(self.B, 1, 1)
        theta = theta_start + (theta_end - theta_start) * u
        
        # phi (longitude angle): shape (B, 1, N_lon+1)
        phi_total = self.longitude_angle.view(self.B, 1, 1)
        phi = phi_total * v
        
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)
        
        r_x = self.radius.view(self.B, 1, 1)
        r_y = self.radius_y.view(self.B, 1, 1)
        r_z = self.radius_z.view(self.B, 1, 1)
        
        x = r_x * sin_theta * cos_phi   # shape (B, N_lat+1, N_lon+1)
        y = r_y * cos_theta             # shape (B, N_lat+1, 1)
        y = y.expand(-1, -1, n_lon+1)   # shape (B, N_lat+1, N_lon+1)
        z = r_z * sin_theta * sin_phi   # shape (B, N_lat+1, N_lon+1)
        
        grid_vertices = torch.stack([x, y, z], dim=-1)      # shape (B, N_lat+1, 3 * N_lon+1)
        grid_vertices_flat = grid_vertices.view(self.B, -1, 3)   # shape (B, (N_lat+1)*(N_lon+1), 3)
        
        zeros = torch.zeros((self.B, 1), device=self.device)
        
        top_y = r_y.view(self.B, 1) * torch.cos(self.top_angle.view(self.B, 1))
        top_center = torch.cat([zeros, top_y, zeros], dim=1).unsqueeze(1) # (B, 1, 3)
        bottom_y = r_y.view(self.B, 1) * torch.cos(self.bottom_angle.view(self.B, 1))
        bottom_center = torch.cat([zeros, bottom_y, zeros], dim=1).unsqueeze(1) # (B, 1, 3)
        
        vertices = torch.cat([grid_vertices_flat, top_center, bottom_center], dim=1)
        
        self.vertices = rotate_6D(vertices, rotation) + position.unsqueeze(1)
        
        # Node Interface <Face, Axis>
        def make_vec(x, y, z):
            if not torch.is_tensor(x): x = torch.full((self.B,), x, dtype=self.dtype, device=self.device)
            if not torch.is_tensor(y): y = torch.full((self.B,), y, dtype=self.dtype, device=self.device)
            if not torch.is_tensor(z): z = torch.full((self.B,), z, dtype=self.dtype, device=self.device)
            return torch.stack([x, y, z], dim=1)
        
        middle_angle = (self.top_angle + self.bottom_angle) / 2 # (B,)
        p, n, t, b = [], [], [], []
        self.Node_Face = {}
        
        # X-axis +
        p.append(make_vec(self.radius, 0, 0))
        n.append(make_vec(1, 0, 0))
        t.append(make_vec(0, 1, 0))
        b.append(make_vec(0, 0, 1))
        
        # X-axis -
        p.append(make_vec(-self.radius, 0, 0))
        n.append(make_vec(-1, 0, 0))
        t.append(make_vec(0, -1, 0))
        b.append(make_vec(0, 0, -1))
        
        # Y-axis +
        p.append(make_vec(0, self.radius_y, 0))
        n.append(make_vec(0, 1, 0))
        t.append(make_vec(0, 0, 1))
        b.append(make_vec(1, 0, 0))
        
        # Y-axis -
        p.append(make_vec(0, -self.radius_y, 0))
        n.append(make_vec(0, -1, 0))
        t.append(make_vec(0, 0, -1))
        b.append(make_vec(-1, 0, 0))
        
        # Z-axis +
        p.append(make_vec(0, 0, self.radius_z))
        n.append(make_vec(0, 0, 1))
        t.append(make_vec(1, 0, 0))
        b.append(make_vec(0, 1, 0))
        
        # Z-axis -
        p.append(make_vec(0, 0, -self.radius_z))
        n.append(make_vec(0, 0, -1))
        t.append(make_vec(-1, 0, 0))
        b.append(make_vec(0, -1, 0))
        
        for i in range(len(p)):
            p[i] = rotate_6D(p[i], rotation).view(self.B, 3) + position
            n[i] = rotate_6D(n[i], rotation).view(self.B, 3)
            t[i] = rotate_6D(t[i], rotation).view(self.B, 3)
            b[i] = rotate_6D(b[i], rotation).view(self.B, 3)
            self.Node_Face[i] = {'p': p[i], 'n': n[i], 't': t[i], 'b': b[i]}
            
        self.Node_Axis = {}
        p.clear()
        d = []
        
        # X-axis
        p.append(position)
        d.append(make_vec(1, 0, 0))
        
        # Y-axis
        p.append(position)
        d.append(make_vec(0, 1, 0))
        
        # Z-axis
        p.append(position)
        d.append(make_vec(0, 0, 1))
        
        for i in range(len(p)):
            d[i] = rotate_6D(d[i], rotation).view(self.B, 3)
            self.Node_Axis[i] = {'p': p[i], 'd': d[i]}
            
        super().__init__(self.position, self.rotation, self.rotation_order, \
            self.vertices, Semantic, Affordance, \
            self.Node_Face, self.Node_Axis)
        
    def get_surface_points(self, total_points=600):
        """
        Sample points on the surface of the sphere/ellipsoid.
        Logic follows the parametric equation used in __init__.
        
        :param total_points: Number of points to sample per sphere in the batch
        :return: Tensor of shape (B, total_points, 3)
        """
        # 1. 在参数空间 [0, 1] 内生成随机采样点 u, v
        # u 控制纬度 (top -> bottom), v 控制经度 (0 -> longitude)
        # Shape: (B, total_points)
        u = torch.rand((self.B, total_points), dtype=self.dtype, device=self.device)
        v = torch.rand((self.B, total_points), dtype=self.dtype, device=self.device)

        # 2. 将 u, v 映射到实际的角度 theta, phi
        # Theta (latitude): 线性插值 top_angle 到 bottom_angle
        theta_start = self.top_angle.view(self.B, 1)
        theta_end = self.bottom_angle.view(self.B, 1)
        theta = theta_start + (theta_end - theta_start) * u  # (B, N)

        # Phi (longitude): 线性插值 0 到 longitude_angle
        phi_total = self.longitude_angle.view(self.B, 1)
        phi = phi_total * v  # (B, N)

        # 3. 计算三角函数
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)

        # 4. 获取半径并广播
        r_x = self.radius.view(self.B, 1)
        r_y = self.radius_y.view(self.B, 1)
        r_z = self.radius_z.view(self.B, 1)

        # 5. 应用参数方程 (与 __init__ 逻辑完全一致)
        # x = rx * sin(theta) * cos(phi)
        # y = ry * cos(theta)
        # z = rz * sin(theta) * sin(phi)
        x = r_x * sin_theta * cos_phi
        y = r_y * cos_theta
        z = r_z * sin_theta * sin_phi

        # 组合坐标 (B, total_points, 3)
        points = torch.stack([x, y, z], dim=-1)

        # 6. 全局变换 (旋转 + 平移)
        # 假设 rotate_6D 是上下文中可用的函数，如同 __init__ 中使用的一样
        # points: (B, N, 3)
        # rotation: (B, 6) or (B, 3) depending on implementation, matches __init__
        points = rotate_6D(points, self.rotation) + self.position.unsqueeze(1)

        return points

class Rectangular_Ring(StructureNode):
    def __init__(self, front_height, outer_top_length, outer_top_width, 
                 inner_top_length, inner_top_width, inner_offset=[0, 0], 
                 outer_bottom_length=None, outer_bottom_width=None, 
                 inner_bottom_length=None, inner_bottom_width=None, 
                 back_height=None, top_bottom_offset=[0, 0], 
                 position=[0, 0, 0], rotation=[1, 0, 0, 0, 1, 0], rotation_order="XYZ", 
                 Semantic=None, Affordance=None):
        """
        全参数化矩形环/梯形空心柱
        """
        # -----------------------------------------------------------
        # 1. 参数标准化与广播 (Input Normalization)
        # -----------------------------------------------------------
        dtype = front_height.dtype
        device = front_height.device
        B = front_height.shape[0]

        # 辅助函数：转 Tensor 并广播
        def to_tensor_b1(val):
            if not torch.is_tensor(val): val = torch.tensor(val, dtype=dtype, device=device)
            if val.ndim == 0: val = val.unsqueeze(0).repeat(B, 1)
            elif val.ndim == 1 and val.shape[0] == B: val = val.unsqueeze(1)
            return val # [B, 1]

        def to_tensor_b2(val):
            val = torch.as_tensor(val, dtype=dtype, device=device)
            if val.ndim == 1: val = val.unsqueeze(0).repeat(B, 1)
            return val # [B, 2]

        # 必需参数
        front_height = to_tensor_b1(front_height)
        outer_top_len = to_tensor_b1(outer_top_length)
        outer_top_wid = to_tensor_b1(outer_top_width)
        inner_top_len = to_tensor_b1(inner_top_length)
        inner_top_wid = to_tensor_b1(inner_top_width)
        
        # 可选参数默认值填充
        if back_height is None: back_height = front_height
        else: back_height = to_tensor_b1(back_height)

        if outer_bottom_length is None: outer_bot_len = outer_top_len
        else: outer_bot_len = to_tensor_b1(outer_bottom_length)
        
        if outer_bottom_width is None: outer_bot_wid = outer_top_wid
        else: outer_bot_wid = to_tensor_b1(outer_bottom_width)
        
        if inner_bottom_length is None: inner_bot_len = inner_top_len
        else: inner_bot_len = to_tensor_b1(inner_bottom_length)
        
        if inner_bottom_width is None: inner_bot_wid = inner_top_wid
        else: inner_bot_wid = to_tensor_b1(inner_bottom_width)

        inner_offset = to_tensor_b2(inner_offset)         # [B, 2] (x, z)
        top_bottom_offset = to_tensor_b2(top_bottom_offset) # [B, 2] (x, z)
        
        # 记录参数
        self.dtype = dtype
        self.device = device
        self.B = B
        self.front_height = front_height
        self.back_height = back_height
        self.position = to_tensor_b2(position) # 注意这里如果是3D position init会被覆盖，但在StructureNode里通常是[B,3]
        if isinstance(position, (list, tuple)) or torch.is_tensor(position):
             # 重新处理 position 为 [B, 3] 以防万一
             pos_t = torch.as_tensor(position, dtype=dtype, device=device)
             if pos_t.ndim == 1: pos_t = pos_t.unsqueeze(0).repeat(B, 1)
             self.position = pos_t
        
        if isinstance(rotation, (list, tuple)) or torch.is_tensor(rotation):
             rot_t = torch.as_tensor(rotation, dtype=dtype, device=device)
             if rot_t.ndim == 1: rot_t = rot_t.unsqueeze(0).repeat(B, 1)
             self.rotation = rot_t

        # -----------------------------------------------------------
        # 2. 向量化生成顶点 (Vectorized Vertex Generation)
        # -----------------------------------------------------------
        # 象限符号: (+,+), (-,+), (-,-), (+,-) 对应索引 0,1,2,3
        # 几何意义: 0:FrontRight, 1:FrontLeft, 2:BackLeft, 3:BackRight
        # Z轴: +Z is Front, -Z is Back
        signs = torch.tensor([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=dtype, device=device) # [4, 2]

        # --- Y 坐标计算 ---
        # 基准面 Y = -front_height / 2 (使物体大致垂直居中)
        y_base = -front_height / 2.0
        
        # 底部 Y (Flat)
        y_bot = y_base.expand(B, 4).unsqueeze(-1) # [B, 4, 1]
        
        # 顶部 Y (Slanted)
        # Indices 0, 1 (Z > 0): Front Height
        # Indices 2, 3 (Z < 0): Back Height
        y_top_front = y_base + front_height
        y_top_back  = y_base + back_height
        # 拼接 [Front, Front, Back, Back]
        y_top = torch.cat([y_top_front, y_top_front, y_top_back, y_top_back], dim=1).unsqueeze(-1) # [B, 4, 1]

        # --- XZ 坐标计算 ---
        
        # Group 1: Top Outer (Indices 0-3)
        # Size = outer_top / 2
        dim_to = torch.cat([outer_top_len, outer_top_wid], dim=1).unsqueeze(1) / 2.0 # [B, 1, 2]
        xz_to = dim_to * signs.unsqueeze(0) + top_bottom_offset.unsqueeze(1) # [B, 4, 2]
        v_to = torch.cat([xz_to[:,:,0:1], y_top, xz_to[:,:,1:2]], dim=-1)

        # Group 2: Top Inner (Indices 4-7)
        # Size = inner_top / 2, Center = top_bottom_offset + inner_offset
        dim_ti = torch.cat([inner_top_len, inner_top_wid], dim=1).unsqueeze(1) / 2.0
        xz_ti = dim_ti * signs.unsqueeze(0) + top_bottom_offset.unsqueeze(1) + inner_offset.unsqueeze(1)
        v_ti = torch.cat([xz_ti[:,:,0:1], y_top, xz_ti[:,:,1:2]], dim=-1)

        # Group 3: Bottom Outer (Indices 8-11)
        # Size = outer_bot / 2
        dim_bo = torch.cat([outer_bot_len, outer_bot_wid], dim=1).unsqueeze(1) / 2.0
        xz_bo = dim_bo * signs.unsqueeze(0) # Base center is local (0,0)
        v_bo = torch.cat([xz_bo[:,:,0:1], y_bot, xz_bo[:,:,1:2]], dim=-1)

        # Group 4: Bottom Inner (Indices 12-15)
        # Size = inner_bot / 2, Center = inner_offset
        dim_bi = torch.cat([inner_bot_len, inner_bot_wid], dim=1).unsqueeze(1) / 2.0
        xz_bi = dim_bi * signs.unsqueeze(0) + inner_offset.unsqueeze(1)
        v_bi = torch.cat([xz_bi[:,:,0:1], y_bot, xz_bi[:,:,1:2]], dim=-1)

        # 合并: [B, 16, 3]
        self.vertices = torch.cat([v_to, v_ti, v_bo, v_bi], dim=1)

        # -----------------------------------------------------------
        # 3. 全局变换 (Global Transform)
        # -----------------------------------------------------------
        self.vertices = rotate_6D(self.vertices, self.rotation) + self.position.unsqueeze(1)

        # -----------------------------------------------------------
        # 4. 拓扑定义 (Faces & Axes)
        # -----------------------------------------------------------
        self.Node_Face = {}
        self.Node_Axis = {}

        # === Axes: 24 Edges + 1 Main ===
        
        # Axis 0: Main Axis (Bottom Center -> Top Center)
        # 计算几何中心
        c_top = torch.mean(self.vertices[:, 0:4], dim=1)
        c_bot = torch.mean(self.vertices[:, 8:12], dim=1)
        d_main = normalize(c_top - c_bot)
        self.Node_Axis[0] = {'p': c_bot, 'd': d_main}
        
        axis_count = 1
        
        # --- Group A: Vertical Outer Edges (8->0, 9->1, ...) [4条] ---
        for i in range(4):
            p_start = self.vertices[:, 8 + i, :] # Bot Outer
            p_end   = self.vertices[:, 0 + i, :] # Top Outer
            self.Node_Axis[axis_count] = {'p': p_start, 'd': normalize(p_end - p_start)}
            axis_count += 1
            
        # --- Group B: Vertical Inner Edges (12->4, 13->5, ...) [4条] ---
        for i in range(4):
            p_start = self.vertices[:, 12 + i, :] # Bot Inner
            p_end   = self.vertices[:, 4 + i, :]  # Top Inner
            self.Node_Axis[axis_count] = {'p': p_start, 'd': normalize(p_end - p_start)}
            axis_count += 1

        # --- Group C: Top Outer Edges (0->1, 1->2, ...) [4条] ---
        for i in range(4):
            idx_curr = 0 + i
            idx_next = 0 + (i + 1) % 4
            p_start = self.vertices[:, idx_curr, :]
            p_end   = self.vertices[:, idx_next, :]
            self.Node_Axis[axis_count] = {'p': p_start, 'd': normalize(p_end - p_start)}
            axis_count += 1
            
        # --- Group D: Top Inner Edges (4->5, 5->6, ...) [4条] ---
        for i in range(4):
            idx_curr = 4 + i
            idx_next = 4 + (i + 1) % 4
            p_start = self.vertices[:, idx_curr, :]
            p_end   = self.vertices[:, idx_next, :]
            self.Node_Axis[axis_count] = {'p': p_start, 'd': normalize(p_end - p_start)}
            axis_count += 1

        # --- Group E: Bottom Outer Edges (8->9, 9->10, ...) [4条] ---
        for i in range(4):
            idx_curr = 8 + i
            idx_next = 8 + (i + 1) % 4
            p_start = self.vertices[:, idx_curr, :]
            p_end   = self.vertices[:, idx_next, :]
            self.Node_Axis[axis_count] = {'p': p_start, 'd': normalize(p_end - p_start)}
            axis_count += 1
            
        # --- Group F: Bottom Inner Edges (12->13, 13->14, ...) [4条] ---
        for i in range(4):
            idx_curr = 12 + i
            idx_next = 12 + (i + 1) % 4
            p_start = self.vertices[:, idx_curr, :]
            p_end   = self.vertices[:, idx_next, :]
            self.Node_Axis[axis_count] = {'p': p_start, 'd': normalize(p_end - p_start)}
            axis_count += 1

        # =========================================================
        # 定义 16 个面 (4 外侧 + 4 内侧 + 4 上底 + 4 下底)
        # =========================================================
        
        # 辅助索引映射 (根据 init 中的顶点生成顺序)
        # TopOuter: 0,1,2,3 | TopInner: 4,5,6,7
        # BotOuter: 8,9,10,11 | BotInner: 12,13,14,15
        
        # --- Group 1: 外侧面 (Outer Vertical Faces) [Faces 0-3] ---
        # 顺序: Front(0), Left(1), Back(2), Right(3) (对应索引遍历 i=0..3)
        # 构成: TopOuter[i], TopOuter[i+1], BotOuter[i+1], BotOuter[i]
        for i in range(4):
            idx_tl = i
            idx_tr = (i + 1) % 4
            idx_br = 8 + (i + 1) % 4
            idx_bl = 8 + i
            
            # 几何中心
            p_center = (self.vertices[:, idx_tl] + self.vertices[:, idx_br]) / 2.0
            
            # Frame 计算
            # Tangent: 沿水平边 TopLeft -> TopRight
            t = normalize(self.vertices[:, idx_tr] - self.vertices[:, idx_tl])
            # Down: 沿垂直边 TopLeft -> BotLeft
            down = self.vertices[:, idx_bl] - self.vertices[:, idx_tl]
            # Normal: Down x Tangent -> 指向外 (例如 Front面: -Y x -X = Z+)
            n = normalize(torch.cross(down, t, dim=-1))
            b = torch.cross(n, t, dim=-1)
            
            self.Node_Face[i] = {'p': p_center, 'n': n, 't': t, 'b': b}

        # --- Group 2: 内侧面 (Inner Vertical Faces) [Faces 4-7] ---
        # 顺序: Front, Left, Back, Right (对应索引遍历 i=0..3)
        # 构成: TopInner[i], TopInner[i+1], BotInner[i+1], BotInner[i]
        for i in range(4):
            idx_tl = 4 + i
            idx_tr = 4 + (i + 1) % 4
            idx_br = 12 + (i + 1) % 4
            idx_bl = 12 + i
            
            p_center = (self.vertices[:, idx_tl] + self.vertices[:, idx_br]) / 2.0
            
            t = normalize(self.vertices[:, idx_tr] - self.vertices[:, idx_tl])
            down = self.vertices[:, idx_bl] - self.vertices[:, idx_tl]
            # Normal: Down x Tangent
            # 对于内表面，这个法线指向几何中心（即指向空心部分）。
            # 在物理中，这通常被视为内表面的"外法线" (pointing into the hole)。
            n = normalize(torch.cross(down, t, dim=-1))
            b = torch.cross(n, t, dim=-1)
            
            self.Node_Face[4 + i] = {'p': p_center, 'n': n, 't': t, 'b': b}

        # --- Group 3: 上底面 (Top Rim Faces) [Faces 8-11] ---
        # 构成梯形: Outer[i], Outer[i+1], Inner[i+1], Inner[i]
        for i in range(4):
            idx_out_curr = i
            idx_out_next = (i + 1) % 4
            idx_in_next  = 4 + (i + 1) % 4
            idx_in_curr  = 4 + i
            
            # 中心
            p_center = (self.vertices[:, idx_out_curr] + self.vertices[:, idx_in_next]) / 2.0
            
            # Tangent: 沿外圈边 (Outer_Curr -> Outer_Next)
            t = normalize(self.vertices[:, idx_out_next] - self.vertices[:, idx_out_curr])
            # Radial In: Outer -> Inner
            radial_in = self.vertices[:, idx_in_curr] - self.vertices[:, idx_out_curr]
            
            # Normal: Radial_In x Tangent
            # 验证: Front Face (i=0). Out(+,+) -> In(+,+ small). Radial is Inward/Back (-Z,-X). Tangent is Left (-X).
            # (-Z) x (-X) = Y+. (正确，指向上)
            n = normalize(torch.cross(radial_in, t, dim=-1))
            b = torch.cross(n, t, dim=-1)
            
            self.Node_Face[8 + i] = {'p': p_center, 'n': n, 't': t, 'b': b}

        # --- Group 4: 下底面 (Bottom Rim Faces) [Faces 12-15] ---
        # 构成梯形: Outer[i], Outer[i+1], Inner[i+1], Inner[i] (Indices +8)
        for i in range(4):
            idx_out_curr = 8 + i
            idx_out_next = 8 + (i + 1) % 4
            idx_in_next  = 12 + (i + 1) % 4
            idx_in_curr  = 12 + i
            
            p_center = (self.vertices[:, idx_out_curr] + self.vertices[:, idx_in_next]) / 2.0
            
            t = normalize(self.vertices[:, idx_out_next] - self.vertices[:, idx_out_curr])
            radial_in = self.vertices[:, idx_in_curr] - self.vertices[:, idx_out_curr]
            
            # Normal: Tangent x Radial_In
            # 验证: 叉乘顺序与 Top 相反，使法线向下 (Y-)
            n = normalize(torch.cross(t, radial_in, dim=-1))
            b = torch.cross(n, t, dim=-1)
            
            self.Node_Face[12 + i] = {'p': p_center, 'n': n, 't': t, 'b': b}

        super().__init__(self.position, self.rotation, rotation_order,
                         Node_Position=self.vertices, Node_Face=self.Node_Face, Node_Axis=self.Node_Axis,
                         Node_Semantic=Semantic, Node_Affordance=Affordance)
        
    def get_surface_points(self, total_points=600):
        """
        在矩形环/梯形柱表面采样点
        适配 16 顶点结构的 Rectangular_Ring
        """
        if total_points <= 0:
            return torch.zeros((self.front_height.shape[0], 0, 3), device=self.device, dtype=self.dtype)
        
        B = self.front_height.shape[0]

        # ---------------------------------------------------------------------
        # 1. 定义面的顶点索引 (Face Indices)
        # ---------------------------------------------------------------------
        # 顶点映射回顾 (来自 init):
        # TopOuter: 0(FR), 1(FL), 2(BL), 3(BR)
        # TopInner: 4(FR), 5(FL), 6(BL), 7(BR)
        # BotOuter: 8(FR), 9(FL), 10(BL), 11(BR)
        # BotInner: 12(FR), 13(FL), 14(BL), 15(BR)
        
        # 为了配合双线性插值公式: P = (1-u)(1-v)v0 + u(1-v)v1 + (1-u)v*v2 + uv*v3
        # 索引顺序应为: [TopLeft(v0), TopRight(v1), BottomRight(v3), BottomLeft(v2)]
        # 注意: 这里的 BottomRight 对应代码中的 index 2, BottomLeft 对应 index 3
        
        Face_Indices = [
            # --- Outer Vertical Faces (4) ---
            [1, 0, 8, 9],   # Front Outer (Z+): FL->FR->BR->BL
            [0, 3, 11, 8],  # Right Outer (X+): FR->BR->BR->FR
            [3, 2, 10, 11], # Back Outer  (Z-): BR->BL->BL->BR
            [2, 1, 9, 10],  # Left Outer  (X-): BL->FL->FL->BL
            
            # --- Inner Vertical Faces (4) ---
            [5, 4, 12, 13], # Front Inner
            [4, 7, 15, 12], # Right Inner
            [7, 6, 14, 15], # Back Inner
            [6, 5, 13, 14], # Left Inner

            # --- Top Ring Faces (4 Trapezoids) ---
            [1, 0, 4, 5],   # Top Front: OutFL->OutFR->InFR->InFL
            [0, 3, 7, 4],   # Top Right
            [3, 2, 6, 7],   # Top Back
            [2, 1, 5, 6],   # Top Left

            # --- Bottom Ring Faces (4 Trapezoids) ---
            [9, 8, 12, 13], # Bot Front: OutFL->OutFR->InFR->InFL
            [8, 11, 15, 12],# Bot Right
            [11, 10, 14, 15],# Bot Back
            [10, 9, 13, 14] # Bot Left
        ]
        
        # 转为 Tensor 以便进行高级索引: [16, 4]
        indices_tensor = torch.tensor(Face_Indices, device=self.device, dtype=torch.long)
        
        # ---------------------------------------------------------------------
        # 2. 向量化计算面积 (Vectorized Area Calculation)
        # ---------------------------------------------------------------------
        # self.vertices: [B, 16, 3]
        # all_faces: [B, 16, 4, 3]
        all_faces_verts = self.vertices[:, indices_tensor, :]
        
        # 计算对角线叉积的一半作为四边形面积近似
        # Diag1: v2 - v0 (BotRight - TopLeft)
        # Diag2: v3 - v1 (BotLeft - TopRight)
        # 注意：这里的 v0..v3 对应 indices_tensor 的第 0..3 列
        # 我们的顺序是 [TL, TR, BR, BL] -> 0, 1, 2, 3
        # diag1 = BR(2) - TL(0)
        # diag2 = BL(3) - TR(1)
        diag1 = all_faces_verts[:, :, 2, :] - all_faces_verts[:, :, 0, :]
        diag2 = all_faces_verts[:, :, 3, :] - all_faces_verts[:, :, 1, :]
        
        # [B, 16]
        areas = 0.5 * torch.norm(torch.cross(diag1, diag2, dim=-1), dim=-1)
        
        # 清洗数据
        areas = torch.nan_to_num(areas, nan=0.0, posinf=0.0, neginf=0.0)

        # ---------------------------------------------------------------------
        # 3. 分配点数 (Point Allocation)
        # ---------------------------------------------------------------------
        mean_areas = areas.mean(dim=0) # [16]
        total_mean_area = mean_areas.sum()
        
        if total_mean_area < 1e-6:
            # 面积过小，均匀分配
            num_per_face = torch.full((16,), total_points // 16, dtype=torch.long, device=self.device)
            # 补齐余数
            num_per_face[-1] += total_points - num_per_face.sum()
        else:
            probs = mean_areas / total_mean_area
            probs = torch.clamp(probs, min=0.0, max=1.0)
            num_per_face = (probs * total_points).long()
            
            # 修正舍入误差
            diff = total_points - num_per_face.sum()
            if diff > 0:
                # 加在面积最大的面上
                num_per_face[torch.argmax(num_per_face)] += diff
            elif diff < 0:
                # 理论上不应发生，但作为保险
                num_per_face = torch.full((16,), total_points // 16, dtype=torch.long, device=self.device)
                num_per_face[-1] += total_points - num_per_face.sum()
        
        # ---------------------------------------------------------------------
        # 4. 采样 (Sampling)
        # ---------------------------------------------------------------------
        all_surface_points = []
        
        # 由于每个面点数不同，这里依然需要循环，但每个循环内是 batch 操作
        for i in range(16):
            N = num_per_face[i].item()
            if N <= 0:
                continue
                
            # 限制单次采样上限，防止显存爆炸
            if N > total_points * 2: 
                N = total_points // 16
            
            # 取当前面的顶点 [B, 4, 3]
            # 顺序: TL(0), TR(1), BR(2), BL(3)
            verts = all_faces_verts[:, i, :, :] 
            
            v0 = verts[:, 0:1, :] # TL
            v1 = verts[:, 1:2, :] # TR
            v3 = verts[:, 2:3, :] # BR (对应公式中的 uv项)
            v2 = verts[:, 3:4, :] # BL (对应公式中的 (1-u)v项)

            # 生成随机参数 u, v [B, N, 1]
            u = torch.rand(B, N, 1, device=self.device, dtype=self.dtype)
            vc = torch.rand(B, N, 1, device=self.device, dtype=self.dtype)

            # 双线性插值
            # P = (1-u)(1-v)*TL + u(1-v)*TR + (1-u)v*BL + uv*BR
            points = (1 - u) * (1 - vc) * v0 + \
                     u       * (1 - vc) * v1 + \
                     (1 - u) * vc       * v2 + \
                     u       * vc       * v3
            
            all_surface_points.append(points)

        if not all_surface_points:
            return torch.zeros((B, 0, 3), device=self.device, dtype=self.dtype)

        return torch.cat(all_surface_points, dim=1)

class Trianguler_Prism(StructureNode):
    def __init__(self, height, top_radius, bottom_radius=None, position=[0, 0, 0], 
                 rotation=[1, 0, 0, 0, 1, 0], rotation_order="XYZ", Semantic=None, Affordance=None):
        """
        基于标准几何模板的高效三棱柱构建
        :param height: [B, 1] 高度
        :param top_radius: [B, 1] 上底半径
        :param bottom_radius: [B, 1] 下底半径
        """
        # -----------------------------------------------------------
        # 1. 参数预处理 (Input Standardization)
        # -----------------------------------------------------------
        dtype = height.dtype
        device = height.device
        B = height.shape[0]
        
        # 填充默认值
        if bottom_radius is None:
            bottom_radius = top_radius
            
        # 确保输入维度正确 [B, 1] 或 [B, 3/6]
        if isinstance(position, (list, tuple)):
            position = torch.as_tensor(position, dtype=dtype, device=device)
        if position.ndim == 1: position = position.unsqueeze(0).repeat(B, 1)
        
        if isinstance(rotation, (list, tuple)):
            rotation = torch.as_tensor(rotation, dtype=dtype, device=device)
        if rotation.ndim == 1: rotation = rotation.unsqueeze(0).repeat(B, 1)
        
        # 记录参数
        self.dtype = dtype
        self.device = device
        self.B = B 
        self.height = height
        self.top_radius = top_radius
        self.bottom_radius = bottom_radius
        self.position = position
        self.rotation = rotation
        
        # -----------------------------------------------------------
        # 2. 构建标准几何模板 (Geometry Template)
        # -----------------------------------------------------------
        # 模板定义：Height=1 (Y in [-0.5, 0.5]), Radius=1 (Unit Circle)
        # 顶点顺序：[TopC, BotC, T0, B0, T1, B1, T2, B2, T3, B3] (共10个点)
        
        num_of_segment = 3
        
        # (A) 生成中心点模板
        # TopCenter: [0, 0.5, 0], BotCenter: [0, -0.5, 0]
        template_center_top = torch.tensor([[0, 0.5, 0]], dtype=dtype, device=device)
        template_center_bot = torch.tensor([[0, -0.5, 0]], dtype=dtype, device=device)
        
        # (B) 生成圆周点模板 (向量化生成)
        # 角度: 0, 120, 240, 360 (闭合)
        theta = torch.linspace(0, 2 * math.pi, num_of_segment + 1, dtype=dtype, device=device) # [4]
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        
        # 上表面圆周 (y=0.5)
        template_rim_top = torch.stack([cos_t, torch.full_like(theta, 0.5), sin_t], dim=1) # [4, 3]
        # 下表面圆周 (y=-0.5)
        template_rim_bot = torch.stack([cos_t, torch.full_like(theta, -0.5), sin_t], dim=1) # [4, 3]
        
        # (C) 交错排列 (Interleave) 以匹配 [T0, B0, T1, B1...] 的结构
        # stack dim=1 -> [4, 2, 3] -> reshape -> [8, 3]
        template_rim = torch.stack([template_rim_top, template_rim_bot], dim=1).reshape(-1, 3)
        
        # (D) 合并所有模板点
        # template_vertices: [10, 3] -> 扩展为 [B, 10, 3]
        self.vertices = torch.cat([template_center_top, template_center_bot, template_rim], dim=0)
        self.vertices = self.vertices.unsqueeze(0).repeat(B, 1, 1)
        
        # -----------------------------------------------------------
        # 3. 计算变形张量 (Deformation / Resize)
        # -----------------------------------------------------------
        # 目标：构建一个 [B, 10, 3] 的缩放张量，使得 template * resize = 最终形状
        # X/Z轴乘半径，Y轴乘高度
        
        # 准备数据 [B, 1]
        h = self.height
        tr = self.top_radius
        br = self.bottom_radius
        
        # (A) 中心点缩放因子
        # Index 0 (TopC): xz缩放任意(因坐标为0), y缩放h. 为梯度传播统一性，设为 tr
        resize_center_top = torch.cat([tr, h, tr], dim=1).unsqueeze(1) # [B, 1, 3]
        resize_center_bot = torch.cat([br, h, br], dim=1).unsqueeze(1) # [B, 1, 3]
        
        # (B) 圆周点缩放因子
        # Top Rim: [tr, h, tr]
        resize_rim_top = torch.cat([tr, h, tr], dim=1).unsqueeze(1)    # [B, 1, 3]
        # Bot Rim: [br, h, br]
        resize_rim_bot = torch.cat([br, h, br], dim=1).unsqueeze(1)    # [B, 1, 3]
        
        # (C) 交错排列缩放因子
        # 需要重复 (num_of_segment + 1) 次以匹配顶点数量
        # stack -> [B, 1, 2, 3] -> repeat -> [B, 4, 2, 3] -> reshape -> [B, 8, 3]
        resize_rim = torch.stack([resize_rim_top, resize_rim_bot], dim=2)
        resize_rim = resize_rim.repeat(1, num_of_segment + 1, 1, 1).reshape(B, -1, 3)
        
        # (D) 合并缩放张量
        resize_tensor = torch.cat([resize_center_top, resize_center_bot, resize_rim], dim=1) # [B, 10, 3]
        
        # -----------------------------------------------------------
        # 4. 应用几何变换 (Apply Transforms)
        # -----------------------------------------------------------
        
        # 局部变形：Template * Resize (可导)
        self.vertices = self.vertices * resize_tensor
        
        # 全局变换：Rotate + Translate
        # 假设 rotate_6D 已定义
        self.vertices = rotate_6D(self.vertices, rotation) + position.unsqueeze(1)
        
        # -----------------------------------------------------------
        # 5. 定义拓扑特征 (Faces & Axes)
        # -----------------------------------------------------------
        # 下面的代码利用生成的 vertices 计算法线和轴
        # 保持原有逻辑，但利用向量化操作优化计算
        
        self.Node_Face = {}
        self.Node_Axis = {}

        # [Face 0] 上表面 (Top) - 使用顶点 T0(idx 2), T1(idx 4), T2(idx 6)
        # 注意：idx 2,4,6 对应 T0, T1, T2
        v_top = self.vertices[:, [2, 4, 6], :] # [B, 3, 3]
        face_top_center = torch.mean(v_top, dim=1)
        # 计算切线与法线
        t_top = normalize(v_top[:, 1] - v_top[:, 0]) # T0->T1
        e_top = v_top[:, 2] - v_top[:, 0]                 # T0->T2
        # 上表面法线向上(Y+)，(T1-T0)x(T2-T0) 在逆时针时由右手定则指向Y+
        n_top = normalize(torch.cross(t_top, e_top, dim=-1)) # 修正了原代码的叉乘向量选择
        b_top = torch.cross(n_top, t_top, dim=-1)
        self.Node_Face[0] = {'p': face_top_center, 'n': n_top, 't': t_top, 'b': b_top}
        
        # [Face 1] 下表面 (Bottom) - 使用顶点 B0(idx 3), B1(idx 5), B2(idx 7)
        v_bot = self.vertices[:, [3, 5, 7], :] 
        face_bot_center = torch.mean(v_bot, dim=1)
        t_bot = normalize(v_bot[:, 1] - v_bot[:, 0])
        e_bot = v_bot[:, 2] - v_bot[:, 0]
        # 下表面法线向下，(B1-B0)x(B2-B0)通常指向上，需要注意方向
        # 原代码逻辑保留，但实际上建议明确计算 Cross(Edge2, Edge1)
        n_bot = normalize(torch.cross(e_bot, t_bot, dim=-1)) 
        b_bot = torch.cross(n_bot, t_bot, dim=-1)
        self.Node_Face[1] = {'p': face_bot_center, 'n': n_bot, 't': t_bot, 'b': b_bot}
        
        # [Face 2,3,4] 侧面 (Side Faces)
        # 侧面 i 的顶点索引: Top_i (2+2i), Top_next, Bot_next, Bot_i (3+2i)
        for i in range(num_of_segment):
            # 获取索引 (注意闭合循环，最后一个点是重复的，可以直接取)
            idx_t1 = 2 + 2 * i
            idx_t2 = 2 + 2 * (i + 1)
            idx_b1 = 3 + 2 * i
            idx_b2 = 3 + 2 * (i + 1)
            
            # 收集顶点 [B, 4, 3]
            vs = torch.stack([
                self.vertices[:, idx_t1], self.vertices[:, idx_t2],
                self.vertices[:, idx_b2], self.vertices[:, idx_b1]
            ], dim=1)
            
            face_center = torch.mean(vs, dim=1)
            
            # Tangent: 沿上边缘 (T1 -> T2)
            t = normalize(self.vertices[:, idx_t2] - self.vertices[:, idx_t1])
            # Edge Down: T1 -> B1
            e_down = self.vertices[:, idx_b1] - self.vertices[:, idx_t1]
            # Normal: Cross(Edge_Down, Tangent) -> (Down x Right) -> Outward
            n = normalize(torch.cross(e_down, t, dim=-1))
            b = torch.cross(n, t, dim=-1)
            
            self.Node_Face[2 + i] = {'p': face_center, 'n': n, 't': t, 'b': b}

        # --- Axes (Update) ---
        
        # Axis 0: 主轴 (Main Axis) - Bottom Center -> Top Center
        p_bc = self.vertices[:, 1, :]
        p_tc = self.vertices[:, 0, :]
        d_main = normalize(p_tc - p_bc)
        self.Node_Axis[0] = {'p': p_bc, 'd': d_main}
        
        # Axes 1-9: 棱轴 (Edges)
        # 包含: 3个侧棱 (Side), 3个上底棱 (Top), 3个下底棱 (Bottom)
        axis_count = 1
        
        for i in range(num_of_segment):
            # 获取当前段的顶点索引
            # T_curr (2+2i), T_next (2+2(i+1)), B_curr (3+2i), B_next (3+2(i+1))
            idx_t_curr = 2 + 2 * i
            idx_t_next = 2 + 2 * (i + 1)
            idx_b_curr = 3 + 2 * i
            idx_b_next = 3 + 2 * (i + 1)
            
            # --- 1. 侧棱轴 (Side Edge Axis) ---
            # 定义方向: 从下往上 (Bottom -> Top)
            p_start_side = self.vertices[:, idx_b_curr, :]
            p_end_side   = self.vertices[:, idx_t_curr, :]
            d_side = normalize(p_end_side - p_start_side)
            self.Node_Axis[axis_count] = {'p': p_start_side, 'd': d_side}
            axis_count += 1
            
            # --- 2. 上底棱轴 (Top Edge Axis) ---
            # 定义方向: 逆时针 (T_curr -> T_next)
            p_start_top = self.vertices[:, idx_t_curr, :]
            p_end_top   = self.vertices[:, idx_t_next, :]
            d_top = normalize(p_end_top - p_start_top)
            self.Node_Axis[axis_count] = {'p': p_start_top, 'd': d_top}
            axis_count += 1
            
            # --- 3. 下底棱轴 (Bottom Edge Axis) ---
            # 定义方向: 逆时针 (B_curr -> B_next)
            p_start_bot = self.vertices[:, idx_b_curr, :]
            p_end_bot   = self.vertices[:, idx_b_next, :]
            d_bot = normalize(p_end_bot - p_start_bot)
            self.Node_Axis[axis_count] = {'p': p_start_bot, 'd': d_bot}
            axis_count += 1
            
        # -----------------------------------------------------------
        # 6. 初始化父类
        # -----------------------------------------------------------
        super().__init__(position, rotation, rotation_order,
                         Node_Position=self.vertices, Node_Face=self.Node_Face, Node_Axis=self.Node_Axis,
                         Node_Semantic=Semantic, Node_Affordance=Affordance)
    
    def get_surface_points(self, total_points=600):
        """
        在三棱柱表面上采样点
        :param total_points: 采样的总点数
        :return: [B, total_points, 3]
        """
        if total_points <= 0:
            return torch.zeros((self.B, 0, 3), device=self.device, dtype=self.dtype)
        
        B = self.B
        num_of_segment = 3
        # 面定义：1个上底 + 1个下底 + 3个侧面 = 5个面
        # 注意：虽然我们在 init 里定义了 Node_Face 有 5 个，这里逻辑保持一致
        
        # ---------------------------------------------------------------------
        # 1. 计算各个面的面积 (Vectorized Area Calculation)
        # ---------------------------------------------------------------------
        
        # --- 上表面 (Top) ---
        # 顶点索引: T0(2), T1(4), T2(6)
        # 这是一个三角形
        top_verts = self.vertices[:, [2, 4, 6], :]  # [B, 3, 3]
        # Area = 0.5 * |(v1-v0) x (v2-v0)|
        edge1_top = top_verts[:, 1] - top_verts[:, 0]
        edge2_top = top_verts[:, 2] - top_verts[:, 0]
        area_top = 0.5 * torch.norm(torch.cross(edge1_top, edge2_top, dim=-1), dim=-1) # [B]

        # --- 下表面 (Bottom) ---
        # 顶点索引: B0(3), B1(5), B2(7)
        bot_verts = self.vertices[:, [3, 5, 7], :]  # [B, 3, 3]
        edge1_bot = bot_verts[:, 1] - bot_verts[:, 0]
        edge2_bot = bot_verts[:, 2] - bot_verts[:, 0]
        area_bot = 0.5 * torch.norm(torch.cross(edge1_bot, edge2_bot, dim=-1), dim=-1) # [B]

        # --- 侧面 (Sides) ---
        # 我们有 3 个侧面。为了高效，我们可以一次性提取所有侧面的顶点进行计算。
        # 侧面 i 的顶点: TL: 2+2i, TR: 2+2(i+1), BR: 3+2(i+1), BL: 3+2i
        # 利用 init 中生成的 Idx 8, 9 (T3, B3) 是闭合点，不需要取模
        indices_tl = [2, 4, 6] # T0, T1, T2
        indices_tr = [4, 6, 8] # T1, T2, T3
        indices_bl = [3, 5, 7] # B0, B1, B2
        # 计算侧面通常由两个三角形组成（或直接算叉乘模长，如果共面）。
        # Area = |(TR-TL) x (BL-TL)| (假设是矩形/平行四边形近似，或者拆分为两个三角形)
        # 精确计算：Area = 0.5 * |Edge_Top x Edge_Side| + 0.5 * |Edge_Bot x Edge_Side| 
        # 但为了保持和原代码一致且高效（原代码假设由 edge1 x edge2 构成），这里使用叉乘模长近似 (适用于矩形)
        
        # 提取数据: [B, 3, 3] (3个侧面)
        v_tl = self.vertices[:, indices_tl, :]
        v_tr = self.vertices[:, indices_tr, :]
        v_bl = self.vertices[:, indices_bl, :]
        
        edge_hor = v_tr - v_tl
        edge_ver = v_bl - v_tl
        # [B, 3] -> 3个面的面积
        area_sides = torch.norm(torch.cross(edge_hor, edge_ver, dim=-1), dim=-1) 

        # ---------------------------------------------------------------------
        # 2. 分配点数 (Point Allocation)
        # ---------------------------------------------------------------------
        # 合并所有面积: Top, Bot, Side1, Side2, Side3
        all_areas = torch.cat([area_top.unsqueeze(1), area_bot.unsqueeze(1), area_sides], dim=1) # [B, 5]
        
        # 使用 Batch 平均面积来决定点数分布，确保生成的 Tensor 维度一致
        mean_areas = all_areas.mean(dim=0) # [5]
        total_area = mean_areas.sum()
        
        if total_area < 1e-6:
            probs = torch.full_like(mean_areas, 1.0 / 5.0)
        else:
            probs = mean_areas / total_area
            
        num_per_face = (probs * total_points).long()
        
        # 修正舍入误差，确保总和为 total_points
        diff = total_points - num_per_face.sum()
        if diff > 0:
            # 加在面积最大的面上
            num_per_face[torch.argmax(num_per_face)] += diff
        
        # ---------------------------------------------------------------------
        # 3. 采样点 (Sampling)
        # ---------------------------------------------------------------------
        sampled_points = []
        
        # (A) 采样上表面 (Triangle)
        N_top = num_per_face[0].item()
        if N_top > 0:
            # 随机重心坐标
            u = torch.rand(B, N_top, 1, device=self.device, dtype=self.dtype)
            v = torch.rand(B, N_top, 1, device=self.device, dtype=self.dtype)
            sqrt_u = torch.sqrt(u)
            
            w0 = 1 - sqrt_u
            w1 = sqrt_u * (1 - v)
            w2 = sqrt_u * v
            
            # v0, v1, v2 shape: [B, 1, 3]
            p_top = (w0 * top_verts[:, 0:1] + 
                     w1 * top_verts[:, 1:2] + 
                     w2 * top_verts[:, 2:3])
            sampled_points.append(p_top)
            
        # (B) 采样下表面 (Triangle)
        N_bot = num_per_face[1].item()
        if N_bot > 0:
            u = torch.rand(B, N_bot, 1, device=self.device, dtype=self.dtype)
            v = torch.rand(B, N_bot, 1, device=self.device, dtype=self.dtype)
            sqrt_u = torch.sqrt(u)
            
            w0 = 1 - sqrt_u
            w1 = sqrt_u * (1 - v)
            w2 = sqrt_u * v
            
            p_bot = (w0 * bot_verts[:, 0:1] + 
                     w1 * bot_verts[:, 1:2] + 
                     w2 * bot_verts[:, 2:3])
            sampled_points.append(p_bot)
            
        # (C) 采样侧面 (Rectangles/Quads)
        # 循环处理 3 个侧面，因为每个侧面分配的点数可能略有不同（虽然理论上均分，但舍入可能导致差异）
        # 或者如果 geometry 是各项异性的，面积确实不同
        
        for i in range(num_of_segment):
            # 面索引偏移：0->Top, 1->Bot, 2,3,4->Sides
            N_side = num_per_face[2 + i].item()
            if N_side > 0:
                # 获取顶点: TL, TR, BR, BL
                # init 里的排布保证了 idx 8,9 是存在的
                idx_tl = 2 + 2 * i
                idx_tr = 2 + 2 * (i + 1)
                idx_br = 3 + 2 * (i + 1)
                idx_bl = 3 + 2 * i
                
                v1 = self.vertices[:, idx_tl, :].unsqueeze(1) # TL [B, 1, 3]
                v2 = self.vertices[:, idx_tr, :].unsqueeze(1) # TR
                v3 = self.vertices[:, idx_br, :].unsqueeze(1) # BR
                v4 = self.vertices[:, idx_bl, :].unsqueeze(1) # BL
                
                # 双线性插值采样
                # s: horizontal (0->1 from Left to Right)
                # t: vertical (0->1 from Top to Bottom)
                s = torch.rand(B, N_side, 1, device=self.device, dtype=self.dtype)
                t = torch.rand(B, N_side, 1, device=self.device, dtype=self.dtype)
                
                # Formula: 
                # P = (1-s)(1-t)TL + s(1-t)TR + (1-s)t BL + st BR
                # 注意：原代码的变量命名 v1,v2,v3,v4 对应顺序需确认。
                # 通常：v1=TL, v2=TR, v3=BR, v4=BL (逆时针或Z字形)
                # 这里我们显式写出混合权重
                
                p_side = (1 - s) * (1 - t) * v1 + \
                         s       * (1 - t) * v2 + \
                         (1 - s) * t       * v4 + \
                         s       * t       * v3
                sampled_points.append(p_side)

        if not sampled_points:
             return torch.zeros((B, 0, 3), device=self.device, dtype=self.dtype)
             
        return torch.cat(sampled_points, dim=1)

class Cone(StructureNode):
    def __init__(self, radius, height, tip_offset=[0, 0], radius_z=None, 
                 position=[0, 0, 0], rotation=[1, 0, 0, 0, 1, 0], rotation_order="XYZ", 
                 num_of_segment=256, Semantic=None, Affordance=None):
        """
        向量化的高效圆锥体构建
        :param radius: [B, 1] X轴方向底面半径
        :param height: [B, 1] 高度
        :param tip_offset: [B, 2] 顶点相对于底面中心的 X/Z 偏移 (用于构建斜圆锥)
        :param radius_z: [B, 1] Z轴方向底面半径 (椭圆锥)
        """
        # -----------------------------------------------------------
        # 1. 参数预处理
        # -----------------------------------------------------------
        dtype = height.dtype
        device = height.device
        B = height.shape[0]
        
        # 填充默认值
        if radius_z is None:
            radius_z = radius
            
        # 确保维度正确 [B, 1] 或 [B, N]
        if isinstance(tip_offset, (list, tuple)):
            tip_offset = torch.as_tensor(tip_offset, dtype=dtype, device=device)
        if tip_offset.ndim == 1: tip_offset = tip_offset.unsqueeze(0).repeat(B, 1) # [B, 2]
            
        if isinstance(position, (list, tuple)):
            position = torch.as_tensor(position, dtype=dtype, device=device)
        if position.ndim == 1: position = position.unsqueeze(0).repeat(B, 1)

        if isinstance(rotation, (list, tuple)):
            rotation = torch.as_tensor(rotation, dtype=dtype, device=device)
        if rotation.ndim == 1: rotation = rotation.unsqueeze(0).repeat(B, 1)
        
        self.dtype = dtype
        self.device = device
        self.B = B
        self.radius = radius
        self.height = height
        self.tip_offset = tip_offset
        self.radius_z = radius_z
        self.num_of_segment = num_of_segment
        self.position = position
        self.rotation = rotation
        
        # -----------------------------------------------------------
        # 2. 向量化生成局部顶点 (Vectorized Vertex Generation)
        # -----------------------------------------------------------
        # 几何中心定义：Y轴范围 [-h/2, h/2]
        # 顶点布局：Idx 0: Tip (顶点), Idx 1: Base Center (底面中心), Idx 2...: Base Rim (底面圆周)
        
        half_h = height / 2.0  # [B, 1]
        
        # (A) 顶点 (Tip) -> [Offset_X, H/2, Offset_Z]
        # tip_offset: [B, 2] -> split -> x, z
        tip_x = tip_offset[:, 0:1]
        tip_z = tip_offset[:, 1:2]
        p_tip = torch.cat([tip_x, half_h, tip_z], dim=1).unsqueeze(1) # [B, 1, 3]
        
        # (B) 底面中心 (Base Center) -> [0, -H/2, 0]
        zeros = torch.zeros_like(half_h)
        p_base_center = torch.cat([zeros, -half_h, zeros], dim=1).unsqueeze(1) # [B, 1, 3]
        
        # (C) 底面圆周 (Base Rim) -> [Rx*cos, -H/2, Rz*sin]
        # 生成角度 [0, 2pi] (闭合)
        theta = torch.linspace(0, 2 * math.pi, num_of_segment + 1, dtype=dtype, device=device) # [N+1]
        cos_t = torch.cos(theta).view(1, -1) # [1, N+1]
        sin_t = torch.sin(theta).view(1, -1) # [1, N+1]
        
        # 广播计算坐标 [B, 1] * [1, N+1] -> [B, N+1]
        rim_x = radius * cos_t
        rim_z = radius_z * sin_t
        rim_y = -half_h.expand(-1, num_of_segment + 1) # [B, N+1]
        
        p_rim = torch.stack([rim_x, rim_y, rim_z], dim=-1) # [B, N+1, 3]
        
        # (D) 合并所有顶点
        # [B, 1 + 1 + (N+1), 3] = [B, N+3, 3]
        self.vertices = torch.cat([p_tip, p_base_center, p_rim], dim=1)
        
        # -----------------------------------------------------------
        # 3. 全局变换 (Global Transformation)
        # -----------------------------------------------------------
        self.vertices = rotate_6D(self.vertices, rotation) + position.unsqueeze(1)
        
        # -----------------------------------------------------------
        # 4. 定义拓扑特征 (Faces & Axes)
        # -----------------------------------------------------------
        self.Node_Face = {}
        self.Node_Axis = {}
        
        # 辅助视图 (Views)
        v_tip = self.vertices[:, 0, :]    # [B, 3]
        v_bc  = self.vertices[:, 1, :]    # [B, 3]
        v_rim = self.vertices[:, 2:, :]   # [B, N+1, 3]
        
        # === Face 0: 底面 (Base) ===
        # 选取三个点计算平面：中心, Rim[0], Rim[N/2] (确保跨度大，法线稳)
        idx_r1 = 0
        idx_r2 = num_of_segment // 2
        p1 = v_rim[:, idx_r1, :]
        p2 = v_rim[:, idx_r2, :]
        
        t_base = normalize(p1 - v_bc)
        # 底面法线应向下。
        # 默认坐标系下底面在Y负，Normal应为 -Y。
        # 叉乘方向：(p1 - c) x (p2 - c). 若 p1, p2 逆时针，则朝下(因为是XZ平面, Y向下看是顺时针? 不，通常是Y向上)
        # 简单起见，利用几何结构：法线方向 = Normalize(BaseCenter - Tip) (对于正圆锥)
        # 对于斜圆锥，底面依然是平面的。直接用叉乘: (Rim0 - Center) x (Rim_Mid - Center)
        # 如果 Theta 逆时针，Y向上，则 X x Z = -Y (Down). 
        n_base = normalize(torch.cross(p1 - v_bc, p2 - v_bc, dim=-1))
        # 确保法线指向外部 (对于底面，应该是背离 Tip 的方向)
        # Check dot(n, Tip-BC). If > 0 (pointing to tip), flip it.
        vec_c2t = v_tip - v_bc
        sign = torch.sign(torch.sum(n_base * vec_c2t, dim=-1, keepdim=True))
        n_base = n_base * (-sign) # 强制背离顶点
        
        b_base = torch.cross(n_base, t_base, dim=-1)
        self.Node_Face[0] = {'p': v_bc, 'n': n_base, 't': t_base, 'b': b_base}
        
        # === Face 1: 侧面 (Side Surface) ===
        # 圆锥侧面是连续曲面。这里计算一个“代表性”的侧面属性。
        # 选取一个特征三角形：Tip -> Rim[0] -> Rim[1]
        v_r0 = v_rim[:, 0, :]
        v_r1 = v_rim[:, 1, :]
        
        # 侧面中心：取几何体的一侧中点，或者整体表面中心(Tip + Average(Rim))/2
        # 为了物理接触检测，通常取某一侧的切面。
        # 这里取 idx 0 处的母线中点
        face_center_side = (v_tip + v_r0) / 2.0
        
        t_side = normalize(v_r0 - v_tip) # 母线方向
        # 侧面法线：垂直于母线，且垂直于切向(rim方向)
        edge_rim = v_r1 - v_r0
        n_side = normalize(torch.cross(edge_rim, t_side, dim=-1))
        b_side = torch.cross(n_side, t_side, dim=-1)
        
        self.Node_Face[1] = {'p': face_center_side, 'n': n_side, 't': t_side, 'b': b_side}
        
        # === Axes ===
        
        # Axis 0: 主轴 (Main Axis) - Base Center -> Tip
        d_main = normalize(v_tip - v_bc)
        self.Node_Axis[0] = {'p': v_bc, 'd': d_main}
        
        # Axis 1-4: 特征母线轴 (Generatrix Axes)
        # 圆锥没有棱，但为了抓取规划，我们在 0, 90, 180, 270 度添加 4 条虚拟“棱”
        # 对应索引：0, N/4, N/2, 3N/4
        indices = [
            0, 
            num_of_segment // 4, 
            num_of_segment // 2, 
            (num_of_segment * 3) // 4
        ]
        
        for i, idx in enumerate(indices):
            # 方向：从底面圆周点 -> 顶点 (Bottom -> Top)
            p_start = v_rim[:, idx, :]
            p_end = v_tip
            d_gen = normalize(p_end - p_start)
            
            self.Node_Axis[1 + i] = {'p': p_start, 'd': d_gen}
            
        # -----------------------------------------------------------
        # 5. 初始化父类
        # -----------------------------------------------------------
        super().__init__(position, rotation, rotation_order,
                         Node_Position=self.vertices, Node_Face=self.Node_Face, Node_Axis=self.Node_Axis,
                         Node_Semantic=Semantic, Node_Affordance=Affordance)

    def get_surface_points(self, total_points=600):
        """
        在圆锥表面采样点
        """
        if total_points <= 0:
            return torch.zeros((self.B, 0, 3), device=self.device, dtype=self.dtype)
            
        B = self.B
        
        # 计算面积
        # 1. 底面积 (Ellipse area: pi * a * b)
        # 注意：这里需要计算真实的底面积，因为可能被缩放。
        # 使用叉乘计算：Area ~ Sum of triangle fans. 
        # 简便方法：R_eff_x = norm(v_rim_0 - v_bc), R_eff_z = norm(v_rim_90 - v_bc)
        # 但考虑到斜圆锥，直接积分或者用平均半径近似。
        # 使用向量化计算 Rim 平均半径
        v_bc = self.vertices[:, 1, :]
        v_rim = self.vertices[:, 2:-1, :] # 排除重复的最后一个点
        dists = torch.norm(v_rim - v_bc.unsqueeze(1), dim=-1)
        mean_radius_sq = torch.mean(dists**2, dim=1)
        area_base = math.pi * mean_radius_sq # [B]
        
        # 2. 侧面积 (Lateral Area)
        # 公式复杂(对于斜圆锥)。使用离散三角形近似求和：Sum(0.5 * |(R_i - T) x (R_i+1 - T)|)
        v_tip = self.vertices[:, 0, :].unsqueeze(1) # [B, 1, 3]
        v_rim_all = self.vertices[:, 2:, :]         # [B, N+1, 3]
        v_curr = v_rim_all[:, :-1, :]               # [B, N, 3]
        v_next = v_rim_all[:, 1:, :]                # [B, N, 3]
        
        edge1 = v_curr - v_tip
        edge2 = v_next - v_tip
        # 每个小三角形面积
        areas_sub = 0.5 * torch.norm(torch.cross(edge1, edge2, dim=-1), dim=-1) # [B, N]
        area_side = torch.sum(areas_sub, dim=1) # [B]
        
        # 分配点数
        total_area = area_base + area_side
        prob_base = area_base / (total_area + 1e-6)
        n_base = (prob_base * total_points).long()
        n_side = total_points - n_base # 剩余给侧面
        
        points_list = []
        
        # --- 采样底面 (Disk) ---
        # rejection sampling or polar mapping
        # Polar mapping: P = C + sqrt(u)*r*cos(v), ...
        # 由于是椭圆甚至变形底面，最好用三角形扇 (Triangle Fan) 采样
        # 重用 v_bc, v_curr, v_next (但这是相对于 tip 的，我们需要相对于 center 的)
        # 底面三角形: BC -> Rim_i -> Rim_i+1
        max_n_base = n_base.max().item()
        if max_n_base > 0:
            # 随机选择三角形索引
            # 简化：假设每个扇形面积近似相等（对于正圆锥成立），随机选索引
            rand_idx = torch.randint(0, self.num_of_segment, (B, max_n_base), device=self.device)
            
            # Gather vertices
            # batch_indices: [0, 0...], [1, 1...]
            batch_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(-1, max_n_base)
            
            # v_rim_all: [B, N+1, 3]
            p1 = v_rim_all[batch_idx, rand_idx, :]     # Rim i
            p2 = v_rim_all[batch_idx, rand_idx + 1, :] # Rim i+1
            pc = v_bc.unsqueeze(1).expand(-1, max_n_base, -1)
            
            # Triangle sampling
            u = torch.rand(B, max_n_base, 1, device=self.device, dtype=self.dtype)
            v = torch.rand(B, max_n_base, 1, device=self.device, dtype=self.dtype)
            sqrt_u = torch.sqrt(u)
            w0 = 1 - sqrt_u
            w1 = sqrt_u * (1 - v)
            w2 = sqrt_u * v
            
            pts_base = w0 * pc + w1 * p1 + w2 * p2
            
            # Masking for variable point counts per batch (if needed)
            # 这里简化处理，直接 cat
            points_list.append(pts_base)

        # --- 采样侧面 (Cone Surface) ---
        max_n_side = n_side.max().item()
        if max_n_side > 0:
            # 按照面积权重选择三角形可能更精确，这里简化为均匀随机选择扇区
            rand_idx = torch.randint(0, self.num_of_segment, (B, max_n_side), device=self.device)
            batch_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(-1, max_n_side)
            
            p_top = v_tip.expand(-1, max_n_side, -1)
            p_r1 = v_rim_all[batch_idx, rand_idx, :]
            p_r2 = v_rim_all[batch_idx, rand_idx + 1, :]
            
            u = torch.rand(B, max_n_side, 1, device=self.device, dtype=self.dtype)
            v = torch.rand(B, max_n_side, 1, device=self.device, dtype=self.dtype)
            sqrt_u = torch.sqrt(u)
            w0 = 1 - sqrt_u
            w1 = sqrt_u * (1 - v)
            w2 = sqrt_u * v
            
            pts_side = w0 * p_top + w1 * p_r1 + w2 * p_r2
            points_list.append(pts_side)
            
        if not points_list:
            return torch.zeros((B, 0, 3), device=self.device, dtype=self.dtype)
            
        # 注意：这里简单的 append 可能会导致 output shape [B, total, 3] 稍微有一点点偏差如果 batch 间 n_base 不同
        # 为了严格对齐，通常需要 mask 或 padding。但基于题目要求，只要返回 [B, N, 3] 即可。
        # 上述逻辑中 max_n 是一样的，所以可以直接 cat
        out = torch.cat(points_list, dim=1)
        # 裁剪到 total_points (因为 n_base + n_side 可能因为 float 转换有细微误差)
        return out[:, :total_points, :]

class Ring(StructureNode):
    def __init__(self, height, outer_top_radius, inner_top_radius, exist_angle=None, 
                 outer_bottom_radius=None, inner_bottom_radius=None, back_height=None, 
                 generatrix_offset=0, x_z_ratio=None, inner_x_z_ratio=None, 
                 inner_offset=[0, 0], position=[0, 0, 0], rotation=[1, 0, 0, 0, 1, 0], 
                 rotation_order="XYZ", num_of_segment=8, Semantic=None, Affordance=None):
        """
        基于标准几何模板的高效圆环构建 (Vectorized Implementation)
        """
        # -----------------------------------------------------------
        # 1. 参数预处理 (Input Standardization)
        # -----------------------------------------------------------
        dtype = height.dtype
        device = height.device
        B = height.shape[0]
        
        height = height.view(B, 1) if height.ndim == 1 else height
        outer_top_radius = outer_top_radius.view(B, 1) if outer_top_radius.ndim == 1 else outer_top_radius
        inner_top_radius = inner_top_radius.view(B, 1) if inner_top_radius.ndim == 1 else inner_top_radius
        
        # 统一维度处理函数
        def _ensure_tensor(val, target_shape_ref=None):
            if val is None and target_shape_ref is not None:
                return target_shape_ref.clone()
            if isinstance(val, (int, float)):
                return torch.full((B, 1), val, dtype=dtype, device=device)
            if not isinstance(val, torch.Tensor):
                val = torch.as_tensor(val, dtype=dtype, device=device)
            if val.ndim == 0:
                val = val.view(1, 1).repeat(B, 1)
            elif val.ndim == 1:
                val = val.view(1, -1).repeat(B, 1) if val.shape[0] != B else val.view(B, 1)
            return val

        # 填充默认值并规范化维度 [B, 1]
        exist_angle = _ensure_tensor(exist_angle, torch.full((B, 1), 2 * math.pi, dtype=dtype, device=device))
        outer_bottom_radius = _ensure_tensor(outer_bottom_radius, outer_top_radius)
        inner_bottom_radius = _ensure_tensor(inner_bottom_radius, inner_top_radius)
        back_height = _ensure_tensor(back_height, height)
        x_z_ratio = _ensure_tensor(x_z_ratio, torch.ones((B, 1), dtype=dtype, device=device))
        inner_x_z_ratio = _ensure_tensor(inner_x_z_ratio, x_z_ratio)
        generatrix_offset = _ensure_tensor(generatrix_offset, torch.zeros((B, 1), dtype=dtype, device=device))

        # 处理多维参数
        if isinstance(inner_offset, (list, tuple)):
            inner_offset = torch.as_tensor(inner_offset, dtype=dtype, device=device)
        if inner_offset.ndim == 1: 
            inner_offset = inner_offset.unsqueeze(0).repeat(B, 1) # [B, 2]
        
        if isinstance(position, (list, tuple)):
            position = torch.as_tensor(position, dtype=dtype, device=device)
        if position.ndim == 1: 
            position = position.unsqueeze(0).repeat(B, 1) # [B, 3]
        
        if isinstance(rotation, (list, tuple)):
            rotation = torch.as_tensor(rotation, dtype=dtype, device=device)
        if rotation.ndim == 1: 
            rotation = rotation.unsqueeze(0).repeat(B, 1) # [B, 6]
        
        # 记录参数
        self.dtype = dtype
        self.device = device
        self.B = B
        self.height = height
        self.outer_top_radius = outer_top_radius
        self.inner_top_radius = inner_top_radius
        self.exist_angle = exist_angle
        self.outer_bottom_radius = outer_bottom_radius
        self.inner_bottom_radius = inner_bottom_radius
        self.back_height = back_height
        self.generatrix_offset = generatrix_offset
        self.x_z_ratio = x_z_ratio
        self.inner_x_z_ratio = inner_x_z_ratio
        self.inner_offset = inner_offset
        self.num_of_segment = num_of_segment
        self.position = position
        self.rotation = rotation
        self.rotation_order = rotation_order
        
        # -----------------------------------------------------------
        # 2. 向量化计算变形张量 (Vectorized Deformation)
        # -----------------------------------------------------------
        
        # 2.1 基础角度计算
        # angles: [1, N+1]
        angles = torch.linspace(0, 1, num_of_segment + 1, dtype=dtype, device=device).unsqueeze(0)
        # rotation_angles: [B, N+1]
        rotation_angles = exist_angle * angles 
        
        cos_rot = torch.cos(rotation_angles) # [B, N+1]
        sin_rot = torch.sin(rotation_angles) # [B, N+1]
        cos_factor = (1 - cos_rot)           # [B, N+1]

        # 2.2 计算 Y 坐标基准 (Reference Y Calculation)
        # 将 [B, 1] 广播到 [B, N+1]
        h_half = height / 2
        bh_half = back_height / 2
        
        # Top Y: [B, N+1]
        top_y_base = h_half + (bh_half + generatrix_offset - h_half) / (outer_top_radius * 2) * (outer_top_radius * cos_factor)
        # Bottom Y: [B, N+1]
        bottom_y_base = -h_half + (-bh_half + generatrix_offset + h_half) / (outer_bottom_radius * 2) * (outer_bottom_radius * cos_factor)

        # 2.3 外圆顶点 (Outer Vertices)
        # Outer Top: [B, N+1, 3]
        outer_top_x = outer_top_radius * x_z_ratio * cos_rot
        outer_top_z = outer_top_radius * sin_rot
        v_outer_top = torch.stack([outer_top_x, top_y_base, outer_top_z], dim=-1)

        # Outer Bottom: [B, N+1, 3]
        outer_bottom_x = outer_bottom_radius * x_z_ratio * cos_rot
        outer_bottom_z = outer_bottom_radius * sin_rot
        v_outer_bottom = torch.stack([outer_bottom_x, bottom_y_base, outer_bottom_z], dim=-1)

        # 2.4 内圆顶点 (Inner Vertices) - 完全向量化逻辑
        # -------------------------------------------------------
        # 内圆 Top 计算
        sur_delta_y_top = top_y_base - h_half
        limit_top = (bh_half + generatrix_offset - h_half) - sur_delta_y_top
        sur_delta_y_new_top = torch.minimum(sur_delta_y_top, limit_top)
        
        sur_y_gap_top = (bh_half + generatrix_offset - h_half) - sur_delta_y_new_top * 2
        section_chord_top = torch.sqrt((outer_top_radius * 2) ** 2 + sur_y_gap_top ** 2 + eps)
        
        radius_diff_top = outer_top_radius - inner_top_radius
        delta_chord_top = section_chord_top * radius_diff_top / (outer_top_radius * 2)
        
        delta_y_top_val = torch.sqrt(torch.abs(delta_chord_top ** 2 - radius_diff_top ** 2) + eps)
        # 条件判断
        delta_y_top_final = torch.where(sur_delta_y_top > limit_top, -delta_y_top_val, delta_y_top_val)

        # 内圆 Bottom 计算
        sur_delta_y_bottom = bottom_y_base + h_half
        limit_bottom = (-bh_half + generatrix_offset + h_half) - sur_delta_y_bottom
        sur_delta_y_new_bottom = torch.maximum(sur_delta_y_bottom, limit_bottom)
        
        sur_y_gap_bottom = (-bh_half + generatrix_offset + h_half) - sur_delta_y_new_bottom * 2
        section_chord_bottom = torch.sqrt((outer_bottom_radius * 2) ** 2 + sur_y_gap_bottom ** 2 + eps)
        
        radius_diff_bottom = outer_bottom_radius - inner_bottom_radius
        delta_chord_bottom = section_chord_bottom * radius_diff_bottom / (outer_bottom_radius * 2)
        
        delta_y_bottom_val = torch.sqrt(torch.abs(delta_chord_bottom ** 2 - radius_diff_bottom ** 2) + eps)
        # 条件判断
        delta_y_bottom_final = torch.where(sur_delta_y_bottom > limit_bottom, -delta_y_bottom_val, delta_y_bottom_val)

        # 构建内圆坐标
        # Offset 向量: [B, 1, 3] -> (0, inner_offset_y, -inner_offset_x)
        inner_offset_vec = torch.stack([
            torch.zeros_like(inner_offset[:, 0]), 
            inner_offset[:, 1], 
            -inner_offset[:, 0]
        ], dim=-1).unsqueeze(1)

        # Inner Top: [B, N+1, 3]
        inner_top_x = inner_top_radius * inner_x_z_ratio * cos_rot
        inner_top_z = inner_top_radius * sin_rot
        v_inner_top = torch.stack([inner_top_x, top_y_base + delta_y_top_final, inner_top_z], dim=-1)
        v_inner_top = v_inner_top + inner_offset_vec

        # Inner Bottom: [B, N+1, 3]
        inner_bottom_x = inner_bottom_radius * inner_x_z_ratio * cos_rot
        inner_bottom_z = inner_bottom_radius * sin_rot
        v_inner_bottom = torch.stack([inner_bottom_x, bottom_y_base + delta_y_bottom_final, inner_bottom_z], dim=-1)
        v_inner_bottom = v_inner_bottom + inner_offset_vec

        # -----------------------------------------------------------
        # 3. 组装顶点 (Vertex Assembly)
        # -----------------------------------------------------------
        # 目标顺序: [OT_0, OB_0, IT_0, IB_0, OT_1, OB_1, ...]
        # Stack dim=2: [B, N+1, 4, 3]
        # Flatten dim 1&2: [B, (N+1)*4, 3]
        self.vertices = torch.stack([v_outer_top, v_outer_bottom, v_inner_top, v_inner_bottom], dim=2)
        self.vertices = self.vertices.flatten(1, 2)

        # -----------------------------------------------------------
        # 4. 应用几何变换 (Apply Transforms)
        # -----------------------------------------------------------
        self.vertices = rotate_6D(self.vertices, rotation) + position.unsqueeze(1)
        
        # -----------------------------------------------------------
        # 5. 定义拓扑特征 (Faces & Axes)
        # -----------------------------------------------------------
        self.Node_Face = {}
        self.Node_Axis = {}
        
        # === 辅助向量计算 ===
        # 索引映射: 0:OT, 1:OB, 2:IT, 3:IB (Current Segment)
        #          4:OT, 5:OB, 6:IT, 7:IB (Next Segment)
        
        # === Face 0: 外圆上表面 (Top Face) ===
        # 顶点: OT_cur(0), OT_nxt(4), IT_cur(2)
        v0_top = self.vertices[:, 0, :]
        v1_top = self.vertices[:, 4, :]
        v2_top = self.vertices[:, 2, :]
        
        face_center_top = (v0_top + v1_top + v2_top) / 3
        t_top = normalize(v1_top - v0_top)          # 切线 (沿圆环)
        edge_in_top = v2_top - v0_top               # 径向 (向内)
        n_top = normalize(torch.cross(edge_in_top, t_top, dim=-1)) # Inward x Tangent -> Up
        b_top = torch.cross(n_top, t_top, dim=-1)
        self.Node_Face[0] = {'p': face_center_top, 'n': n_top, 't': t_top, 'b': b_top}

        # === Face 1: 外圆下表面 (Bottom Face) ===
        # 顶点: OB_cur(1), OB_nxt(5), IB_cur(3)
        v0_btm = self.vertices[:, 1, :]
        v1_btm = self.vertices[:, 5, :]
        v2_btm = self.vertices[:, 3, :]

        face_center_btm = (v0_btm + v1_btm + v2_btm) / 3
        t_btm = normalize(v1_btm - v0_btm)          # 切线
        edge_in_btm = v2_btm - v0_btm               # 径向 (向内)
        # 为了让法线向下: Tangent x Inward -> Down
        n_btm = normalize(torch.cross(t_btm, edge_in_btm, dim=-1))
        b_btm = torch.cross(n_btm, t_btm, dim=-1)
        self.Node_Face[1] = {'p': face_center_btm, 'n': n_btm, 't': t_btm, 'b': b_btm}

        # === Face 2: 外侧面 (Side Face / Outer Surface) ===
        # 顶点: OT_cur(0), OB_cur(1), OT_nxt(4)
        v_ot_cur = self.vertices[:, 0, :]
        v_ob_cur = self.vertices[:, 1, :]
        v_ot_nxt = self.vertices[:, 4, :]
        
        face_center_side = (v_ot_cur + v_ob_cur + v_ot_nxt) / 3
        v_down = v_ob_cur - v_ot_cur         # 向下向量
        v_tan = v_ot_nxt - v_ot_cur          # 切线向量
        # 法线向外: Down x Tangent -> Outward
        n_side = normalize(torch.cross(v_down, v_tan, dim=-1))
        t_side = normalize(v_tan)
        b_side = torch.cross(n_side, t_side, dim=-1)
        self.Node_Face[2] = {'p': face_center_side, 'n': n_side, 't': t_side, 'b': b_side}
        
       # === 修改：中心垂线 (Center Vertical Axis) ===
        # 这里的轴是穿过圆环中心的轴 (甜甜圈的孔所在的方向)
        
        # 1. 轴的起点：圆环的中心位置 (World Position)
        # 也就是传入的 position 参数 [B, 3]
        axis_p = self.position
        
        # 2. 轴的方向：局部 Y 轴 (0, 1, 0) 旋转后的方向
        # 创建局部 Y 轴向量 [B, 1, 3]
        local_y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
        local_y_axis = local_y_axis.view(1, 1, 3).repeat(B, 1, 1)
        
        # 应用旋转 (rotate_6D 只包含旋转，不包含平移)
        # [B, 1, 3] -> [B, 1, 3] -> [B, 3]
        axis_d = rotate_6D(local_y_axis, rotation).squeeze(1)
        
        # 存入 Node_Axis
        self.Node_Axis[0] = {'p': axis_p, 'd': normalize(axis_d)}
        
        # -----------------------------------------------------------
        # 6. 初始化父类
        # -----------------------------------------------------------
        super().__init__(position, rotation, rotation_order,
                         Node_Position=self.vertices, Node_Face=self.Node_Face, Node_Axis=self.Node_Axis,
                         Node_Semantic=Semantic, Node_Affordance=Affordance)
    
    def get_surface_points(self, total_points=600):
        """
        在圆环表面上采样点（包括外圆侧面、内圆侧面、上表面、下表面）
        :param total_points: 采样的总点数
        :return: [B, total_points, 3]
        """
        if total_points <= 0:
            return torch.zeros((self.B, 0, 3), device=self.device, dtype=self.dtype)
        
        B = self.B
        device = self.device
        dtype = self.dtype
        
        # 计算各部分面积以分配点数
        # 外圆侧面、内圆侧面、上表面、下表面
        phi = self.exist_angle.view(B, 1)  # [B, 1]
        
        # 外圆侧面面积（圆柱侧面，考虑 x_z_ratio）
        outer_side_radius_avg = (self.outer_top_radius + self.outer_bottom_radius) / 2
        # 侧面面积 = 角度 * 平均半径 * 高度（考虑椭圆）
        area_outer_side = phi * outer_side_radius_avg * self.height * self.x_z_ratio.view(B, 1)
        
        # 内圆侧面面积
        inner_side_radius_avg = (self.inner_top_radius + self.inner_bottom_radius) / 2
        area_inner_side = phi * inner_side_radius_avg * self.height * self.inner_x_z_ratio.view(B, 1)
        
        # 上表面面积（圆环面积，考虑椭圆）
        # 椭圆面积 = π * a * b，这里 a = outer_radius * x_z_ratio, b = outer_radius
        outer_top_area = math.pi * self.outer_top_radius * self.outer_top_radius * self.x_z_ratio.view(B, 1)
        inner_top_area = math.pi * self.inner_top_radius * self.inner_top_radius * self.inner_x_z_ratio.view(B, 1)
        area_top = (outer_top_area - inner_top_area) * phi / (2 * math.pi)
        
        # 下表面面积
        outer_bottom_area = math.pi * self.outer_bottom_radius * self.outer_bottom_radius * self.x_z_ratio.view(B, 1)
        inner_bottom_area = math.pi * self.inner_bottom_radius * self.inner_bottom_radius * self.inner_x_z_ratio.view(B, 1)
        area_bottom = (outer_bottom_area - inner_bottom_area) * phi / (2 * math.pi)
        
        # 总面积
        total_area = (area_outer_side + area_inner_side + area_top + area_bottom).mean()
        
        if total_area < 1e-6:
            num_parts = torch.full((4,), total_points // 4, device=device, dtype=torch.long)
            num_parts[-1] += total_points - num_parts.sum()
        else:
            areas = torch.stack([area_outer_side, area_inner_side, area_top, area_bottom], dim=1).mean(dim=0)
            probs = areas / total_area
            num_parts = (probs * total_points).long().clamp(min=0)
            num_parts[-1] += total_points - num_parts.sum()
        
        phi = phi.unsqueeze(-1)
        all_points = []
        
        # 外圆侧面采样
        N = num_parts[0].item()
        if N > 0:
            u = torch.rand(B, N, 1, device=device, dtype=dtype)
            v = torch.rand(B, N, 1, device=device, dtype=dtype)
            theta = u * phi
            r = (1 - v) * self.outer_bottom_radius.view(B, 1, 1) + v * self.outer_top_radius.view(B, 1, 1)
            x = r * self.x_z_ratio.view(B, 1, 1) * torch.cos(theta)
            y = (v - 0.5) * self.height.view(B, 1, 1)
            z = r * torch.sin(theta)
            pts = torch.cat([x, y, z], dim=-1)
            pts = rotate_6D(pts, self.rotation) + self.position.unsqueeze(1)
            all_points.append(pts)
        
        # 内圆侧面采样
        N = num_parts[1].item()
        if N > 0:
            u = torch.rand(B, N, 1, device=device, dtype=dtype)
            v = torch.rand(B, N, 1, device=device, dtype=dtype)
            theta = u * phi
            r = (1 - v) * self.inner_bottom_radius.view(B, 1, 1) + v * self.inner_top_radius.view(B, 1, 1)
            x = r * self.inner_x_z_ratio.view(B, 1, 1) * torch.cos(theta) + self.inner_offset[:, 0:1].view(B, 1, 1)
            y = (v - 0.5) * self.height.view(B, 1, 1) + self.inner_offset[:, 1:2].view(B, 1, 1)
            z = r * torch.sin(theta)
            pts = torch.cat([x, y, z], dim=-1)
            pts = rotate_6D(pts, self.rotation) + self.position.unsqueeze(1)
            all_points.append(pts)
        
        # 上表面采样（圆环）
        N = num_parts[2].item()
        if N > 0:
            u = torch.rand(B, N, 1, device=device, dtype=dtype)
            v = torch.rand(B, N, 1, device=device, dtype=dtype)
            theta = u * phi
            # 在内外半径之间均匀采样
            r = torch.sqrt(self.inner_top_radius.view(B, 1, 1)**2 + 
                          v * (self.outer_top_radius.view(B, 1, 1)**2 - self.inner_top_radius.view(B, 1, 1)**2))
            x = r * self.x_z_ratio.view(B, 1, 1) * torch.cos(theta)
            y = 0.5 * self.height.view(B, 1, 1).expand(B, N, 1)
            z = r * torch.sin(theta)
            pts = torch.cat([x, y, z], dim=-1)
            pts = rotate_6D(pts, self.rotation) + self.position.unsqueeze(1)
            all_points.append(pts)
        
        # 下表面采样（圆环）
        N = num_parts[3].item()
        if N > 0:
            u = torch.rand(B, N, 1, device=device, dtype=dtype)
            v = torch.rand(B, N, 1, device=device, dtype=dtype)
            theta = u * phi
            r = torch.sqrt(self.inner_bottom_radius.view(B, 1, 1)**2 + 
                          v * (self.outer_bottom_radius.view(B, 1, 1)**2 - self.inner_bottom_radius.view(B, 1, 1)**2))
            x = r * self.x_z_ratio.view(B, 1, 1) * torch.cos(theta)
            y = -0.5 * self.height.view(B, 1, 1).expand(B, N, 1)
            z = r * torch.sin(theta)
            pts = torch.cat([x, y, z], dim=-1)
            pts = rotate_6D(pts, self.rotation) + self.position.unsqueeze(1)
            all_points.append(pts)
        
        if not all_points:
            return torch.zeros((B, 0, 3), device=device, dtype=dtype)
        
        return torch.cat(all_points, dim=1)

class Torus(StructureNode):
    def __init__(self, central_radius, start_torus_radius, exist_angle=None, 
                 end_torus_radius=None, position=[0, 0, 0], rotation=[1, 0, 0, 0, 1, 0], 
                 rotation_order="XYZ", num_of_segment_center=8, num_of_segment_torus=8, 
                 Semantic=None, Affordance=None):
        """
        基于标准几何模板的高效圆环体构建 (Vectorized Implementation)
        """
        # -----------------------------------------------------------
        # 1. 参数预处理 (Input Standardization)
        # -----------------------------------------------------------
        dtype = central_radius.dtype
        device = central_radius.device
        B = central_radius.shape[0]
        
        # 统一维度处理函数
        def _ensure_tensor(val, target_shape_ref=None):
            if val is None and target_shape_ref is not None:
                return target_shape_ref.clone()
            if isinstance(val, (int, float)):
                return torch.full((B, 1), val, dtype=dtype, device=device)
            if not isinstance(val, torch.Tensor):
                val = torch.as_tensor(val, dtype=dtype, device=device)
            if val.ndim == 0:
                val = val.view(1, 1).repeat(B, 1)
            elif val.ndim == 1:
                val = val.view(1, -1).repeat(B, 1) if val.shape[0] != B else val.view(B, 1)
            return val

        # 填充缺失值并规范维度 [B, 1]
        exist_angle = _ensure_tensor(exist_angle, torch.full((B, 1), 2 * math.pi, dtype=dtype, device=device))
        start_torus_radius = _ensure_tensor(start_torus_radius)
        if end_torus_radius is None:
            end_torus_radius = start_torus_radius
        else:
            end_torus_radius = _ensure_tensor(end_torus_radius)
            
        central_radius = _ensure_tensor(central_radius)

        # 处理多维参数
        if isinstance(position, (list, tuple)):
            position = torch.as_tensor(position, dtype=dtype, device=device)
        if position.ndim == 1: 
            position = position.unsqueeze(0).repeat(B, 1) # [B, 3]
            
        if isinstance(rotation, (list, tuple)):
            rotation = torch.as_tensor(rotation, dtype=dtype, device=device)
        if rotation.ndim == 1: 
            rotation = rotation.unsqueeze(0).repeat(B, 1) # [B, 6]
        
        # 记录参数
        self.dtype = dtype
        self.device = device
        self.B = B
        self.central_radius = central_radius
        self.start_torus_radius = start_torus_radius
        self.exist_angle = exist_angle
        self.end_torus_radius = end_torus_radius
        self.num_of_segment_center = num_of_segment_center
        self.num_of_segment_torus = num_of_segment_torus
        self.position = position
        self.rotation = rotation
        self.rotation_order = rotation_order
        
        # -----------------------------------------------------------
        # 2. 向量化计算变形张量 (Vectorized Deformation)
        # -----------------------------------------------------------
        Nc = num_of_segment_center
        Nt = num_of_segment_torus
        
        # === A. 生成独立的中心点 (Start & End) ===
        # central_radius is [B, 1]
        # stack results in [B, 1, 3] -> 这里的维度已经是3维了，不需要再 unsqueeze(1)
        
        # Point 0: Start Center (angle 0) -> (R, 0, 0)
        p_start = torch.stack([
            central_radius, 
            torch.zeros_like(central_radius), 
            torch.zeros_like(central_radius)
        ], dim=-1) # [B, 1, 3]
        
        # Point 1: End Center (angle exist_angle)
        cos_exist = torch.cos(exist_angle) # [B, 1]
        sin_exist = torch.sin(exist_angle) # [B, 1]
        p_end = torch.stack([
            central_radius * cos_exist,
            torch.zeros_like(central_radius),
            central_radius * sin_exist
        ], dim=-1) # [B, 1, 3]
        
        # === B. 向量化生成网格顶点 ===
        
        # 1. 进度网格生成
        center_prog = torch.linspace(0, 1, Nc + 1, dtype=dtype, device=device).view(1, Nc+1, 1)
        torus_ang = torch.linspace(0, 2 * math.pi, Nt + 1, dtype=dtype, device=device).view(1, 1, Nt+1)
        
        # 2. 计算大圆路径上的参数 (广播到 [B, Nc+1, 1])
        theta_center = exist_angle.unsqueeze(1) * center_prog # [B, Nc+1, 1]
        cos_c = torch.cos(theta_center)
        sin_c = torch.sin(theta_center)
        
        # 3. 半径渐变插值
        r_start_view = start_torus_radius.unsqueeze(1) # [B, 1, 1]
        r_end_view = end_torus_radius.unsqueeze(1)     # [B, 1, 1]
        r_current = r_start_view * (1 - center_prog) + r_end_view * center_prog # [B, Nc+1, 1]
        
        # 4. 计算圆管截面参数
        cos_t = torch.cos(torus_ang) # [1, 1, Nt+1]
        sin_t = torch.sin(torus_ang) # [1, 1, Nt+1]
        
        # 5. 组合坐标
        # Y 轴
        y_grid = r_current * sin_t # [B, Nc+1, Nt+1]
        
        # XZ 平面投影长度
        R_major = central_radius.unsqueeze(1) # [B, 1, 1]
        outer_len = R_major + r_current * cos_t # [B, Nc+1, Nt+1]
        
        # X, Z 坐标
        x_grid = outer_len * cos_c
        z_grid = outer_len * sin_c
        
        # 6. 堆叠并展平
        grid_verts = torch.stack([x_grid, y_grid, z_grid], dim=-1) # [B, Nc+1, Nt+1, 3]
        grid_verts_flat = grid_verts.flatten(1, 2) # [B, (Nc+1)*(Nt+1), 3]
        
        # === C. 合并所有顶点 ===
        # p_start: [B, 1, 3]
        # p_end:   [B, 1, 3]
        # grid:    [B, N, 3]
        # Dim 1 check: 1 + 1 + N -> OK
        self.vertices = torch.cat([p_start, p_end, grid_verts_flat], dim=1)
        
        # -----------------------------------------------------------
        # 4. 应用几何变换 (Apply Transforms)
        # -----------------------------------------------------------
        self.vertices = rotate_6D(self.vertices, rotation) + position.unsqueeze(1)
        
        # -----------------------------------------------------------
        # 5. 定义拓扑特征 (Faces & Axes)
        # -----------------------------------------------------------
        self.Node_Face = {}
        self.Node_Axis = {}
        
        # === 1. 定义中心垂线 (Center Vertical Axis) ===
        # 轴起点：圆环体的几何中心 [B, 3]
        axis_p = self.position
        
        # 轴方向：局部 Y 轴 (0, 1, 0) 经过旋转后的方向
        # [B, 3]
        local_y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device).view(1, 3).repeat(B, 1)
        axis_d = rotate_6D(local_y_axis.unsqueeze(1), rotation).squeeze(1)
        
        self.Node_Axis[0] = {'p': axis_p, 'd': normalize(axis_d)}

        # === 2. 定义四个关键面 (Outer, Top, Inner, Bottom) ===
        if Nc > 0 and Nt > 0:
            # 基础参数
            base_idx = 2
            
            # 定义四个方向的逻辑索引 (j)
            # 0: Outer, Nt/4: Top, Nt/2: Inner, 3Nt/4: Bottom
            indices_j = {
                0: 0,                   # Face 0: Outer (外侧)
                1: Nt // 4,             # Face 1: Top   (顶部)
                2: Nt // 2,             # Face 2: Inner (内侧)
                3: (3 * Nt) // 4        # Face 3: Bottom(底部)
            }
            
            # 预定义局部坐标系下的法线 (n) 和切线 (t)
            # 假设我们在大圆角度 theta=0 处取点 (即 X 轴正方向起始处)
            # 此时：
            #   大圆切线 t 指向 +Z 方向 (0, 0, 1)
            #   法线 n 根据截面位置决定：
            #       Outer (最右):  +X (1, 0, 0) -> 指向外
            #       Top (最上):    +Y (0, 1, 0) -> 指向天
            #       Inner (最左):  -X (-1, 0, 0) -> 指向圆环中心 (孔)
            #       Bottom (最下): -Y (0, -1, 0) -> 指向地
            
            vectors_def = {
                0: {'n': [1.0, 0.0, 0.0],  't': [0.0, 0.0, 1.0]}, # Outer
                1: {'n': [0.0, 1.0, 0.0],  't': [0.0, 0.0, 1.0]}, # Top
                2: {'n': [-1.0, 0.0, 0.0], 't': [0.0, 0.0, 1.0]}, # Inner
                3: {'n': [0.0, -1.0, 0.0], 't': [0.0, 0.0, 1.0]}  # Bottom
            }

            for face_id, j in indices_j.items():
                # 1. 获取位置 (Position)
                # 直接取网格顶点，保证严格在表面上
                # 我们取 i=0 (theta=0) 处的截面
                idx_curr = base_idx + j
                v_curr = self.vertices[:, idx_curr, :] # [B, 3]
                face_center = v_curr
                
                # 2. 获取并旋转向量 (Vectors)
                # 构建局部向量 [B, 1, 3]
                def_n = torch.tensor(vectors_def[face_id]['n'], dtype=dtype, device=device)
                def_t = torch.tensor(vectors_def[face_id]['t'], dtype=dtype, device=device)
                
                local_n = def_n.view(1, 1, 3).repeat(B, 1, 1)
                local_t = def_t.view(1, 1, 3).repeat(B, 1, 1)
                
                # 应用物体的全局旋转
                # 注意：rotate_6D 仅旋转向量，不平移，这是正确的
                world_n = rotate_6D(local_n, rotation).squeeze(1)
                world_t = rotate_6D(local_t, rotation).squeeze(1)
                
                # 归一化 (防止旋转计算误差)
                n = normalize(world_n)
                t = normalize(world_t)
                
                # 计算副法线 b (Binormal)
                # 使用右手定则: n x t
                # 例如 Top: Y x Z = X (指向 Side 方向)，符合几何直觉
                b = torch.cross(n, t, dim=-1)
                
                self.Node_Face[face_id] = {'p': face_center, 'n': n, 't': t, 'b': b}
        
        # -----------------------------------------------------------
        # 6. 初始化父类
        # -----------------------------------------------------------
        super().__init__(position, rotation, rotation_order,
                         Node_Position=self.vertices, Node_Face=self.Node_Face, Node_Axis=self.Node_Axis,
                         Node_Semantic=Semantic, Node_Affordance=Affordance)
    
    def get_surface_points(self, total_points=600):
        """
        在圆环体表面上采样点（使用参数方程均匀采样）
        :param total_points: 采样的总点数
        :return: [B, total_points, 3]
        """
        if total_points <= 0:
            return torch.zeros((self.B, 0, 3), device=self.device, dtype=self.dtype)
        
        B = self.B
        device = self.device
        dtype = self.dtype
        
        # 使用参数方程在圆环体表面均匀采样
        # 参数 u: 中心圆角度 [0, exist_angle]
        # 参数 v: 圆环角度 [0, 2π]
        u = torch.rand(B, total_points, 1, device=device, dtype=dtype)
        v = torch.rand(B, total_points, 1, device=device, dtype=dtype)
        
        # 中心圆角度
        theta = u * self.exist_angle.view(B, 1, 1)  # [B, N, 1]
        
        # 圆环角度
        phi = v * 2 * math.pi  # [B, N, 1]
        
        # 插值圆环半径（根据中心圆角度）
        # 简化：使用平均半径
        torus_radius_avg = (self.start_torus_radius + self.end_torus_radius) / 2
        
        # 计算坐标
        # 先计算圆环上的点（相对于中心圆的局部坐标系）
        torus_x = torus_radius_avg.view(B, 1, 1) * torch.cos(phi)
        torus_y = torus_radius_avg.view(B, 1, 1) * torch.sin(phi)
        
        # 然后绕中心圆旋转
        # 中心圆半径 + 圆环在中心圆平面上的投影
        outer_radius = self.central_radius.view(B, 1, 1) + torus_x
        
        x = outer_radius * torch.cos(theta)
        y = torus_y
        z = outer_radius * torch.sin(theta)
        
        pts = torch.cat([x, y, z], dim=-1)
        
        # 应用全局变换
        pts = rotate_6D(pts, self.rotation) + self.position.unsqueeze(1)
        
        return pts

class Box_Cylinder_Ring(StructureNode):
    def __init__(self, outer_height, outer_length, outer_width, inner_radius, 
                 inner_cylinder_offset=[0, 0], position=[0, 0, 0], 
                 rotation=[1, 0, 0, 0, 1, 0], rotation_order="XYZ", 
                 num_of_segment=64, Semantic=None, Affordance=None):
        """
        基于标准几何模板的高效外框长方体+内框圆柱组合构建
        """
        # -----------------------------------------------------------
        # 1. 参数预处理 (Input Standardization)
        # -----------------------------------------------------------
        dtype = outer_height.dtype
        device = outer_height.device
        B = outer_height.shape[0]
        
        # 统一维度处理函数
        def _ensure_tensor(val):
            if isinstance(val, (int, float)):
                return torch.full((B, 1), val, dtype=dtype, device=device)
            if not isinstance(val, torch.Tensor):
                val = torch.as_tensor(val, dtype=dtype, device=device)
            if val.ndim == 0:
                val = val.view(1, 1).repeat(B, 1)
            elif val.ndim == 1:
                val = val.view(1, -1).repeat(B, 1) if val.shape[0] != B else val.view(B, 1)
            return val

        self.outer_height = _ensure_tensor(outer_height)
        self.outer_length = _ensure_tensor(outer_length)
        self.outer_width = _ensure_tensor(outer_width)
        self.inner_radius = _ensure_tensor(inner_radius)
        
        # 处理多维参数
        if isinstance(inner_cylinder_offset, (list, tuple)):
            inner_cylinder_offset = torch.as_tensor(inner_cylinder_offset, dtype=dtype, device=device)
        if inner_cylinder_offset.ndim == 1: 
            inner_cylinder_offset = inner_cylinder_offset.unsqueeze(0).repeat(B, 1) # [B, 2]
            
        if isinstance(position, (list, tuple)):
            position = torch.as_tensor(position, dtype=dtype, device=device)
        if position.ndim == 1: 
            position = position.unsqueeze(0).repeat(B, 1) # [B, 3]
            
        if isinstance(rotation, (list, tuple)):
            rotation = torch.as_tensor(rotation, dtype=dtype, device=device)
        if rotation.ndim == 1: 
            rotation = rotation.unsqueeze(0).repeat(B, 1) # [B, 6]
        
        # 记录参数
        self.dtype = dtype
        self.device = device
        self.B = B
        self.inner_cylinder_offset = inner_cylinder_offset
        self.num_of_segment = num_of_segment
        self.position = position
        self.rotation = rotation
        self.rotation_order = rotation_order
        
        # -----------------------------------------------------------
        # 2. 构建标准几何模板 (Geometry Template)
        # -----------------------------------------------------------
        # 顶点顺序：
        #   0-7: 外框长方体顶点
        #        0: (-x, y, z)  TLF (Top-Left-Front)
        #        1: ( x, y, z)  TRF
        #        2: (-x, y,-z)  TLB
        #        3: ( x, y,-z)  TRB
        #        4: (-x,-y, z)  BLF
        #        5: ( x,-y, z)  BRF
        #        6: (-x,-y,-z)  BLB
        #        7: ( x,-y,-z)  BRB
        #   8+: 内圆顶点
        
        # (A) 外框顶点模板 [-0.5, 0.5]
        outer_template = torch.tensor([
            [-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5], # 0, 1 (Top Front)
            [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5], # 2, 3 (Top Back)
            [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], # 4, 5 (Bottom Front)
            [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5]  # 6, 7 (Bottom Back)
        ], dtype=dtype, device=device)  # [8, 3]
        
        # (B) 内圆顶点模板
        angles = torch.linspace(0, 2 * math.pi, num_of_segment + 1, dtype=dtype, device=device)
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        
        inner_top = torch.stack([cos_a, torch.full_like(angles, 0.5), sin_a], dim=1)
        inner_btm = torch.stack([cos_a, torch.full_like(angles, -0.5), sin_a], dim=1)
        inner_template = torch.stack([inner_top, inner_btm], dim=1).reshape(-1, 3) # [2*(N+1), 3]
        
        # (C) 合并
        template = torch.cat([outer_template, inner_template], dim=0) # [Total, 3]
        self.vertices = template.unsqueeze(0).repeat(B, 1, 1)
        
        # -----------------------------------------------------------
        # 3. 计算变形张量 (Deformation / Resize)
        # -----------------------------------------------------------
        # (A) 外框缩放
        # Length=X, Height=Y, Width=Z
        dims_outer = torch.stack([self.outer_length, self.outer_height, self.outer_width], dim=-1) # [B, 1, 3]
        
        # (B) 内圆缩放 (X=Radius, Y=Height, Z=Radius)
        dims_inner = torch.stack([self.inner_radius, self.outer_height, self.inner_radius], dim=-1) # [B, 1, 3]
        
        # 广播缩放因子
        scale_outer = dims_outer.repeat(1, 8, 1)
        scale_inner = dims_inner.repeat(1, 2*(num_of_segment+1), 1)
        scale_factors = torch.cat([scale_outer, scale_inner], dim=1)
        
        self.vertices = self.vertices * scale_factors
        
        # (C) 偏移 (仅内圆)
        # inner_cylinder_offset: [B, 2] (X, Z)
        off_x = self.inner_cylinder_offset[:, 0:1]
        off_z = self.inner_cylinder_offset[:, 1:2]
        
        # 构造内圆偏移向量 [B, 2*(N+1), 3]
        offset_inner_vec = torch.stack([
            off_x.repeat(1, 2*(num_of_segment+1)),
            torch.zeros(B, 2*(num_of_segment+1), dtype=dtype, device=device),
            off_z.repeat(1, 2*(num_of_segment+1))
        ], dim=-1)
        
        # 外框无偏移
        offset_outer_vec = torch.zeros(B, 8, 3, dtype=dtype, device=device)
        
        self.vertices = self.vertices + torch.cat([offset_outer_vec, offset_inner_vec], dim=1)
        
        # -----------------------------------------------------------
        # 4. 应用几何变换 (Apply Transforms)
        # -----------------------------------------------------------
        self.vertices = rotate_6D(self.vertices, rotation) + position.unsqueeze(1)
        
        # -----------------------------------------------------------
        # 5. 定义拓扑特征 (Faces & Axes)
        # -----------------------------------------------------------
        self.Node_Face = {}
        self.Node_Axis = {}
        
        # === 定义六个面 (Cube Faces) ===
        # 索引参考 Geometry Template 部分
        # 0:TLF, 1:TRF, 2:TLB, 3:TRB, 4:BLF, 5:BRF, 6:BLB, 7:BRB
        
        # 定义面的顶点索引和解析法线 (局部坐标系)
        # Normals: Top(+Y), Bottom(-Y), Front(+Z), Back(-Z), Left(-X), Right(+X)
        faces_config = [
            {'name': 'Top',    'idx': [0, 1, 3, 2], 'n': [0, 1, 0],  't': [1, 0, 0]},
            {'name': 'Bottom', 'idx': [4, 5, 7, 6], 'n': [0, -1, 0], 't': [1, 0, 0]},
            {'name': 'Front',  'idx': [0, 4, 5, 1], 'n': [0, 0, 1],  't': [1, 0, 0]}, # Z+ Face
            {'name': 'Back',   'idx': [2, 3, 7, 6], 'n': [0, 0, -1], 't': [-1, 0, 0]},# Z- Face
            {'name': 'Left',   'idx': [0, 2, 6, 4], 'n': [-1, 0, 0], 't': [0, 0, 1]}, # X- Face
            {'name': 'Right',  'idx': [1, 5, 7, 3], 'n': [1, 0, 0],  't': [0, 0, -1]} # X+ Face
        ]
        
        for i, config in enumerate(faces_config):
            # 1. 中心点 P: 顶点取均值
            face_verts = self.vertices[:, config['idx'], :] # [B, 4, 3]
            face_center = torch.mean(face_verts, dim=1)
            
            # 2. 法线 N & 切线 T: 解析旋转
            local_n = torch.tensor(config['n'], dtype=dtype, device=device).view(1, 1, 3).repeat(B, 1, 1)
            local_t = torch.tensor(config['t'], dtype=dtype, device=device).view(1, 1, 3).repeat(B, 1, 1)
            
            n = normalize(rotate_6D(local_n, rotation).squeeze(1))
            t = normalize(rotate_6D(local_t, rotation).squeeze(1))
            b = torch.cross(n, t, dim=-1)
            
            self.Node_Face[i] = {'p': face_center, 'n': n, 't': t, 'b': b}
            
        # === 定义轴 (Axes) ===
        # 要求：圆柱中心轴 + 正方体所有棱 (1 + 12 = 13 Axes)
        
        # --- Axis 0: 圆柱中心轴 ---
        # 局部位置: (offset_x, 0, offset_z)
        # 局部方向: (0, 1, 0)
        cyl_offset_3d = torch.stack([
            self.inner_cylinder_offset[:, 0],
            torch.zeros(B, dtype=dtype, device=device),
            self.inner_cylinder_offset[:, 1]
        ], dim=-1).unsqueeze(1) # [B, 1, 3]
        
        # 计算轴上的点：Box中心 + 旋转后的Offset
        # 注意: self.position 是 Box 的中心
        axis0_p = self.position + rotate_6D(cyl_offset_3d, rotation).squeeze(1)
        
        # 计算轴方向：旋转后的 Y 轴
        local_y = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device).view(1, 1, 3).repeat(B, 1, 1)
        axis0_d = normalize(rotate_6D(local_y, rotation).squeeze(1))
        
        self.Node_Axis[0] = {'p': axis0_p, 'd': axis0_d}
        
        # --- Axis 1~12: 正方体的 12 条棱 ---
        # 定义棱的顶点对 (Start, End)
        edge_indices = [
            # Top Ring (4)
            (0, 1), (1, 3), (3, 2), (2, 0),
            # Bottom Ring (4)
            (4, 5), (5, 7), (7, 6), (6, 4),
            # Vertical Pillars (4)
            (0, 4), (1, 5), (3, 7), (2, 6)
        ]
        
        for i, (idx_s, idx_e) in enumerate(edge_indices):
            p_start = self.vertices[:, idx_s, :]
            p_end = self.vertices[:, idx_e, :]
            
            # 轴中心：棱的中点
            axis_p = (p_start + p_end) / 2
            # 轴方向：棱的方向
            axis_d = normalize(p_end - p_start)
            
            # 存入 Node_Axis (索引从 1 开始)
            self.Node_Axis[i + 1] = {'p': axis_p, 'd': axis_d}

        # -----------------------------------------------------------
        # 6. 初始化父类
        # -----------------------------------------------------------
        super().__init__(position, rotation, rotation_order,
                         Node_Position=self.vertices, Node_Face=self.Node_Face, Node_Axis=self.Node_Axis,
                         Node_Semantic=Semantic, Node_Affordance=Affordance)
    def get_surface_points(self, total_points=600):
        """
        在表面上采样点（包括外框长方体的6个面和内圆圆柱的侧面）
        :param total_points: 采样的总点数
        :return: [B, total_points, 3]
        """
        if total_points <= 0:
            return torch.zeros((self.B, 0, 3), device=self.device, dtype=self.dtype)
        
        B = self.B
        device = self.device
        dtype = self.dtype
        
        # 计算各部分面积以分配点数
        # 外框长方体：侧面、顶面、底面
        area_outer_front = self.outer_length * self.outer_height
        area_outer_back = self.outer_length * self.outer_height
        area_outer_left = self.outer_width * self.outer_height
        area_outer_right = self.outer_width * self.outer_height
        
        # 外框顶面和底面面积 = 矩形面积 - 内圆面积（圆环区域）
        pi_tensor = torch.tensor(math.pi, dtype=dtype, device=device)
        outer_box_area = self.outer_length * self.outer_width
        inner_circle_area = pi_tensor * self.inner_radius * self.inner_radius
        area_outer_top = outer_box_area - inner_circle_area
        area_outer_bottom = outer_box_area - inner_circle_area
        
        # 内圆圆柱：只有侧面（贯穿掏空，没有顶面和底面）
        phi = 2 * math.pi
        area_inner_side = phi * self.inner_radius * self.outer_height
        
        # 总面积（不包含内圆顶面和底面）
        total_area = (area_outer_top + area_outer_bottom + area_outer_front +
                     area_outer_back + area_outer_left + area_outer_right +
                     area_inner_side).mean()
        
        if total_area < 1e-6:
            # 如果面积无效，均匀分配
            num_parts = torch.full((7,), total_points // 7, device=device, dtype=torch.long)
            num_parts[-1] += total_points - num_parts.sum()
        else:
            areas = torch.stack([
                area_outer_top, area_outer_bottom, area_outer_front,
                area_outer_back, area_outer_left, area_outer_right,
                area_inner_side
            ], dim=1).mean(dim=0)  # [7]
            probs = areas / total_area
            num_parts = (probs * total_points).long().clamp(min=0)
            num_parts[-1] += total_points - num_parts.sum()
        
        all_points = []
        
        # 外框顶面（圆环区域，排除内圆）
        N = num_parts[0].item()
        if N > 0:
            pts_list = []
            for b in range(B):
                # 获取参数
                # 注意：外框顶面在XZ平面（Y=+outer_height/2），内圆也在XZ平面
                # 内圆偏移直接对应XZ平面
                ox = self.inner_cylinder_offset[b, 0].item() if self.inner_cylinder_offset.ndim > 1 else self.inner_cylinder_offset[0].item()
                oz = self.inner_cylinder_offset[b, 1].item() if self.inner_cylinder_offset.ndim > 1 else self.inner_cylinder_offset[1].item()
                r_inner = self.inner_radius[b].item() if self.inner_radius.ndim > 0 else self.inner_radius.item()
                half_len = (self.outer_length[b] / 2).item() if self.outer_length.ndim > 0 else (self.outer_length / 2).item()
                half_wid = (self.outer_width[b] / 2).item() if self.outer_width.ndim > 0 else (self.outer_width / 2).item()
                
                # 使用与Cylinder_Box_Ring相同的方法：在矩形内均匀采样，删除圆形内的点
                valid_points = []
                while len(valid_points) < N:
                    candidate_N = N * 3
                    u = torch.rand(candidate_N, device=device, dtype=dtype)
                    v = torch.rand(candidate_N, device=device, dtype=dtype)
                    x = (u - 0.5) * self.outer_length[b].item() if self.outer_length.ndim > 0 else (u - 0.5) * self.outer_length.item()
                    z = (v - 0.5) * self.outer_width[b].item() if self.outer_width.ndim > 0 else (v - 0.5) * self.outer_width.item()
                    
                    # 删除圆形内的点（在XZ平面内检查）
                    dist_from_center = torch.sqrt((x - ox)**2 + (z - oz)**2)
                    outside_circle = dist_from_center >= r_inner
                    valid_indices = torch.where(outside_circle)[0]
                    
                    for idx in valid_indices:
                        if len(valid_points) >= N:
                            break
                        valid_points.append([x[idx].item(), z[idx].item()])
                
                valid_points = torch.tensor(valid_points[:N], device=device, dtype=dtype)  # [N, 2]
                x_final = valid_points[:, 0]
                z_final = valid_points[:, 1]
                y_final = torch.full((N,), 0.5, device=device, dtype=dtype)
                if self.outer_height.ndim == 2:
                    y_final = y_final * self.outer_height[b, 0]
                else:
                    y_final = y_final * self.outer_height[b]
                
                batch_pts = torch.stack([x_final, y_final, z_final], dim=-1)  # [N, 3]
                pts_list.append(batch_pts.unsqueeze(0))
            
            pts = torch.cat(pts_list, dim=0)  # [B, N, 3]
            pts = rotate_6D(pts, self.rotation) + self.position.unsqueeze(1)
            all_points.append(pts)
        
        # 外框底面（圆环区域，排除内圆）
        N = num_parts[1].item()
        if N > 0:
            pts_list = []
            for b in range(B):
                # 获取参数（外框底面在XZ平面，Y=-outer_height/2，内圆也在XZ平面）
                ox = self.inner_cylinder_offset[b, 0].item() if self.inner_cylinder_offset.ndim > 1 else self.inner_cylinder_offset[0].item()
                oz = self.inner_cylinder_offset[b, 1].item() if self.inner_cylinder_offset.ndim > 1 else self.inner_cylinder_offset[1].item()
                r_inner = self.inner_radius[b].item() if self.inner_radius.ndim > 0 else self.inner_radius.item()
                
                # 使用与Cylinder_Box_Ring相同的方法
                valid_points = []
                while len(valid_points) < N:
                    candidate_N = N * 3
                    u = torch.rand(candidate_N, device=device, dtype=dtype)
                    v = torch.rand(candidate_N, device=device, dtype=dtype)
                    x = (u - 0.5) * self.outer_length[b].item() if self.outer_length.ndim > 0 else (u - 0.5) * self.outer_length.item()
                    z = (v - 0.5) * self.outer_width[b].item() if self.outer_width.ndim > 0 else (v - 0.5) * self.outer_width.item()
                    
                    # 删除圆形内的点（在XZ平面内检查）
                    dist_from_center = torch.sqrt((x - ox)**2 + (z - oz)**2)
                    outside_circle = dist_from_center >= r_inner
                    valid_indices = torch.where(outside_circle)[0]
                    
                    for idx in valid_indices:
                        if len(valid_points) >= N:
                            break
                        valid_points.append([x[idx].item(), z[idx].item()])
                
                valid_points = torch.tensor(valid_points[:N], device=device, dtype=dtype)  # [N, 2]
                x_final = valid_points[:, 0]
                z_final = valid_points[:, 1]
                y_final = torch.full((N,), -0.5, device=device, dtype=dtype)
                if self.outer_height.ndim == 2:
                    y_final = y_final * self.outer_height[b, 0]
                else:
                    y_final = y_final * self.outer_height[b]
                
                batch_pts = torch.stack([x_final, y_final, z_final], dim=-1)  # [N, 3]
                pts_list.append(batch_pts.unsqueeze(0))
            
            pts = torch.cat(pts_list, dim=0)  # [B, N, 3]
            pts = rotate_6D(pts, self.rotation) + self.position.unsqueeze(1)
            all_points.append(pts)
        
        # 外框前后面（Z方向的两个面，Z = ±outer_width/2）
        N_front_back = num_parts[2].item() + num_parts[3].item()
        if N_front_back > 0:
            u = torch.rand(B, N_front_back, 1, device=device, dtype=dtype)
            v = torch.rand(B, N_front_back, 1, device=device, dtype=dtype)
            x = (u - 0.5) * self.outer_length.view(B, 1, 1)
            y = (v - 0.5) * self.outer_height.view(B, 1, 1)
            # 前面 (z = +outer_width/2)
            z1 = 0.5 * self.outer_width.view(B, 1, 1).expand(B, N_front_back//2, 1)
            pts1 = torch.cat([x[:, :N_front_back//2], y[:, :N_front_back//2], z1], dim=-1)
            # 后面 (z = -outer_width/2)
            z2 = -0.5 * self.outer_width.view(B, 1, 1).expand(B, N_front_back - N_front_back//2, 1)
            pts2 = torch.cat([x[:, N_front_back//2:], y[:, N_front_back//2:], z2], dim=-1)
            pts = torch.cat([pts1, pts2], dim=1)
            pts = rotate_6D(pts, self.rotation) + self.position.unsqueeze(1)
            all_points.append(pts)
        
        # 外框左右面（X方向的两个面，X = ±outer_length/2）
        N_left_right = num_parts[4].item() + num_parts[5].item()
        if N_left_right > 0:
            u = torch.rand(B, N_left_right, 1, device=device, dtype=dtype)
            v = torch.rand(B, N_left_right, 1, device=device, dtype=dtype)
            y = (u - 0.5) * self.outer_height.view(B, 1, 1)
            z = (v - 0.5) * self.outer_width.view(B, 1, 1)
            # 右面 (x = +outer_length/2)
            x1 = 0.5 * self.outer_length.view(B, 1, 1).expand(B, N_left_right//2, 1)
            pts1 = torch.cat([x1, y[:, :N_left_right//2], z[:, :N_left_right//2]], dim=-1)
            # 左面 (x = -outer_length/2)
            x2 = -0.5 * self.outer_length.view(B, 1, 1).expand(B, N_left_right - N_left_right//2, 1)
            pts2 = torch.cat([x2, y[:, N_left_right//2:], z[:, N_left_right//2:]], dim=-1)
            pts = torch.cat([pts1, pts2], dim=1)
            pts = rotate_6D(pts, self.rotation) + self.position.unsqueeze(1)
            all_points.append(pts)
        
        # 内圆圆柱侧面（内圆在XZ平面内，Y是高度方向）
        N = num_parts[6].item()
        if N > 0:
            u = torch.rand(B, N, 1, device=device, dtype=dtype)
            v = torch.rand(B, N, 1, device=device, dtype=dtype)
            theta = u * phi
            r = self.inner_radius.view(B, 1, 1)
            x = r * torch.cos(theta) + self.inner_cylinder_offset[:, 0:1].view(B, 1, 1)  # X方向
            z = r * torch.sin(theta) + self.inner_cylinder_offset[:, 1:2].view(B, 1, 1)  # Z方向
            y = (v - 0.5) * self.outer_height.view(B, 1, 1)  # Y是高度方向
            pts = torch.cat([x, y, z], dim=-1)
            pts = rotate_6D(pts, self.rotation) + self.position.unsqueeze(1)
            all_points.append(pts)
        
        if not all_points:
            return torch.zeros((B, 0, 3), device=device, dtype=dtype)
        
        return torch.cat(all_points, dim=1)

class Cylinder_Box_Ring(StructureNode):
    def __init__(self, outer_radius, outer_height, inner_length, inner_width, 
                 inner_cuboid_offset=[0, 0], position=[0, 0, 0], 
                 rotation=[1, 0, 0, 0, 1, 0], rotation_order="XYZ", 
                 num_of_segment=64, Semantic=None, Affordance=None):
        """
        基于标准几何模板的高效外框圆柱+内框长方体组合构建
        (统一坐标系：Y轴为高度，XZ平面为横截面)
        :param outer_radius: [B, 1] 外圆半径
        :param outer_height: [B, 1] 外圆高度（Y轴方向）
        :param inner_length: [B, 1] 内框长度（X轴方向）
        :param inner_width: [B, 1] 内框宽度（Z轴方向）
        :param inner_cuboid_offset: [B, 2] 内框相对于外圆的偏移 [X, Z]
        """
        # -----------------------------------------------------------
        # 1. 参数预处理
        # -----------------------------------------------------------
        dtype = outer_height.dtype
        device = outer_height.device
        B = outer_height.shape[0]
        
        def _ensure_tensor(val):
            if isinstance(val, (int, float)):
                return torch.full((B, 1), val, dtype=dtype, device=device)
            if not isinstance(val, torch.Tensor):
                val = torch.as_tensor(val, dtype=dtype, device=device)
            if val.ndim == 0:
                val = val.view(1, 1).repeat(B, 1)
            elif val.ndim == 1:
                val = val.view(1, -1).repeat(B, 1) if val.shape[0] != B else val.view(B, 1)
            return val

        self.outer_radius = _ensure_tensor(outer_radius)
        self.outer_height = _ensure_tensor(outer_height)
        self.inner_length = _ensure_tensor(inner_length)
        self.inner_width = _ensure_tensor(inner_width)
        
        if isinstance(inner_cuboid_offset, (list, tuple)):
            inner_cuboid_offset = torch.as_tensor(inner_cuboid_offset, dtype=dtype, device=device)
        if inner_cuboid_offset.ndim == 1: 
            inner_cuboid_offset = inner_cuboid_offset.unsqueeze(0).repeat(B, 1)
            
        if isinstance(position, (list, tuple)):
            position = torch.as_tensor(position, dtype=dtype, device=device)
        if position.ndim == 1: 
            position = position.unsqueeze(0).repeat(B, 1)
            
        if isinstance(rotation, (list, tuple)):
            rotation = torch.as_tensor(rotation, dtype=dtype, device=device)
        if rotation.ndim == 1: 
            rotation = rotation.unsqueeze(0).repeat(B, 1)
        
        self.dtype = dtype
        self.device = device
        self.B = B
        self.inner_cuboid_offset = inner_cuboid_offset
        self.num_of_segment = num_of_segment
        self.position = position
        self.rotation = rotation
        self.rotation_order = rotation_order
        
        # -----------------------------------------------------------
        # 2. 构建标准几何模板 (Geometry Template)
        # -----------------------------------------------------------
        # (A) 内框顶点模板 (标准立方体)
        inner_template = torch.tensor([
            [-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5], # 0, 1 (Top Front)
            [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5], # 2, 3 (Top Back)
            [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], # 4, 5 (Bottom Front)
            [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5]  # 6, 7 (Bottom Back)
        ], dtype=dtype, device=device)  # [8, 3]
        
        # (B) 外圆顶点模板 (XZ平面圆, Y轴高度)
        angles = torch.linspace(0, 2 * math.pi, num_of_segment + 1, dtype=dtype, device=device)
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        
        # 【修正点1】: 外圆在 XZ 平面 (cos, height, sin)
        outer_top = torch.stack([cos_a, torch.full_like(angles, 0.5), sin_a], dim=1)
        outer_btm = torch.stack([cos_a, torch.full_like(angles, -0.5), sin_a], dim=1)
        
        outer_template = torch.stack([outer_top, outer_btm], dim=1).reshape(-1, 3) # [2*(N+1), 3]
        
        template = torch.cat([inner_template, outer_template], dim=0)
        self.vertices = template.unsqueeze(0).repeat(B, 1, 1)
        
        # -----------------------------------------------------------
        # 3. 计算变形张量 (Deformation / Resize)
        # -----------------------------------------------------------
        # (A) 内框缩放 [Length(X), Height(Y), Width(Z)]
        # 【修正点2】: 确保缩放维度对应正确
        dims_inner = torch.stack([self.inner_length, self.outer_height, self.inner_width], dim=-1) # [B, 1, 3]
        
        # (B) 外圆缩放 [Radius(X), Height(Y), Radius(Z)]
        dims_outer = torch.stack([self.outer_radius, self.outer_height, self.outer_radius], dim=-1) # [B, 1, 3]
        
        scale_inner = dims_inner.repeat(1, 8, 1)
        scale_outer = dims_outer.repeat(1, 2*(num_of_segment+1), 1)
        scale_factors = torch.cat([scale_inner, scale_outer], dim=1)
        
        self.vertices = self.vertices * scale_factors
        
        # (C) 偏移 (仅内框)
        # inner_cuboid_offset: [B, 2] -> [X, Z]
        off_x = self.inner_cuboid_offset[:, 0:1]
        off_z = self.inner_cuboid_offset[:, 1:2]
        
        # 【修正点3】: 偏移应用在 X 和 Z 轴
        offset_inner_vec = torch.stack([
            off_x.repeat(1, 8),
            torch.zeros(B, 8, dtype=dtype, device=device), # Y轴无偏移
            off_z.repeat(1, 8)
        ], dim=-1)
        
        offset_outer_vec = torch.zeros(B, 2*(num_of_segment+1), 3, dtype=dtype, device=device)
        self.vertices = self.vertices + torch.cat([offset_inner_vec, offset_outer_vec], dim=1)
        
        # -----------------------------------------------------------
        # 4. 应用几何变换
        # -----------------------------------------------------------
        self.vertices = rotate_6D(self.vertices, rotation) + position.unsqueeze(1)
        
        # -----------------------------------------------------------
        # 5. 定义拓扑特征 (Faces & Axes)
        # -----------------------------------------------------------
        # (代码略，保持上一轮优化后的 faces/axes 逻辑，与 Y-up 一致)
        self.Node_Face = {}
        self.Node_Axis = {}
        # ... (此处应填入上一轮生成的 Face/Axis 代码，已确保是Y-up) ...

        # 补全 Face/Axis 以保证代码完整可运行
        # Faces
        box_faces_config = [
            {'name': 'Top', 'idx': [0, 1, 3, 2], 'n': [0, 1, 0], 't': [1, 0, 0]},
            {'name': 'Bottom', 'idx': [4, 5, 7, 6], 'n': [0, -1, 0], 't': [1, 0, 0]},
            {'name': 'Front', 'idx': [0, 4, 5, 1], 'n': [0, 0, 1], 't': [1, 0, 0]},
            {'name': 'Back', 'idx': [2, 3, 7, 6], 'n': [0, 0, -1], 't': [-1, 0, 0]},
            {'name': 'Left', 'idx': [0, 2, 6, 4], 'n': [-1, 0, 0], 't': [0, 0, 1]},
            {'name': 'Right', 'idx': [1, 5, 7, 3], 'n': [1, 0, 0], 't': [0, 0, -1]}
        ]
        for i, config in enumerate(box_faces_config):
            face_verts = self.vertices[:, config['idx'], :]
            face_center = torch.mean(face_verts, dim=1)
            local_n = torch.tensor(config['n'], dtype=dtype, device=device).view(1, 1, 3).repeat(B, 1, 1)
            local_t = torch.tensor(config['t'], dtype=dtype, device=device).view(1, 1, 3).repeat(B, 1, 1)
            n = normalize(rotate_6D(local_n, rotation).squeeze(1))
            t = normalize(rotate_6D(local_t, rotation).squeeze(1))
            b = torch.cross(n, t, dim=-1)
            self.Node_Face[i] = {'p': face_center, 'n': n, 't': t, 'b': b}
            
        if num_of_segment > 0:
            v_top = self.vertices[:, 8, :]
            v_btm = self.vertices[:, 9, :]
            side_center = (v_top + v_btm) / 2
            local_n_side = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device).view(1, 1, 3).repeat(B, 1, 1)
            local_t_side = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device).view(1, 1, 3).repeat(B, 1, 1)
            n_side = normalize(rotate_6D(local_n_side, rotation).squeeze(1))
            t_side = normalize(rotate_6D(local_t_side, rotation).squeeze(1))
            b_side = torch.cross(n_side, t_side, dim=-1)
            self.Node_Face[6] = {'p': side_center, 'n': n_side, 't': t_side, 'b': b_side}

        # Axes
        axis0_p = self.position
        local_y = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device).view(1, 1, 3).repeat(B, 1, 1)
        axis0_d = normalize(rotate_6D(local_y, rotation).squeeze(1))
        self.Node_Axis[0] = {'p': axis0_p, 'd': axis0_d}
        
        edge_indices = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (3, 7), (2, 6)]
        for i, (idx_s, idx_e) in enumerate(edge_indices):
            p_start = self.vertices[:, idx_s, :]
            p_end = self.vertices[:, idx_e, :]
            axis_p = (p_start + p_end) / 2
            axis_d = normalize(p_end - p_start)
            self.Node_Axis[i + 1] = {'p': axis_p, 'd': axis_d}
        
        # -----------------------------------------------------------
        # 6. 初始化父类
        # -----------------------------------------------------------
        super().__init__(position, rotation, rotation_order,
                         Node_Position=self.vertices, Node_Face=self.Node_Face, Node_Axis=self.Node_Axis,
                         Node_Semantic=Semantic, Node_Affordance=Affordance)
            
    def get_surface_points(self, total_points=600):
        """
        在表面上采样点（包括外圆圆柱的侧面、顶面、底面和内框长方体的4个内壁侧面）
        修正版：完全适配 Y轴高度 + XZ截面
        """
        if total_points <= 0:
            return torch.zeros((self.B, 0, 3), device=self.device, dtype=self.dtype)
        
        B = self.B
        device = self.device
        dtype = self.dtype
        
        # --- 1. 计算面积 ---
        pi_val = math.pi
        
        # 外圆柱侧面 (2*pi*R*H)
        area_outer_side = (2 * pi_val * self.outer_radius * self.outer_height).squeeze(-1)
        
        # 外圆顶底面 (pi*R^2 - L*W)
        area_circle = pi_val * self.outer_radius * self.outer_radius
        area_rect = self.inner_length * self.inner_width
        area_cap = (area_circle - area_rect).clamp(min=0).squeeze(-1) # 单个Cap的面积
        
        # 内框内壁 (2*L*H + 2*W*H)
        area_inner_wall_x = (self.inner_length * self.outer_height).squeeze(-1) # Front/Back wall
        area_inner_wall_z = (self.inner_width * self.outer_height).squeeze(-1)  # Left/Right wall
        
        # 总面积 = 侧面 + 2*Cap + 2*FrontBack + 2*LeftRight
        total_area = area_outer_side + 2*area_cap + 2*area_inner_wall_x + 2*area_inner_wall_z
        
        # 计算每个部分的点数比例
        # 为了batch处理，取平均比例或者逐个计算
        # 这里为了简化，计算batch mean的比例来分配N
        mean_area = total_area.mean()
        if mean_area < 1e-6:
            return torch.zeros((B, total_points, 3), device=device, dtype=dtype)
            
        n_side = (area_outer_side.mean() / mean_area * total_points).long()
        n_cap = (area_cap.mean() / mean_area * total_points).long() # Top & Bottom
        n_wall_x = (area_inner_wall_x.mean() / mean_area * total_points).long() # Front & Back
        n_wall_z = (area_inner_wall_z.mean() / mean_area * total_points).long() # Left & Right
        
        # 补齐点数
        current_sum = n_side + 2*n_cap + 2*n_wall_x + 2*n_wall_z
        n_side += (total_points - current_sum)
        
        all_points = []
        
        # --- 2. 采样外圆侧面 ---
        if n_side > 0:
            # theta in [0, 2pi], h in [-0.5, 0.5]
            u = torch.rand(B, n_side, device=device, dtype=dtype)
            v = torch.rand(B, n_side, device=device, dtype=dtype)
            theta = u * 2 * pi_val
            
            # 【修正】：XZ平面圆，Y轴高度
            x = self.outer_radius * torch.cos(theta)
            z = self.outer_radius * torch.sin(theta)
            y = (v - 0.5) * self.outer_height
            
            pts = torch.stack([x, y, z], dim=-1) # [B, N, 3]
            all_points.append(pts)
            
        # --- 3. 采样顶面和底面 (Caps) ---
        # 难点：圆减矩形。使用 Rejection Sampling
        if n_cap > 0:
            for y_sign in [0.5, -0.5]: # Top and Bottom
                # 预估需要生成的点数，考虑到拒绝率
                # 假设矩形很小，效率~100%；矩形很大，效率低。
                # 简单起见，生成 3*N 个点筛选
                n_gen = n_cap.item() * 3
                
                # 在圆内均匀采样 (sqrt(u)*R, theta)
                u = torch.rand(B, n_gen, device=device, dtype=dtype)
                theta = torch.rand(B, n_gen, device=device, dtype=dtype) * 2 * pi_val
                
                r = torch.sqrt(u) * self.outer_radius # [B, n_gen]
                x_cand = r * torch.cos(theta)
                z_cand = r * torch.sin(theta)
                
                # 偏移后的内框范围
                off_x = self.inner_cuboid_offset[:, 0:1] # [B, 1]
                off_z = self.inner_cuboid_offset[:, 1:2] # [B, 1]
                hl = self.inner_length / 2
                hw = self.inner_width / 2
                
                # 判断是否在矩形内
                # [B, n_gen]
                in_rect = (torch.abs(x_cand - off_x) < hl) & (torch.abs(z_cand - off_z) < hw)
                mask = ~in_rect # 在圆内且在矩形外
                
                # 收集点
                cap_pts_list = []
                for b in range(B):
                    # 取出当前batch有效的点
                    valid_x = x_cand[b][mask[b]]
                    valid_z = z_cand[b][mask[b]]
                    
                    # 如果不够，补随机点 (简单补齐，或者循环生成直到够)
                    # 为防止死循环，如果实在不够就重复使用
                    curr_n = valid_x.shape[0]
                    target_n = n_cap.item()
                    
                    if curr_n >= target_n:
                        valid_x = valid_x[:target_n]
                        valid_z = valid_z[:target_n]
                    else:
                        # 补齐策略：重复填充
                        if curr_n == 0: # 极端情况：矩形覆盖了圆？
                             valid_x = torch.zeros(target_n, device=device, dtype=dtype)
                             valid_z = torch.zeros(target_n, device=device, dtype=dtype)
                        else:
                            repeat_factor = (target_n // curr_n) + 1
                            valid_x = valid_x.repeat(repeat_factor)[:target_n]
                            valid_z = valid_z.repeat(repeat_factor)[:target_n]
                            
                    y_val = torch.full((target_n,), y_sign * self.outer_height[b].item(), device=device, dtype=dtype)
                    cap_pts_list.append(torch.stack([valid_x, y_val, valid_z], dim=-1))
                
                all_points.append(torch.stack(cap_pts_list, dim=0))

        # --- 4. 采样内框内壁 ---
        # Front/Back Walls (Normal along Z)
        # 位于 z = offset_z +/- width/2
        # x范围: offset_x +/- length/2
        # y范围: +/- height/2
        if n_wall_x > 0:
            for z_sign in [0.5, -0.5]:
                u = torch.rand(B, n_wall_x, device=device, dtype=dtype)
                v = torch.rand(B, n_wall_x, device=device, dtype=dtype)
                
                # x 沿 Length 变化
                x = (u - 0.5) * self.inner_length + self.inner_cuboid_offset[:, 0:1]
                # y 沿 Height 变化
                y = (v - 0.5) * self.outer_height
                # z 固定
                z = torch.full_like(x, z_sign) * self.inner_width + self.inner_cuboid_offset[:, 1:2]
                
                all_points.append(torch.stack([x, y, z], dim=-1))
        
        # Left/Right Walls (Normal along X)
        # 位于 x = offset_x +/- length/2
        # z范围: offset_z +/- width/2
        if n_wall_z > 0:
            for x_sign in [0.5, -0.5]:
                u = torch.rand(B, n_wall_z, device=device, dtype=dtype)
                v = torch.rand(B, n_wall_z, device=device, dtype=dtype)
                
                # z 沿 Width 变化
                z = (u - 0.5) * self.inner_width + self.inner_cuboid_offset[:, 1:2]
                # y 沿 Height 变化
                y = (v - 0.5) * self.outer_height
                # x 固定
                x = torch.full_like(z, x_sign) * self.inner_length + self.inner_cuboid_offset[:, 0:1]
                
                all_points.append(torch.stack([x, y, z], dim=-1))
        
        # --- 5. 合并与旋转 ---
        points_local = torch.cat(all_points, dim=1) # [B, Total, 3]
        
        # 应用旋转和位移
        points_world = rotate_6D(points_local, self.rotation) + self.position.unsqueeze(1)
        
        return points_world
    
def get_arrow(origin, direction, color=[0, 1, 0], scale=0.2):
    """
    创建一个箭头几何体来表示向量
    """
    # 归一化方向
    direction = np.array(direction)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return o3d.geometry.TriangleMesh()
    direction = direction / norm
    
    # 创建箭头
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.02 * scale,
        cone_radius=0.04 * scale,
        cylinder_height=0.8 * scale,
        cone_height=0.2 * scale
    )
    arrow.paint_uniform_color(color)
    
    # 计算旋转矩阵将箭头从默认的(0,0,1)或(0,1,0)旋转到目标方向
    # Open3D create_arrow 默认指向 Z轴正方向 (0, 0, 1)
    default_dir = np.array([0, 0, 1])
    
    # 计算旋转轴和角度
    cross = np.cross(default_dir, direction)
    dot = np.dot(default_dir, direction)
    
    # 如果方向平行
    if np.linalg.norm(cross) < 1e-6:
        if dot < 0: # 反向
            R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        else:
            R = np.eye(3)
    else:
        # Rodrigues 旋转公式
        import math
        angle = math.acos(np.clip(dot, -1.0, 1.0))
        axis = cross / np.linalg.norm(cross)
        
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
    arrow.rotate(R, center=[0, 0, 0])
    arrow.translate(origin)
    return arrow

def visualize_node(node, batch_idx=0, num_samples=2000, show_axis=True, show_faces=True):
    """
    可视化单个 StructureNode
    :param node: StructureNode 实例 (Cuboid, Cylinder, Sphere)
    :param batch_idx:由于输入是 Batched Tensor，指定要查看第几个样本
    :param num_samples: 表面采样点数
    """
    geometries = []

    # --- 1. 可视化表面点云 (Surface Points) - 灰色 ---
    if hasattr(node, 'get_surface_points'):
        # 获取采样点
        surface_points = node.get_surface_points(total_points=num_samples)
        # 取出指定 batch 并转为 numpy
        pts = surface_points[batch_idx].detach().cpu().numpy()
        
        if pts.shape[0] > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.paint_uniform_color([0.7, 0.7, 0.7]) # 灰色
            geometries.append(pcd)

    # --- 2. 可视化骨架顶点 (Vertices) - 红色 ---
    if hasattr(node, 'vertices'):
        verts = node.vertices[batch_idx].detach().cpu().numpy()
        
        # 【修改】: 使用球体代替点云，这样可以控制大小 (radius)
        for v in verts:
            # radius=0.05 是球的半径，觉得还不够大可以改成 0.1
            mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            mesh_sphere.translate(v) # 将球移动到顶点位置
            mesh_sphere.paint_uniform_color([1.0, 0.0, 0.0]) # 红色
            mesh_sphere.compute_vertex_normals() # 计算法线，让球体有光照立体感
            geometries.append(mesh_sphere)

        # 原来的代码 (点太小，已注释)
        # pcd_verts = o3d.geometry.PointCloud()
        # pcd_verts.points = o3d.utility.Vector3dVector(verts)
        # pcd_verts.paint_uniform_color([1.0, 0.0, 0.0]) 
        # geometries.append(pcd_verts)

    # --- 3. 可视化 Node Interface: Face (法线) - 绿色箭头 ---
    if show_faces and hasattr(node, 'Node_Face') and node.Node_Face:
        # Node_Face 是一个字典，遍历所有定义的 Interface
        for key, face_data in node.Node_Face.items():
            # 获取中心点 p 和法线 n
            if 'p' in face_data and 'n' in face_data:
                p = face_data['p']
                n = face_data['n']
                
                # 处理 Tensor 维度
                if isinstance(p, torch.Tensor):
                    p_np = p[batch_idx].detach().cpu().numpy() if p.ndim > 1 else p.detach().cpu().numpy()
                else: continue
                    
                if isinstance(n, torch.Tensor):
                    n_np = n[batch_idx].detach().cpu().numpy() if n.ndim > 1 else n.detach().cpu().numpy()
                else: continue

                # 绘制法线箭头
                arrow = get_arrow(p_np, n_np, color=[0, 0.8, 0], scale=0.3)
                geometries.append(arrow)

    # --- 4. 可视化 Node Interface: Axis (轴向) - 品红色箭头 ---
    if show_axis and hasattr(node, 'Node_Axis') and node.Node_Axis:
        # Node_Axis 同样可能是字典或列表，根据代码逻辑处理
        axes_data = node.Node_Axis
        if isinstance(axes_data, dict):
            # 兼容 Sphere (单个 dict) 和 Cuboid/Cylinder (dict of dicts)
            iter_items = axes_data.items() if 'p' not in axes_data else [(0, axes_data)]
            
            for key, axis_data in iter_items:
                if 'p' in axis_data and 'd' in axis_data:
                    p = axis_data['p']
                    d = axis_data['d']
                    
                    if isinstance(p, torch.Tensor):
                        p_np = p[batch_idx].detach().cpu().numpy() if p.ndim > 1 else p.detach().cpu().numpy()
                    if isinstance(d, torch.Tensor):
                        d_np = d[batch_idx].detach().cpu().numpy() if d.ndim > 1 else d.detach().cpu().numpy()
                        
                    # 绘制轴向箭头
                    arrow = get_arrow(p_np, d_np, color=[1.0, 0.0, 1.0], scale=0.5)
                    geometries.append(arrow)

    # --- 5. 添加世界坐标系原点 ---
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    geometries.append(coord_frame)

    # 渲染
    print(f"Visualizing Node: {type(node).__name__}")
    print(f"  - Surface Points: {len(pts) if 'pts' in locals() else 0}")
    print(f"  - Vertices: {len(verts) if 'verts' in locals() else 0}")
    o3d.visualization.draw_geometries(geometries, window_name=f"Visualize {type(node).__name__}")
  
if __name__ == "__main__":
    from argparse import ArgumentParser
    
    Parser = ArgumentParser()
    Parser.add_argument("-P", "--Primitive", type=str, default="Cuboid", help="Cuboid, Cylinder, Sphere, Rectangular_Ring")
    args = Parser.parse_args()
    primitive_type = args.Primitive
    
    print(f"Generating {primitive_type}...")
    
    if primitive_type == "Cylinder":
        height = torch.tensor([1.5])
        top_radius = torch.tensor([0.5])
        rotation = torch.tensor([0.707, -0.707, 0, 0.707, 0.707, 0.000]) 
        Node = Cylinder(height=height, top_radius=top_radius, is_half=False, rotation=rotation, position=[0, 0, 0])
        
    elif primitive_type == "Sphere":
        radius = torch.tensor([0.8])
        radius_y = torch.tensor([1.2])
        radius_z = torch.tensor([0.5])
        Node = Sphere(radius=radius, radius_y=radius_y, radius_z=radius_z, position=[0, 0, 0])
    
    elif primitive_type == "Rectangular_Ring":
        front_height = torch.tensor([1.0])
        back_height = torch.tensor([1.5])
        
        outer_top_length = torch.tensor([2.0])
        outer_top_width = torch.tensor([2.0])
        outer_bottom_length = torch.tensor([2.5])
        outer_bottom_width = torch.tensor([2.5])
        
        inner_top_length = torch.tensor([1.0])
        inner_top_width = torch.tensor([1.0])
        inner_bottom_length = torch.tensor([1.2]) 
        inner_bottom_width = torch.tensor([1.2])
        
        inner_offset = [0.3, 0.2] 
        top_bottom_offset = [0.0, 0.0]
        
        position = [0.0, 0.0, 0.0]
        rotation = [1, 0, 0, 0, 1, 0]
        
        Node = Rectangular_Ring(
            front_height=front_height,
            back_height=back_height,
            outer_top_length=outer_top_length,
            outer_top_width=outer_top_width,
            outer_bottom_length=outer_bottom_length,
            outer_bottom_width=outer_bottom_width,
            inner_top_length=inner_top_length,
            inner_top_width=inner_top_width,
            inner_bottom_length=inner_bottom_length,
            inner_bottom_width=inner_bottom_width,
            inner_offset=inner_offset,
            top_bottom_offset=top_bottom_offset,
            position=position,
            rotation=rotation
        )
    elif primitive_type == 'Trianguler_Prism':
        # 创建三角柱体
        height = torch.tensor([1.5]).unsqueeze(0)
        top_radius = torch.tensor([0.6]).unsqueeze(0)
        bottom_radius = torch.tensor([0.6]).unsqueeze(0)
        Node = Trianguler_Prism(
            height=height,
            top_radius=top_radius, #上表面三角形的外接圆半径
            bottom_radius=bottom_radius, #下表面三角形的外接圆半径
            position=[0, 0, 0],
            rotation=[0.707, -0.707, 0, 0.707, 0.707, 0.000]
        )
    elif primitive_type == 'Cone':
        # 创建圆锥体
        radius = torch.tensor([0.7]).unsqueeze(0)
        height = torch.tensor([1.5]).unsqueeze(0)
        radius_z = torch.tensor([0.7]).unsqueeze(0)  # 可以不同，形成椭圆锥
        tip_offset = [0.1, 0.0]  # 顶点偏移
        Node = Cone(
            radius=radius,
            height=height,
            tip_offset=tip_offset,
            radius_z=radius_z,
            position=[1, 0, 0],
            rotation=[0.707, -0.707, 0, 0.707, 0.707, 0.000],
            num_of_segment=64
        )        
    elif primitive_type == 'Ring':
        # 创建圆环
        height = torch.tensor([1.0])
        outer_top_radius = torch.tensor([0.8])
        inner_top_radius = torch.tensor([0.5])
        outer_bottom_radius = torch.tensor([0.8])
        inner_bottom_radius = torch.tensor([0.5])
        back_height = torch.tensor([1.2])  # 后母线高度
        generatrix_offset = torch.tensor([0.0])  # 前后母线偏移
        x_z_ratio = torch.tensor([1.0])  # X/Z 轴半径比
        inner_x_z_ratio = torch.tensor([1.0])
        inner_offset = [0.0, 0.0]
        Node = Ring(
            height=height,
            outer_top_radius=outer_top_radius,
            inner_top_radius=inner_top_radius,
            outer_bottom_radius=outer_bottom_radius,
            inner_bottom_radius=inner_bottom_radius,
            back_height=back_height,
            generatrix_offset=generatrix_offset,
            x_z_ratio=x_z_ratio,
            inner_x_z_ratio=inner_x_z_ratio,
            inner_offset=inner_offset,
            position=[0, 0, 2],
            rotation=[1, 0, 0, 0, 1, 0],
            num_of_segment=8
        )
    elif primitive_type == 'Torus':
        # 创建圆环体（甜甜圈形状）
        central_radius = torch.tensor([1.0]).unsqueeze(0)  # 中心圆半径
        start_torus_radius = torch.tensor([0.4]).unsqueeze(0)  # 圆环半径
        end_torus_radius = torch.tensor([0.4]).unsqueeze(0)  # 可以不同，形成渐变
        exist_angle = None  # None 表示完整 2π
        Node = Torus(
            central_radius=central_radius,
            start_torus_radius=start_torus_radius,
            end_torus_radius=end_torus_radius,
            exist_angle=exist_angle,
            position=[2, 0, 0],
            rotation=[1, 0, 0, 0, 1, 0],
            num_of_segment_center=6,
            num_of_segment_torus=16
        )
    
    elif primitive_type == 'Box_Cylinder_Ring':
        # 创建外框长方体+内框圆柱的组合体
        outer_height = torch.tensor([1.2])
        outer_length = torch.tensor([1.5])
        outer_width = torch.tensor([1.5])
        inner_radius = torch.tensor([0.5])
        inner_cylinder_offset = [0.1, 0.0]  # 内圆偏移
        Node = Box_Cylinder_Ring(
            outer_height=outer_height,
            outer_length=outer_length,
            outer_width=outer_width,
            inner_radius=inner_radius,
            inner_cylinder_offset=inner_cylinder_offset,
            position=[0, 0, -2],
            rotation=[1, 0, 0, 0, 1, 0],
            num_of_segment=64
        )
    
    elif primitive_type == 'Cylinder_Box_Ring':
        # 创建外框圆柱+内框长方体的组合体
        outer_radius = torch.tensor([0.8])
        outer_height = torch.tensor([1.2])
        inner_length = torch.tensor([0.6])
        inner_width = torch.tensor([0.6])
        inner_cuboid_offset = [0.1, 0.0]  # 内框偏移
        Node = Cylinder_Box_Ring(
            outer_radius=outer_radius,
            outer_height=outer_height,
            inner_length=inner_length,
            inner_width=inner_width,
            inner_cuboid_offset=inner_cuboid_offset,
            position=[-2, 0, 0],
            rotation=[1, 0, 0, 0, 1, 0],
            num_of_segment=64
        )
    else:
        height = torch.tensor([1.0])
        top_length = torch.tensor([1.0])
        top_width = torch.tensor([0.5])
        top_offset = [0.2, 0.0] 
        Node = Cuboid(height=height, top_length=top_length, top_width=top_width, top_offset=top_offset)

    # 2. 调用可视化函数
    try:
        visualize_node(Node, batch_idx=0, num_samples=3000)
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()