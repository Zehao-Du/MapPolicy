import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean
from mappolicy.helper.pytorch import safe_normalize

MAP_NODE_VOCAB = {
    # metaworld
    "basketball": 4,
    "bin-picking": 5,
    "box-close": 6,
    "coffee-pull": 4,
    "coffee-push": 4,
    "disassemble": 3,
    "hammer": 3,
    "hand-insert": 2,
    "handle-pull": 3,
    "handle-pull-side": 3,
    "lever-pull": 4,
    "peg-insert-side": 3,
    "peg-unplug-side": 3,
    "pick-out-of-hole": 3,
    "pick-place": 1,
    "pick-place-wall": 3,
    "push": 1,
    "push-back": 1,
    "push-wall": 3,
    "reach-wall": 2,
    "shelf-place": 4,
    "soccer": 5,
    "stick-pull": 4,
    "sweep": 1,
    "sweep-into": 3,

    # rlbench
    "close_box": 3,
    "close_laptop_lid": 3,
    "put_rubbish_in_bin": 4,
    "toilet_seat_down": 4,
    "unplug_charger": 3,
    "water_plants": 5,

    # maniskill
    "PickCube-v1": 1,
    "PegInsertionSide-v1": 3,
    "StackCube-v1": 3,
    "PlugCharger-v1": 6,
    "StackPyramid-v1": 3,
    "PullCubeTool-v1": 3,
}

# -----------------------------------------------------------------------------
# 1. 基础组件: PointNet 用于处理不定长点云
# -----------------------------------------------------------------------------
class PointNetFeatureExtractor(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, out_dim=128):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x, batch_idx, num_nodes):
        """
        x: [Total_Points, 3]
        batch_idx: [Total_Points] (指向 Flatten 后的 Global Node ID)
        num_nodes: B * N
        """
        if x.size(0) == 0:
            return torch.zeros((num_nodes, self.mlp2[-1].out_features), device=x.device)

        # Point-wise MLP
        x = self.mlp1(x) # [Total_Points, H]
        
        # Max Pooling per Node (Scatter Max)
        # out: [Total_Nodes, H]
        x_global, _ = scatter_max(x, batch_idx, dim=0, dim_size=num_nodes)
        
        # Post MLP
        x_out = self.mlp2(x_global) # [Total_Nodes, Out]
        return x_out

# -----------------------------------------------------------------------------
# 2. Encoders: 将原始特征映射到隐空间
# -----------------------------------------------------------------------------

class RoboNodeEncoder(nn.Module):
    def __init__(self, sem_dim=512, hidden_dim=768):
        super().__init__()
        # Position PointNet
        self.pos_encoder = PointNetFeatureExtractor(input_dim=3, hidden_dim=64, out_dim=hidden_dim)
        # Affordance PointNet
        self.aff_encoder = PointNetFeatureExtractor(input_dim=3, hidden_dim=64, out_dim=hidden_dim)
        # Semantic Projection
        self.sem_proj = nn.Linear(sem_dim, hidden_dim)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, data):
        # 1. Point Clouds -> Node Features [B*N, H]
        h_pos = self.pos_encoder(data.x_pos, data.pos_batch_idx, data.num_nodes)
        h_aff = self.aff_encoder(data.x_aff, data.aff_batch_idx, data.num_nodes)
        
        # 2. Semantic -> Node Features [B*N, H]
        h_sem = self.sem_proj(data.x_sem)
        
        # 3. Fuse
        h_node = self.fusion(torch.cat([h_pos, h_aff, h_sem], dim=-1))
        return h_node

class RoboEdgeEncoder(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        # Inputs from StructureGraph._Batch_Graph:
        # edge_type: [M, 1] (Indices)
        # edge_param: [M, 3]
        # edge_anchor: [M, 24] (12 for src + 12 for dst)
        # edge_pose: [M, 9] (3 pos + 6 rot)
        
        self.type_embed = nn.Embedding(10, hidden_dim) # 7 types + padding
        self.param_proj = nn.Linear(3, hidden_dim)
        self.anchor_proj = nn.Linear(24, hidden_dim)
        self.pose_proj = nn.Linear(9, hidden_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, data):
        h_type = self.type_embed(data.edge_type.squeeze(-1).long())
        h_param = self.param_proj(data.edge_param.float())
        h_anchor = self.anchor_proj(data.edge_anchor.float())
        h_pose = self.pose_proj(data.edge_pose.float())
        
        h_edge = self.fusion(torch.cat([h_type, h_param, h_anchor, h_pose], dim=-1))
        return h_edge # [Total_Edges, H]

# -----------------------------------------------------------------------------
# 3. Backbone: Graphormer Layer (Dense Calculation)
# -----------------------------------------------------------------------------
class RoboGraphormerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # Edge Update
        self.edge_update_E = nn.Linear(hidden_dim, hidden_dim)
        self.edge_update_Src = nn.Linear(hidden_dim, hidden_dim)
        self.edge_update_Dst = nn.Linear(hidden_dim, hidden_dim)
        
        # Attention Bias Projection
        self.bias_proj = nn.Linear(hidden_dim, num_heads)
        
        # Self Attention
        self.attn_ln = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Node FFN
        self.ffn_ln = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, h, e, mask=None):
        """
        h: [B, N, D]
        e: [B, N, N, D]
        mask: [B, N, N] (Optional, for padding or non-existing edges)
        """
        B, N, D = h.shape
        
        # --- 1. Edge Update ---
        # e_new = ReLU(W_e*e + W_src*h_i + W_dst*h_j)
        h_src = self.edge_update_Src(h).unsqueeze(2) # [B, N, 1, D]
        h_dst = self.edge_update_Dst(h).unsqueeze(1) # [B, 1, N, D]
        e_feat = self.edge_update_E(e)
        
        e_new = F.relu(e_feat + h_src + h_dst) # [B, N, N, D]
        
        # --- 2. Biased Attention ---
        # Bias: [B, N, N, Heads] -> [B, Heads, N, N] -> [B*Heads, N, N]
        attn_bias = self.bias_proj(e_new).permute(0, 3, 1, 2).reshape(B * self.num_heads, N, N)
        
        # Residual Node Feature
        residual = h
        h = self.attn_ln(h)
        
        # PyTorch MHA: attn_mask shape [N*B, L, S] or [B*Heads, N, N]
        # 注意: 如果某些边不存在(mask)，attn_bias 对应位置应为 -inf
        h, _ = self.self_attn(h, h, h, attn_mask=attn_bias)
        h = residual + h
        
        # --- 3. FFN ---
        residual = h
        h = self.ffn_ln(h)
        h = self.ffn(h)
        h = residual + h
        
        return h, e_new

# -----------------------------------------------------------------------------
# 4. Auxiliary Head: 物理约束预测
# -----------------------------------------------------------------------------
class MathConstraintHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # 目标：从 Edge Feature 中恢复出该边连接的两个几何 Anchor
        # 每个 Anchor 包含: Normal(3), Tangent(3), Bitangent(3), Position(3) = 12 dims
        # 输出总维度: 12 (Anchor_i) + 12 (Anchor_j) = 24
        
        self.anchor_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 24) 
        )

    def forward(self, e_final):
        """
        Input:
            e_final: [B, N, N, D] (Dense Edge Embeddings from the last layer)
        Output:
            pred_anchors: [B, N, N, 24]
        """
        return self.anchor_generator(e_final)

# -----------------------------------------------------------------------------
# 5. 主模型: RoboGraphormer
# -----------------------------------------------------------------------------
class RoboGraphormer(nn.Module):
    def __init__(self, map_name, hidden_dim=768, num_layers=4, num_heads=8):
        super().__init__()
        if map_name not in MAP_NODE_VOCAB:
            raise ValueError(f"Unknown map_name: {map_name}. Available: {list(MAP_NODE_VOCAB.keys())}")
        self.map_name = map_name
        self.N = MAP_NODE_VOCAB[map_name] # 每个 Graph 固定的节点数
        self.hidden_dim = hidden_dim
        self.feature_dim = hidden_dim
        
        # Encoders
        self.node_encoder = RoboNodeEncoder(hidden_dim=hidden_dim)
        self.edge_encoder = RoboEdgeEncoder(hidden_dim=hidden_dim)
        
        # Transformer Layers
        self.layers = nn.ModuleList([
            RoboGraphormerLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        
        # Auxiliary Head
        self.math_head = MathConstraintHead(hidden_dim)
        
        # Readout
        self.gate_nn = nn.Linear(hidden_dim, 1)
        self.readout_proj = nn.Linear(hidden_dim, hidden_dim)

    def to_dense(self, h_sparse, e_sparse, data):
        """
        Helper: Convert flattened PyG features to Dense [B, N, ...] tensors.
        """
        B = h_sparse.shape[0] // self.N
        
        # 1. Node Dense: [B, N, D]
        h_dense = h_sparse.view(B, self.N, self.hidden_dim)
        
        # 2. Edge Dense: [B, N, N, D]
        # 创建空 Tensor
        e_dense = torch.zeros(B, self.N, self.N, self.hidden_dim, device=h_sparse.device)
        
        # 使用 edge_index 填充
        # data.edge_index 是 [2, B*M]
        # 我们需要计算每个 edge 在 Dense Tensor 中的 batch 索引
        
        if data.edge_index.size(1) > 0:  # 如果有边
            src, dst = data.edge_index
            # 计算 Batch Index: src // N
            batch_idx = src // self.N 
            # 计算 Local Node Index: src % N
            local_src = src % self.N
            local_dst = dst % self.N
            
            # 赋值
            e_dense[batch_idx, local_src, local_dst] = e_sparse
            # 双向
            e_dense = e_dense + e_dense.transpose(1, 2)
        
        return h_dense, e_dense

    def forward(self, data):
        # 1. Encode Features
        h_sparse = self.node_encoder(data) 
        if data.edge_index.size(1) > 0:  # 如果有边
            e_sparse = self.edge_encoder(data)
        else:
            e_sparse = torch.zeros(0, self.hidden_dim, device=h_sparse.device)
        
        # 2. Convert to Dense for Transformer
        h, e = self.to_dense(h_sparse, e_sparse, data)
        
        # 3. Backbone Flow (Edge Update + Biased Attention)
        for layer in self.layers:
            h, e = layer(h, e)
        
        # ==========================================================
        # 4. Math Loss Calculation (Before Aggregation)
        # ==========================================================
        loss_dict = {}
        if self.training:
            # 4.1 生成 Predicted Anchors
            dense_preds = self.math_head(e)
            
            # 检查是否有边
            if data.edge_index.size(1) > 0:
                src, dst = data.edge_index
                batch_idx = src // self.N
                local_src = src % self.N
                local_dst = dst % self.N
                
                # [Total_Edges, 24]
                valid_preds = dense_preds[batch_idx, local_src, local_dst]
                
                # 4.3 计算具体的几何约束 Loss
                math_loss, ortho_loss = self.compute_math_loss(valid_preds, data)
                loss_dict['math_loss'] = math_loss
                loss_dict['ortho_loss'] = ortho_loss
            else:
                # 没有边时返回0损失
                loss_dict['math_loss'] = torch.tensor(0.0, device=h_sparse.device)
                loss_dict['ortho_loss'] = torch.tensor(0.0, device=h_sparse.device)

        # 5. Readout (Aggregation)
        score = self.gate_nn(torch.tanh(self.readout_proj(h)))
        weights = F.softmax(score, dim=1)
        z_graph = torch.sum(h * weights, dim=1)
        
        return z_graph, loss_dict

    def compute_math_loss(self, preds, data):
        """
        preds: [M, 24] Predicted anchors for node i and node j
        data: PyG batch containing edge_type, edge_param
        """
        # --- A. 解析预测向量 ---
        # 约定: 前12位是 Node i 的 Anchor, 后12位是 Node j 的 Anchor
        # Anchor 结构: [Normal(3), Tangent(3), Bitangent(3), Position(3)]
        
        # 提取 Node i 的向量
        n_i, t_i, b_i, p_i = preds[:, 0:3], preds[:, 3:6], preds[:, 6:9], preds[:, 9:12]
        
        # 提取 Node j 的向量
        n_j, t_j, b_j, p_j = preds[:, 12:15], preds[:, 15:18], preds[:, 18:21], preds[:, 21:24]
        
        # 归一化方向向量 (几何约束的前提)
        n_i, t_i, b_i = F.normalize(n_i), F.normalize(t_i), F.normalize(b_i)
        n_j, t_j, b_j = F.normalize(n_j), F.normalize(t_j), F.normalize(b_j)
        
        # --- B. Orthogonality Loss (自我一致性约束) ---
        # 强迫预测出的 n, t, b 构成正交基 (Frame Validity)
        # Loss = ||n.t|| + ||t.b|| + ||b.n|| + ||n x t - b||
        def frame_loss(n, t, b):
            l_dot = (n*t).sum(-1)**2 + (t*b).sum(-1)**2 + (b*n).sum(-1)**2
            l_cross = torch.sum((torch.cross(n, t, dim=-1) - b)**2, dim=-1)
            return (l_dot + l_cross).mean()
            
        ortho_loss = frame_loss(n_i, t_i, b_i) + frame_loss(n_j, t_j, b_j)
        
        # --- C. Constraint Loss (基于 Edge Type) ---
        constraint_loss = 0.0
        edge_types = data.raw_edge_type.squeeze()
        params = data.raw_edge_param # [M, 3] (例如: [delta, phi_sin, phi_cos])
        
        # 辅助函数: 向量点积误差
        def dot_loss(v1, v2, target): 
            pred = (v1 * v2).sum(dim=-1)
            
            if target.dim() == 0:
                target = target.expand_as(pred) 
                
            return F.mse_loss(pred, target)
        
        # -----------------------------------------------------------
        # Case 1: Fixed (Type 1)
        # 约束: n_i 对齐 n_j (反向), t_i 对齐 t_j (带旋转 phi)
        # -----------------------------------------------------------
        mask = (edge_types == 1)
        if mask.any():
            # Rotation
            # n_i dot n_j = -1 (Face-to-Face contact usually opposite normals)
            loss_n = dot_loss(n_i[mask], n_j[mask], torch.tensor(-1.0, device=n_i.device))
            
            # t_i dot t_j = cos(phi)
            # t_i dot b_j = -sin(phi) (Assuming standard rotation around normal)
            phi_cos = params[mask, 1] # 假设 param[1] 是 cos
            phi_sin = params[mask, 2] # 假设 param[2] 是 sin
            
            loss_t = dot_loss(t_i[mask], t_j[mask], phi_cos)
            loss_tb = dot_loss(t_i[mask], b_j[mask], -phi_sin)
            
            # Position
            # (p_j - p_i) dot n_i = delta (offset along normal)
            delta = params[mask, 0]
            diff = p_j[mask] - p_i[mask]
            loss_p_n = dot_loss(diff, n_i[mask], delta)
            
            # (p_j - p_i) dot t_i = 0 (Assuming centered alignment)
            loss_p_t = dot_loss(diff, t_i[mask], torch.tensor(0.0, device=n_i.device))
            
            constraint_loss += (loss_n + loss_t + loss_tb + loss_p_n + loss_p_t)

        # -----------------------------------------------------------
        # Case 2: Revolute (Type 2)
        # 约束: 轴线重合。对于 Axis 类型的 Anchor，我们复用 Normal n 作为 Axis d
        # d_i = n_i, d_j = n_j
        # -----------------------------------------------------------
        mask = (edge_types == 2)
        if mask.any():
            d_i, d_j = n_i[mask], n_j[mask]
            diff = p_j[mask] - p_i[mask]
            
            # 1. Parallel: || d_i x d_j || = 0
            loss_par = torch.mean(torch.norm(torch.cross(d_i, d_j, dim=-1), dim=-1))
            
            # 2. Co-linear: Point j must be on axis i -> || (p_j - p_i) x d_i || = 0
            delta = params[mask, 0]
            loss_col = torch.mean(torch.norm(torch.cross(diff, d_i, dim=-1), dim=-1))
            
            constraint_loss += (loss_par + loss_col)

        # -----------------------------------------------------------
        # Case 3: Prismatic (Type 3) - 滑轨
        # 约束: 轴线平行但不一定重合，相对旋转固定
        # -----------------------------------------------------------
        mask = (edge_types == 3)
        if mask.any():
            # 轴线 (n) 平行
            # n_i dot n_j = -1 or 1 (Depending on definition)
            loss_n = dot_loss(n_i[mask], n_j[mask], torch.tensor(-1.0, device=n_i.device))
            
            # 侧向约束: t_i dot t_j = 1 (No rotation around sliding axis)
            loss_t = dot_loss(t_i[mask], t_j[mask], torch.tensor(1.0, device=n_i.device))
            
            # Position: (p_j - p_i) 在非滑动方向上的分量受限
            # 假设沿着 t_i 滑动，那么在 n_i 和 b_i 方向上的距离固定
            diff = p_j[mask] - p_i[mask]
            delta_n = params[mask, 0] # Offset in normal
            loss_p_n = dot_loss(diff, n_i[mask], delta_n)
            loss_p_b = dot_loss(diff, b_i[mask], torch.tensor(0.0, device=n_i.device))
            
            constraint_loss += (loss_n + loss_t + loss_p_n + loss_p_b)

        # -----------------------------------------------------------
        # Case 4: Cylindrical (Type 4)
        # 约束: 同 Revolute，但允许沿轴移动
        # -----------------------------------------------------------
        mask = (edge_types == 4)
        if mask.any():
            d_i, d_j = n_i[mask], n_j[mask]
            diff = p_j[mask] - p_i[mask]
            
            # Parallel Axis
            loss_par = torch.mean(torch.norm(torch.cross(d_i, d_j, dim=-1), dim=-1))
            
            # Position: Co-linear (Distance to axis is 0)
            loss_col = torch.mean(torch.norm(torch.cross(diff, d_i, dim=-1), dim=-1))
            
            constraint_loss += (loss_par + loss_col)

        # -----------------------------------------------------------
        # Case 5: Planar (Type 5)
        # 约束: 面平行，距离固定，但允许面内平移和旋转
        # -----------------------------------------------------------
        mask = (edge_types == 5)
        if mask.any():
            # Normals oppose
            loss_n = dot_loss(n_i[mask], n_j[mask], torch.tensor(-1.0, device=n_i.device))
            
            # Distance along normal is fixed
            delta = params[mask, 0]
            diff = p_j[mask] - p_i[mask]
            loss_p = dot_loss(diff, n_i[mask], delta)
            
            constraint_loss += (loss_n + loss_p)
            
        # -----------------------------------------------------------
        # Case 6: Alignment (Type 6)
        # 约束: 轴线平行 (Parallel Alignment)
        # -----------------------------------------------------------
        mask = (edge_types == 6)
        if mask.any():
            # 复用 Normal (n) 作为 Axis (d)
            # 假设对于 Alignment 类型的边，Anchor 的 Normal 通道代表对齐轴
            d_i = n_i[mask]
            d_j = n_j[mask]

            # 计算叉积 (Cross Product)
            # 如果平行，叉积应为 0
            cross_prod = torch.cross(d_i, d_j, dim=-1)
            
            # 计算 L2 Norm 的平方: || d_i x d_j ||^2
            # sum(dim=-1) 计算每个样本的平方和，mean() 对 Batch 求平均
            loss_align = torch.mean(torch.sum(cross_prod ** 2, dim=-1))
            
            constraint_loss += loss_align
        
        return constraint_loss, ortho_loss