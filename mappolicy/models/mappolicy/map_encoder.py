# Encode Structure Map
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean 

### 3D Map is represented as a graph, which has nodes and edges
### Node: P[k,3], S[texture], A[n,3]
### Edge: C[1], R[2,2], Param[3], Math[0-5]

class Node_PointCloudEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        # 对每个点独立作用的 MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x, batch_idx, num_nodes):
        """
        x: [Total_Points, 3]
        batch_idx: [Total_Points] (属于哪个全局节点)
        num_nodes: int (Batch中总节点数，用于对齐输出维度)
        """
        # 1. 判空处理 (防止某个模态完全没有数据)
        if x.numel() == 0:
            return torch.zeros((num_nodes, self.mlp[-2].out_features), device=batch_idx.device)

        # 2. Point-wise MLP
        point_feats = self.mlp(x) # [Total_Points, H]

        # 3. Aggregation (Max Pooling)
        # dim_size=num_nodes 保证输出形状固定为 [B*N, H]，即使某些节点没有点云，也会填0
        node_feats, _ = scatter_max(point_feats, batch_idx, dim=0, dim_size=num_nodes)
        
        return node_feats

class Edge_ParamEncoder(nn.Module):
    """Encodes physical parameters [delta, theta, phi]"""
    def __init__(self, output_dim, scale=10.0):
        super().__init__()
        self.scale = scale
        # Input: 5 dims (delta_norm, sin_t, cos_t, sin_p, cos_p)
        self.mlp = nn.Sequential(
            nn.Linear(5, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, x):
        delta = x[:, 0]
        theta = x[:, 1]
        phi = x[:, 2]
        delta_norm = torch.clamp(delta * self.scale, -5.0, 5.0)
        v = torch.stack([
            delta_norm,
            torch.sin(theta), torch.cos(theta),
            torch.sin(phi), torch.cos(phi)
        ], dim=1)
        return self.mlp(v)  

class StructureFeatureEncoder(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # ====================
        # Node Encoders
        # ====================
        # 1. Semantic
        self.sem_emb = nn.Linear(512, hidden_dim)
        
        # 2. Position Point Cloud
        self.pos_encoder = Node_PointCloudEncoder(in_dim=3, hidden_dim=hidden_dim)
        
        # 3. Affordance Point Cloud
        self.aff_encoder = Node_PointCloudEncoder(in_dim=3, hidden_dim=hidden_dim)
        
        # Node Fusion (将三个模态融合)
        self.node_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # ====================
        # Edge Encoders
        # ====================
        # 1. Type (离散值: Planar, Alignment, Free)
        self.edge_type_emb = nn.Embedding(num_embeddings=7, embedding_dim=hidden_dim)
        
        # 2. Parameters (连续值: [B*M, 3])
        self.edge_param_mlp = Edge_ParamEncoder(output_dim=hidden_dim)
        
        # 3. Anchor (连续值: [B*M, 24])
        self.edge_anchor_mlp = nn.Linear(24, hidden_dim)
        
        # 4. Pose (连续值: [B*M, 6])
        self.edge_pose_mlp = nn.Linear(9, hidden_dim)
        
        # Edge Fusion
        self.edge_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, data):
        """
        data: StructureGraph._Batch_Graph 生成的 PyG Data 对象
        """
        num_nodes = data.num_nodes
        
        # --------------------
        # 1. Encode Nodes
        # --------------------
        # Semantic: [B*N, 1] -> [B*N, H]
        h_sem = self.sem_emb(data.x_sem.float()) 
        
        # Position: [Total_Pos_Points, 3] -> [B*N, H]
        h_pos = self.pos_encoder(data.x_pos, data.pos_batch_idx, num_nodes)
        
        # Affordance: [Total_Aff_Points, 3] -> [B*N, H]
        h_aff = self.aff_encoder(data.x_aff, data.aff_batch_idx, num_nodes)
        
        # Fusion
        # 拼接三个特征
        node_cat = torch.cat([h_sem, h_pos, h_aff], dim=-1) # [B*N, 3*H]
        h_nodes = self.node_fusion(node_cat) # [B*N, H]

        # --------------------
        # 2. Encode Edges
        # --------------------
        # Type: [B*M, 1] -> [B*M, H]
        h_type = self.edge_type_emb(data.edge_type.long().squeeze(-1))
        
        # Continuous features
        h_param = self.edge_param_mlp(data.edge_param)     # [B*M, H]
        h_anchor = self.edge_anchor_mlp(data.edge_anchor)  # [B*M, H]
        h_pose = self.edge_pose_mlp(data.edge_pose)        # [B*M, H]
        
        # Fusion
        edge_cat = torch.cat([h_type, h_param, h_anchor, h_pose], dim=-1) # [B*M, 4*H]
        h_edges = self.edge_fusion(edge_cat) # [B*M, H]
        
        return h_nodes, h_edges
    
class MapEncoder_MLP(nn.Module):
    def __init__ (self, hidden_dim, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.feature_encoder = StructureFeatureEncoder(hidden_dim=feature_dim)
        
        # 此时输入维度由 feature_dim 决定，不再由 N 决定
        # 假设 h_nodes 是 dim, h_edges 也是 dim
        # 拼接后是 2 * dim
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, structure_map):
        data = structure_map.data
        h_nodes, h_edges = self.feature_encoder(data) # [B*N, dim], [B*M, dim]
        
        # --- 1. 获取当前的 B (推理时可能是 1) ---
        current_B = structure_map.B 
        
        # --- 2. 使用 Global Pooling (按 Batch 聚合) ---
        # 我们把属于同一个 Batch 的所有 Node 特征取平均或求和，变成 [B, dim]
        # data.batch 是我们在 _Batch_Graph 里定义的 [0,0,1,1,2,2...]
        
        # 聚合节点特征: [B*N, dim] -> [B, dim]
        h_nodes_pooled = scatter_mean(h_nodes, data.batch, dim=0, dim_size=current_B)
        
        # 聚合边特征: [B*M, dim] -> [B, dim]
        # 我们需要知道每条边属于哪个 batch
        # 在 _Batch_Graph 里我们定义了 edge_index，我们可以根据源节点计算边的 batch
        edge_batch = data.batch[data.edge_index[0]] 
        h_edges_pooled = scatter_mean(h_edges, edge_batch, dim=0, dim_size=current_B)
        
        # --- 3. 拼接聚合后的特征 [B, dim * 2] ---
        h = torch.cat([h_nodes_pooled, h_edges_pooled], dim=-1)
        
        # --- 4. 通过 MLP ---
        feature_map = self.mlp(h)
        return feature_map