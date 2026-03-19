import numpy as np
from knowledge_utils import *
from utils_torch import relative_pose_6d
import torch
from torch_geometric.data import Data
import clip
import re

TYPE_VOCAB = {'Free': 0,'Fixed': 1, 'Revolute': 2, 'Prismatic': 3, 'Cylindrical': 4, 'Planar-Contact': 5, 'Alignment': 6}

class GeometryTemplate:

    def __init__(self, position, rotation, rotation_order):
        self.position = position
        self.rotation = rotation
        self.rotation_order = rotation_order
      
      
class ConceptTemplate:

    def __init__(self, position, rotation):
        self.position = position
        self.rotation = rotation

    def proximation(self, pt):
        dist = np.linalg.norm(self.overall_obj_pts - pt, axis=1)
        return np.min(dist) < PROXIMITY_THRES
    
        
class StructureNode:
    
    def __init__ (self, position, rotation, rotation_order = "XYZ", Node_Position=None, Node_Semantic=None, Node_Affordance=None, Node_Face=None, Node_Axis=None):
        '''
        Node in Structure Map (3D)
        Consisting three keys: <Position[n, 3], Semantic[1], Affordance[k]>
        Also interfaces: <Face, Axis>
        '''
        self.position = position    # [B, 3]
        self.rotation = rotation    # [B, 6]
        self.rotation_order = rotation_order    # "XYZ"
        
        self.Node_Position = Node_Position  # [B, n, 3]
        self.Node_Semantic = Node_Semantic  # [B, 1] texture
        self.Node_Affordance = Node_Affordance
        
        self.Node_Face = Node_Face
        self.Node_Axis = Node_Axis
        
        self.Refrence_Anchor = []
        self.Refrence_Anchor.append(self.Node_Face)
        self.Refrence_Anchor.append(self.Node_Axis)

class StructureEdge:
    
    def __init__ (self, Node_idx1, Node_idx2, Constraint_Type, Refrence_Anchor_1, Refrence_Anchor_2, Parameters):
        '''
        Edge in Struture Map (3D)
        Consisting four keys: <Constraint-Type[1], Refrence-Anchor[2], Geometric-Parameters[3], Function[5]>
        '''
        self.Node_idx = [Node_idx1, Node_idx2]
        self.C_Type = Constraint_Type
        self.Refrence_Anchor = {}
        self.Refrence_Anchor[Node_idx1] = Refrence_Anchor_1
        self.Refrence_Anchor[Node_idx2] = Refrence_Anchor_2
        self.Anchor = []
        self.Parameter = Parameters
        self.Relative_Pose = None
    def update_node_idx(self, add):
        self.Node_idx = [i + add for i in self.Node_idx]
        self.Refrence_Anchor = {k + add: v for k, v in self.Refrence_Anchor.items()}

class StructureGraph:
    
    def __init__ (self, Node, Edge, clip_model):
        '''
        Graph representation of Struture Map (3D)
        '''
        self.Edge = Edge
        self.Node = Node
        
        self.N = len(self.Node) # Number of Nodes
        self.B = self.Node[0].position.shape[0]
        self.device = self.Node[0].position.device
        self.dtype = self.Node[0].position.dtype
        
        self.clip_encoder = clip_model
        
        self.sem_cache = self._precompute_semantics()
        self._Find_Anchors()
        self._Add_Free_Edge()
        
        # calculate relative pose for all edges
        for edge in self.Edge:
            i_idx = edge.Node_idx[0]
            j_idx = edge.Node_idx[1]
            pos, rot = self._Relative_Pose(i_idx, j_idx)    # [B, 3]
            edge.Relative_pose = torch.cat([pos, rot], dim=1)   # [B, 6]
        
        self.M = len(self.Edge) # Number of Edges
        self._Batch_Graph() # self.data is used to train the model

    @staticmethod
    def _normalize_semantic_text(text):
        text = str(text).strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    @classmethod
    def _infer_object_prompt_from_nodes(cls, nodes):
        semantics = []
        for node in nodes:
            sem = getattr(node, "Node_Semantic", None)
            if sem is not None and str(sem).strip() != "":
                semantics.append(cls._normalize_semantic_text(sem))

        if len(semantics) == 0:
            return "object"
        if len(semantics) == 1:
            return semantics[0]

        token_lists = [s.split(" ") for s in semantics]
        min_len = min(len(tokens) for tokens in token_lists)

        prefix = []
        for i in range(min_len):
            token_i = token_lists[0][i]
            if all(tokens[i] == token_i for tokens in token_lists):
                prefix.append(token_i)
            else:
                break

        if len(prefix) > 0:
            return " ".join(prefix)
        return "object"

    def _build_subgraph_prompts(self, objects):
        """
        由 map 显式传入的 object 列表构建粗粒度子图语义。

        Returns:
            List[Dict[str, Any]]: [{"text_prompt": str, "node_indices": List[int]}, ...]
        """
        subgraph_prompts = []
        node_offset = 0

        for obj in objects:
            nodes = getattr(obj, "Nodes", None)
            if nodes is None or len(nodes) == 0:
                continue

            text_prompt = getattr(obj, "Object_Prompt", None)
            if text_prompt is None or str(text_prompt).strip() == "":
                text_prompt = self._infer_object_prompt_from_nodes(nodes)
                # 若节点语义无法形成好的粗粒度描述，则回退到对象类名
                # 例如: "base", "handle", "link" -> "handle pull"
                if text_prompt in {"object"}:
                    cls_name = obj.__class__.__name__
                    text_prompt = re.sub(r"_+", " ", cls_name)
                    text_prompt = re.sub(r"(?<!^)(?=[A-Z])", " ", text_prompt)
                    text_prompt = self._normalize_semantic_text(text_prompt)
            else:
                text_prompt = self._normalize_semantic_text(text_prompt)

            subgraph_prompts.append(
                {
                    "text_prompt": text_prompt,
                    "node_indices": list(range(node_offset, node_offset + len(nodes))),
                }
            )
            node_offset += len(nodes)

        return subgraph_prompts
        
    def _precompute_semantics(self):
        # 提取所有节点的唯一语义字符串
        unique_texts = list(set([node.Node_Semantic for node in self.Node]))
        
        cache = {}
        with torch.no_grad():
            # 一次性处理所有唯一文本，效率最高
            tokens = clip.tokenize(unique_texts).to(self.device)
            embeddings = self.clip_encoder(tokens, "text").float() # [Num_Unique, 512]
            
        for text, emb in zip(unique_texts, embeddings):
            cache[text] = emb # emb 形状是 [512]
            
        return cache

    def _Find_Anchors(self):
        for edge in self.Edge:
            for node_idx, refrence_anchor in edge.Refrence_Anchor.items():
                edge.Anchor.append(self.Node[node_idx].Refrence_Anchor[refrence_anchor['type']][refrence_anchor['idx']])

    def _Relative_Pose(self, i_idx, j_idx):
        """
        calculate relative pose of node j to node i
        
        :param i_idx: idx for node i
        :param j_idx: idx for node j
        """
        Node_i = self.Node[i_idx]
        Node_j = self.Node[j_idx]
        pos_rel, rot_rel = relative_pose_6d(Node_i.position, Node_i.rotation, Node_j.position, Node_j.rotation)
        return pos_rel, rot_rel
    
    def _Add_Free_Edge(self):
        """
        add free links between objects
        TODO: KNN
        """
        existing_edges = set()
        for edge in self.Edge:
            i, j = edge.Node_idx
            existing_edges.add(tuple(sorted((i, j))))
        
        new_edges = []
        for i in range(self.N):
            for j in range(i+1, self.N):
                if (i, j) not in existing_edges:
                    new_edge = StructureEdge(
                        Node_idx1=i, 
                        Node_idx2=j, 
                        Constraint_Type='Free',
                        Refrence_Anchor_1=None,
                        Refrence_Anchor_2=None,
                        Parameters=None
                    )
                    new_edges.append(new_edge)
        self.Edge.extend(new_edges)
    
    def _Encode_Semantic(self, sem_raw):
        emb = self.sem_cache[sem_raw]
        return emb.unsqueeze(0).repeat(self.B, 1)
    
    def _Flatten_Anchor(self, anchor_dict_list):
        """
        将 Anchor 的 List[Dict] 展平为 Tensor [B, 24]
        """
        flat_anchors = []
        zero_vec = None 
        
        for anchor_data in anchor_dict_list:
            # Case 1: 空锚点 (如 Free Edge)
            if anchor_data is None:
                flat_anchors.append(torch.zeros((self.B, 12), device=self.device))
                continue
            
            p = anchor_data['p']
            
            # 初始化零向量 (确保 device 和 dtype 与 p 一致)
            if zero_vec is None:
                zero_vec = torch.zeros_like(p)
            
            # --- 构建 4 个特征向量 ---
            # 1. Position
            vec_0 = p
            
            # 2. Primary Direction (Face 的 n 或 Axis 的 d)
            # 优先取 'n', 如果没有则取 'd', 再没有则填 0 (容错)
            if 'n' in anchor_data:
                vec_1 = anchor_data['n']
            elif 'd' in anchor_data:
                vec_1 = anchor_data['d']
            else:
                vec_1 = zero_vec
            
            vec_2 = anchor_data.get('t', zero_vec)
            vec_3 = anchor_data.get('b', zero_vec)
            
            # --- 拼接当前 Anchor [B, 12] ---
            flat_anchors.append(torch.cat([vec_0, vec_1, vec_2, vec_3], dim=1))
        
        if not flat_anchors:
            return torch.zeros((self.B, 24), device=self.device)
            
        return torch.cat(flat_anchors, dim=1)
    
    def _Batch_Graph(self):
        """
        flatten graph to a BIG BATCH graph
        
        After process: 
        
            self.Node_Coordinates:
            self.Node_Semantic:
            self.Node_Affordance:
            
            self.Edge_Constraint
            self.Edge_Anchors
            self.Edge_Parameters
            self.Edge_RelaPoses
        """
        # ==========================================
        # 1. 处理 Node Data (Flattening)
        # ==========================================
            # --- A. Semantic ---
            # 处理语义 [B, 1]
        all_node_embs = torch.stack([self.sem_cache[node.Node_Semantic] for node in self.Node])
        x_sem = all_node_embs.unsqueeze(0).expand(self.B, self.N, 512).reshape(self.B * self.N, 512)
        
        # 点云数据容器
        all_pos_points = []
        all_pos_indices = [] # Map point -> Global Node ID
        
        all_aff_points = []
        all_aff_indices = []
        
        for i, node in enumerate(self.Node):   
            # --- B. 计算全局 Node ID ---
            # 对于第 i 个节点，它在 batch 0 的 ID 是 i
            # 在 batch 1 的 ID 是 N + i ... 在 batch b 的 ID 是 b*N + i
            global_ids = torch.arange(self.B, device=self.device) * self.N + i  # [B] -> [i, N+i, 2N+i, ...]

            # --- C. Position Point Cloud [B, k, 3] ---
            # node.Node_Position: [B, num_points, 3]    k=num_pts
            if node.Node_Position is not None:
                p_pts = node.Node_Position # [B, k, 3]
                num_pts = p_pts.shape[1]
                
                # 1. 展平点: [B*k, 3]
                flat_pts = p_pts.reshape(self.B * num_pts, 3)
                all_pos_points.append(flat_pts)
                
                # 2. 生成索引
                # 我们需要让每个 batch 的 k 个点都指向对应的 global_id
                # ids_expanded: [B, k]
                ids_expanded = global_ids.unsqueeze(1).expand(self.B, num_pts)  # [B, k]
                all_pos_indices.append(ids_expanded.reshape(-1))
                
            # --- D. Affordance Point Cloud ---
            if node.Node_Affordance is not None:
                a_pts = node.Node_Affordance # [B, m, 3]
                num_pts = a_pts.shape[1]
                
                flat_pts = a_pts.reshape(self.B * num_pts, 3)
                all_aff_points.append(flat_pts)
                
                ids_expanded = global_ids.unsqueeze(1).expand(self.B, num_pts)
                all_aff_indices.append(ids_expanded.reshape(-1))
        # 拼接 Node 特征
        
        # 拼接点云 (List 中的顺序没关系，只要 index 对就行，但为了整洁建议 sort)
        if all_pos_points:
            x_pos = torch.cat(all_pos_points, dim=0)    # [Total_Points, 3]
            pos_idx = torch.cat(all_pos_indices, dim=0) # [Total_Points]
        else:
            x_pos = torch.empty((0,3), device=self.device)
            pos_idx = torch.empty((0,), dtype=torch.long, device=self.device)
            
        if all_aff_points:
            x_aff = torch.cat(all_aff_points, dim=0)
            aff_idx = torch.cat(all_aff_indices, dim=0)
        else:
            x_aff = torch.empty((0,3), device=self.device)
            aff_idx = torch.empty((0,), dtype=torch.long, device=self.device)
            
        # ==========================================
        # 2. 处理 Edge Data (处理空边情况)
        # ==========================================
        if self.M == 0:  # 没有边的情况
            # 创建空的边索引 [2, 0]
            final_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            
            # 创建空的边特征张量
            edge_type = torch.zeros((0, 1), device=self.device)
            edge_param = torch.zeros((0, 3), device=self.device)
            edge_anchor = torch.zeros((0, 24), device=self.device)
            edge_pose = torch.zeros((0, 9), device=self.device)
        else:
            base_src = []
            base_dst = []
            raw_types = []
            raw_params = []
            raw_anchors = []
            raw_poses = []
            
            for edge in self.Edge:
                base_src.append(edge.Node_idx[0])
                base_dst.append(edge.Node_idx[1])
                
                t_idx = TYPE_VOCAB.get(edge.C_Type, 2)
                raw_types.append(torch.full((self.B, 1), t_idx, device=self.device))
                
                if isinstance(edge.Parameter, torch.Tensor):
                    raw_params.append(edge.Parameter)
                else:
                    raw_params.append(torch.zeros((self.B, 3), device=self.device))
                    
                raw_anchors.append(self._Flatten_Anchor(edge.Anchor))
                raw_poses.append(edge.Relative_pose)
            
            # 构建基础边索引
            base_edge_index = torch.tensor([base_src, base_dst], dtype=torch.long, device=self.device)
            
            # 广播生成 Batched Edge Index
            offsets = (torch.arange(self.B, device=self.device) * self.N).view(self.B, 1, 1)
            batched_edges = base_edge_index.unsqueeze(0) + offsets
            final_edge_index = batched_edges.permute(1, 0, 2).reshape(2, -1)
            
            # 处理特征展平
            edge_type = torch.stack(raw_types, dim=1).reshape(self.B * self.M, -1)
            edge_param = torch.stack(raw_params, dim=1).reshape(self.B * self.M, -1)
            edge_anchor = torch.stack(raw_anchors, dim=1).reshape(self.B * self.M, -1)
            edge_pose = torch.stack(raw_poses, dim=1).reshape(self.B * self.M, -1)
        
        
        # # ==========================================
        # # 边索引越界检测 (调试用)
        # # ==========================================
        # max_edge_idx = final_edge_index.max().item()
        # total_nodes = self.B * self.N
        # if max_edge_idx >= total_nodes:
        #     print(f"\n[ERROR] 边索引 (Edge Index) 越界!")
        #     print(f"final_edge_index 中的最大索引: {max_edge_idx}")
        #     print(f"允许的最大节点索引: {total_nodes - 1}")
        #     print(f"Batch Size (B): {self.B}, 单图节点数 (N): {self.N}")
            
        #     # 进一步分析是哪条边出错了
        #     # 检查原始的 base_edge_index
        #     raw_max = base_edge_index.max().item()
        #     print(f"原始 base_edge_index 的最大值: {raw_max} (应小于 N={self.N})")
            
        #     raise IndexError(f"Edge index {max_edge_idx} exceeds total nodes {total_nodes}")
        
        # ==========================================
        # 3. 创建 Data 对象
        # ==========================================
        
        # 标记每个节点属于哪个 Batch [B*N]
        batch_vec = torch.arange(self.B, device=self.device).repeat_interleave(self.N)

        self.data = Data(
            # Nodes
            x_sem=x_sem,
            
            # Point Clouds
            x_pos=x_pos,
            pos_batch_idx=pos_idx, # 指向 Global Node ID
            
            x_aff=x_aff,
            aff_batch_idx=aff_idx, # 指向 Global Node ID
            
            # Edge Topology
            edge_index=final_edge_index,
            
            # Edge Features
            edge_type=edge_type,
            edge_param=edge_param,
            edge_anchor=edge_anchor,
            edge_pose=edge_pose,
            
            # === Raw Ground Truth for Loss Calculation ===
            # 保留原始数据的副本，确保 Loss 计算使用的是绝对正确的标签/参数
            raw_edge_type=edge_type.clone(), 
            raw_edge_param=edge_param.clone(),
            
            # Meta
            num_nodes=self.B*self.N,
            batch=batch_vec
        )
        
    def complete_point_cloud(self):
        total_num = 3000
        num_primitive = total_num // self.N
        
        points = []
        for node in self.Node:
            point = node.get_surface_points(num_primitive)
            points.append(point)
        
        points = torch.cat(points, dim=1)   # [B, n, 3]
        return points
    
    def get_prompt(self):
        return self.Subgraph_Prompts