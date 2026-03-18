import torch
import torch.nn as nn
import torch.nn.functional as F
from Structure_Primitive import Cuboid, Cylinder
from base_template import StructureEdge, StructureGraph

class Cup:
    def __init__ (self, size, position, rotation):
        semantic = 'cup'
        
        Nodes = []
        Edges = []
        
        Nodes.append(Cylinder(size[:, 0], size[:, 1], position=position, rotation=rotation, Semantic=semantic))
        
        self.Nodes = Nodes
        self.Edges = Edges
        
class CoffeMachine:
    def __init__ (self, sizes, positions, rotations):
        semantic1 = "coffee machine body"
        semantic2 = "coffee machine body"
        semantic3 = "coffee machine button"
        # semantic4 = "coffee machine spout"
        
        size1 = sizes[:, 0:3]
        size2 = sizes[:, 3:6]
        size3 = sizes[:, 6:8]
        # size4 = sizes[:, 8:10]
        
        position1 = positions[:, 0:3]
        position2 = positions[:, 3:6]
        position3 = positions[:, 6:9]
        position4 = positions[:, 9:12]
        
        rotation1 = rotations[:, 0:1*6]
        rotation2 = rotations[:, 1*6:2*6]
        rotation3 = rotations[:, 2*6:3*6]
        # rotation4 = rotations[:, 3*6:4*6]
        
        Nodes = []
        Edges = []
        
        Nodes.append(Cuboid(size1[:, 0], size1[:, 1], size1[:, 2], position=position1, rotation=rotation1, Semantic=semantic1))
        Nodes.append(Cuboid(size2[:, 0], size2[:, 1], size2[:, 2], position=position2, rotation=rotation2, Semantic=semantic2))
        Nodes.append(Cylinder(size3[:, 0], size3[:, 1], position=position3, rotation=rotation3, Semantic=semantic3))
        # Nodes.append(Cylinder(size4[:, 0], size4[:, 1], position=position4, rotation=rotation4, Semantic=semantic4))
        
        Edges.append(StructureEdge(0, 1, "Planar-Contact", {"type": 0, "idx": 2}, {"type": 0, "idx": 3}, [0, 0, 0]))
        Edges.append(StructureEdge(0, 1, "Alignment", {"type": 0, "idx": 1}, {"type": 0, "idx": 1}, [0, 0, 0]))
        Edges.append(StructureEdge(0, 1, "Alignment", {"type": 0, "idx": 4}, {"type": 0, "idx": 4}, [0, 0, 0]))
        Edges.append(StructureEdge(1, 2, "Fixed", {"type": 0, "idx": 5}, {"type": 0, "idx": 1}, [0, 0, 0]))
        
        self.Nodes = Nodes
        self.Edges = Edges
        
class StructureMap_CoffeePush(StructureGraph):
    def __init__(self, sizes, positions, rotations, clip_model, preprocess=False):
        """        
        :param sizes: [B, 11]
        :param positions: [B,12] -- 23
        :param rotations: [B, 4*6] -- 47
        Total: [B, 47], Node:4
        """
        if preprocess:
            sizes = self._preprocess_parameters(sizes)
            
        Objects = []
        Objects.append(Cup(sizes[:, 0:3], positions[:, 0:3], rotations[:, 0:6*1]))
        Objects.append(CoffeMachine(sizes[:, 3:11], positions[:, 3:12], rotations[:, 1*6:4*6]))
        
        Nodes = []
        Edges = []
        
        num_node = 0
        for object in Objects:
            for node in object.Nodes:
                Nodes.append(node)
            for edge in object.Edges:
                edge.update_node_idx(num_node)
                Edges.append(edge)
            num_node += len(object.Nodes)
        
        super().__init__(Nodes, Edges, clip_model)
        
    def _preprocess_parameters(self, sizes):
        """
        对网络输出的参数进行预处理，使其符合物理约束。
        
        Args:
            sizes: [B, 8] 网络原始输出
            size_range: (min_val, max_val) 尺寸的最小值和最大值约束
            
        Returns:
            constrained_sizes
        """
        size_range=(0.02, 5)
        min_s, max_s = size_range
        sizes = torch.sigmoid(sizes) * (max_s - min_s) + min_s
        
        return sizes
 
if __name__ == "__main__":
    import os
    import math
    os.environ["WAYLAND_DISPLAY"] = "" 
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Initializing Coffee-Pull Scenario (Corrected Logic) on {device}...")

    # ==========================================
    # 1. 定义物理尺寸 (Sizes) - 总计 11 个参数
    # ==========================================
    # Cup [0:3]: [Height, Radius, Placeholder]
    h_cup, r_cup = 0.12, 0.045
    size_cup = [h_cup, r_cup, r_cup] 

    # Machine Body 1 (背部立柱) [3:6]: [H1, L1, W1]
    h_body1, l_body1, w_body1 = 0.35, 0.20, 0.15 # 较窄的立柱
    size_m1 = [h_body1, l_body1, w_body1]

    # Machine Body 2 (顶部出水头) [6:9]: [H2, L2, W2]
    h_body2, l_body2, w_body2 = 0.06, 0.20, 0.25 # 较长且向 Z 轴延伸
    size_m2 = [h_body2, l_body2, w_body2]

    # Machine Button [9:11]: [Height, Radius]
    h_btn, r_btn = 0.02, 0.02
    size_m3 = [h_btn, r_btn]

    sizes = torch.tensor([size_cup + size_m1 + size_m2 + size_m3], 
                         dtype=torch.float32, device=device)

    # ==========================================
    # 2. 定义旋转 (Rotations)
    # ==========================================
    identity_6d = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    rotations = torch.tensor([identity_6d * 4], dtype=torch.float32, device=device)

    # ==========================================
    # 3. 定义位置 (Positions) - 注意空间堆叠关系
    # ==========================================
    
    # --- Node 1: Body 1 (立柱) ---
    # 放在原点，底部着地
    pos_body1 = [0.0, h_body1 / 2.0, 0.0]
    
    # --- Node 2: Body 2 (悬空头部) ---
    # Y: 在立柱顶部 -> h_body1 + h_body2/2
    # Z: Body 1 的 Z 轴范围是 [-w_body1/2, w_body1/2]
    # 我们让 Body 2 向前（Z轴正方向）偏移，使其前方悬空
    y_body2 = h_body1 + (h_body2 / 2.0)
    z_body2 = 0.08 # 偏移量，使得 Body 2 的一部分伸到前面
    pos_body2 = [0.0, y_body2, z_body2]
    
    # --- Node 3: Button (按钮) ---
    # 放在 Body 2 的顶部前方
    y_btn = h_body1 + h_body2 + (h_btn / 2.0)
    z_btn = z_body2 + 0.05
    pos_btn = [0.0, y_btn, z_btn]

    # --- Node 0: Cup (杯子) ---
    # Y: 放在地面上 (Y = H/2)
    # Z: 必须在 Body 2 悬空部分的下方，也就是“出水口”的位置
    # Body 2 延伸到了 z = z_body2 + w_body2/2 (约 0.08 + 0.125 = 0.205)
    # 我们把杯子放在 Z = 0.18 左右
    y_cup = h_cup / 2.0
    z_cup = z_body2 + 0.06 
    pos_cup = [0.0, y_cup, z_cup]
    
    positions_list = pos_cup + pos_body1 + pos_body2 + pos_btn
    positions = torch.tensor([positions_list], dtype=torch.float32, device=device)

    # ==========================================
    # 4. 初始化与可视化
    # ==========================================
    from utils import CLIPEncoder
    from visualization_helper import visualize_structure_graph
    
    try:
        clip_encoder = CLIPEncoder("ViT-B/32").to(device)
        graph = StructureMap_CoffeePush(sizes, positions, rotations, clip_encoder)
        
        # 配色方案
        graph.Node[0].visual_color = [0.9, 0.9, 0.9]     # Cup: 白色
        graph.Node[1].visual_color = [0.15, 0.15, 0.15] # Body1: 黑色/深灰
        graph.Node[2].visual_color = [0.15, 0.15, 0.15] # Body2: 黑色/深灰
        graph.Node[3].visual_color = [0.0, 0.6, 1.0]    # Button: 蓝色

        print("Graph Initialized. Visualizing...")
        visualize_structure_graph(graph)
        
    except Exception as e:
        import traceback
        traceback.print_exc()