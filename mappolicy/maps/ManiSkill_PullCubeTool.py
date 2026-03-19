import torch
from Structure_Primitive import Cuboid
from base_template import StructureEdge, StructureGraph


class BlueCube:
    """蓝色正方体：单个 Cuboid，边长一致。"""

    def __init__(self, sizes, positions, rotations):
        semantic = "blue cube"
        self.Object_Prompt = "blue cube"

        Nodes = []
        Edges = []

        # sizes: [B, 1] -> 正方体边长
        side = sizes[:, 0]
        position = positions[:, 0:3]
        rotation = rotations[:, 0:6]

        Nodes.append(
            Cuboid(side, side, side, position=position, rotation=rotation, Semantic=semantic)
        )

        self.Nodes = Nodes
        self.Edges = Edges


class TwoRedCuboids:
    """
    两个红色长方体组装体：
    - 长方体1 下底面 与 长方体2 上底面 Planar-Contact；
    - 长方体1 下底面的一条侧棱 与 长方体2 上底面的一条侧棱 Alignment。
    蓝色正方体与所有节点之间无显式边，由 _Add_Free_Edge 补全为 Free。
    """

    def __init__(self, sizes, positions, rotations):
        semantic1 = "red cuboid"
        semantic2 = "red cuboid"
        self.Object_Prompt = "red cuboids tool"

        Nodes = []
        Edges = []

        # sizes: [B, 6] = (h1, l1, w1, h2, l2, w2)
        h1, l1, w1 = sizes[:, 0], sizes[:, 1], sizes[:, 2]
        h2, l2, w2 = sizes[:, 3], sizes[:, 4], sizes[:, 5]

        position1 = positions[:, 0:3]
        position2 = positions[:, 3:6]
        rotation1 = rotations[:, 0:6]
        rotation2 = rotations[:, 6:12]

        Nodes.append(Cuboid(h1, l1, w1, position=position1, rotation=rotation1, Semantic=semantic1))
        Nodes.append(Cuboid(h2, l2, w2, position=position2, rotation=rotation2, Semantic=semantic2))

        # Cuboid Face: 0=Back, 1=Front, 2=Top, 3=Bottom, 4=Left, 5=Right
        # Cuboid Axis: 0..11 为 12 条棱，12 为主轴
        # 下底面(Face 3) 的棱: Axis 2,6,10,11；上底面(Face 2) 的棱: Axis 0,4,8,9

        # 红色长方体1 下底面 与 红色长方体2 上底面 接触
        Edges.append(
            StructureEdge(0, 1, "Planar-Contact", {"type": 0, "idx": 3}, {"type": 0, "idx": 2}, [0, 0, 0])
        )
        # 长方体1 下底面的一条侧棱 与 长方体2 上底面的一条侧棱 对齐（同向）
        Edges.append(
            StructureEdge(0, 1, "Alignment", {"type": 1, "idx": 2}, {"type": 1, "idx": 0}, [0, 0, 0])
        )

        self.Nodes = Nodes
        self.Edges = Edges


class StructureMap_PullCubeTool(StructureGraph):
    """
    Map：一个蓝色正方体 + 两个红色长方体（组装体）。
    - 蓝色正方体与所有节点：Free（由 _Add_Free_Edge 生成）。
    - 红色长方体1 下底面 与 红色长方体2 上底面：Planar-Contact；
      两体各一条侧棱：Alignment。

    :param sizes: [B, 7] = cube_side(1) + cuboid1(3) + cuboid2(3)
    :param positions: [B, 9] = cube(3) + cuboid1(3) + cuboid2(3)
    :param rotations: [B, 18] = cube(6) + cuboid1(6) + cuboid2(6)
    """

    def __init__(self, sizes, positions, rotations, clip_model, preprocess=False):
        if preprocess:
            sizes = self._preprocess_parameters(sizes)

        Objects = []
        Objects.append(BlueCube(sizes[:, 0:1], positions[:, 0:3], rotations[:, 0:6]))
        Objects.append(TwoRedCuboids(sizes[:, 1:7], positions[:, 3:9], rotations[:, 6:18]))

        Nodes = []
        Edges = []

        num_node = 0
        for obj in Objects:
            for node in obj.Nodes:
                Nodes.append(node)
            for edge in obj.Edges:
                edge.update_node_idx(num_node)
                Edges.append(edge)
            num_node += len(obj.Nodes)

        self.Subgraph_Prompts = self._build_subgraph_prompts(Objects)

        super().__init__(Nodes, Edges, clip_model)

    def _preprocess_parameters(self, sizes):
        size_range = (0.02, 5)
        min_s, max_s = size_range
        sizes = torch.sigmoid(sizes) * (max_s - min_s) + min_s
        return sizes

if __name__ == "__main__":
    import os
    os.environ["WAYLAND_DISPLAY"] = "" 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 1
    
    print(f"Initializing Scenario on {device}...")
    
    # -----------------------------------------------------------------
    # A. Define Physical Dimensions (Meters)
    # -----------------------------------------------------------------
    # sizes: [B, 7] = cube_side(1) + cuboid1(3) + cuboid2(3)
    cube_side = 0.05
    red1_dims = [0.1, 0.2, 0.2]
    red2_dims = [0.1, 0.2, 0.2]
    
    sizes_list = [cube_side] + red1_dims + red2_dims
    sizes = torch.tensor([sizes_list], dtype=torch.float32, device=device)
    
    # -----------------------------------------------------------------
    # B. Define Rotations
    # -----------------------------------------------------------------
    identity_6d = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    rotations = torch.tensor([identity_6d * 3], dtype=torch.float32, device=device)
    
    # -----------------------------------------------------------------
    # C. Define Positions
    # -----------------------------------------------------------------
    # Y is UP. Grounds are Y=0.
    # Center = Bottom_Y + Height/2
    
    # 1. Blue Cube (left)
    pos_blue = [-0.2, cube_side / 2.0, 0.0]
    
    # 2. Red Cuboid 2 (bottom, right)
    pos_red2 = [0.2, red2_dims[0] / 2.0, 0.0]
    
    # 3. Red Cuboid 1 (top of red cuboid 2)
    pos_red1 = [0.2, red2_dims[0] + (red1_dims[0] / 2.0), 0.0]
    
    # Order: cube(3) + cuboid1(3) + cuboid2(3)
    positions_list = pos_blue + pos_red1 + pos_red2
    positions = torch.tensor([positions_list], dtype=torch.float32, device=device)
    
    # -----------------------------------------------------------------
    # D. Initialization & Visualization
    # -----------------------------------------------------------------
    from utils import CLIPEncoder
    
    try:
        clip_encoder = CLIPEncoder("ViT-B/32").to(device)
        graph = StructureMap_PullCubeTool(sizes, positions, rotations, clip_encoder)
        
        # Node 0: Blue Cube
        graph.Node[0].visual_color = [0.2, 0.2, 0.8]
        # Node 1: Red Cuboid 1 (Top)
        graph.Node[1].visual_color = [0.8, 0.2, 0.2]
        # Node 2: Red Cuboid 2 (Bottom)
        graph.Node[2].visual_color = [0.5, 0.1, 0.1]
        
        print("Graph constructed successfully.")
        from visualization_helper import visualize_structure_graph
        visualize_structure_graph(graph)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()