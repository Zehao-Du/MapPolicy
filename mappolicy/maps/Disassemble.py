import torch

from Structure_Primitive import Cylinder, Torus
from base_template import StructureEdge, StructureGraph


class Peg:
    def __init__(self, sizes, positions, rotations):
        semantic = "peg"
        self.Object_Prompt = "peg"

        Nodes = []
        Edges = []

        # sizes: [B, 2] = [peg_height, peg_radius]
        if sizes.shape[1] != 2:
            raise ValueError(f"Expected peg sizes shape [B, 2], got {tuple(sizes.shape)}")
        if positions.shape[1] != 3:
            raise ValueError(f"Expected peg positions shape [B, 3], got {tuple(positions.shape)}")
        if rotations.shape[1] != 6:
            raise ValueError(f"Expected peg rotations shape [B, 6], got {tuple(rotations.shape)}")

        peg_height = sizes[:, 0:1]
        peg_radius = sizes[:, 1:2]
        size_peg = torch.cat([peg_height, peg_radius, peg_radius], dim=1)

        Nodes.append(
            Cylinder(
                height=size_peg[:, 0],
                top_radius=size_peg[:, 1],
                position=positions,
                rotation=rotations,
                Semantic=semantic,
            )
        )

        self.Nodes = Nodes
        self.Edges = Edges


class Nut:
    def __init__(self, sizes, positions, rotations):
        semantic_nut = "nut"
        semantic_handle = "nut handle"
        self.Object_Prompt = "nut"

        Nodes = []
        Edges = []

        # sizes: [B, 4] = [nut_central_radius, nut_tube_radius, handle_radius, handle_height]
        if sizes.shape[1] != 4:
            raise ValueError(f"Expected nut sizes shape [B, 4], got {tuple(sizes.shape)}")
        # positions: [B, 6] = [nut_xyz, handle_xyz]
        if positions.shape[1] != 6:
            raise ValueError(f"Expected nut positions shape [B, 6], got {tuple(positions.shape)}")
        # rotations: [B, 12] = [nut_6d, handle_6d]
        if rotations.shape[1] != 12:
            raise ValueError(f"Expected nut rotations shape [B, 12], got {tuple(rotations.shape)}")

        nut_central_radius = sizes[:, 0]
        nut_tube_radius = sizes[:, 1]
        handle_radius = sizes[:, 2]
        handle_height = sizes[:, 3]
        nut_position = positions[:, 0:3]
        handle_pos = positions[:, 3:6]
        nut_rotation = rotations[:, 0:6]
        handle_rotation = rotations[:, 6:12]
        
        # Nut: Torus (donut) around local Y-axis
        Nodes.append(
            Torus(
                central_radius=nut_central_radius,
                start_torus_radius=nut_tube_radius,
                position=nut_position,
                rotation=nut_rotation,
                Semantic=semantic_nut,
            )
        )

        Nodes.append(
            Cylinder(
                height=handle_height,
                top_radius=handle_radius,
                position=handle_pos,
                rotation=handle_rotation,
                Semantic=semantic_handle,
            )
        )

        # Fixed connection between nut (torus) and handle (cylinder)
        # Torus Face idx=0 (Outer), Cylinder Face idx=0 (Top)
        B = sizes.shape[0]
        device = sizes.device
        Edges.append(
            StructureEdge(
                0,
                1,
                "Fixed",
                {"type": 0, "idx": 0},
                {"type": 0, "idx": 0},
                torch.zeros((B, 3), device=device),
            )
        )

        self.Nodes = Nodes
        self.Edges = Edges


class StructureMap_Disassemble(StructureGraph):
    def __init__(self, sizes, positions, rotations, clip_model, preprocess=False):
        """
        Minimal structure map for MetaWorld disassemble.

        sizes: [B, 6]
            [peg_height, peg_radius, nut_central_radius, nut_tube_radius, handle_radius, handle_height]
        positions: [B, 9] (3 nodes * 3)
        rotations: [B, 18] (3 nodes * 6D rotation)
        total: [B, 33]
        """
        if preprocess:
            sizes = self._preprocess_parameters(sizes)

        # sizes: [B, 6] = [peg_height, peg_radius, nut_central_radius, nut_tube_radius, handle_radius, handle_height]
        if sizes.shape[1] != 6:
            raise ValueError(f"Expected sizes shape [B, 6], got {tuple(sizes.shape)}")
        if positions.shape[1] != 9:
            raise ValueError(f"Expected positions shape [B, 9], got {tuple(positions.shape)}")
        if rotations.shape[1] != 18:
            raise ValueError(f"Expected rotations shape [B, 18], got {tuple(rotations.shape)}")

        peg_obj = Peg(sizes[:, 0:2], positions[:, 0:3], rotations[:, 0:6])
        nut_obj = Nut(sizes[:, 2:6], positions[:, 3:9], rotations[:, 6:18])
        
        Objects = [peg_obj, nut_obj]

        Nodes = []
        Edges = []
        num_node = 0
        for obj in Objects:
            Nodes.extend(obj.Nodes)
            for edge in obj.Edges:
                edge.update_node_idx(num_node)
                Edges.append(edge)
            num_node += len(obj.Nodes)

        # Cylindrical constraint between peg (node 0) and nut main body (node 1)
        B = sizes.shape[0]
        device = sizes.device
        Edges.append(
            StructureEdge(
                0,
                1,
                "Cylindrical",
                {"type": 1, "idx": 0},
                {"type": 1, "idx": 0},
                torch.zeros((B, 3), device=device),
            )
        )

        self.Subgraph_Prompts = self._build_subgraph_prompts(Objects)

        super().__init__(Nodes, Edges, clip_model)

    def _preprocess_parameters(self, sizes):
        size_range = (0.02, 5)
        min_s, max_s = size_range
        return torch.sigmoid(sizes) * (max_s - min_s) + min_s


if __name__ == "__main__":
    import argparse
    import os

    os.environ["WAYLAND_DISPLAY"] = ""

    parser = argparse.ArgumentParser(description="Disassemble structure-map demo")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--batch", type=int, default=1, help="Batch size B")
    parser.add_argument("--no-vis", action="store_true", help="Skip Open3D visualization")
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    B = int(args.batch)

    print(f"Initializing Disassemble Scenario on {device} (B={B})...")

    # -----------------------------------------------------------------
    # A. Define Sizes (Meters)
    # sizes: [B, 6]
    #   [peg_height, peg_radius, nut_central_radius, nut_tube_radius, handle_radius, handle_height]
    # -----------------------------------------------------------------
    peg_height = 0.30
    peg_radius = 0.01
    nut_central_radius = 0.03
    nut_tube_radius = 0.01
    handle_radius = 0.01
    handle_height = 0.10

    sizes_list = [
        peg_height,
        peg_radius,
        nut_central_radius,
        nut_tube_radius,
        handle_radius,
        handle_height,
    ]
    sizes = torch.tensor([sizes_list], dtype=torch.float32, device=device).repeat(B, 1)

    # -----------------------------------------------------------------
    # B. Define Rotations (6D)
    # rotations: [B, 18] = [peg_6d, nut_6d, handle_6d]
    # 6D 表示: rotation matrix 的前两列拼接 [col1, col2]
    # 下面三个是“局部 +Y 轴”对齐到“世界 +X/+Y/+Z”的标准旋转
    # -----------------------------------------------------------------
    x_pos_6d = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0]  # Rz(-90deg): local +Y -> world +X
    y_pos_6d = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]   # Identity:  local +Y -> world +Y
    z_pos_6d = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]   # Rx(+90deg): local +Y -> world +Z

    rotations = torch.tensor([y_pos_6d * 2 + x_pos_6d], dtype=torch.float32, device=device).repeat(B, 1)

    # -----------------------------------------------------------------
    # C. Define Positions
    # positions: [B, 9] = [peg_xyz, nut_xyz, handle_xyz]
    # -----------------------------------------------------------------
    pos_peg = [0.0, peg_height / 2.0, 0.0]
    pos_nut = [0.0, peg_height * 0.75, 0.0]
    # Put handle on top of nut: cylinder bottom touches torus top (approx y + tube_radius)
    handle_center_y = peg_height * 0.75
    pos_handle = [0.1, handle_center_y, 0.0]

    positions = torch.tensor([pos_peg + pos_nut + pos_handle], dtype=torch.float32, device=device).repeat(B, 1)

    # -----------------------------------------------------------------
    # D. Initialization & Visualization
    # -----------------------------------------------------------------
    from utils import CLIPEncoder

    try:
        clip_encoder = CLIPEncoder("ViT-B/32").to(device)
        graph = StructureMap_Disassemble(sizes, positions, rotations, clip_encoder)

        # Node indices: 0 = peg, 1 = nut, 2 = handle
        graph.Node[0].visual_color = [0.2, 0.8, 0.2]
        graph.Node[1].visual_color = [0.8, 0.2, 0.2]
        graph.Node[2].visual_color = [0.2, 0.2, 0.8]

        print("Graph constructed successfully.")
        print(f"Nodes: {len(graph.Node)}")
        print(f"Edges: {len(graph.Edge)}")

        if not args.no_vis:
            print("Visualizing...")
            from visualization_helper import visualize_structure_graph

            visualize_structure_graph(graph)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()
