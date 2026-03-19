import torch
from Structure_Primitive import Cuboid, Cylinder, Sphere
from base_template import StructureEdge, StructureGraph


class Lever_Mechanism:
    """
    四部件：灰色长方体主体、深灰细圆柱1（与主体 Fixed）、深灰细圆柱2（底面与圆柱1侧面 Planar-Contact）、
    球（与圆柱2 Fixed，绕圆柱1轴线 Revolute）。
    """

    def __init__(self, sizes, positions, rotations):
        self.Object_Prompt = "lever"
        # sizes: [B, 8]  ->  cuboid(3) + cyl1(2: height, top_radius) + cyl2(2) + sphere(1: radius)
        # positions: [B, 12],  rotations: [B, 24]

        size_cuboid = sizes[:, 0:3]
        size_cyl1 = sizes[:, 3:5]
        size_cyl2 = sizes[:, 5:7]
        size_sphere = sizes[:, 7:8]

        pos_cuboid = positions[:, 0:3]
        pos_cyl1 = positions[:, 3:6]
        pos_cyl2 = positions[:, 6:9]
        pos_sphere = positions[:, 9:12]

        rot_cuboid = rotations[:, 0:6]
        rot_cyl1 = rotations[:, 6:12]
        rot_cyl2 = rotations[:, 12:18]
        rot_sphere = rotations[:, 18:24]

        Nodes = []

        Nodes.append(
            Cuboid(
                size_cuboid[:, 0],
                size_cuboid[:, 1],
                size_cuboid[:, 2],
                position=pos_cuboid,
                rotation=rot_cuboid,
                Semantic="lever base body",
            )
        )
        Nodes.append(
            Cylinder(
                size_cyl1[:, 0],
                size_cyl1[:, 1],
                position=pos_cyl1,
                rotation=rot_cyl1,
                Semantic="lever cylinder 1",
            )
        )
        Nodes.append(
            Cylinder(
                size_cyl2[:, 0],
                size_cyl2[:, 1],
                position=pos_cyl2,
                rotation=rot_cyl2,
                Semantic="lever cylinder 2",
            )
        )
        Nodes.append(
            Sphere(
                size_sphere[:, 0],
                position=pos_sphere,
                rotation=rot_sphere,
                Semantic="lever handle sphere",
            )
        )

        Edges = []
        # 0=主体(Cuboid), 1=圆柱1(Cylinder), 2=圆柱2(Cylinder), 3=球(Sphere)
        # 根据文档，Planar-Contact 约束需要 Face $i$ + Face $j$，要求两个面平行且接触
        # Cylinder Faces: 0=Top, 1=Bottom, 2=Side
        # Sphere Faces: 0=X+, 1=X-, 2=Y+, 3=Y-, 4=Z+, 5=Z-
        # Cuboid Faces: 0=Back, 1=Front, 2=Top, 3=Bottom, 4=Left, 5=Right
        
        # 球 — 圆柱2: Fixed（球的外侧面 — 圆柱2 顶面 Face 0）
        # 使用球的 +Z 方向面（Face 4）与圆柱2的顶面
        Edges.append(
            StructureEdge(3, 2, "Fixed", {"type": 0, "idx": 4}, {"type": 0, "idx": 0}, [0, 0, 0])
        )
        
        # 圆柱2 — 圆柱1: Planar-Contact（圆柱2 底面 Face 1 — 圆柱1 侧面 Face 2）
        Edges.append(
            StructureEdge(2, 1, "Planar-Contact", {"type": 0, "idx": 1}, {"type": 0, "idx": 2}, [0, 0, 0])
        )
        
        # 球 — 圆柱1: Revolute（球穿过中心、与圆柱1轴线平行的轴绕圆柱1轴线旋转）
        # Revolute 要求两轴平行：圆柱1 Axis 0（沿 Y）↔ 球 Axis 1（Y 方向，过球心）
        Edges.append(
            StructureEdge(3, 1, "Revolute", {"type": 1, "idx": 1}, {"type": 1, "idx": 0}, [0, 0, 0])
        )
        
        # 圆柱1 — 主体: Fixed（圆柱1与主体刚性连接）
        # Cuboid Faces: 2=Top, 3=Bottom; Cylinder Faces: 0=Top, 1=Bottom
        # 圆柱1底面贴合主体顶面
        Edges.append(
            StructureEdge(1, 0, "Fixed", {"type": 0, "idx": 1}, {"type": 0, "idx": 2}, [0, 0, 0])
        )

        self.Nodes = Nodes
        self.Edges = Edges


class StructureMap_LeverPull(StructureGraph):
    def __init__(self, sizes, positions, rotations, clip_model, preprocess=False):
        """
        :param sizes: [B, 8]  (cuboid [3] + cyl1 [2] + cyl2 [2] + sphere [1])
        :param positions: [B, 12]
        :param rotations: [B, 24]  (4 * 6D)
        Node: 4。Edges: 球–圆柱2 Fixed，圆柱2–圆柱1 Planar-Contact，球–圆柱1 Revolute，圆柱1–主体 Fixed；其余 Free。
        """
        if preprocess:
            sizes = self._preprocess_parameters(sizes)

        Objects = [Lever_Mechanism(sizes, positions, rotations)]

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