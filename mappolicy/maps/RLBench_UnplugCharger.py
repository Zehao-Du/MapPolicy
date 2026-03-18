import torch
from Structure_Primitive import Cuboid, Cylinder, Rectangular_Ring
from base_template import StructureEdge, StructureGraph


class Peg_Three_Parts_Mechanism:
    """
    三部件：深灰圆柱、黄色小长方体、白色矩形环（孔板）。
    - 黄色长方体 — 圆柱: Fixed（黄色底面 — 圆柱顶面）
    - 其他节点对: Free（由 _Add_Free_Edge 自动添加）
    """

    def __init__(self, sizes, positions, rotations):
        # sizes: [B, 7] -> cyl(2: h, r) + yellow(3: height, length, width) + ring_outer(2: L, W)
        # 注：ring 的 front_height = yellow height, inner_L/W = yellow length/width
        # positions: [B, 9],  rotations: [B, 18]

        size_cyl = sizes[:, 0:2]      # [h, r]
        size_yellow = sizes[:, 2:5]   # [height, length, width]
        size_ring_outer = sizes[:, 5:7]  # [outer_L, outer_W]

        pos_cyl = positions[:, 0:3]
        pos_yellow = positions[:, 3:6]
        pos_ring = positions[:, 6:9]

        rot_cyl = rotations[:, 0:6]
        rot_yellow = rotations[:, 6:12]
        rot_ring = rotations[:, 12:18]

        # 黄色长方体的高、长、宽（高沿 Y，长 X，宽 Z）
        h_yellow = size_yellow[:, 0]
        len_yellow = size_yellow[:, 1]
        wid_yellow = size_yellow[:, 2]

        # Rectangular_Ring: 墙厚=黄高, 孔=黄长×宽, 外框=给定
        front_height = h_yellow
        inner_top_length = len_yellow
        inner_top_width = wid_yellow
        outer_top_length = size_ring_outer[:, 0]
        outer_top_width = size_ring_outer[:, 1]

        Nodes = []

        Nodes.append(
            Cylinder(
                size_cyl[:, 0],
                size_cyl[:, 1],
                position=pos_cyl,
                rotation=rot_cyl,
                Semantic="peg cylinder",
            )
        )
        Nodes.append(
            Cuboid(
                h_yellow,
                len_yellow,
                wid_yellow,
                position=pos_yellow,
                rotation=rot_yellow,
                Semantic="peg yellow tab",
            )
        )
        Nodes.append(
            Rectangular_Ring(
                front_height,
                outer_top_length,
                outer_top_width,
                inner_top_length,
                inner_top_width,
                position=pos_ring,
                rotation=rot_ring,
                Semantic="peg wall with hole",
            )
        )

        self.Nodes = Nodes
        Edges = []
        # Fixed 需要 Face+Face，法向反向、切向对齐、位置重合
        # 0=圆柱(Cylinder), 1=黄色长方体(Cuboid), 2=墙(Rectangular_Ring)
        # Cylinder Faces: 0=Top, 1=Bottom, 2=Side
        # Cuboid Faces: 0=Back, 1=Front, 2=Top, 3=Bottom, 4=Left, 5=Right
        
        # 黄色长方体 (1) — 圆柱 (0): Fixed（黄色底面 Face 3 — 圆柱顶面 Face 0）
        Edges.append(
            StructureEdge(1, 0, "Fixed", {"type": 0, "idx": 3}, {"type": 0, "idx": 0}, [0, 0, 0])
        )
        
        # 注意：其他节点对之间没有显式约束，将由 _Add_Free_Edge 自动添加 Free 边
        # 包括：墙-黄色长方体 (2, 1)、墙-圆柱 (2, 0) 之间都是 Free

        self.Edges = Edges


class StructureMap_UnplugCharger(StructureGraph):
    def __init__(self, sizes, positions, rotations, clip_model, preprocess=False):
        """
        :param sizes: [B, 7]  (cyl [2] + yellow [3] + ring_outer [2])
        :param positions: [B, 9]
        :param rotations: [B, 18]
        Node: 3。Edges: 黄色长方体–圆柱 Fixed；其余 Free。
        """
        if preprocess:
            sizes = self._preprocess_parameters(sizes)

        Objects = [Peg_Three_Parts_Mechanism(sizes, positions, rotations)]

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

        super().__init__(Nodes, Edges, clip_model)

    def _preprocess_parameters(self, sizes):
        size_range = (0.02, 5)
        min_s, max_s = size_range
        sizes = torch.sigmoid(sizes) * (max_s - min_s) + min_s
        return sizes

