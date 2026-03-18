import torch
from Structure_Primitive import Cuboid, Sphere
from base_template import StructureEdge, StructureGraph


class Soccer_Mechanism:
    """
    五部件：球、左柱、右柱、横梁、球网（薄板）。
    - 左柱—横梁、右柱—横梁: Planar-Contact；球网一面—横梁后面: Planar-Contact；两门柱轴线平行；球网顶面边‖横梁轴线；球与球门: Free。
    """

    def __init__(self, sizes, positions, rotations):
        # sizes: [B, 13]  ->  ball(1) + post_L(3) + post_R(3) + bar(3) + net(3)
        # positions: [B, 15],  rotations: [B, 30]

        size_ball = sizes[:, 0:1]
        size_post_L = sizes[:, 1:4]
        size_post_R = sizes[:, 4:7]
        size_bar = sizes[:, 7:10]
        size_net = sizes[:, 9:12]

        pos_ball = positions[:, 0:3]
        pos_post_L = positions[:, 3:6]
        pos_post_R = positions[:, 6:9]
        pos_bar = positions[:, 9:12]
        pos_net = positions[:, 12:15]

        rot_ball = rotations[:, 0:6]
        rot_post_L = rotations[:, 6:12]
        rot_post_R = rotations[:, 12:18]
        rot_bar = rotations[:, 18:24]
        rot_net = rotations[:, 24:30]

        Nodes = []

        Nodes.append(
            Sphere(
                size_ball[:, 0],
                position=pos_ball,
                rotation=rot_ball,
                Semantic="soccer ball",
            )
        )
        Nodes.append(
            Cuboid(
                size_post_L[:, 0],
                size_post_L[:, 1],
                size_post_L[:, 2],
                position=pos_post_L,
                rotation=rot_post_L,
                Semantic="goal post left",
            )
        )
        Nodes.append(
            Cuboid(
                size_post_R[:, 0],
                size_post_R[:, 1],
                size_post_R[:, 2],
                position=pos_post_R,
                rotation=rot_post_R,
                Semantic="goal post right",
            )
        )
        Nodes.append(
            Cuboid(
                size_bar[:, 0],
                size_bar[:, 1],
                size_bar[:, 2],
                position=pos_bar,
                rotation=rot_bar,
                Semantic="goal crossbar",
            )
        )
        Nodes.append(
            Cuboid(
                size_net[:, 0],
                size_net[:, 1],
                size_net[:, 2],
                position=pos_net,
                rotation=rot_net,
                Semantic="goal net",
            )
        )

        self.Nodes = Nodes
        Edges = []
        # 0=球, 1=左柱, 2=右柱, 3=横梁, 4=球网
        # Cuboid Faces: 0=Back, 1=Front, 2=Top, 3=Bottom, 4=Left, 5=Right
        
        # 左柱 (1) — 横梁 (3): Planar-Contact（柱顶 Face 2 — 横梁前面 Face 1）
        Edges.append(
            StructureEdge(1, 3, "Planar-Contact", {"type": 0, "idx": 2}, {"type": 0, "idx": 1}, [0, 0, 0])
        )
        # 右柱 (2) — 横梁 (3): Planar-Contact（柱顶 Face 2 — 横梁前面 Face 1）
        Edges.append(
            StructureEdge(2, 3, "Planar-Contact", {"type": 0, "idx": 2}, {"type": 0, "idx": 1}, [0, 0, 0])
        )
        # 球网 (4) — 横梁 (3): Planar-Contact（球网前面 Face 1 — 横梁后面 Face 0）
        Edges.append(
            StructureEdge(4, 3, "Planar-Contact", {"type": 0, "idx": 1}, {"type": 0, "idx": 0}, [0, 0, 0])
        )
        # 左柱 (1) — 右柱 (2): Alignment（两门柱轴线平行，Cuboid Axis 12 为主轴/竖轴）
        Edges.append(
            StructureEdge(1, 2, "Alignment", {"type": 1, "idx": 12}, {"type": 1, "idx": 12}, [0, 0, 0])
        )
        # 球网 (4) — 横梁 (3): Alignment（球网顶面边 ‖ 横梁轴线）
        # Cuboid 顶面边: Axis 0~3；横梁沿 X 的轴线用 Axis 0（顶面一条边）
        Edges.append(
            StructureEdge(4, 3, "Alignment", {"type": 1, "idx": 0}, {"type": 1, "idx": 0}, [0, 0, 0])
        )
        self.Edges = Edges


class StructureMap_Soccer(StructureGraph):
    def __init__(self, sizes, positions, rotations, clip_model, preprocess=False):
        """
        :param sizes: [B, 13]  (ball [1] + post_L [3] + post_R [3] + bar [3] + net [3])
        :param positions: [B, 15]
        :param rotations: [B, 30]
        Node: 5。Edges: 左柱–横梁、右柱–横梁、球网–横梁 Planar-Contact；左柱–右柱、球网–横梁 Alignment；其余 Free。
        """
        if preprocess:
            sizes = self._preprocess_parameters(sizes)

        Objects = [Soccer_Mechanism(sizes, positions, rotations)]

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