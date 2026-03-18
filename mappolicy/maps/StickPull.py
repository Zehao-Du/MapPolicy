import torch
from Structure_Primitive import Cylinder, Rectangular_Ring
from base_template import StructureEdge, StructureGraph


class Stick_Pull_Mechanism:
    """
    四部件：杆(细长圆柱)、保温杯底座、瓶身、把手(Rectangular_Ring，孔轴 X，贴杯壁)。
    Edges: 底座—杯体 Fixed；把手—杯体 Planar-Contact + Alignment（顶面开洞棱‖杯体轴线）；其他 Free。
    """

    def __init__(self, sizes, positions, rotations):
        # sizes: [B, 11]  ->  stick(2: height, radius) + base(2) + body(2) + handle(5)
        # positions: [B, 12],  rotations: [B, 24]

        size_stick = sizes[:, 0:2]
        size_base = sizes[:, 2:4]
        size_body = sizes[:, 4:6]
        size_handle = sizes[:, 6:11]

        pos_stick = positions[:, 0:3]
        pos_base = positions[:, 3:6]
        pos_body = positions[:, 6:9]
        pos_handle = positions[:, 9:12]

        rot_stick = rotations[:, 0:6]
        rot_base = rotations[:, 6:12]
        rot_body = rotations[:, 12:18]
        rot_handle = rotations[:, 18:24]

        Nodes = []

        # 杆：细长圆柱，Cylinder 默认轴向 Y，需 rot 使轴向 X 以穿过把手孔
        Nodes.append(
            Cylinder(
                size_stick[:, 0],   # height: 沿轴向长度
                size_stick[:, 1],   # top_radius (= bottom_radius)
                position=pos_stick,
                rotation=rot_stick,
                Semantic="stick",
            )
        )
        Nodes.append(
            Cylinder(
                size_base[:, 0],
                size_base[:, 1],
                position=pos_base,
                rotation=rot_base,
                Semantic="thermos base",
            )
        )
        Nodes.append(
            Cylinder(
                size_body[:, 0],
                size_body[:, 1],
                position=pos_body,
                rotation=rot_body,
                Semantic="thermos body",
            )
        )
        # Rectangular_Ring: 孔轴(local Y)经 rot_z_neg90 对齐 world X；local -X 面贴杯壁
        # 参数: front_height(孔沿X的延伸), outer_L(径向,贴杯侧到外缘), outer_W(Z向), inner_L, inner_W
        Nodes.append(
            Rectangular_Ring(
                size_handle[:, 0],   # front_height: 孔沿 X 的延伸
                size_handle[:, 1],   # outer_top_length: 外框 径向(rot 后=Y 向，贴杯侧到外缘)
                size_handle[:, 2],   # outer_top_width: 外框 Z 向
                size_handle[:, 3],   # inner_top_length: 孔 X 向
                size_handle[:, 4],   # inner_top_width: 孔 Z 向
                position=pos_handle,
                rotation=rot_handle,
                Semantic="thermos handle",
            )
        )

        self.Nodes = Nodes
        # 根据文档，Planar-Contact 约束需要 Face $i$ + Face $j$，要求两个面平行且接触
        # 0=杆(Cylinder), 1=底座(Cylinder), 2=杯体(Cylinder), 3=把手(Rectangular_Ring)
        # Cylinder Faces: 0=Top, 1=Bottom, 2=Side
        # Rectangular_Ring Faces: 0-3=外侧面, 4-7=内侧面, 8-11=上底, 12-15=下底
        
        Edges = []
        
        # 底座—杯体: Fixed（底座顶面 Face 0 — 杯体底面 Face 1，两圆柱刚性连接）
        Edges.append(
            StructureEdge(1, 2, "Fixed", {"type": 0, "idx": 0}, {"type": 0, "idx": 1}, [0, 0, 0])
        )
        
        # 把手—杯体: Planar-Contact（把手外侧面 Face 1 — 杯体侧面 Face 2）
        Edges.append(
            StructureEdge(3, 2, "Planar-Contact", {"type": 0, "idx": 1}, {"type": 0, "idx": 2}, [0, 0, 0])
        )
        
        # 把手—杯体: Alignment（顶面开洞的棱 ‖ 大圆柱轴线）
        # Rectangular_Ring Axis 13~16: 顶面内孔棱（Top Inner Edges）；Cylinder Axis 0: 中心轴
        Edges.append(
            StructureEdge(3, 2, "Alignment", {"type": 1, "idx": 14}, {"type": 1, "idx": 0}, [0, 0, 0])
        )
        
        # 注意：其他节点对之间没有显式约束，将由 _Add_Free_Edge 自动添加 Free 边
        # 包括：杆子(0)与底座(1)、杆子(0)与杯体(2)、杆子(0)与把手(3)之间都是 Free
        
        self.Edges = Edges


class StructureMap_StickPull(StructureGraph):
    def __init__(self, sizes, positions, rotations, clip_model, preprocess=False):
        """
        :param sizes: [B, 11]  (stick [2: height, radius] + base [2] + body [2] + handle [5])
        :param positions: [B, 12]
        :param rotations: [B, 24]
        Node: 4。Edges: 底座–杯体 Fixed，把手–杯体 Planar-Contact + Alignment；其余 Free。
        """
        if preprocess:
            sizes = self._preprocess_parameters(sizes)

        Objects = [Stick_Pull_Mechanism(sizes, positions, rotations)]

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
