import torch
from Structure_Primitive import Sphere, Torus, Cuboid, Cylinder
from base_template import StructureEdge, StructureGraph


class Basketball:
    """篮球：单个 Sphere 节点，无内部边。"""

    def __init__(self, sizes, positions, rotations):
        semantic1 = "basketball"
        self.Object_Prompt = "basketball"

        Nodes = []
        Edges = []

        # sizes: [B, 1] -> radius [B]
        radius = sizes[:, 0]
        position1 = positions[:, 0:3]
        rotation1 = rotations[:, 0:6]

        Nodes.append(
            Sphere(radius, position=position1, rotation=rotation1, Semantic=semantic1)
        )

        self.Nodes = Nodes
        self.Edges = Edges


class Basket:
    """
    篮筐组装：篮圈 (Torus)、篮板 (Cuboid)、支撑杆 (Cylinder)。

    尺寸与布局取自 Meta-World 源码：metaworld/assets/objects/assets/basketballhoop.xml
    - 篮圈：大圆半径 cr（TARGET_RADIUS=0.08），环管半径 tr
    - 篮板：backboard 碰撞 box size="0.1 0.01 0.07" (half) → 高 0.14、宽 0.2、厚 0.02；pos 0,0,0.29
    - 支撑杆：pole 碰撞 cylinder pos 0,0,0.118 size="0.007 0.108" (r, half-h) → 高 0.216、半径 0.007
    - hooplink pos 0,-0.083,0.25（篮圈在篮板前方 0.083，中心 z=0.25）
    """

    def __init__(self, sizes, positions, rotations):
        self.Object_Prompt = "basket"
        Nodes = []
        Edges = []

        # sizes: [B, 7] = hoop(2) + backboard(3) + pole(2)
        cr = sizes[:, 0]       # 篮圈开口半径
        tr = sizes[:, 1]       # 篮圈管半径
        bb_h = sizes[:, 2]     # 篮板高 (Y)
        bb_w = sizes[:, 3]     # 篮板宽 (X)
        bb_t = sizes[:, 4]     # 篮板厚 (Z)
        pole_h = sizes[:, 5]   # 杆高 (Y)
        pole_r = sizes[:, 6]   # 杆半径

        # positions: [B, 9] = hoop(3) + backboard(3) + pole(3)
        pos_hoop = positions[:, 0:3]
        pos_backboard = positions[:, 3:6]
        pos_pole = positions[:, 6:9]

        # rotations: [B, 18] = 6*3
        rot_hoop = rotations[:, 0:6]
        rot_backboard = rotations[:, 6:12]
        rot_pole = rotations[:, 12:18]

        Nodes.append(Torus(cr, tr, position=pos_hoop, rotation=rot_hoop, Semantic="basketball hoop"))
        Nodes.append(Cuboid(bb_h, bb_w, bb_t, position=pos_backboard, rotation=rot_backboard, Semantic="backboard"))
        Nodes.append(Cylinder(pole_h, pole_r, position=pos_pole, rotation=rot_pole, Semantic="backboard pole"))

        # 篮筐、篮板、杆子的约束关系
        # Fixed 需要 Face+Face，法向反向、切向对齐、位置重合；Planar-Contact 需要两面平行且接触
        # 0=篮圈(Torus), 1=篮板(Cuboid), 2=支撑杆(Cylinder)
        # 
        # Torus Faces: 0=Outer, 1=Top, 2=Inner(指向中心), 3=Bottom
        # Cuboid Faces: 0=Back, 1=Front, 2=Top, 3=Bottom, 4=Left, 5=Right
        # Cylinder Faces: 0=Top, 1=Bottom, 2=Side
        # Torus Axis 0: 穿过圆环中心的垂直线；Cuboid Axis 12: 主轴（底到顶/竖棱方向）
        
        # 篮板-杆子: 杆子支撑篮板，杆子顶部与篮板底部 Fixed
        # 使用篮板的底部(Face 3) 和 杆子的顶部(Face 0)
        Edges.append(StructureEdge(1, 2, "Fixed", {"type": 0, "idx": 3}, {"type": 0, "idx": 0}, [0, 0, 0]))
        
        # 篮圈-篮板: 篮圈通过连接件固定在篮板前面
        # 使用篮圈的外侧面(Face 0, Outer) 和 篮板的前面(Face 1, Front)
        Edges.append(StructureEdge(0, 1, "Planar-Contact", {"type": 0, "idx": 0}, {"type": 0, "idx": 1}, [0, 0, 0]))
        
        # 篮圈-篮板: 篮球框穿过中心的轴线与篮板竖着的棱平行
        # Torus Axis 0: 穿过圆环中心的垂直线；Cuboid Axis 12: 篮板主轴（底到顶，竖棱方向）
        Edges.append(StructureEdge(0, 1, "Alignment", {"type": 1, "idx": 0}, {"type": 1, "idx": 12}, [0, 0, 0]))
        
        # 注意：其他节点对之间没有显式约束，将由 _Add_Free_Edge 自动添加 Free 边
        # 包括：篮圈-杆子 (0, 2) 之间也是 Free

        self.Nodes = Nodes
        self.Edges = Edges


class StructureMap_Basketball(StructureGraph):
    """
    Basketball 任务的结构图。

    :param sizes: [B, 8] = [ball_radius, hoop_cr, hoop_tr, bb_h, bb_w, bb_t, pole_h, pole_r]
        - ball_radius: 篮球半径 0.025（与 MW caging obj_radius 一致）
        - hoop_cr, hoop_tr: 篮圈大圆 0.08（TARGET_RADIUS）、环管约 0.015；见 basketballhoop.xml
        - bb_h, bb_w, bb_t: 篮板高、宽、厚，取自 MW backboard 碰撞 box size 0.1,0.01,0.07 (half) → 0.14, 0.2, 0.02
        - pole_h, pole_r: 支撑杆高、半径，取自 MW pole 碰撞 cylinder 0.007, 0.108 (r, half-h) → 0.216, 0.007
    :param positions: [B, 12] = ball(3) + hoop(3) + backboard(3) + pole(3)
        - 篮球: x∈[-0.1,0.1], y∈[0.6,0.7], z=0.03
        - 篮圈: hooplink pos (0,-0.083,0.25) 相对 basket_goal → 若 basket 在 [0,0.9,0] 则 [0, 0.817, 0.25]
        - 篮板: backboard pos (0,0,0.29) → [0, 0.9, 0.29]
        - 支撑杆: pole 圆柱中心 (0,0,0.118) → [0, 0.9, 0.118]
    :param rotations: [B, 24] = ball(6) + hoop(6) + backboard(6) + pole(6)，各 6D 旋转
    :param clip_model: CLIP 编码器
    :param preprocess: 是否对 sizes 做 sigmoid 等预处理
    """

    def __init__(self, sizes, positions, rotations, clip_model, preprocess=False):
        if preprocess:
            sizes = self._preprocess_parameters(sizes)

        Objects = []
        Objects.append(
            Basketball(sizes[:, 0:1], positions[:, 0:3], rotations[:, 0:6])
        )
        Objects.append(
            Basket(sizes[:, 1:8], positions[:, 3:12], rotations[:, 6:24])
        )

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
        """
        对网络输出的参数进行预处理，使其符合物理约束。
        """
        size_range = (0.02, 5)
        min_s, max_s = size_range
        sizes = torch.sigmoid(sizes) * (max_s - min_s) + min_s
        return sizes