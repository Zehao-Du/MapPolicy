import torch

from Structure_Primitive import Cylinder
from base_template import StructureEdge, StructureGraph

class Hammer:
    def __init__ (self, sizes, positions, rotations):
        semantic1 = "hammer head"
        semantic2 = "hammer handle"

        Nodes = []
        Edges = []

        # size1: Hammer head [H, R] (2 params)
        # size2: Hammer handle [H, R] (2 params)
        size1 = sizes[:, 0:2]
        size2 = sizes[:, 2:4]

        position1 = positions[:, 0:3]
        position2 = positions[:, 3:6]

        rotation1 = rotations[:, 0:6]
        rotation2 = rotations[:, 6:12]

        Nodes.append(Cylinder(size1[:, 0], size1[:, 1], position=position1, rotation=rotation1, Semantic=semantic1))
        Nodes.append(Cylinder(size2[:, 0], size2[:, 1], position=position2, rotation=rotation2, Semantic=semantic2))

        B = sizes.shape[0]
        device = sizes.device
        Edges.append(
            StructureEdge(
                0,
                1,
                "Fixed",
                {"type": 0, "idx": 2},
                {"type": 0, "idx": 0},
                torch.zeros((B, 3), device=device),
            )
        )

        self.Nodes = Nodes
        self.Edges = Edges

class Nail:
    def __init__ (self, sizes, positions, rotations):
        semantic = "nail"

        Nodes = []
        Edges = []

        # size: Nail [H, R] (2 params)
        size = sizes[:, 0:2]

        position = positions[:, 0:3]

        rotation = rotations[:, 0:6]

        Nodes.append(Cylinder(size[:, 0], size[:, 1], position=position, rotation=rotation, Semantic=semantic))

        self.Nodes = Nodes
        self.Edges = Edges

class StructureMap_Hammer(StructureGraph):
    def __init__(self, sizes, positions, rotations, clip_model, preprocess=False):
        """        
        :param sizes: [B, 4(Hammer) + 2(Nail) = 6]
        :param positions: [B, 6(Hammer) + 3(Nail) = 9]
        :param rotations: [B, 12(Hammer) + 6(Nail) = 18]
        """
        if preprocess:
            sizes = self._preprocess_parameters(sizes)
            
        Objects = []
        Objects.append(Hammer(sizes[:, 0:4], positions[:, 0:6], rotations[:, 0:12]))
        Objects.append(Nail(sizes[:, 4:6], positions[:, 6:9], rotations[:, 12:18]))
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
            sizes: [B, 6] 网络原始输出
            size_range: (min_val, max_val) 尺寸的最小值和最大值约束
            
        Returns:
            constrained_sizes
        """
        size_range=(0.02, 5)
        min_s, max_s = size_range
        sizes = torch.sigmoid(sizes) * (max_s - min_s) + min_s
        
        return sizes
