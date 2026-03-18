import torch
import torch.nn as nn
import torch.nn.functional as F
from Structure_Primitive import Cuboid, Cylinder
from base_template import StructureEdge, StructureGraph

class Handle:
    def __init__ (self, sizes, positions, rotations):
        semantic1 = "base"
        semantic2 = "handle"
        semantic3 = "link"
        
        size1 = sizes[:, 0:2]
        size2 = sizes[:, 2:4]
        size3 = sizes[:, 4:6]
        
        position1 = positions[:, 0:3]
        position2 = positions[:, 3:6]
        position3 = positions[:, 6:9]
        
        rotation1 = rotations[:, 0:1*6]
        rotation2 = rotations[:, 1*6:2*6]
        rotation3 = rotations[:, 2*6:3*6]
        
        Nodes = []
        Edges = []
        
        Nodes.append(Cylinder(size1[:, 0], size1[:, 1], is_quarter=True, position=position1, rotation=rotation1, Semantic=semantic1))
        Nodes.append(Cylinder(size2[:, 0], size2[:, 1], position=position2, rotation=rotation2, Semantic=semantic2))
        Nodes.append(Cylinder(size3[:, 0], size3[:, 1], position=position3, rotation=rotation3, Semantic=semantic3))
        
        R = torch.linalg.norm(position2 - position1, dim=-1, keepdim=True)
        ZERO = torch.zeros_like(R)
        Param1 = torch.cat((R, ZERO, ZERO), dim=-1)
        Edges.append(StructureEdge(0, 1, "Revolute", {"type": 1, "idx": 0}, {"type": 1, "idx": 0}, Param1))
        Edges.append(StructureEdge(1, 2, "Fixed", {"type": 0, "idx": 2}, {"type": 0, "idx": 1}, [0, 0, 0]))
        
        self.Nodes = Nodes
        self.Edges = Edges
        
class StructureMap_HandlePull(StructureGraph):
    def __init__(self, sizes, positions, rotations, clip_model, preprocess=False):
        """        
        :param sizes: [B, 6]
        :param positions: [B, 9] -- 15
        :param rotations: [B, 3*6] -- 33
        Total: [B, 47], Node:3
        """
        if preprocess:
            sizes = self._preprocess_parameters(sizes)
            
        Objects = []
        Objects.append(Handle(sizes, positions, rotations))
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
    
