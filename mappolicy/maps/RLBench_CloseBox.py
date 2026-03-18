import torch
import torch.nn as nn
import torch.nn.functional as F
from Structure_Primitive import Cuboid, Cylinder, Rectangular_Ring
from base_template import StructureEdge, StructureGraph

class Box:
    def __init__ (self, sizes, positions, rotations):
        semantic1 = 'box bottom'
        semantic2 = 'box body'
        semantic3 = 'box cover'
        
        Nodes = []
        Edges = []
        
        size1 = sizes[:, 0:3]
        size2 = sizes[:, 3:8]
        size3 = sizes[:, 8:11]
        position1 = positions[:, 0:3]
        position2 = positions[:, 3:6]
        position3 = positions[:, 6:9]
        rotation1 = rotations[:, 0:1*6]
        rotation2 = rotations[:, 1*6:2*6]
        rotation3 = rotations[:, 2*6:3*6]
        
        Nodes.append(Cuboid(size1[:, 0], size1[:, 1], size1[:, 2], position=position1, rotation=rotation1, Semantic=semantic1))
        Nodes.append(Rectangular_Ring(size2[:, 0], size2[:, 1], size2[:, 2], size2[:, 3], size2[:, 4], position=position2, rotation=rotation2, Semantic=semantic2))
        Nodes.append(Cuboid(size3[:, 0], size3[:, 1], size3[:, 2], position=position3, rotation=rotation3, Semantic=semantic3))
        
        Edges.append(StructureEdge(0, 1, "Cylindrical", {"type": 1, "idx": 12}, {"type": 1, "idx": 0}, [0, 0, 0]))
        Edges.append(StructureEdge(0, 1, "Planar-Contact", {"type": 0, "idx": 2}, {"type": 0, "idx": 12}, [0, 0, 0]))
        Edges.append(StructureEdge(0, 1, "Planar-Contact", {"type": 0, "idx": 2}, {"type": 0, "idx": 13}, [0, 0, 0]))
        Edges.append(StructureEdge(0, 1, "Planar-Contact", {"type": 0, "idx": 2}, {"type": 0, "idx": 14}, [0, 0, 0]))
        Edges.append(StructureEdge(0, 1, "Planar-Contact", {"type": 0, "idx": 2}, {"type": 0, "idx": 15}, [0, 0, 0]))
        Edges.append(StructureEdge(0, 1, "Alignment", {"type": 0, "idx": 1}, {"type": 0, "idx": 0}, [0, 0, 0]))
        
        Edges.append(StructureEdge(1, 2, "Revolute", {"type": 1, "idx": 8}, {"type": 1, "idx": 0}, [0, 0, 0]))
        
        self.Nodes = Nodes
        self.Edges = Edges
        
class StructureMap_CloseBox(StructureGraph):
    def __init__(self, sizes, positions, rotations, clip_model, preprocess=False):
        """        
        :param sizes: [B, 11]
        :param positions: [B, 3*3=9]
        :param rotations: [B, 3*6=18]
        Total: [B,38], Node:3
        """
        if preprocess:
            sizes = self._preprocess_parameters(sizes)
            
        Objects = []
        Objects.append(Box(sizes[:, 0:11], positions[:, 0:9], rotations[:, 0:3*6]))
        
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