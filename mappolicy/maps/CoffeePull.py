import torch
import torch.nn as nn
import torch.nn.functional as F
from Structure_Primitive import Cuboid, Cylinder
from base_template import StructureEdge, StructureGraph

class Cup:
    def __init__ (self, size, position, rotation):
        semantic = 'cup'
        
        Nodes = []
        Edges = []
        
        Nodes.append(Cylinder(size[:, 0], size[:, 1], position=position, rotation=rotation, Semantic=semantic))
        
        self.Nodes = Nodes
        self.Edges = Edges
        
class CoffeMachine:
    def __init__ (self, sizes, positions, rotations):
        semantic1 = "coffee machine body"
        semantic2 = "coffee machine body"
        semantic3 = "coffee machine button"
        # semantic4 = "coffee machine spout"
        
        size1 = sizes[:, 0:3]
        size2 = sizes[:, 3:6]
        size3 = sizes[:, 6:8]
        # size4 = sizes[:, 8:10]
        
        position1 = positions[:, 0:3]
        position2 = positions[:, 3:6]
        position3 = positions[:, 6:9]
        position4 = positions[:, 9:12]
        
        rotation1 = rotations[:, 0:1*6]
        rotation2 = rotations[:, 1*6:2*6]
        rotation3 = rotations[:, 2*6:3*6]
        # rotation4 = rotations[:, 3*6:4*6]
        
        Nodes = []
        Edges = []
        
        Nodes.append(Cuboid(size1[:, 0], size1[:, 1], size1[:, 2], position=position1, rotation=rotation1, Semantic=semantic1))
        Nodes.append(Cuboid(size2[:, 0], size2[:, 1], size2[:, 2], position=position2, rotation=rotation2, Semantic=semantic2))
        Nodes.append(Cylinder(size3[:, 0], size3[:, 1], position=position3, rotation=rotation3, Semantic=semantic3))
        # Nodes.append(Cylinder(size4[:, 0], size4[:, 1], position=position4, rotation=rotation4, Semantic=semantic4))
        
        Edges.append(StructureEdge(0, 1, "Planar-Contact", {"type": 0, "idx": 2}, {"type": 0, "idx": 3}, [0, 0, 0]))
        Edges.append(StructureEdge(0, 1, "Alignment", {"type": 0, "idx": 1}, {"type": 0, "idx": 1}, [0, 0, 0]))
        Edges.append(StructureEdge(0, 1, "Alignment", {"type": 0, "idx": 4}, {"type": 0, "idx": 4}, [0, 0, 0]))
        Edges.append(StructureEdge(1, 2, "Fixed", {"type": 0, "idx": 5}, {"type": 0, "idx": 1}, [0, 0, 0]))
        
        self.Nodes = Nodes
        self.Edges = Edges
        
class StructureMap_CoffeePull(StructureGraph):
    def __init__(self, sizes, positions, rotations, clip_model, preprocess=False):
        """        
        :param sizes: [B, 11]
        :param positions: [B,12] -- 23
        :param rotations: [B, 4*6] -- 47
        Total: [B, 47], Node:4
        """
        if preprocess:
            sizes = self._preprocess_parameters(sizes)
            
        Objects = []
        Objects.append(Cup(sizes[:, 0:3], positions[:, 0:3], rotations[:, 0:6*1]))
        Objects.append(CoffeMachine(sizes[:, 3:11], positions[:, 3:12], rotations[:, 1*6:4*6]))
        
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
 
