import torch
import torch.nn as nn
import torch.nn.functional as F
from Structure_Primitive import Cuboid, Cylinder, Rectangular_Ring
from base_template import StructureEdge, StructureGraph

class Bucket:
    def __init__ (self, size, position, rotation):
        semantic = 'bucket'
        
        Nodes = []
        Edges = []
        
        Nodes.append(Cylinder(size[:, 0], size[:, 1], position=position, rotation=rotation, Semantic=semantic))
        
        self.Nodes = Nodes
        self.Edges = Edges
        
class Rubbish:
    def __init__ (self, sizes, positions, rotations):
        semantic1 = "rubbish"

        size1 = sizes[:, 0:3]
        position1 = positions[:, 0:3]
        rotation1 = rotations[:, 0:1*6]
        
        Nodes = []
        Edges = []
        
        Nodes.append(Cuboid(size1[:, 0], size1[:, 1], size1[:, 2], position=position1, rotation=rotation1, Semantic=semantic1))

        self.Nodes = Nodes
        self.Edges = Edges
        
class Fruit:
    def __init__ (self, sizes, positions, rotations):
        semantic1 = "fruit"
        semantic2 = "fruit"

        size1 = sizes[:, 0:3]
        size2 = sizes[:, 3:6]
        position1 = positions[:, 0:3]
        position2 = positions[:, 3:6]
        rotation1 = rotations[:, 0:1*6]
        rotation2 = rotations[:, 1*6:2*6]
        
        Nodes = []
        Edges = []
        
        Nodes.append(Cuboid(size1[:, 0], size1[:, 1], size1[:, 2], position=position1, rotation=rotation1, Semantic=semantic1))
        Nodes.append(Cuboid(size2[:, 0], size2[:, 1], size2[:, 2], position=position2, rotation=rotation2, Semantic=semantic2))       

        self.Nodes = Nodes
        self.Edges = Edges
        
class StructureMap_PutRubbishIn(StructureGraph):
    def __init__(self, sizes, positions, rotations, clip_model, preprocess=False):
        """        
        :param sizes: [B, 2+3+6=11]
        :param positions: [B, 4*3=12] -- 23
        :param rotations: [B, 4*6=24] -- 47
        Total: [B, 47], Node:4
        """
        if preprocess:
            sizes = self._preprocess_parameters(sizes)
            
        Objects = []
        Objects.append(Bucket(sizes[:, 0:2], positions[:, 0:3], rotations[:, 0:6*1]))
        Objects.append(Rubbish(sizes[:, 2:5], positions[:, 3:6], rotations[:, 1*6:2*6]))
        Objects.append(Fruit(sizes[:, 5:11], positions[:, 6:12], rotations[:, 2*6:4*6]))
        
        
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