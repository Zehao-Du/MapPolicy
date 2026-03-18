import torch
import torch.nn as nn
import torch.nn.functional as F
from Structure_Primitive import Cuboid, Cylinder, Ring
from base_template import StructureEdge, StructureGraph
import math

class Plant:
    def __init__ (self, size, position, rotation):
        semantic = 'plant'
        
        Nodes = []
        Edges = []
        
        Nodes.append(Cylinder(size[:, 0], size[:, 1], position=position, rotation=rotation, Semantic=semantic))
        
        self.Nodes = Nodes
        self.Edges = Edges
        
class Kettle:
    def __init__ (self, sizes, positions, rotations):
        semantic1 = 'kettle body'           # cylinder 2
        semantic2 = 'kettle top handle'     # ring 3
        semantic3 = 'kettle side handle'    # ring 4
        semantic4 = 'kettle spout'          # cylinder 2
        
        Nodes = []
        Edges = []
        
        size1 = sizes[:, 0:2]
        size2 = sizes[:, 2:5]
        size3 = sizes[:, 5:9]
        size4 = sizes[:, 9:11]
        position1 = positions[:, 0:3]
        position2 = positions[:, 3:6]
        position3 = positions[:, 6:9]
        position4 = positions[:, 9:12]
        rotation1 = rotations[:, 0:1*6]
        rotation2 = rotations[:, 1*6:2*6]
        rotation3 = rotations[:, 2*6:3*6]
        rotation4 = rotations[:, 3*6:4*6]
        
        Nodes.append(Cylinder(size1[:, 0], size1[:, 1], position=position1, rotation=rotation1, Semantic=semantic1))
        Nodes.append(Ring(size2[:, 0], size2[:, 1], size2[:, 2], math.pi, position=position2, rotation=rotation2, Semantic=semantic2))
        Nodes.append(Ring(size3[:, 0], size3[:, 1], size3[:, 2], math.pi, x_z_ratio=size3[:, 3], inner_x_z_ratio=size3[:, 3],position=position3, rotation=rotation3, Semantic=semantic3))
        Nodes.append(Cylinder(size4[:, 0], size4[:, 1], position=position4, rotation=rotation4, Semantic=semantic4))
                        
        Edges.append(StructureEdge(0, 1, "Fixed", {"type": 0, "idx": 0}, {"type": 0, "idx": 1}, [0,0,0.5*math.pi]))
        Edges.append(StructureEdge(0, 2, "Fixed", {"type": 0, "idx": 2}, {"type": 0, "idx": 1}, [0,0,0.5*math.pi]))
        
        self.Nodes = Nodes
        self.Edges = Edges

       
class StructureMap_WaterPlants(StructureGraph):
    def __init__(self, sizes, positions, rotations, clip_model, preprocess=False):
        """        
        :param sizes: [B, 2+11=13]
        :param positions: [B, 5*3=15]
        :param rotations: [B, 5*6=30]
        Total: [B,58], Node:5
        """
        if preprocess:
            sizes = self._preprocess_parameters(sizes)
            
        Objects = []
        Objects.append(Plant(sizes[:, 0:2], positions[:, 0:3], rotations[:, 0:1*6]))
        Objects.append(Kettle(sizes[:, 2:13], positions[:, 3:15], rotations[:, 1*6:5*6]))
        
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
    