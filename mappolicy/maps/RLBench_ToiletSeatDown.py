import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from Structure_Primitive import Cuboid, Cylinder, Sphere
from base_template import StructureEdge, StructureGraph
        
class Toilet:
    def __init__ (self, sizes, positions, rotations):
        semantic1 = "toilet cistern"    # cuboid 3
        semantic2 = "toilet bowl"       # sphere 1   
        semantic3 = "toilet lid"        # cylinder 3
        semantic4 = "toilet base"       # cuboid 3
        self.Object_Prompt = "toilet"

        size1 = sizes[:, 0:3]
        size2 = sizes[:, 3:4]
        size3 = sizes[:, 4:7]
        size4 = sizes[:, 7:10]
        position1 = positions[:, 0:3]
        position2 = positions[:, 3:6]
        position3 = positions[:, 6:9]
        position4 = positions[:, 9:12]
        rotation1 = rotations[:, 0:1*6]
        rotation2 = rotations[:, 1*6:2*6]
        rotation3 = rotations[:, 2*6:3*6]
        rotation4 = rotations[:, 3*6:4*6]
        
        Nodes = []
        Edges = []
        
        Nodes.append(Cuboid(size1[:, 0], size1[:, 1], size1[:, 2], position=position1, rotation=rotation1, Semantic=semantic1))
        Nodes.append(Sphere(size2[:, 0], math.pi/2, position=position2, rotation=rotation2, Semantic=semantic2))
        Nodes.append(Cylinder(size3[:, 0], size3[:, 1], top_radius_z=size3[:, 2], is_half=True, position=position3, rotation=rotation3, Semantic=semantic3))
        Nodes.append(Cuboid(size4[:, 0], size4[:, 1], size4[:, 2], position=position4, rotation=rotation4, Semantic=semantic4))       

        self.Nodes = Nodes
        self.Edges = Edges
        
class StructureMap_ToiletSeatDown(StructureGraph):
    def __init__(self, sizes, positions, rotations, clip_model, preprocess=False):
        """        
        :param sizes: [B, 10]
        :param positions: [B, 4*3=12] -- 22
        :param rotations: [B, 4*6=24] -- 46
        Total: [B, 46], Node:4
        """
        if preprocess:
            sizes = self._preprocess_parameters(sizes)
            
        Objects = []
        Objects.append(Toilet(sizes[:, 0:10], positions[:, 0:12], rotations[:, 0:6*4]))
        
        
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

        self.Subgraph_Prompts = self._build_subgraph_prompts(Objects)
        
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