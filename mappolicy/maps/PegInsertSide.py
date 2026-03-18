import torch
import torch.nn as nn
import torch.nn.functional as F
from Structure_Primitive import Cuboid, Cylinder
from base_template import StructureEdge, StructureGraph

import math

class Target_Block:
    def __init__ (self, sizes, positions, rotations):
        semantic1 = 'target block'
        semantic2 = 'target area'
        
        Nodes = []
        Edges = []
        
        size1 = sizes[:, 0:3]
        size2 = sizes[:, 3:6]
        position1 = positions[:, 0:3]
        position2 = positions[:, 3:6]
        rotation1 = rotations[:, 0:6*1]
        rotation2 = rotations[:, 6*1:6*2]
        
        Nodes.append(Cuboid(size1[:, 0], size1[:, 1], size1[:, 2], position=position1, rotation=rotation1, Semantic=semantic1))
        Nodes.append(Cuboid(size2[:, 0], size2[:, 1], size2[:, 2], position=position2, rotation=rotation2, Semantic=semantic2))
        
        Edges.append(StructureEdge(0, 1, "Planar-Contact", {"type": 0, "idx": 1}, {"type": 0, "idx": 1}, [0, 0, 0])) # Front
        Edges.append(StructureEdge(0, 1, "Alignment ", {"type": 0, "idx": 2}, {"type": 0, "idx": 2}, [0, 0, 0])) # Top
        Edges.append(StructureEdge(0, 1, "Alignment ", {"type": 0, "idx": 5}, {"type": 0, "idx": 5}, [0, 0, 0])) # Right

        self.Nodes = Nodes
        self.Edges = Edges

class Peg:
    def __init__ (self, size, position, rotation):
        semantic = 'peg'
        
        Nodes = []
        Edges = []
        
        Nodes.append(Cylinder(size[:, 0], size[:, 1], position=position, rotation=rotation, Semantic=semantic))
        
        self.Nodes = Nodes
        self.Edges = Edges
        
class StructureMap_PegInsertSide(StructureGraph):
    def __init__(self, sizes, positions, rotations, clip_model, preprocess=False):
        """        
        :param sizes: [B, 8]
        :param positions: [B, 9]
        :param rotations: [B, 3*6]
        Total: [B, 35], Node:3
        """
        if preprocess:
            sizes = self._preprocess_parameters(sizes)
        target_block = Target_Block(sizes[:, 0:6], positions[:, 0:6], rotations[:, 0:6*2])
        peg = Peg(sizes[:, 6:8], positions[:, 6:9], rotations[:, 6*2:6*3])
        
        Nodes = []
        Edges = []
        
        num_node = 0
        for node in target_block.Nodes:
            Nodes.append(node)
        for edge in target_block.Edges:
            edge.Node_idx[0] += num_node 
            edge.Node_idx[1] += num_node 
            Edges.append(edge)
        num_node += len(target_block.Nodes)
        
        for node in peg.Nodes:
            Nodes.append(node)
        for edge in peg.Edges:
            edge.Node_idx[0] += num_node 
            edge.Node_idx[1] += num_node 
            Edges.append(edge)
        num_node += len(peg.Nodes)
        
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
        # 1. Size 处理: 限制在 [min, max] 之间
        # 使用 sigmoid 将 (-inf, +inf) 映射到 (0, 1)
        # 然后线性缩放到 (min, max)
        size_range=(0.02, 5)
        min_s, max_s = size_range
        sizes = torch.sigmoid(sizes) * (max_s - min_s) + min_s
        
        return sizes