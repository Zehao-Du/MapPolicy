import torch
import torch.nn as nn
import torch.nn.functional as F
from Structure_Primitive import Cuboid
from base_template import StructureEdge, StructureGraph

class RedCube:
    def __init__ (self, sizes, positions, rotations):
        semantic1 = 'red cube'
        self.Object_Prompt = 'red cube'
        
        Nodes = []
        Edges = []
        
        size1 = sizes[:, 0:3]
        position1 = positions[:, 0:3]
        rotation1 = rotations[:, 0:6*1]
        
        Nodes.append(Cuboid(size1[:, 0], size1[:, 1], size1[:, 2], position=position1, rotation=rotation1, Semantic=semantic1))

        self.Nodes = Nodes
        self.Edges = Edges

class GreenCube:
    def __init__ (self, sizes, positions, rotations):
        semantic1 = 'green cube'
        self.Object_Prompt = 'green cube'
        
        Nodes = []
        Edges = []
        
        size1 = sizes[:, 0:3]
        position1 = positions[:, 0:3]
        rotation1 = rotations[:, 0:6*1]
        
        Nodes.append(Cuboid(size1[:, 0], size1[:, 1], size1[:, 2], position=position1, rotation=rotation1, Semantic=semantic1))

        self.Nodes = Nodes
        self.Edges = Edges
        
class Desk:
    def __init__ (self, sizes, positions, rotations):
        semantic1 = 'desk'
        self.Object_Prompt = 'table top'
        
        Nodes = []
        Edges = []
        
        size1 = sizes[:, 0:3]
        position1 = positions[:, 0:3]
        rotation1 = rotations[:, 0:6*1]
        
        Nodes.append(Cuboid(size1[:, 0], size1[:, 1], size1[:, 2], position=position1, rotation=rotation1, Semantic=semantic1))

        self.Nodes = Nodes
        self.Edges = Edges

class StructureMap_StackCube(StructureGraph):
    def __init__(self, sizes, positions, rotations, clip_model, preprocess=False):
        """        
        :param sizes: [B, 9]
        :param positions: [B, 9]
        :param rotations: [B, 3*6]
        Total: [B, 24], Node:3, Edge:0
        """
        if preprocess:
            sizes = self._preprocess_parameters(sizes)
            
        Objects = []
        Objects.append(RedCube(sizes[:, 0:3], positions[:, 0:3], rotations[:, 0:6*1]))
        Objects.append(GreenCube(sizes[:, 3:6], positions[:, 3:6], rotations[:, 6*1:6*2]))
        Objects.append(Desk(sizes[:, 6:9], positions[:, 6:9], rotations[:, 6*2:6*3]))

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
        
        self.Nodes = Nodes
        self.Edges = Edges
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
        # 1. Size 处理: 限制在 [min, max] 之间
        # 使用 sigmoid 将 (-inf, +inf) 映射到 (0, 1)
        # 然后线性缩放到 (min, max)
        size_range=(0.02, 5)
        min_s, max_s = size_range
        sizes = torch.sigmoid(sizes) * (max_s - min_s) + min_s
        
        return sizes