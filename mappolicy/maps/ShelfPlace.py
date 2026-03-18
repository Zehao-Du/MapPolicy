import torch
import torch.nn as nn
import torch.nn.functional as F
from Structure_Primitive import Cuboid, Rectangular_Ring
from base_template import StructureEdge, StructureGraph

class Target_Block:
    def __init__(self, sizes, positions, rotations):
        semantic1 = "puck"
        
        Nodes = []
        Edges = []
        
        size1 = sizes[:, 0:3]
        position1 = positions[:, 0:3]
        rotation1 = rotations[:, 0:6*1]
        
        Nodes.append(Cuboid(size1[:, 0], size1[:, 1], size1[:, 2], position=position1, rotation=rotation1, Semantic=semantic1))
        
        self.Nodes = Nodes
        self.Edges = Edges
        
class Shelf:
    def __init__(self, sizes, positions, rotations):
        semantic1 = 'bookshelf\'s back panel'
        semantic2 = 'bookshelf\'s shelf '
        semantic3 = 'bookshelf\'s top, bottom and side panel'
        
        Nodes = []
        Edges = []
        
        size1 = sizes[:, 0:3]
        size2 = sizes[:, 3:6]
        size3 = sizes[:, 6:11]
        position1 = positions[:, 0:3]
        position2 = positions[:, 3:6]
        position3 = positions[:, 6:9]
        rotation1 = rotations[:, 0:6*1]
        rotation2 = rotations[:, 6*1:6*2]
        rotation3 = rotations[:, 6*2:6*3]
        
        Nodes.append(Cuboid(size1[:, 0], size1[:, 1], size1[:, 2], position=position1, rotation=rotation1, Semantic=semantic1))
        Nodes.append(Cuboid(size2[:, 0], size2[:, 1], size2[:, 2], position=position2, rotation=rotation2, Semantic=semantic2))
        Nodes.append(Rectangular_Ring(size3[:, 0], size3[:, 1], size3[:, 2], size3[:, 3], size3[:, 4], position=position3, rotation=rotation3, Semantic=semantic3))
        
        Edges.append(StructureEdge(0, 1, "Planar-Contact", {"type": 0, "idx": 1}, {"type": 0, "idx": 0}, [0, 0, 0]))
        
        Edges.append(StructureEdge(1, 2, "Planar-Contact", {"type": 0, "idx": 4}, {"type": 0, "idx": 6}, [0, 0, 0]))
        Edges.append(StructureEdge(1, 2, "Planar-Contact", {"type": 0, "idx": 5}, {"type": 0, "idx": 7}, [0, 0, 0]))
        
        Edges.append(StructureEdge(0, 2, "Planar-Contact", {"type": 0, "idx": 1}, {"type": 0, "idx": 12}, [0, 0, 0]))
        Edges.append(StructureEdge(0, 2, "Planar-Contact", {"type": 0, "idx": 1}, {"type": 0, "idx": 13}, [0, 0, 0]))
        Edges.append(StructureEdge(0, 2, "Planar-Contact", {"type": 0, "idx": 1}, {"type": 0, "idx": 14}, [0, 0, 0]))
        Edges.append(StructureEdge(0, 2, "Planar-Contact", {"type":     0, "idx": 1}, {"type": 0, "idx": 15}, [0, 0, 0]))
        

        
        self.Nodes = Nodes
        self.Edges = Edges
        
class StructureMap_ShelfPlace(StructureGraph):
    def __init__(self, sizes, positions, rotations, clip_model, preprocess=False):
        """        
        :param sizes: [B, 14]
        :param positions: [B, 12]
        :param rotations: [B, 4*6]
        Total: [B, 26+4*6=50], Node:4
        """
        if preprocess:
            sizes = self._preprocess_parameters(sizes)
        target_block = Target_Block(sizes[:, 0:3], positions[:, 0:3], rotations[:, 0:6*1])
        shelf = Shelf(sizes[:, 3:14], positions[:, 3:12], rotations[:, 6*1:6*4])
        
        Nodes = []
        Edges = []
        
        num_node = 0
        for node in target_block.Nodes:
            Nodes.append(node)
        for edge in target_block.Edges:
            edge.update_node_idx(num_node)
            Edges.append(edge)
        num_node += len(target_block.Nodes)
        
        for node in shelf.Nodes:
            Nodes.append(node)
        for edge in shelf.Edges:
            edge.update_node_idx(num_node) 
            Edges.append(edge)
        num_node += len(shelf.Nodes)
        
        # print(len(Edges))
        # for edge in Edges:
        #     print(edge.Node_idx[0], edge.Node_idx[1])
        
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
