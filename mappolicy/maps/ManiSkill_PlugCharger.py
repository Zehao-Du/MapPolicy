import torch
import torch.nn as nn
import torch.nn.functional as F
from Structure_Primitive import Cuboid
from base_template import StructureEdge, StructureGraph

class Charger:
    def __init__ (self, size, position, rotation):
        semantic1 = 'charger body'
        semantic2 = 'charger plug'
        Nodes = []
        Edges = []
        
        Nodes.append(Cuboid(size[:, 0], size[:, 1], size[:, 2], position=position[:, 0:3], rotation=rotation[:, 0:6], Semantic=semantic1))
        Nodes.append(Cuboid(size[:, 3], size[:, 4], size[:, 5], position=position[:, 3:6], rotation=rotation[:, 6:12], Semantic=semantic2))
        Nodes.append(Cuboid(size[:, 6], size[:, 7], size[:, 8], position=position[:, 6:9], rotation=rotation[:, 12:18], Semantic=semantic2))

        Edges.append(StructureEdge(0, 1, "Planar-Contact", {"type": 0, "idx": 0}, {"type": 0, "idx": 0}, [0, 0, 0]))
        Edges.append(StructureEdge(0, 2, "Planar-Contact", {"type": 0, "idx": 0}, {"type": 0, "idx": 0}, [0, 0, 0]))
        Edges.append(StructureEdge(1, 2, "Alignment", {"type": 1, "idx": 0}, {"type": 1, "idx": 0}, [0, 0, 0]))
        
        
        self.Nodes = Nodes
        self.Edges = Edges

class socket:
    def __init__ (self, size, position, rotation):
        semantic1 = 'socket shell'
        semantic2 = 'socket hole'
        Nodes = []
        Edges = []
        
        Nodes.append(Cuboid(size[:, 0], size[:, 1], size[:, 2], position=position[:, 0:3], rotation=rotation[:, 0:6], Semantic=semantic1))
        Nodes.append(Cuboid(size[:, 3], size[:, 4], size[:, 5], position=position[:, 3:6], rotation=rotation[:, 6:12], Semantic=semantic2))
        Nodes.append(Cuboid(size[:, 6], size[:, 7], size[:, 8], position=position[:, 6:9], rotation=rotation[:, 12:18], Semantic=semantic2))
        
        Edges.append(StructureEdge(0, 1, "Planar-Contact", {"type": 0, "idx": 0}, {"type": 0, "idx": 0}, [0, 0, 0]))
        Edges.append(StructureEdge(0, 2, "Planar-Contact", {"type": 0, "idx": 0}, {"type": 0, "idx": 0}, [0, 0, 0]))
        Edges.append(StructureEdge(1, 2, "Alignment", {"type": 1, "idx": 0}, {"type": 1, "idx": 0}, [0, 0, 0]))
        
        self.Nodes = Nodes
        self.Edges = Edges


class StructureMap_PlugCharger(StructureGraph):
    def __init__(self, sizes, positions, rotations, clip_model, preprocess=False):
        """        
        :param sizes: [B, 18]  (charger:9 + socket:9)
        :param positions: [B, 18]
        :param rotations: [B, 36]
        Total: [B, 72], Node:6
        """
        if preprocess:
            sizes = self._preprocess_parameters(sizes)

        charger_obj = Charger(sizes[:, 0:9], positions[:, 0:9], rotations[:, 0:18])
        socket_obj  = socket(sizes[:, 9:18], positions[:, 9:18], rotations[:, 18:36])

        Nodes = []
        Edges = []
        num_node = 0
        for obj in (charger_obj, socket_obj):
            for node in obj.Nodes:
                Nodes.append(node)
            for edge in obj.Edges:
                edge.update_node_idx(num_node)
                Edges.append(edge)
            num_node += len(obj.Nodes)

        super().__init__(Nodes, Edges, clip_model)

    def _preprocess_parameters(self, sizes):
        """
        Apply simple sigmoid scaling to ensure positive dimensions. Adapt as needed.
        """
        size_range = (0.001, 1.0)
        min_s, max_s = size_range
        return torch.sigmoid(sizes) * (max_s - min_s) + min_s