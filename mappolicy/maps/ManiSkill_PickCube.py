import torch
import torch.nn as nn
import torch.nn.functional as F
from Structure_Primitive import Cuboid
from base_template import StructureGraph


class Cube:
    def __init__(self, sizes, positions, rotations):
        semantic1 = 'cube'
        self.Object_Prompt = 'cube'

        Nodes = []
        Edges = []

        size1 = sizes[:, 0:3]
        position1 = positions[:, 0:3]
        rotation1 = rotations[:, 0:1*6]

        Nodes.append(
            Cuboid(
                size1[:, 0],
                size1[:, 1],
                size1[:, 2],
                position=position1,
                rotation=rotation1,
                Semantic=semantic1,
            )
        )

        self.Nodes = Nodes
        self.Edges = Edges


class StructureMap_PickCube(StructureGraph):
    def __init__(self, sizes, positions, rotations, clip_model, preprocess=False):
        """
        :param sizes: [B, 3]
        :param positions: [B, 3]
        :param rotations: [B, 6]
        Total: [B, 12], Node:1
        """
        if preprocess:
            sizes = self._preprocess_parameters(sizes)

        Objects = []
        Objects.append(Cube(sizes[:, 0:3], positions[:, 0:3], rotations[:, 0:1*6]))

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
        size_range = (0.02, 5)
        min_s, max_s = size_range
        sizes = torch.sigmoid(sizes) * (max_s - min_s) + min_s
        return sizes