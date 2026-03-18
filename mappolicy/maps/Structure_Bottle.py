import numpy as np
from base_template import ConceptTemplate, StructureEdge
from MapPolicy.maps.structure_primitive_numpy import Cylinder, Ring
from utils import apply_transformation
from knowledge_utils import *
import trimesh

class Multilevel_Body(ConceptTemplate):
    def __init__(self, num_levels, level_1_size, level_2_size, level_3_size, level_4_size, position = [0, 0, 0], rotation = [0, 0, 0]):

        self.semantic = 'Body'
        
        Nodes = []
        Edges = []
        
        # Process rotation param
        rotation = [x / 180 * np.pi for x in rotation]
        super().__init__(position, rotation)

        # Record Parameters
        self.num_levels = num_levels
        self.level_1_size = level_1_size
        self.level_2_size = level_2_size
        self.level_3_size = level_3_size
        self.level_4_size = level_4_size

        # Instantiate component geometries
        vertices_list = []
        faces_list = []
        total_num_vertices = 0

        self.bottom_mesh = Cylinder(level_1_size[1], level_1_size[0], level_1_size[2])
        Nodes.append(Cylinder(level_1_size[1], level_1_size[0], level_1_size[2], Semantic=self.semantic))
        vertices_list.append(self.bottom_mesh.vertices)
        faces_list.append(self.bottom_mesh.faces + total_num_vertices)
        total_num_vertices += len(self.bottom_mesh.vertices)

        delta_height = level_1_size[1] / 2
        for i in range(num_levels[0] - 1):
            delta_height += locals()['level_'+ str(i+2) +'_size'][1] / 2
            mesh_position = [0, delta_height, 0]
            delta_height += locals()['level_'+ str(i+2) +'_size'][1] / 2
            self.mesh = Cylinder(locals()['level_'+ str(i+2) +'_size'][1], locals()['level_'+ str(i+2) +'_size'][0], locals()['level_'+ str(i+1) +'_size'][0], 
                                 position = mesh_position)
            Nodes.append(Cylinder(locals()['level_'+ str(i+2) +'_size'][1], locals()['level_'+ str(i+2) +'_size'][0], locals()['level_'+ str(i+1) +'_size'][0], 
                                 position = mesh_position, Semantic=self.semantic))
            R_Anchor1 = {"type": 0, "idx": 0}
            R_Anchor2 = {"type": 0, "idx": 1}
            P_Geometry = [0, 0, 0]
            Edges.append(StructureEdge(i, i+1, "Fixed", R_Anchor1, R_Anchor2, P_Geometry))
            vertices_list.append(self.mesh.vertices)
            faces_list.append(self.mesh.faces + total_num_vertices)
            total_num_vertices += len(self.mesh.vertices)

        self.vertices = np.concatenate(vertices_list)
        self.faces = np.concatenate(faces_list)

        # Global Transformation
        self.vertices = apply_transformation(self.vertices, position, rotation)

        self.overall_obj_mesh = trimesh.Trimesh(self.vertices, self.faces)
        self.overall_obj_pts = np.array(self.overall_obj_mesh.sample(SAMPLENUM))
        
        self.Nodes = Nodes
        self.Edges = Edges


class Cylindrical_Lid(ConceptTemplate):
    def __init__(self, outer_size, inner_size, position = [0, 0, 0], rotation = [0, 0, 0]):
        
        self.semantic = 'Lid'

        # Process rotation param
        rotation = [x / 180 * np.pi for x in rotation]
        super().__init__(position, rotation)
        
        Nodes = []
        Edges = []

        # Record Parameters
        self.outer_size = outer_size
        self.inner_size = inner_size

        # Instantiate component geometries
        vertices_list = []
        faces_list = []
        total_num_vertices = 0

        middle_radius = outer_size[1] * (1 - inner_size[2] / outer_size[2]) + outer_size[0] * inner_size[2] / outer_size[2]
        top_height = outer_size[2] - inner_size[2]

        bottom_mesh_position = [0, -(outer_size[2] - inner_size[2]) / 2, 0]
        self.bottom_mesh = Ring(inner_size[2], middle_radius, inner_size[0], 
                                outer_bottom_radius = outer_size[1],
                                inner_bottom_radius = inner_size[1],
                                position=bottom_mesh_position)
        Nodes.append(Ring(inner_size[2], middle_radius, inner_size[0], 
                                outer_bottom_radius = outer_size[1],
                                inner_bottom_radius = inner_size[1],
                                position=bottom_mesh_position), Semantic=self.semantic)
        vertices_list.append(self.bottom_mesh.vertices)
        faces_list.append(self.bottom_mesh.faces + total_num_vertices)
        total_num_vertices += len(self.bottom_mesh.vertices)

        top_mesh_position = [0, inner_size[2] / 2, 0]
        self.top_mesh = Cylinder(top_height, outer_size[0], middle_radius,
                                 position=top_mesh_position)
        Nodes.append(Cylinder(top_height, outer_size[0], middle_radius,
                                 position=top_mesh_position), Semantic=self.semantic)
        R_Anchor1 = {"type": 0, "idx": 1}
        R_Anchor2 = {"type": 0, "idx": 0}
        P_Geometry = [0, 0, 0]
        Edges.append(StructureEdge(0, 1, "Fixed", R_Anchor1, R_Anchor2, P_Geometry))
        vertices_list.append(self.top_mesh.vertices)
        faces_list.append(self.top_mesh.faces + total_num_vertices)
        total_num_vertices += len(self.top_mesh.vertices)

        self.vertices = np.concatenate(vertices_list)
        self.faces = np.concatenate(faces_list)

        # Global Transformation
        self.vertices = apply_transformation(self.vertices, position, rotation)

        self.overall_obj_mesh = trimesh.Trimesh(self.vertices, self.faces)
        self.overall_obj_pts = np.array(self.overall_obj_mesh.sample(SAMPLENUM))
        
        self.Nodes = Nodes
        self.Edges = Edges
        
class SructureMap_Bottle:
    def __init__ (self, Param_Body, Param_Lid):
        multilevel_body = Multilevel_Body(Param_Body)
        cylinder_lid = Cylindrical_Lid(Param_Lid)
        
        total_num_nodes = 0
        Nodes = []
        Edges = []
        
        # Extend subgraph multilevel_body
        Nodes.extend(multilevel_body.Nodes)
        Edges.extend(multilevel_body.Edges)
        total_num_nodes += len(multilevel_body.Nodes)
        
        # Extend subgraph cylinder_lid
        Nodes.extend(cylinder_lid)
        for edge in cylinder_lid.Edges:
            for idx in edge.Node:
                idx += total_num_nodes
        Edges.extend(cylinder_lid)
        # add edges between multilevel_body and cylinder_lid
        # cylinder_lid-top and multilevel_body-top_body
        R_Anchor1_1 = {"type": 1, "idx": 0}
        R_Anchor1_2 = {"type": 1, "idx": 0}
        P_1 = [0, 0, 0]
        Edges.append(StructureEdge(total_num_nodes-1, total_num_nodes, "Cylindrical", R_Anchor1_1, R_Anchor1_2, P_1))
        R_Anchor2_1 = {"type": 1, "idx": 0}
        R_Anchor2_2 = {"type": 1, "idx": 0}
        P_2 = [0, 0, 0]
        Edges.append(StructureEdge(total_num_nodes-1, total_num_nodes+1, "Cylindrical", R_Anchor2_1, R_Anchor2_2, P_2))
        
        self.Nodes = Nodes
        self.Edges = Edges