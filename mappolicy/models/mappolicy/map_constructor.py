# Construct Structure Map
import torch
import torch.nn as nn
import torch.nn.functional as F

import pathlib
import sys
models_path = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(
    str(models_path / "maps")
)

# MetaWorld
from mappolicy.maps.Basketball import StructureMap_Basketball
from mappolicy.maps.BinPicking import StructureMap_BinPicking
from mappolicy.maps.BoxClose import StructureMap_BoxClose
from mappolicy.maps.CoffeePull import StructureMap_CoffeePull
from mappolicy.maps.CoffeePush import StructureMap_CoffeePush
from mappolicy.maps.Disassemble import StructureMap_Disassemble
from mappolicy.maps.Hammer import StructureMap_Hammer
from mappolicy.maps.HandInsert import StructureMap_HandInsert
from mappolicy.maps.HandlePull import StructureMap_HandlePull
from mappolicy.maps.HandlePullSide import StructureMap_HandlePullSide
from mappolicy.maps.LeverPull import StructureMap_LeverPull
from mappolicy.maps.PegInsertSide import StructureMap_PegInsertSide
from mappolicy.maps.PegUnplugSide import StructureMap_PegUnplugSide
from mappolicy.maps.PickOutOfHole import StructureMap_PickOutOfHole
from mappolicy.maps.PickPlace import StructureMap_PickPlace
from mappolicy.maps.PickPlaceWall import StructureMap_PickPlaceWall
from mappolicy.maps.Push import StructureMap_Push
from mappolicy.maps.PushBack import StructureMap_PushBack
from mappolicy.maps.PushWall import StructureMap_PushWall
from mappolicy.maps.ReachWall import StructureMap_ReachWall
from mappolicy.maps.ShelfPlace import StructureMap_ShelfPlace
from mappolicy.maps.Soccer import StructureMap_Soccer
from mappolicy.maps.StickPull import StructureMap_StickPull
from mappolicy.maps.Sweep import StructureMap_Sweep
from mappolicy.maps.SweepInto import StructureMap_SweepInto

# RLBench
from mappolicy.maps.RLBench_CloseBox import StructureMap_CloseBox
from mappolicy.maps.RLBench_LaptopLid import StructureMap_LaptopLid
from mappolicy.maps.RLBench_PutRubbishIn import StructureMap_PutRubbishIn
from mappolicy.maps.RLBench_ToiletSeatDown import StructureMap_ToiletSeatDown
from mappolicy.maps.RLBench_UnplugCharger import StructureMap_UnplugCharger
from mappolicy.maps.RLBench_WaterPlants import StructureMap_WaterPlants

# ManiSkill
from mappolicy.maps.ManiSkill_PickCube import StructureMap_PickCube
from mappolicy.maps.ManiSkill_StackCube import StructureMap_StackCube
from mappolicy.maps.ManiSkill_PlugCharger import StructureMap_PlugCharger
from mappolicy.maps.ManiSkill_StackPyramid import StructureMap_StackPyramid
from mappolicy.maps.ManiSkill_PullCubeTool import StructureMap_PullCubeTool


##### clip
from mappolicy.models.Clip.clip_encoder import CLIPEncoder
# fast_sam
from mappolicy.models.fast_sam.fastsam_loader import FastSAM_Loader

MAP_DIM_VOCAB = {
    # metaworld
    "basketball": [8, 20, 44],
    "bin-picking": [19, 34, 64],
    "box-close": [20, 38, 74],
    "coffee-pull": [11, 23, 47],
    "coffee-push": [11, 23, 47],
    "disassemble": [6, 15, 33],
    "hammer": [6, 15, 33],
    "hand-insert": [6, 12, 24],
    "handle-pull": [6, 15, 33],
    "handle-pull-side": [6, 15, 33],
    "lever-pull": [8, 20, 44],
    "peg-insert-side": [8, 17, 35],
    "peg-unplug-side": [7, 16, 34],
    "pick-out-of-hole": [9, 18, 36],
    "pick-place": [3, 6, 12],
    "pick-place-wall": [9, 18, 36],
    "push": [3, 6, 12],
    "push-back": [3, 6, 12],
    "push-wall": [9, 18, 36],
    "reach-wall": [6, 12, 24],
    "shelf-place": [14, 26, 50],
    "soccer": [13, 28, 58],
    "stick-pull": [11, 23, 47],
    "sweep": [3, 6, 12],
    "sweep-into": [9, 18, 36],
    
    # rlbench
    "close_box": [11, 20, 38],
    "close_laptop_lid": [9, 18, 36],
    "put_rubbish_in_bin": [11, 23, 47],
    "toilet_seat_down": [10, 22, 46],
    "unplug_charger": [7, 16, 34],
    "water_plants": [13, 28, 58],
    
    # Maniskill
    "PickCube-v1": [3, 6, 12],
    "PegInsertionSide-v1": [8, 17, 35],
    "StackCube-v1": [9, 18, 36],
    "PlugCharger-v1": [18, 36, 72],
    "StackPyramid-v1": [9, 18, 36],
    "PullCubeTool-v1": [7, 16, 34],
}
MAP_CLASS_VOCAB = {
    # metaworld
    "basketball": StructureMap_Basketball,
    "bin-picking": StructureMap_BinPicking,
    "box-close": StructureMap_BoxClose,
    "coffee-pull": StructureMap_CoffeePull,
    "coffee-push": StructureMap_CoffeePush,
    "disassemble": StructureMap_Disassemble,
    "hammer": StructureMap_Hammer,
    "hand-insert": StructureMap_HandInsert,
    "handle-pull": StructureMap_HandlePull,
    "handle-pull-side": StructureMap_HandlePullSide,
    "lever-pull": StructureMap_LeverPull,
    "peg-insert-side": StructureMap_PegInsertSide,
    "peg-unplug-side": StructureMap_PegUnplugSide,
    "pick-out-of-hole": StructureMap_PickOutOfHole,
    "pick-place": StructureMap_PickPlace,
    "pick-place-wall": StructureMap_PickPlaceWall,
    "push": StructureMap_Push,
    "push-back": StructureMap_PushBack,
    "push-wall": StructureMap_PushWall,
    "reach-wall": StructureMap_ReachWall,
    "shelf-place": StructureMap_ShelfPlace,
    "soccer": StructureMap_Soccer,
    "stick-pull": StructureMap_StickPull,
    "sweep": StructureMap_Sweep,
    "sweep-into": StructureMap_SweepInto,
    
    # rlbench
    "close_box": StructureMap_CloseBox,
    "close_laptop_lid": StructureMap_LaptopLid,
    "put_rubbish_in_bin": StructureMap_PutRubbishIn,
    "toilet_seat_down": StructureMap_ToiletSeatDown,
    "unplug_charger": StructureMap_UnplugCharger,
    "water_plants": StructureMap_WaterPlants,

    # Maniskill
    "PickCube-v1": StructureMap_PickCube,
    "PegInsertionSide-v1": StructureMap_PegInsertSide,
    "StackCube-v1": StructureMap_StackCube,
    "PlugCharger-v1": StructureMap_PlugCharger,
    "StackPyramid-v1": StructureMap_StackPyramid,
    "PullCubeTool-v1": StructureMap_PullCubeTool
}

class ParameterEstimator_SingleFrame(nn.Module):
    def __init__ (self,
                  point_cloud_encoder: nn.Module,
                  task_name,
                  device,
                  ):
        if task_name not in MAP_DIM_VOCAB or task_name not in MAP_CLASS_VOCAB:
            raise ValueError(f"Unknown task_name: {task_name}. Available: {list(MAP_CLASS_VOCAB.keys())}")
        self.map_name = task_name
        self.dims = MAP_DIM_VOCAB[task_name]
        self.device = device
        self.MapClass = MAP_CLASS_VOCAB[task_name]
        super(ParameterEstimator_SingleFrame, self).__init__()
        self.clip_encoder = CLIPEncoder("ViT-B/32").to(self.device)
        self.point_cloud_encoder = point_cloud_encoder
        self.estimation_head = nn.Sequential(
            nn.Linear(point_cloud_encoder.feature_dim, self.dims[2]),
        )
        
    def forward(self, point_cloud):
        features = self.point_cloud_encoder(point_cloud)
        parameters = self.estimation_head(features)
        sizes = parameters[:, 0:self.dims[0]]
        positions = parameters[:, self.dims[0]:self.dims[1]]
        rotations = parameters[:, self.dims[1]:self.dims[2]]
        scene_map = self.MapClass(sizes, positions, rotations, self.clip_encoder, preprocess=False)
        return scene_map
    
class ParameterEstimator_SingleFrame_Regularization(nn.Module):
    def __init__ (self,
                  point_cloud_encoder: nn.Module,
                  task_name: str,
                  device,
                  ):
        if task_name not in MAP_DIM_VOCAB or task_name not in MAP_CLASS_VOCAB:
            raise ValueError(f"Unknown task_name: {task_name}. Available: {list(MAP_CLASS_VOCAB.keys())}")
        self.map_name = task_name
        self.dims = MAP_DIM_VOCAB[task_name]
        self.device = device
        self.MapClass = MAP_CLASS_VOCAB[task_name]
        super().__init__()
        self.clip_encoder = CLIPEncoder("ViT-B/32").to(self.device)
        self.point_cloud_encoder = point_cloud_encoder
        self.estimation_head = nn.Sequential(
            nn.Linear(point_cloud_encoder.feature_dim, self.dims[2]),
        )
        
    def forward(self, point_cloud):
        features = self.point_cloud_encoder(point_cloud)
        parameters = self.estimation_head(features)
        sizes = parameters[:, 0:self.dims[0]]
        positions = parameters[:, self.dims[0]:self.dims[1]]
        rotations = parameters[:, self.dims[1]:self.dims[2]]
        scene_map = self.MapClass(sizes, positions, rotations, self.clip_encoder, preprocess=True)
        return scene_map
    
class ParameterEstimator_SingleFrame_Segmentation(nn.Module):
    def __init__ (self,
                  point_cloud_encoder: nn.Module,
                  task_name: str,
                  fastsam_ckpt_path: str,
                  image_size: int,
                  device,
                  ):
        if task_name not in MAP_DIM_VOCAB or task_name not in MAP_CLASS_VOCAB:
            raise ValueError(f"Unknown task_name: {task_name}. Available: {list(MAP_CLASS_VOCAB.keys())}")
        self.map_name = task_name
        self.dims = MAP_DIM_VOCAB[task_name]
        self.device = device
        self.MapClass = MAP_CLASS_VOCAB[task_name]
        super().__init__()
        self.clip_encoder = CLIPEncoder("ViT-B/32").to(self.device)
        self.segmentation_model = FastSAM_Loader(ckpt_path=fastsam_ckpt_path, device=device, imgsz=image_size)
        self.point_cloud_encoder = point_cloud_encoder
        self.estimation_head = nn.Sequential(
            nn.Linear(point_cloud_encoder.feature_dim, self.dims[2]),
        )
        
    def forward(self, point_cloud):
        mask = self.segmentation_model.segment_by_text_prompt(rgb, prompt)
        features = self.point_cloud_encoder(point_cloud)
        parameters = self.estimation_head(features)
        sizes = parameters[:, 0:self.dims[0]]
        positions = parameters[:, self.dims[0]:self.dims[1]]
        rotations = parameters[:, self.dims[1]:self.dims[2]]
        scene_map = self.MapClass(sizes, positions, rotations, self.clip_encoder, preprocess=True)
        return scene_map