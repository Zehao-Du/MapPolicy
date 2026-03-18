from typing import List

import clip
import numpy as np
import torch
import torch.nn as nn

from mappolicy.helper.graphics import PointCloud
from mappolicy.models.Clip.clip_encoder import CLIPEncoder
from mappolicy.models.mlp.batchnorm_mlp import BatchNormMLP
from mappolicy.models.mlp.mlp import MLP


class RGB_CLIP_MLP_Policy(nn.Module):
    def __init__(
        self,
        robot_state_dim: int = 8,
        robot_state_encoder_dim: int = 128,
        action_dim: int = 7,
        mlp_hidden_dims: List[int] = [1024, 512, 256],
        mlp_activation: str = "relu",
        mlp_use_batchnorm: bool = False,
        mlp_dropout: float = 0.1,
    ):
        super(RGB_CLIP_MLP_Policy, self).__init__()

        # Load CLIP model
        self.clip_model = CLIPEncoder("ViT-B/32", freeze=True)
        clip_feature_dim = self.clip_model.feature_dim
        
        self.robot_state_encoder = nn.Linear(in_features=robot_state_dim, out_features=robot_state_encoder_dim)

        # Create MLP for policy output
        if mlp_use_batchnorm:
            self.mlp = BatchNormMLP(
                input_dim=clip_feature_dim + robot_state_encoder_dim,
                hidden_dims=mlp_hidden_dims,
                output_dim=action_dim,
                nonlinearity=mlp_activation,
                dropout=mlp_dropout,
            )
        else:
            self.mlp = MLP(
                input_dim=clip_feature_dim + robot_state_encoder_dim,
                hidden_dims=mlp_hidden_dims,
                output_dim=action_dim,
                init_method="orthogonal",
            )

    def forward(self, image, depth, pcd, pcd_no_robot, robot_state, text) -> torch.Tensor:
        B, C, H, W = image.shape
        assert C == 3, "Input images must have 3 channels (RGB)"
        # Extract CLIP features
        clip_features = self.clip_model(image, type="image")  # Shape: (B, clip_feature_dim)
        # Encode robot state
        robot_state_features = self.robot_state_encoder(robot_state)  # Shape: (B, robot_state_encoder_dim)
        # Concatenate features
        combined_features = torch.cat([clip_features, robot_state_features], dim=-1)  # Shape: (B, clip_feature_dim + robot_state_encoder_dim)
        action = self.mlp(combined_features)  # Shape: (B, action_dim)
        return action
   
class RGBD_Map_CLIP_PointNet_MLP_Policy(nn.Module):
    def __init__(self,
            map_constructor,
            map_encoder,
            loss_map_construction=None,
            fusion_dim: int = 256,
            robot_state_dim: int = 8,
            robot_state_encoder_dim: int = 128,
            action_dim: int = 7,
            mlp_hidden_dims: List[int] = [1024, 512, 256],
            mlp_activation: str = "relu",
            mlp_use_batchnorm: bool = False,
            mlp_dropout: float = 0.1
        ):
        super(RGBD_Map_CLIP_PointNet_MLP_Policy, self).__init__()
        
        # RGB
        self.rgb_encoder = CLIPEncoder("ViT-B/32", freeze=True)
        
        # map
        self.map_constructor = map_constructor
        self.map_encoder = map_encoder
        self.loss_map_construction = loss_map_construction
        
        # robot state
        self.robot_state_encoder = nn.Linear(in_features=robot_state_dim, out_features=robot_state_encoder_dim)
        
        # fusion
        self.fusion_layer = nn.Linear(in_features=self.map_encoder.feature_dim + self.rgb_encoder.feature_dim, out_features=fusion_dim)
        
        # policy head
        if mlp_use_batchnorm:
            self.policy_head = BatchNormMLP(
                input_dim=fusion_dim + robot_state_encoder_dim,
                hidden_dims=mlp_hidden_dims,
                output_dim=action_dim,
                nonlinearity=mlp_activation,
                dropout=mlp_dropout,
            )
        else:   
            self.policy_head = MLP(
                input_dim=fusion_dim + robot_state_encoder_dim,
                hidden_dims=mlp_hidden_dims,
                output_dim=action_dim,
                init_method="orthogonal",
            )
    def forward(self, image, depth, pcd, pcd_no_robot, robot_state, text) -> torch.Tensor:
        B, C, H, W = image.shape
        assert C == 3, "Input images must have 3 channels (RGB)"
        
        # RGB feature extraction
        rgb_features = self.rgb_encoder(image, type="image")  # Shape: (B, rgb_feature_dim)
        
        # Map construction and encoding
        sence_map = self.map_constructor(pcd)  # Construct maps from RGB-D and point cloud data
        map_features, math_loss = self.map_encoder(sence_map.data)  # Encode maps to get map features
        
        # Feature fusion
        fused_features = self.fusion_layer(torch.cat([rgb_features, map_features], dim=-1))  # Shape: (B, fusion_dim)
        
        # Robot state encoding
        robot_state_features = self.robot_state_encoder(robot_state)  # Shape: (B, robot_state_encoder_dim)
        
        # Concatenate fused features with robot state features
        combined_features = torch.cat([fused_features, robot_state_features], dim=-1)  # Shape: (B, fusion_dim + robot_state_encoder_dim)
        
        # Policy head to get action output
        action = self.policy_head(combined_features)  # Shape: (B, action_dim)
        
        if self.training:
            # Map Loss
            point_cloud_map = sence_map.complete_point_cloud()
            loss_map = self.loss_map_construction(point_cloud_map, pcd_no_robot)
            # Math Loss
            loss_math = math_loss['math_loss'] + math_loss['ortho_loss']
            return action, loss_map, loss_math
        
        return action
        
    
if __name__ == "__main__":
    import pathlib

    from hydra.utils import instantiate
    from mappolicy.helper.logger import Logger
    from mappolicy.models.pointnet.model_loader import PointnetEnc
    from omegaconf import OmegaConf

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # RGB_CLIP_MLP_Policy test
    policy = RGB_CLIP_MLP_Policy(
        robot_state_dim=8,
        robot_state_encoder_dim=128,
        action_dim=7,
        mlp_hidden_dims=[1024, 512, 256],
        mlp_activation="relu",
        mlp_use_batchnorm=False,
        mlp_dropout=0.1,
    ).to(device)
    
    image = torch.randn(4, 3, 224, 224).to(device)  # Batch of 4 RGB images
    depth = torch.randn(4, 1, 224, 224).to(device)  # Batch of 4 depth images
    pcd = torch.randn(4, 1024, 3).to(device)  # Batch of 4 point clouds with 1024 points each
    pcd_no_robot = torch.randn(4, 1024, 3).to(device)  # Batch of 4 point clouds without robot points
    robot_state = torch.randn(4, 8).to(device)  # Batch of 4 robot states
    text = ["Example instruction"] * 4  # Batch of 4 text instructions
    
    action = policy(image, depth, pcd, pcd_no_robot, robot_state, text)
    Logger.log_info(f"input image shape: {image.shape}, depth shape: {depth.shape}, pcd shape: {pcd.shape}, pcd_no_robot shape: {pcd_no_robot.shape}, robot_state shape: {robot_state.shape}")
    Logger.log_info(f"Action shape: {action.shape}")

    # RGBD_Map_CLIP_PointNet_MLP_Policy test (load from config)
    config_path = pathlib.Path(__file__).resolve().parents[1] / "config" / "agent" / "mappolicy_rgbd" / "CLIP_PointNet_MLP.yaml"
    cfg = OmegaConf.load(config_path)

    # fill interpolation values used in config
    cfg.task_name = "close_box"
    cfg.device = device
    cfg.camera_name = "front"

    # compatibility fix for legacy target path in config
    cfg.instantiate_config.map_constructor._target_ = str(cfg.instantiate_config.map_constructor._target_).replace(
        "mappolicy.models.map_models", "mappolicy.models.mappolicy"
    )
    cfg.instantiate_config.map_encoder._target_ = str(cfg.instantiate_config.map_encoder._target_).replace(
        "mappolicy.models.map_models", "mappolicy.models.mappolicy"
    )

    point_cloud_encoder = PointnetEnc().to(device)
    map_constructor_factory = instantiate(cfg.instantiate_config.map_constructor, point_cloud_encoder=point_cloud_encoder)
    map_constructor = map_constructor_factory()
    map_encoder = instantiate(cfg.instantiate_config.map_encoder)
    loss_map_construction = instantiate(cfg.instantiate_config.loss_map_construction)

    rgbd_policy = instantiate(
        cfg.instantiate_config,
        map_constructor=map_constructor,
        map_encoder=map_encoder,
        loss_map_construction=loss_map_construction,
        robot_state_dim=8,
        action_dim=7,
    ).to(device)
    rgbd_policy.eval()

    image2 = torch.randn(2, 3, 224, 224).to(device)
    depth2 = torch.randn(2, 1, 224, 224).to(device)
    pcd2 = torch.randn(2, 1024, 6).to(device)
    pcd_no_robot2 = torch.randn(2, 1024, 3).to(device)
    robot_state2 = torch.randn(2, 8).to(device)
    text2 = ["Example instruction"] * 2

    with torch.no_grad():
        action2 = rgbd_policy(image2, depth2, pcd2, pcd_no_robot2, robot_state2, text2)

    Logger.log_info(f"[RGBD config] config path: {config_path}")
    Logger.log_info(
        f"[RGBD config] input image shape: {image2.shape}, depth shape: {depth2.shape}, "
        f"pcd shape: {pcd2.shape}, robot_state shape: {robot_state2.shape}"
    )
    Logger.log_info(f"[RGBD config] Action shape: {action2.shape}")