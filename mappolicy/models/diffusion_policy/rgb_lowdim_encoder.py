import torch
import torch.nn as nn

import os
import sys

from mappolicy.models.Clip.clip_encoder import CLIPEncoder

class MultiModalEncoder(nn.Module):
    def __init__(self, shape_meta, feature_dim=256):
        super().__init__()
        # 1. RGB Encoder (使用 CLIP)
        self.rgb_encoder = CLIPEncoder(model_name="ViT-B/32", freeze=True)
        self.rgb_proj = nn.Linear(self.rgb_encoder.feature_dim, feature_dim // 2)

        # 2. Low-Dim Encoder (MLP)
        low_dim_shape = shape_meta['obs']['robot_state']['shape'][0]
        self.low_dim_encoder = nn.Sequential(
            nn.Linear(low_dim_shape, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim // 2)
        )

        self.output_dim = feature_dim

    def forward(self, obs_dict):
        img_features = self.rgb_encoder(obs_dict['image'], type="image")
        img_features = img_features.to(self.rgb_proj.weight.dtype)
        img_out = self.rgb_proj(img_features)
        
        low_out = self.low_dim_encoder(obs_dict['robot_state'])
        
        return torch.cat([img_out, low_out], dim=-1)

    def output_shape(self):
        return (self.output_dim,)
    

if __name__ == "__main__":
    
    device = "cuda:0"
    
    # 1. Load config and get shape_meta
    from omegaconf import OmegaConf
    project_root = os.getenv("MAPPOLICY_ROOT", "your_path_to_project_root")
    config_path = f"{project_root}/mappolicy/models/diffusion_policy/diffusion_unet_hybrid_image.yaml"
    config = OmegaConf.load(config_path)
    shape_meta = config['shape_meta']
    
    # 2. Create encoder and test forward pass
    encoder = MultiModalEncoder(shape_meta).to(device)
    
    # Create dummy input
    obs_dict = {
        'image': torch.rand(1, 3, 224, 224).to(device),  # Batch of 1 RGB image
        'robot_state': torch.rand(1, shape_meta['obs']['robot_state']['shape'][0]).to(device)  # Batch of 1 low-dim state
    }
    
    output = encoder(obs_dict)
    print(f"Output shape: {output.shape}")
    # Output shape: torch.Size([1, 256])