import os
import pathlib
import sys

import clip
import torch
import torch.nn as nn
from torchvision.transforms import Normalize


class CLIPEncoder(nn.Module):
    '''
    Input: 
        - image: Tensor of shape [B, 3, 224, 224]
        - text: List of strings, tokenized by clip.tokenize
    Output:
        - features: Tensor of shape [B, self.feature_dim]
    '''
    def __init__(self, model_name, freeze=True):
        super(CLIPEncoder, self).__init__()
        self.model, self.preprocess = clip.load(model_name)
        # see: https://github.com/openai/CLIP/blob/main/clip/clip.py line 79
        self.preprocess = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        self.feature_dim = {
            "ViT-B/32": 512,
        }[model_name]
        if freeze:
            self.freeze()

    def forward(self, x, type="image"):
        # x = self.preprocess(x)
        if type == "image":
            return self.model.encode_image(x)
        elif type == "text":
            return self.model.encode_text(x)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
            
            
if __name__ == "__main__":
    model_name = "ViT-B/32"
    device = "cuda:0"
    encoder = CLIPEncoder(model_name).to(device)
    
    # test image encoding
    input = torch.rand(size=(1, 3, 224, 224)).to(device)
    output = encoder(input, type="image")
    print(f"Output shape: {output.shape}") 
    # Output shape: torch.Size([1, 512])
    
    # test text encoding
    text_input = clip.tokenize(["Hello, CLIP!"]).to(device)
    text_output = encoder(text_input, type="text")
    print(f"Text output shape: {text_output.shape}")
    # Text output shape: torch.Size([1, 512])