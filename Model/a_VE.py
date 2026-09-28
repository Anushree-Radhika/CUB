import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    
    
    def forward(self, *args, **kwargs):
        """
           Expects a list of image paths [image_path1, imagepath2, .....]
           \n
           Outputs a BatchSize x Number of Patches x Encoder Embedding Dimension 
        """
        
        return self.model.forward(*args, **kwargs)
