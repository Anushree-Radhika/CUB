import torch
import torch.nn as nn


class TextDecoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image_features, captions):
        return self.model.forward(
            image_features,
            captions
        )

    @torch.no_grad()
    def generate(self, image_features):
        return self.model.generate(
            image_features
        )
