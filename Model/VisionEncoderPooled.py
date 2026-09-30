import open_clip
import torch
import torch.nn as nn

class VisionEncoder(nn.Module):
    """
    OpenCLIP Vision Encoder.

    This module is responsible for:
        - Loading the OpenCLIP visual encoder.
        - Freezing the visual backbone (optional).
        - Extracting image patch embeddings (excluding [CLS] token).
    """

    def __init__(self, args):
        super().__init__()

        self.args = args

        clip, _, self.preprocess = open_clip.create_model_and_transforms(args.encoder_model)
        self.visual_encoder = clip.visual

        for p in self.visual_encoder.parameters():
            p.requires_grad = False


    def forward(self, pixel_values: torch.Tensor,) -> torch.Tensor:
        with torch.no_grad():
            features = self.visual_encoder.forward_intermediates(pixel_values)
            image_features=features["image_intermediates"][11]
            image_features = image_features.reshape(image_features.shape[0], image_features.shape[1], -1)

        return image_features