import open_clip
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

class VisionEncoder(nn.Module):
    """
    OpenCLIP Vision Encoder.

    This module is responsible for:
        - Loading the OpenCLIP visual encoder.
        - Applying LoRA to the visual backbone.
        - Extracting image patch embeddings (excluding [CLS] token).
    """

    def __init__(self, args):
        super().__init__()

        self.args = args

        clip, _, self.preprocess = open_clip.create_model_and_transforms(args.encoder_model)
        self.visual_encoder = clip.visual

        # Freeze the base encoder completely first
        for p in self.visual_encoder.parameters():
            p.requires_grad = False

        # Apply LoRA to the visual encoder (using lightweight in_proj and out_proj)
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["in_proj", "out_proj"],
            lora_dropout=0.05,
        )
        self.visual_encoder = get_peft_model(self.visual_encoder, peft_config)

        # Ensure LoRA weights are trainable and fp32 for stability
        for p in self.visual_encoder.parameters():
            if p.requires_grad:
                p.data = p.data.float()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # No torch.no_grad() here so gradients flow through the LoRA adapters
        # We access base_model.model because PEFT wraps the original visual_encoder
        features = self.visual_encoder.base_model.model.forward_intermediates(pixel_values)
        image_features = features["image_intermediates"][11]
        image_features = image_features.reshape(image_features.shape[0], image_features.shape[1], -1)

        return image_features