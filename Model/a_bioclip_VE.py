import torch
import open_clip
from PIL import Image


class BioCLIP(torch.nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()

        self.device = device

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "hf-hub:imageomics/bioclip"
        )

        self.model = self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def forward(self, image_paths):
        images = [
            self.preprocess(Image.open(path).convert("RGB"))
            for path in image_paths
        ]

        images = torch.stack(images).to(self.device)

        visual = self.model.visual

        # Convert image into patch embeddings
        x = visual.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        # Add CLS token
        cls = visual.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(
            x.shape[0], 1, x.shape[-1],
            dtype=x.dtype,
            device=x.device
        )

        x = torch.cat([cls, x], dim=1)

        # Add positional embeddings
        x = x + visual.positional_embedding.to(x.dtype)

        # Transformer
        x = visual.patch_dropout(x)
        x = visual.ln_pre(x)
        x = visual.transformer(x)
        x = visual.ln_post(x)

        # Remove CLS
        x = x[:, 1:, :]

        return x
