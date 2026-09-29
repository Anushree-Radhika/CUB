"""
CUB-200-2011 dataset for TraitGen.

Each sample provides an image path and a ground-truth attribute description.
The JSON files (train.json / test.json) contain records of the form:
    {"imagePath": "<class_folder>/<filename>.jpg", "gt": "<description>"}
"""

import json
import os

import torch
from torch.utils.data import Dataset
from PIL import Image


class CUBDataset(Dataset):
    """
    Reads a JSON annotation file and yields (image_path, ground_truth) pairs.

    Parameters
    ----------
    json_path : str
        Path to the JSON file (a list of dicts with 'imagePath' and 'gt').
    image_root : str
        Root directory containing the species sub-folders with images.
    """

    def __init__(self, json_path: str, image_root: str):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.image_root = image_root

        # Sanity-check: make sure the expected keys exist
        if len(self.data) == 0:
            raise ValueError(f"JSON file is empty: {json_path}")
        sample = self.data[0]
        if "imagePath" not in sample or "gt" not in sample:
            raise ValueError(
                f"Expected keys 'imagePath' and 'gt' in {json_path}, "
                f"but found: {list(sample.keys())}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_root, item["imagePath"])
        return {
            "image_path": image_path,
            "imagePath": item["imagePath"],  # relative path, for logging
            "gt": item["gt"],
        }


def collate_fn(batch):
    """Custom collate: keep strings as lists, don't try to tensor-ify them."""
    return {
        "image_paths": [b["image_path"] for b in batch],
        "relative_paths": [b["imagePath"] for b in batch],
        "captions": [b["gt"] for b in batch],
    }
