#!/usr/bin/env python3
"""
TraitGen evaluation / inference script.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python evaluate.py --checkpoint checkpoints/epoch_15
    CUDA_VISIBLE_DEVICES=0 python evaluate.py --checkpoint checkpoints/epoch_15 --num_samples 20

Loads BioCLIP, Gemma 3 base, LoRA adapter, and the trained projector from a
checkpoint directory, then generates descriptions for test images.
"""

import argparse
import json
import os
import random
import sys

import torch

import config as cfg
from dataset import CUBDataset
from model import TraitGenModel


def main():
    parser = argparse.ArgumentParser(description="TraitGen evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory (e.g. checkpoints/epoch_15)")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of test images to generate descriptions for")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Override max_new_tokens for generation")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: <checkpoint>/eval_generations.json)")
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint}")

    # ---- Device ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # ---- Build model ----
    print("Building model...")
    model = TraitGenModel(device=device)

    # ---- Load checkpoint (LoRA + projector) ----
    print(f"Loading checkpoint: {args.checkpoint}")
    model.load_checkpoint(args.checkpoint)

    # ---- Dataset ----
    test_dataset = CUBDataset(cfg.TEST_JSON, cfg.IMAGE_ROOT)
    print(f"Test samples: {len(test_dataset)}")

    # ---- Generate ----
    model.gemma.eval()
    model.projector.eval()

    max_new = args.max_new_tokens or cfg.MAX_NEW_TOKENS
    num = min(args.num_samples, len(test_dataset))

    rng = random.Random(cfg.SEED)
    indices = rng.sample(range(len(test_dataset)), num)

    results = []
    print(f"\nGenerating descriptions for {num} test images (max_new_tokens={max_new})...\n")

    for i, idx in enumerate(indices):
        sample = test_dataset[idx]

        with torch.no_grad():
            generated = model.generate(
                [sample["image_path"]], max_new_tokens=max_new
            )[0].strip()

        record = {
            "imagePath": sample["imagePath"],
            "ground_truth": sample["gt"].strip(),
            "generated_text": generated,
        }
        results.append(record)

        print(f"[{i+1}/{num}] {sample['imagePath']}")
        print(f"  GT:  {record['ground_truth'][:150]}...")
        print(f"  Gen: {record['generated_text'][:150]}...")
        print()

    # ---- Save ----
    output_path = args.output or os.path.join(args.checkpoint, "eval_generations.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {output_path}")


if __name__ == "__main__":
    main()
