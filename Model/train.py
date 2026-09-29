#!/usr/bin/env python3
"""
TraitGen training script.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python train.py
    CUDA_VISIBLE_DEVICES=0 python train.py --resume checkpoints/epoch_07

All hyper-parameters are read from config.py (which reads environment variables).
See config.py for the full list of settings.

IMPORTANT — test-set caveat
----------------------------
Because the test loss is monitored after every epoch and a random test image is
used for qualitative generation, the test set is no longer a completely untouched
hold-out.  If a truly unseen evaluation is needed, reserve a separate set.
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch
from torch.utils.data import DataLoader

import config as cfg
from dataset import CUBDataset, collate_fn
from model import TraitGenModel
from utils import (
    print_system_info,
    safety_check_trainable_params,
    save_loss_history,
    load_loss_history,
    save_epoch_generation,
    save_final_generations,
    format_time,
)


# ---------------------------------------------------------------------------
# Test-loss evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_test_loss(model, test_loader):
    """
    Compute average causal-LM loss over the entire test set.

    Model is set to eval mode, then restored to train mode.
    No gradients are computed.
    """
    model.gemma.eval()
    model.projector.eval()

    total_loss = 0.0
    total_tokens = 0

    for batch in test_loader:
        loss, n_tokens = model.forward_loss_components(
            batch["image_paths"], batch["captions"]
        )
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    model.gemma.train()
    model.projector.train()

    if total_tokens == 0:
        raise RuntimeError("Test set produced 0 tokens — check data.")
    return total_loss / total_tokens


# ---------------------------------------------------------------------------
# Random test-image generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_random_test_sample(model, test_dataset, rng):
    """Pick a random test image, generate a description, return the record."""
    idx = rng.randint(0, len(test_dataset) - 1)
    sample = test_dataset[idx]

    model.gemma.eval()
    model.projector.eval()

    generated = model.generate([sample["image_path"]])[0].strip()

    model.gemma.train()
    model.projector.train()

    return {
        "imagePath": sample["imagePath"],
        "ground_truth": sample["gt"].strip(),
        "generated": generated,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TraitGen training")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint directory to resume from "
                             "(e.g. checkpoints/epoch_07)")
    args = parser.parse_args()

    # ---- Reproducibility ----
    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)

    # ---- Device ----
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, running on CPU (this will be very slow)")
        device = "cpu"
    else:
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # ---- Build model ----
    model = TraitGenModel(device=device)

    # ---- Print system info & safety check ----
    params = print_system_info(model, cfg)
    safety_check_trainable_params(params, cfg.MAX_TRAINABLE_PARAMS)

    # ---- Dataset ----
    train_dataset = CUBDataset(cfg.TRAIN_JSON, cfg.IMAGE_ROOT)
    test_dataset = CUBDataset(cfg.TEST_JSON, cfg.IMAGE_ROOT)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=0,   # image loading is done inside the model (BioCLIP preprocess)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=0,
    )

    # ---- Optimizer ----
    trainable_params = list(model.trainable_parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    # ---- Scheduler (linear warmup + cosine decay) ----
    steps_per_epoch = math.ceil(len(train_loader) / cfg.GRADIENT_ACCUMULATION)
    total_steps = cfg.EPOCHS * steps_per_epoch
    warmup_steps = max(1, int(cfg.WARMUP_RATIO * total_steps))

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- AMP scaler ----
    scaler = None
    if cfg.USE_AMP and device == "cuda":
        scaler = torch.amp.GradScaler("cuda")
        print("AMP (FP16) enabled")
    else:
        print(f"AMP disabled (USE_AMP={cfg.USE_AMP})")

    # ---- Output directory ----
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # ---- Resume ----
    start_epoch = 0
    global_step = 0
    loss_history = {"epochs": [], "train_loss": [], "test_loss": []}

    if args.resume:
        if not os.path.isdir(args.resume):
            raise FileNotFoundError(f"Resume directory not found: {args.resume}")
        print(f"\nResuming from: {args.resume}")
        state = model.load_checkpoint(
            args.resume, optimizer=optimizer, scheduler=scheduler, scaler=scaler
        )
        start_epoch = state["epoch"]
        global_step = state["global_step"]
        if state.get("loss_history"):
            loss_history = state["loss_history"]
        print(f"  Resuming training from epoch {start_epoch + 1}\n")

    # ---- RNG for random test samples (changes each epoch) ----
    sample_rng = random.Random(cfg.SEED + 1000)

    # ---- Training loop ----
    print(f"\nStarting training: {cfg.EPOCHS} epochs, "
          f"{len(train_loader)} batches/epoch, "
          f"grad_accum={cfg.GRADIENT_ACCUMULATION}\n")

    model.vision_encoder.eval()   # always frozen + eval
    model.gemma.train()
    model.projector.train()

    for epoch in range(start_epoch, cfg.EPOCHS):
        epoch_num = epoch + 1
        epoch_start = time.time()

        running_loss = 0.0
        running_tokens = 0
        optimizer.zero_grad(set_to_none=True)
        step_in_epoch = 0

        for batch_idx, batch in enumerate(train_loader):
            # Forward
            if scaler is not None:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    loss = model(batch["image_paths"], batch["captions"])
                loss = loss / cfg.GRADIENT_ACCUMULATION
                scaler.scale(loss).backward()
            else:
                loss = model(batch["image_paths"], batch["captions"])
                loss = loss / cfg.GRADIENT_ACCUMULATION
                loss.backward()

            running_loss += loss.item() * cfg.GRADIENT_ACCUMULATION
            running_tokens += 1

            # Check for NaN
            if not math.isfinite(loss.item()):
                raise RuntimeError(
                    f"NaN/Inf loss at epoch {epoch_num}, batch {batch_idx}. "
                    "Check data, learning rate, or precision settings."
                )

            # Optimizer step every GRADIENT_ACCUMULATION batches
            if (batch_idx + 1) % cfg.GRADIENT_ACCUMULATION == 0 or (batch_idx + 1) == len(train_loader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, cfg.GRAD_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, cfg.GRAD_CLIP_NORM)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                step_in_epoch += 1

                # Check gradient norm
                if not torch.isfinite(grad_norm):
                    raise RuntimeError(
                        f"Non-finite gradient norm at epoch {epoch_num}, step {global_step}."
                    )

                # Logging
                if global_step % cfg.LOG_EVERY_STEPS == 0:
                    avg_loss = running_loss / running_tokens
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - epoch_start
                    mem = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0
                    print(
                        f"  Epoch {epoch_num}/{cfg.EPOCHS} | "
                        f"Step {step_in_epoch} | "
                        f"Global {global_step}/{total_steps} | "
                        f"Loss {avg_loss:.4f} | "
                        f"LR {lr:.2e} | "
                        f"Grad {grad_norm:.3f} | "
                        f"Mem {mem:.1f}G | "
                        f"{format_time(elapsed)}",
                        flush=True,
                    )

        train_loss = running_loss / max(running_tokens, 1)
        epoch_time = time.time() - epoch_start

        # ---- Test loss ----
        print(f"\n  Computing test loss ...", flush=True)
        test_loss = evaluate_test_loss(model, test_loader)

        # ---- Record ----
        loss_history["epochs"].append(epoch_num)
        loss_history["train_loss"].append(train_loss)
        loss_history["test_loss"].append(test_loss)

        # ---- Epoch summary ----
        print(f"\n{'=' * 50}")
        print(f"  Epoch {epoch_num}/{cfg.EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test Loss:  {test_loss:.4f}")
        print(f"  Time:       {format_time(epoch_time)}")
        print(f"{'=' * 50}")

        # ---- Save checkpoint ----
        ckpt_dir = os.path.join(cfg.OUTPUT_DIR, f"epoch_{epoch_num:02d}")
        model.save_checkpoint(
            ckpt_dir, epoch_num, global_step,
            optimizer, scheduler, scaler, loss_history,
        )

        # ---- Update loss plot ----
        csv_path, json_path, png_path = save_loss_history(loss_history, cfg.OUTPUT_DIR)
        print(f"  Loss curves: {png_path}")

        # ---- Random test generation ----
        print(f"\n  Generating random test sample ...", flush=True)
        gen = generate_random_test_sample(model, test_dataset, sample_rng)
        save_epoch_generation(cfg.OUTPUT_DIR, epoch_num, gen["imagePath"],
                              gen["ground_truth"], gen["generated"])

        print(f"\n{'=' * 40}")
        print(f"  Epoch {epoch_num}")
        print(f"{'=' * 40}")
        print(f"\n  Random test image:")
        print(f"  {gen['imagePath']}")
        print(f"\n  Ground Truth:")
        print(f"  {gen['ground_truth']}")
        print(f"\n  Generated:")
        print(f"  {gen['generated']}")
        print(f"{'=' * 40}\n")

        # Ensure train mode for next epoch
        model.gemma.train()
        model.projector.train()

    # ---- Final generation (10 random test images) ----
    print(f"\nFinal generation: {cfg.FINAL_GENERATION_SAMPLES} random test images ...\n")
    final_rng = random.Random(cfg.SEED + 9999)
    final_generations = []

    model.gemma.eval()
    model.projector.eval()

    for i in range(cfg.FINAL_GENERATION_SAMPLES):
        idx = final_rng.randint(0, len(test_dataset) - 1)
        sample = test_dataset[idx]

        with torch.no_grad():
            generated = model.generate([sample["image_path"]])[0].strip()

        record = {
            "imagePath": sample["imagePath"],
            "ground_truth": sample["gt"].strip(),
            "generated_text": generated,
        }
        final_generations.append(record)

        print(f"  [{i+1}/{cfg.FINAL_GENERATION_SAMPLES}] {sample['imagePath']}")
        print(f"    GT:  {record['ground_truth'][:120]}...")
        print(f"    Gen: {record['generated_text'][:120]}...")
        print()

    save_final_generations(cfg.OUTPUT_DIR, final_generations)

    # ---- Done ----
    print(f"\n{'=' * 60}")
    print(f"  Training complete!")
    print(f"  Epochs:         {cfg.EPOCHS}")
    print(f"  Final train:    {loss_history['train_loss'][-1]:.4f}")
    print(f"  Final test:     {loss_history['test_loss'][-1]:.4f}")
    print(f"  Checkpoints:    {cfg.OUTPUT_DIR}")
    print(f"  Loss plot:      {os.path.join(cfg.OUTPUT_DIR, 'train_vs_test_loss.png')}")
    print(f"  Generations:    {os.path.join(cfg.OUTPUT_DIR, 'epoch_generations.json')}")
    print(f"  Final gens:     {os.path.join(cfg.OUTPUT_DIR, 'final_generations.json')}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
