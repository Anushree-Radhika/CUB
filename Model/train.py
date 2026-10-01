#!/usr/bin/env python3
"""
TraitGen training script.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python train.py
    CUDA_VISIBLE_DEVICES=0 python train.py --resume checkpoints/epoch_07

All hyper-parameters are read from config.py (which reads environment variables).

IMPORTANT - test-set caveat
---------------------------
Because the test loss is monitored after every epoch and a random test image is
used for qualitative generation, the test set is no longer a completely untouched
hold-out.  If a truly unseen evaluation is needed, reserve a separate set.
"""

import argparse
import math
import os
import random
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
    save_epoch_generation,
    save_final_generations,
    format_time,
)


def _sample_path(sample):
    """The dataset may expose the path as 'image_path' or 'imagePath'."""
    p = sample.get("image_path") or sample.get("imagePath")
    if p is None:
        raise KeyError("Dataset sample has neither 'image_path' nor 'imagePath'.")
    return p


def _set_train(model):
    model.gemma.train()
    model.projector.train()
    model.vision_encoder.eval()   # always frozen + eval


def _set_eval(model):
    model.gemma.eval()
    model.projector.eval()
    model.vision_encoder.eval()


# ---------------------------------------------------------------------------
# Loss evaluation (used for both val and test)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_loss(model, loader):
    """Token-weighted average causal-LM loss over a loader."""
    _set_eval(model)
    try:
        total_loss = 0.0
        total_tokens = 0
        for batch in loader:
            loss, n_tokens = model.forward_loss_components(
                batch["image_paths"], batch["captions"]
            )
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
    finally:
        _set_train(model)

    if total_tokens == 0:
        raise RuntimeError("Loader produced 0 tokens - check data.")
    return total_loss / total_tokens


# ---------------------------------------------------------------------------
# Random test-image generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_one(model, test_dataset, rng):
    """Pick a random test image, generate a description, return the record."""
    idx = rng.randint(0, len(test_dataset) - 1)
    sample = test_dataset[idx]
    path = _sample_path(sample)

    _set_eval(model)
    try:
        generated = model.generate([path])[0].strip()
    finally:
        _set_train(model)

    return {
        "imagePath": path,
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

    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, running on CPU (this will be very slow)")
        device = "cpu"
    else:
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # ---- Model ----
    model = TraitGenModel(device=device)
    params = print_system_info(model, cfg)
    safety_check_trainable_params(params, cfg.MAX_TRAINABLE_PARAMS)

    # ---- Data ----
    full_train_dataset = CUBDataset(cfg.TRAIN_JSON, cfg.IMAGE_ROOT)

    if getattr(cfg, "VAL_JSON", "") and os.path.exists(cfg.VAL_JSON):
        train_dataset = full_train_dataset
        val_dataset = CUBDataset(cfg.VAL_JSON, cfg.IMAGE_ROOT)
        print("Using provided VAL_JSON for validation.")
    else:
        val_size = int(len(full_train_dataset) * getattr(cfg, "VAL_SPLIT", 0.1))
        train_size = len(full_train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(cfg.SEED),
        )
        print(f"Split train dataset into {train_size} train and {val_size} val samples.")

    test_dataset = CUBDataset(cfg.TEST_JSON, cfg.IMAGE_ROOT)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    def make_loader(ds, shuffle):
        return DataLoader(
            ds, batch_size=cfg.BATCH_SIZE, shuffle=shuffle,
            collate_fn=collate_fn, drop_last=False,
            num_workers=0,   # image loading happens inside the model (BioCLIP preprocess)
        )

    train_loader = make_loader(train_dataset, True)
    val_loader = make_loader(val_dataset, False)
    test_loader = make_loader(test_dataset, False)

    # ---- Optimizer ----
    trainable_params = list(model.trainable_parameters())
    optimizer = torch.optim.AdamW(
        trainable_params, lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY,
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
        print("AMP (FP16 autocast) enabled - watch for non-finite loss with Gemma 3")
    else:
        print(f"AMP disabled (USE_AMP={cfg.USE_AMP})")

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # ---- Resume ----
    start_epoch = 0
    global_step = 0
    loss_history = {"epochs": [], "train_loss": [], "val_loss": [], "test_loss": []}

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
        for k in ("epochs", "train_loss", "val_loss", "test_loss"):
            loss_history.setdefault(k, [])
        print(f"  Resuming training from epoch {start_epoch + 1}\n")

    sample_rng = random.Random(cfg.SEED + 1000 + start_epoch)

    print(f"\nStarting training: {cfg.EPOCHS} epochs, "
          f"{len(train_loader)} batches/epoch, "
          f"grad_accum={cfg.GRADIENT_ACCUMULATION}\n")

    _set_train(model)

    for epoch in range(start_epoch, cfg.EPOCHS):
        epoch_num = epoch + 1
        epoch_start = time.time()

        running_loss = 0.0
        running_batches = 0
        optimizer.zero_grad(set_to_none=True)
        step_in_epoch = 0

        for batch_idx, batch in enumerate(train_loader):
            if scaler is not None:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    loss = model(batch["image_paths"], batch["captions"])
            else:
                loss = model(batch["image_paths"], batch["captions"])

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                raise RuntimeError(
                    f"NaN/Inf loss at epoch {epoch_num}, batch {batch_idx}. "
                    "Check data, learning rate, or precision settings."
                )

            scaled = loss / cfg.GRADIENT_ACCUMULATION
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

            running_loss += loss_value
            running_batches += 1

            is_last = (batch_idx + 1) == len(train_loader)
            if (batch_idx + 1) % cfg.GRADIENT_ACCUMULATION == 0 or is_last:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, cfg.GRAD_CLIP_NORM)

                if not torch.isfinite(grad_norm):
                    raise RuntimeError(
                        f"Non-finite gradient norm at epoch {epoch_num}, step {global_step + 1}."
                    )

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                step_in_epoch += 1

                if global_step % cfg.LOG_EVERY_STEPS == 0:
                    avg_loss = running_loss / running_batches
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - epoch_start
                    mem = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0
                    print(
                        f"  Epoch {epoch_num}/{cfg.EPOCHS} | "
                        f"Step {step_in_epoch} | "
                        f"Global {global_step}/{total_steps} | "
                        f"Loss {avg_loss:.4f} | "
                        f"LR {lr:.2e} | "
                        f"Grad {float(grad_norm):.3f} | "
                        f"Mem {mem:.1f}G | "
                        f"{format_time(elapsed)}",
                        flush=True,
                    )

        train_loss = running_loss / max(running_batches, 1)
        epoch_time = time.time() - epoch_start

        # ---- Val / test loss ----
        print("\n  Computing val loss ...", flush=True)
        val_loss = evaluate_loss(model, val_loader)
        print("  Computing test loss ...", flush=True)
        test_loss = evaluate_loss(model, test_loader)

        loss_history["epochs"].append(epoch_num)
        loss_history["train_loss"].append(train_loss)
        loss_history["val_loss"].append(val_loss)
        loss_history["test_loss"].append(test_loss)

        print(f"\n{'=' * 50}")
        print(f"  Epoch {epoch_num}/{cfg.EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Test Loss:  {test_loss:.4f}")
        print(f"  Time:       {format_time(epoch_time)}")
        print(f"{'=' * 50}")

        # ---- Checkpoint + plots ----
        ckpt_dir = os.path.join(cfg.OUTPUT_DIR, f"epoch_{epoch_num:02d}")
        model.save_checkpoint(
            ckpt_dir, epoch_num, global_step,
            optimizer, scheduler, scaler, loss_history,
        )
        _, _, png_path = save_loss_history(loss_history, cfg.OUTPUT_DIR)
        print(f"  Loss curves: {png_path}")

        # ---- Random test generation ----
        print("\n  Generating random test sample ...", flush=True)
        gen = generate_one(model, test_dataset, sample_rng)
        save_epoch_generation(cfg.OUTPUT_DIR, epoch_num, gen["imagePath"],
                              gen["ground_truth"], gen["generated"])

        print(f"\n{'=' * 40}")
        print(f"  Epoch {epoch_num}")
        print(f"{'=' * 40}")
        print(f"\n  Random test image:\n  {gen['imagePath']}")
        print(f"\n  Ground Truth:\n  {gen['ground_truth']}")
        print(f"\n  Generated:\n  {gen['generated']}")
        print(f"{'=' * 40}\n")

    # ---- Final generation ----
    print(f"\nFinal generation: {cfg.FINAL_GENERATION_SAMPLES} random test images ...\n")
    final_rng = random.Random(cfg.SEED + 9999)
    final_generations = []

    for i in range(cfg.FINAL_GENERATION_SAMPLES):
        gen = generate_one(model, test_dataset, final_rng)
        record = {
            "imagePath": gen["imagePath"],
            "ground_truth": gen["ground_truth"],
            "generated_text": gen["generated"],
        }
        final_generations.append(record)

        print(f"  [{i + 1}/{cfg.FINAL_GENERATION_SAMPLES}] {record['imagePath']}")
        print(f"    GT:  {record['ground_truth'][:120]}...")
        print(f"    Gen: {record['generated_text'][:120]}...\n")

    save_final_generations(cfg.OUTPUT_DIR, final_generations)

    print(f"\n{'=' * 60}")
    print("  Training complete!")
    print(f"  Epochs:         {cfg.EPOCHS}")
    print(f"  Final train:    {loss_history['train_loss'][-1]:.4f}")
    print(f"  Final val:      {loss_history['val_loss'][-1]:.4f}")
    print(f"  Final test:     {loss_history['test_loss'][-1]:.4f}")
    print(f"  Checkpoints:    {cfg.OUTPUT_DIR}")
    print(f"  Loss plot:      {os.path.join(cfg.OUTPUT_DIR, 'train_vs_test_loss.png')}")
    print(f"  Generations:    {os.path.join(cfg.OUTPUT_DIR, 'epoch_generations.json')}")
    print(f"  Final gens:     {os.path.join(cfg.OUTPUT_DIR, 'final_generations.json')}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()