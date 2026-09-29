"""
TraitGen utilities: logging, loss plotting, and epoch-generation recording.
"""

import csv
import json
import os
import time

import matplotlib
matplotlib.use("Agg")  # headless server: write PNGs, never open a window
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# System info banner
# ---------------------------------------------------------------------------

def print_system_info(model, cfg):
    """Print environment, dataset, model, and training configuration."""
    import torch

    params = model.count_parameters()

    print("\n" + "=" * 60)
    print("  TraitGen — BioCLIP + Gemma 3 + LoRA")
    print("=" * 60)

    print(f"\nDevice:             {model.device}")
    if torch.cuda.is_available():
        print(f"GPU:                {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"GPU memory:         {mem:.1f} GiB")
    print(f"PyTorch version:    {torch.__version__}")
    print(f"CUDA version:       {torch.version.cuda}")

    print(f"\nDataset:")
    print(f"  Image root:       {cfg.IMAGE_ROOT}")
    print(f"  Train JSON:       {cfg.TRAIN_JSON}")
    print(f"  Test JSON:        {cfg.TEST_JSON}")

    print(f"\nBioCLIP:")
    print(f"  Model:            {cfg.BIOCLIP_MODEL}")
    print(f"  Output dim:       {model.vision_dim}")

    print(f"\nGemma:")
    print(f"  Model:            {cfg.GEMMA_MODEL}")
    print(f"  Hidden size:      {model.hidden_size}")

    print(f"\nParameters:")
    print(f"  Total:            {params['total']:>14,}")
    print(f"  Trainable:        {params['trainable']:>14,}")
    print(f"  Trainable %:      {params['trainable_pct']:>13.4f}%")
    print(f"  ---")
    print(f"  BioCLIP trainable:{params['bioclip_trainable']:>14,}")
    print(f"  Gemma base train: {params['gemma_base_trainable']:>14,}")
    print(f"  Gemma LoRA:       {params['gemma_lora']:>14,}")
    print(f"  Projector:        {params['projector_trainable']:>14,}")

    print(f"\nTraining:")
    print(f"  Epochs:           {cfg.EPOCHS}")
    print(f"  Batch size:       {cfg.BATCH_SIZE}")
    print(f"  Grad accumulation:{cfg.GRADIENT_ACCUMULATION}")
    print(f"  Effective batch:  {cfg.BATCH_SIZE * cfg.GRADIENT_ACCUMULATION}")
    print(f"  Learning rate:    {cfg.LEARNING_RATE}")
    print(f"  Weight decay:     {cfg.WEIGHT_DECAY}")
    print(f"  Warmup ratio:     {cfg.WARMUP_RATIO}")
    print(f"  Grad clip norm:   {cfg.GRAD_CLIP_NORM}")
    print(f"  Dtype:            {cfg.DTYPE}")
    print(f"  AMP (FP16):       {cfg.USE_AMP}")
    print(f"  Grad checkpoint:  {cfg.GRADIENT_CHECKPOINTING}")

    print(f"\nLoRA:")
    print(f"  r:                {cfg.LORA_R}")
    print(f"  alpha:            {cfg.LORA_ALPHA}")
    print(f"  dropout:          {cfg.LORA_DROPOUT}")
    print(f"  targets:          {cfg.LORA_TARGET_MODULES}")

    print(f"\nTokenisation:")
    print(f"  Max text tokens:  {cfg.MAX_TEXT_TOKENS}")
    print(f"  Max new tokens:   {cfg.MAX_NEW_TOKENS}")
    print(f"  Steering prompt:  \"{cfg.STEERING_PROMPT}\"")

    print("=" * 60 + "\n")

    return params


# ---------------------------------------------------------------------------
# Safety check
# ---------------------------------------------------------------------------

def safety_check_trainable_params(params, max_allowed):
    """Abort if the trainable count is unexpectedly large."""
    trainable = params["trainable"]
    if trainable > max_allowed:
        raise RuntimeError(
            f"SAFETY ABORT: {trainable:,} trainable parameters detected, "
            f"which exceeds the safety limit of {max_allowed:,}. "
            f"This likely means base model weights are NOT frozen. "
            f"Breakdown: BioCLIP={params['bioclip_trainable']:,}, "
            f"Gemma base={params['gemma_base_trainable']:,}, "
            f"Gemma LoRA={params['gemma_lora']:,}, "
            f"Projector={params['projector_trainable']:,}"
        )
    if params["bioclip_trainable"] > 0:
        raise RuntimeError("BioCLIP has trainable parameters — it should be frozen.")
    if params["gemma_base_trainable"] > 0:
        raise RuntimeError("Gemma base model has trainable parameters — it should be frozen.")
    print(f"Safety check passed: {trainable:,} trainable params "
          f"(limit: {max_allowed:,})\n")


# ---------------------------------------------------------------------------
# Loss history I/O
# ---------------------------------------------------------------------------

def save_loss_history(history, output_dir):
    """
    Save loss history as CSV, JSON, and a PNG plot.

    history = {"epochs": [1,2,...], "train_loss": [...], "test_loss": [...]}
    """
    os.makedirs(output_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(output_dir, "loss_history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_loss"])
        for e, tl, vl in zip(history["epochs"], history["train_loss"], history["test_loss"]):
            writer.writerow([e, f"{tl:.6f}", f"{vl:.6f}"])

    # JSON
    json_path = os.path.join(output_dir, "loss_history.json")
    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)

    # PNG
    png_path = os.path.join(output_dir, "train_vs_test_loss.png")
    plt.figure(figsize=(8, 5))
    plt.plot(history["epochs"], history["train_loss"], "o-", label="Training Loss", color="#2563eb")
    plt.plot(history["epochs"], history["test_loss"], "s-", label="Test Loss", color="#dc2626")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training vs Test Loss", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    return csv_path, json_path, png_path


def load_loss_history(output_dir):
    """Load loss history from JSON if it exists."""
    json_path = os.path.join(output_dir, "loss_history.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return {"epochs": [], "train_loss": [], "test_loss": []}


# ---------------------------------------------------------------------------
# Epoch generation log
# ---------------------------------------------------------------------------

def save_epoch_generation(output_dir, epoch, image_path, ground_truth, generated):
    """Append one epoch's random generation to epoch_generations.json."""
    path = os.path.join(output_dir, "epoch_generations.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            records = json.load(f)
    else:
        records = []

    records.append({
        "epoch": epoch,
        "imagePath": image_path,
        "ground_truth": ground_truth,
        "generated": generated,
    })

    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def save_final_generations(output_dir, generations):
    """Save the final batch of test-image generations."""
    path = os.path.join(output_dir, "final_generations.json")
    with open(path, "w") as f:
        json.dump(generations, f, indent=2)
    print(f"  Final generations saved: {path}")


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_time(seconds):
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    else:
        return f"{seconds / 3600:.1f}h"
