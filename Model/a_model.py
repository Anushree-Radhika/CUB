import os
import json
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")  # headless server: write PNGs, never open a window
import matplotlib.pyplot as plt

import a_VE
import a_TD


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGES_ROOT = os.path.expanduser("~/DATASET/CUB_200_2011/images")

# Kaggle's /kaggle/working doesn't exist on this box — checkpoints and plots
# save under the user's own home dir instead.
CHECKPOINT_DIR = os.path.expanduser("~/checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "model2.pt")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

TEST_JSON = "./test3.json"

BATCH_SIZE = 8
LEARNING_RATE = 1e-4
GRAD_CLIP_NORM = 1.0
EPOCHS = 8

# Validation is carved out of train3.json (test3.json stays untouched as the
# test set). The split is stratified per species and seeded, so a_play.py and
# aa_play.py always produce the SAME split — otherwise resuming in aa_play.py
# would leak validation images into training.
VAL_FRACTION = 0.1
SPLIT_SEED = 42

# Set to an int to evaluate on a random subset (faster epochs); None = full set.
EVAL_MAX_SAMPLES = None


class CUBDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(IMAGES_ROOT, item["imagePath"])
        return image_path, item["gt"]


def _collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


def split_train_val(data, val_fraction=VAL_FRACTION, seed=SPLIT_SEED):
    """Stratified split by species folder (imagePath = '<class_folder>/<file>.jpg')."""
    by_class = {}
    for item in data:
        by_class.setdefault(item["imagePath"].split("/")[0], []).append(item)

    rng = random.Random(seed)
    train, val = [], []
    for cls in sorted(by_class):
        items = by_class[cls][:]
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_fraction))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val


def save_loss_curves(history, png_path, json_path):
    epochs = range(1, len(history["train"]) + 1)

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, history["train"], marker="o", label="train")
    plt.plot(epochs, history["val"], marker="o", label="val")
    plt.plot(epochs, history["test"], marker="o", label="test")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss per epoch")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)


class Model(nn.Module):
    def __init__(self, vision_encoder, text_decoder):
        super().__init__()

        self.ve = a_VE.VisionEncoder(vision_encoder).to(DEVICE)
        self.td = a_TD.TextDecoder(text_decoder).to(DEVICE)

    def forward(self, image_paths, captions):
        image_features = self.ve(image_paths)
        return self.td(image_features, captions)

    @torch.no_grad()
    def generate(self, image_paths):
        image_features = self.ve(image_paths)
        return self.td.generate(image_features)

    def _evaluate(self, call_target, data, max_samples=EVAL_MAX_SAMPLES):
        if max_samples is not None and len(data) > max_samples:
            data = random.Random(SPLIT_SEED).sample(data, max_samples)  # same subset every epoch

        loader = DataLoader(CUBDataset(data), batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate)

        self.td.eval()  # LoRA dropout off
        total, count = 0.0, 0
        with torch.no_grad():
            for image_paths, captions in loader:
                loss = call_target(image_paths, captions)
                total += loss.item() * len(image_paths)
                count += len(image_paths)
        self.td.train()

        return total / count

    def _print_sample(self, epoch, data):
        item = random.choice(data)
        image_path = os.path.join(IMAGES_ROOT, item["imagePath"])

        self.td.eval()
        prediction = self.generate([image_path])[0]
        self.td.train()

        print(f"\n--- Sample after epoch {epoch} (random validation image) ---")
        print(f"image:      {item['imagePath']}")
        print(f"prediction: {prediction.strip()}")
        print(f"reference:  {item['gt'].strip()}\n")

    def start_training(self, json_path, accelerator=None):
        is_main = accelerator is None or accelerator.is_main_process

        with open(json_path, "r") as f:
            all_train = json.load(f)
        with open(TEST_JSON, "r") as f:
            test_data = json.load(f)

        train_data, val_data = split_train_val(all_train)
        if is_main:
            print(f"train={len(train_data)}  val={len(val_data)}  test={len(test_data)}")

        loader = DataLoader(
            CUBDataset(train_data),
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=_collate,
        )

        self.ve.eval()
        self.td.train()

        optimizer = torch.optim.AdamW(self.parameters(), lr=LEARNING_RATE)

        call_target = self
        if accelerator is not None:
            call_target, optimizer, loader = accelerator.prepare(self, optimizer, loader)

        # Sanity check: one forward pass, no graph kept. Loss should be finite.
        first_batch = next(iter(loader))
        with torch.no_grad():
            first_loss = call_target(first_batch[0], first_batch[1])
        if is_main:
            print(f"sanity check — first-batch loss: {first_loss.item():.4f}")

        run_tag = time.strftime("%Y%m%d_%H%M%S")
        png_path = os.path.join(CHECKPOINT_DIR, f"loss_curves_{run_tag}.png")
        json_path = os.path.join(CHECKPOINT_DIR, f"loss_history_{run_tag}.json")
        history = {"train": [], "val": [], "test": []}

        best_val = float("inf")
        for epoch in range(EPOCHS):
            total_loss = 0.0

            for image_paths, captions in loader:
                optimizer.zero_grad()

                loss = call_target(image_paths, captions)

                if accelerator is not None:
                    accelerator.backward(loss)
                else:
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                total_loss += loss.item()

            train_loss = total_loss / len(loader)
            val_loss = self._evaluate(call_target, val_data)
            test_loss = self._evaluate(call_target, test_data)

            history["train"].append(train_loss)
            history["val"].append(val_loss)
            history["test"].append(test_loss)

            if is_main:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"train={train_loss:.4f} | val={val_loss:.4f} | test={test_loss:.4f}"
                )
                save_loss_curves(history, png_path, json_path)
                self._print_sample(epoch + 1, val_data)

            # Checkpoint on VALIDATION loss (never test loss — that would leak the test set).
            if val_loss < best_val:
                if accelerator is None:
                    torch.save(self.state_dict(), CHECKPOINT_PATH)
                elif accelerator.is_main_process:
                    unwrapped = accelerator.unwrap_model(call_target)
                    torch.save(unwrapped.state_dict(), CHECKPOINT_PATH)
                best_val = val_loss
                if is_main:
                    print(f"  saved checkpoint (best val loss {best_val:.4f})")
