import os
import glob
import gc
import json
import math
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


# --------------------------------------------------------------------------- config
def _env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v else default


def _env_float(name, default):
    v = os.environ.get(name)
    return float(v) if v else default


DEVICE = torch.device("cuda" if torch.cuda.is_available() and not os.environ.get("FORCE_CPU") else "cpu")


def _find_images_root():
    """CUB images folder: env override > auto-detect on Kaggle > local server path."""
    override = os.environ.get("CUB_IMAGES_ROOT")
    if override:
        return override
    if os.path.isdir("/kaggle/input"):
        hits = glob.glob("/kaggle/input/**/001.Black_footed_Albatross", recursive=True)
        if hits:
            return os.path.dirname(hits[0])
    return os.path.expanduser("~/DATASET/CUB_200_2011/images")


IMAGES_ROOT = _find_images_root()

_default_ckpt_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.expanduser("~/checkpoints")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", _default_ckpt_dir)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "model2.pt")   # trainable weights only (small)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

TEST_JSON = os.environ.get("TEST_JSON", "./test3.json")

# Default: Gemma 3 4B pretrained. (google/gemma-3-1b-pt is a faster, smaller alternative.)
DECODER_ID = os.environ.get("DECODER_ID", "google/gemma-3-4b-pt")

EPOCHS = _env_int("EPOCHS", 6)
LEARNING_RATE = _env_float("LEARNING_RATE", 1e-4)
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
WARMUP_FRACTION = 0.05
MIN_LR_FRACTION = 0.1

# Batch size: auto-selected by measuring peak memory on a worst-case batch, unless
# BATCH_SIZE is set. MEMORY_FRACTION is the share of currently-free GPU memory that
# training may use; the rest is headroom so the run doesn't die mid-epoch.
BATCH_SIZE_MAX = _env_int("BATCH_SIZE_MAX", 16)
MEMORY_FRACTION = _env_float("MEMORY_FRACTION", 0.85)

VAL_FRACTION = 0.1
SPLIT_SEED = 42

# Per-epoch validation uses the full validation split. The per-epoch test curve uses a
# fixed random subset (set TEST_EVAL_SUBSET=0 for the full test set every epoch).
# The FULL test set is always evaluated once at the end, with the best checkpoint.
_subset = _env_int("TEST_EVAL_SUBSET", 1000)
TEST_EVAL_SUBSET = _subset if _subset > 0 else None

LOG_EVERY = _env_int("LOG_EVERY", 25)


# ---------------------------------------------------------------------------- data
class CUBDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return os.path.join(IMAGES_ROOT, item["imagePath"]), item["gt"]


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

    plt.figure(figsize=(7.5, 4.8))
    plt.plot(epochs, history["train"], marker="o", label="train")
    plt.plot(epochs, history["val"], marker="o", label="val")
    test_label = "test (subset)" if history["config"].get("test_eval_subset") else "test"
    plt.plot(epochs, history["test"], marker="o", label=test_label)
    if history["val"]:
        best = min(range(len(history["val"])), key=lambda i: history["val"][i])
        plt.axvline(best + 1, color="gray", linestyle="--", alpha=0.6, label=f"best val (epoch {best + 1})")
    plt.xlabel("epoch")
    plt.ylabel("weighted token loss")
    plt.title("Loss per epoch")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)


# --------------------------------------------------------------------------- model
class Model(nn.Module):
    def __init__(self, vision_encoder, text_decoder):
        super().__init__()

        self.ve = a_VE.VisionEncoder(vision_encoder).to(DEVICE)
        self.td = a_TD.TextDecoder(text_decoder).to(DEVICE)

    def forward(self, image_paths, captions):
        image_features = self.ve(image_paths)
        return self.td(image_features, captions)

    def forward_sums(self, image_paths, captions):
        image_features = self.ve(image_paths)
        return self.td.model.forward_sums(image_features, captions)

    @torch.no_grad()
    def generate(self, image_paths):
        image_features = self.ve(image_paths)
        return self.td.generate(image_features)

    # ------------------------------------------------------------- checkpoints
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def save_trainable(self, path, **meta):
        """Save only what training changes (LoRA adapters + image projector): ~100s of MB, not GBs."""
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        state = {k: v.detach().cpu() for k, v in self.state_dict().items() if k in names}
        tmp = path + ".tmp"
        torch.save({"state": state, "meta": meta}, tmp)
        os.replace(tmp, path)

    def load_trainable(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        result = self.load_state_dict(ckpt["state"], strict=False)
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        missing = [k for k in result.missing_keys if k in names]
        if result.unexpected_keys or missing:
            raise RuntimeError(
                f"Checkpoint {path} does not match this model "
                f"(unexpected={result.unexpected_keys[:3]}, missing={missing[:3]}). "
                "Was it trained with a different DECODER_ID or LoRA setting?"
            )
        return ckpt.get("meta", {})

    # ------------------------------------------------------------------ memory
    def _try_probe(self, batch_size, num_patches, text_len, opt_state_bytes):
        """Forward+backward on a worst-case synthetic batch. Returns (fits, peak_bytes)."""
        loss = None
        fits, peak = True, 0
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            self.td.train()
            loss = self.td.model.probe_loss(batch_size, num_patches, text_len, DEVICE)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss on the probe batch")
            loss.backward()
            peak = torch.cuda.max_memory_reserved() + opt_state_bytes
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
                raise
            fits = False
        finally:
            for p in self.trainable_parameters():
                p.grad = None
            del loss
            gc.collect()
            torch.cuda.empty_cache()
        return fits, peak

    def _choose_batch_size(self, num_patches, text_len):
        if DEVICE.type != "cuda":
            return _env_int("BATCH_SIZE", 2)

        params = self.trainable_parameters()
        opt_state_bytes = 2 * sum(p.numel() * p.element_size() for p in params)   # AdamW m and v

        free, _ = torch.cuda.mem_get_info()
        budget = MEMORY_FRACTION * (free + torch.cuda.memory_reserved())
        gb = 1024 ** 3
        print(f"GPU memory budget for training: {budget / gb:.1f} GiB "
              f"({MEMORY_FRACTION:.0%} of {(free + torch.cuda.memory_reserved()) / gb:.1f} GiB available)")

        forced = os.environ.get("BATCH_SIZE")
        candidates = [int(forced)] if forced else [b for b in (1, 2, 4, 6, 8, 12, 16, 24, 32) if b <= BATCH_SIZE_MAX]

        best = None
        for b in candidates:
            fits, peak = self._try_probe(b, num_patches, text_len, opt_state_bytes)
            if fits:
                print(f"  batch size {b:>2}: worst-case peak {peak / gb:.1f} GiB "
                      f"{'OK' if peak <= budget else 'over budget'}")
            else:
                print(f"  batch size {b:>2}: out of memory")
            if not fits or peak > budget:
                if forced:
                    raise RuntimeError(f"BATCH_SIZE={b} does not fit in the GPU memory budget.")
                break
            best = b

        if best is None:
            raise RuntimeError(
                "Not even batch size 1 fits in the memory budget. Use a smaller DECODER_ID "
                "(e.g. google/gemma-3-1b-pt) or free GPU memory."
            )
        return best

    # ---------------------------------------------------------------- evaluation
    def _evaluate(self, data, batch_size, max_samples=None):
        if max_samples is not None and len(data) > max_samples:
            data = random.Random(SPLIT_SEED).sample(data, max_samples)   # same subset every epoch

        loader = DataLoader(CUBDataset(data), batch_size=batch_size, shuffle=False, collate_fn=_collate)

        self.td.eval()
        total, wsum = 0.0, 0.0
        with torch.no_grad():
            for image_paths, captions in loader:
                s, w = self.forward_sums(image_paths, captions)
                total += s.item()
                wsum += w.item()
        self.td.train()
        return total / wsum

    def _sample(self, label, item):
        image_path = os.path.join(IMAGES_ROOT, item["imagePath"])
        self.td.eval()
        prediction = self.generate([image_path])[0].strip()
        self.td.train()
        return (
            f"[{label}] {item['imagePath']}\n"
            f"  prediction: {prediction}\n"
            f"  reference:  {item['gt'].strip()}\n"
        )

    # ------------------------------------------------------------------ training
    def start_training(self, json_path):
        torch.manual_seed(0)
        random.seed(0)

        for p in self.ve.parameters():
            p.requires_grad_(False)
        self.ve.eval()

        with open(json_path, "r") as f:
            all_train = json.load(f)
        with open(TEST_JSON, "r") as f:
            test_data = json.load(f)
        train_data, val_data = split_train_val(all_train)
        print(f"train={len(train_data)}  val={len(val_data)}  test={len(test_data)}")

        # ---- pre-flight: everything below runs BEFORE the long training loop ----
        dec = self.td.model
        with torch.no_grad():
            feats = self.ve([os.path.join(IMAGES_ROOT, train_data[0]["imagePath"])])
        num_patches, vision_dim = feats.shape[1], feats.shape[2]
        if vision_dim != dec.vision_dim:
            raise RuntimeError(f"Vision encoder outputs {vision_dim}-dim tokens but the decoder expects {dec.vision_dim}.")

        lengths = dec.caption_lengths([d["gt"] for d in (train_data + val_data + test_data)])
        max_len = max(lengths)
        dec.max_new_tokens = int(max_len * 1.1) + 8
        print(f"image tokens={num_patches}  caption tokens: mean={sum(lengths) / len(lengths):.0f} max={max_len}  "
              f"max_new_tokens={dec.max_new_tokens}")

        trainable = self.trainable_parameters()
        n_train = sum(p.numel() for p in trainable)
        print(f"trainable parameters: {n_train / 1e6:.1f}M")

        batch_size = self._choose_batch_size(num_patches, max_len)
        print(f"==> using batch size {batch_size}")

        self.td.eval()
        with torch.no_grad():
            check_items = val_data[:min(batch_size, 4)]
            s, w = self.forward_sums([os.path.join(IMAGES_ROOT, d["imagePath"]) for d in check_items],
                                     [d["gt"] for d in check_items])
        pre_loss = (s / w).item()
        if not math.isfinite(pre_loss):
            raise FloatingPointError("Validation loss is not finite before training (check DECODER_DTYPE).")
        print(f"pre-flight: loss on real validation batch = {pre_loss:.4f}")

        samples_log = os.path.join(CHECKPOINT_DIR, f"samples_{time.strftime('%Y%m%d_%H%M%S')}.txt")
        rng = random.Random(1234)
        first = self._sample("before training", rng.choice(val_data))
        print("\n" + first)
        with open(samples_log, "w") as f:
            f.write("=== before training ===\n" + first + "\n")

        probe_file = os.path.join(CHECKPOINT_DIR, ".write_test")
        with open(probe_file, "w") as f:
            f.write("ok")
        os.remove(probe_file)
        # ---- end pre-flight ----

        loader = DataLoader(CUBDataset(train_data), batch_size=batch_size, shuffle=True,
                            collate_fn=_collate, drop_last=False)
        steps_per_epoch = len(loader)
        total_steps = EPOCHS * steps_per_epoch
        warmup = max(1, int(WARMUP_FRACTION * total_steps))

        optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        def lr_lambda(step):
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total_steps - warmup)
            return MIN_LR_FRACTION + (1 - MIN_LR_FRACTION) * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        run_tag = time.strftime("%Y%m%d_%H%M%S")
        png_path = os.path.join(CHECKPOINT_DIR, f"loss_curves_{run_tag}.png")
        json_out = os.path.join(CHECKPOINT_DIR, f"loss_history_{run_tag}.json")
        history = {
            "train": [], "val": [], "test": [],
            "config": {
                "decoder_id": DECODER_ID, "epochs": EPOCHS, "batch_size": batch_size,
                "learning_rate": LEARNING_RATE, "test_eval_subset": TEST_EVAL_SUBSET,
                "trainable_params": n_train,
            },
        }

        best_val = float("inf")
        best_epoch = -1
        t0 = time.time()
        step = 0

        self.td.train()
        for epoch in range(EPOCHS):
            running, seen = 0.0, 0

            for image_paths, captions in loader:
                optimizer.zero_grad(set_to_none=True)

                loss = self(image_paths, captions)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at step {step}. Aborting instead of training on garbage.")
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP_NORM)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"Non-finite gradient norm at step {step}.")
                optimizer.step()
                scheduler.step()

                running += loss.item()
                seen += 1
                step += 1

                if step % LOG_EVERY == 0 or step == total_steps:
                    elapsed = time.time() - t0
                    eta = elapsed / step * (total_steps - step)
                    mem = torch.cuda.max_memory_allocated() / 1024 ** 3 if DEVICE.type == "cuda" else 0.0
                    print(f"  epoch {epoch + 1}/{EPOCHS} step {step}/{total_steps} | loss {running / seen:.4f} | "
                          f"lr {scheduler.get_last_lr()[0]:.2e} | peak mem {mem:.1f} GiB | "
                          f"elapsed {elapsed / 60:.0f} min | ETA {eta / 60:.0f} min", flush=True)

            train_loss = running / seen
            val_loss = self._evaluate(val_data, batch_size)
            test_loss = self._evaluate(test_data, batch_size, TEST_EVAL_SUBSET)

            history["train"].append(train_loss)
            history["val"].append(val_loss)
            history["test"].append(test_loss)
            save_loss_curves(history, png_path, json_out)

            print(f"\nEpoch {epoch + 1}/{EPOCHS} | train={train_loss:.4f} | val={val_loss:.4f} | test={test_loss:.4f}")

            text = (self._sample("random validation image", rng.choice(val_data)) + "\n"
                    + self._sample("random training image", rng.choice(train_data)))
            print("\n" + text)
            with open(samples_log, "a") as f:
                f.write(f"=== after epoch {epoch + 1} (train={train_loss:.4f} val={val_loss:.4f} test={test_loss:.4f}) ===\n"
                        + text + "\n")

            # Best checkpoint is chosen on VALIDATION loss, never on test loss.
            if val_loss < best_val:
                best_val, best_epoch = val_loss, epoch + 1
                self.save_trainable(CHECKPOINT_PATH, epoch=epoch + 1, val_loss=val_loss,
                                    decoder_id=DECODER_ID, batch_size=batch_size)
                print(f"  saved checkpoint (best val loss {best_val:.4f}, epoch {best_epoch})\n", flush=True)

        # Final: restore the best checkpoint and score the FULL test set once.
        if best_epoch > 0:
            self.load_trainable(CHECKPOINT_PATH)
        final_test = self._evaluate(test_data, batch_size)
        history["best_epoch"] = best_epoch
        history["best_val"] = best_val
        history["final_test_full"] = final_test
        save_loss_curves(history, png_path, json_out)
        print(f"Done. Best epoch {best_epoch} (val {best_val:.4f}); full-test loss with best checkpoint: {final_test:.4f}")
        print(f"Plot: {png_path}\nSamples: {samples_log}\nCheckpoint: {CHECKPOINT_PATH}")
        return history
