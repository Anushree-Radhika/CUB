# TraitGen

Image-to-text generation of bird attribute descriptions on CUB-200-2011.

**BioCLIP (frozen) → MLP projector (trainable) → Gemma 3 + LoRA (trainable adapters)**

```
image → BioCLIP patch tokens → projector → [BOS] + image tokens + steering prompt → Gemma 3 → description
```

Loss is cross-entropy on the description tokens only (the visual prefix is not trained on). EOS is appended to every caption, so generation stops on its own.

## Setup

```bash
conda create -n gen python=3.10 -y && conda activate gen
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
huggingface-cli login      # accept the Gemma license on Hugging Face first
```

## Train

```bash
CUDA_VISIBLE_DEVICES=0 python train.py
CUDA_VISIBLE_DEVICES=0 python train.py --resume checkpoints/epoch_07
```

## Configure

All settings are in `config.py` and can be overridden with environment variables.

| Variable | Default | Notes |
|---|---|---|
| `GEMMA_MODEL` | `google/gemma-3-4b-pt` | Use `google/gemma-3-1b-pt` on 16 GB GPUs (e.g. Kaggle T4) |
| `BIOCLIP_MODEL` | `hf-hub:imageomics/bioclip` | |
| `DTYPE` | `fp32` | `bf16` only on Ampere or newer |
| `USE_AMP` | `0` | Keep off on V100/T4: Gemma 3 overflows in fp16 |
| `BATCH_SIZE` / `GRADIENT_ACCUMULATION_STEPS` | `2` / `2` | |
| `LEARNING_RATE` | `2e-4` | |
| `EPOCHS` | `15` | |
| `LORA_R` / `LORA_ALPHA` | `16` / `32` | Targets: q, k, v, o, gate, up, down |
| `MAX_TEXT_TOKENS` | `800` | Set slightly above your longest caption (about 320 if captions ≤ 300 tokens) |
| `MAX_NEW_TOKENS` | `800` | Keep ≥ `MAX_TEXT_TOKENS` |
| `OUTPUT_DIR` | `./checkpoints` | |

Data paths: `IMAGE_ROOT`, `TRAIN_JSON`, `TEST_JSON`, optional `VAL_JSON` (otherwise 10% of train is held out via `VAL_SPLIT`).

## Outputs

In `OUTPUT_DIR`:

- `epoch_XX/`: LoRA adapter, projector, tokenizer, optimizer state, config snapshot
- `loss_history.{csv,json}`, `train_vs_test_loss.png`
- `epoch_generations.json`: one random test generation per epoch
- `final_generations.json`: final samples from the test set

Every epoch is checkpointed, but training does not stop early. Pick the `epoch_XX` with the lowest validation loss.

## Notes

- Requires `transformers>=4.56,<5` (uses `dtype=` and `logits_to_keep`) and Python 3.10+.
- Test loss is evaluated every epoch, so the test set is not a clean hold-out.
- Checkpoints trained before the EOS fix will not stop generation correctly. Retrain from scratch.
- Files: `model.py` (model, checkpoints), `train.py` (loop), `config.py`, `utils.py` (logging, plots), `dataset.py`.
