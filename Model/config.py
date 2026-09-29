"""
TraitGen configuration.

Every setting is read from an environment variable (if set) or falls back to
a default.  To override, either ``export VAR=value`` in the shell or create a
small wrapper script that sets the variables before ``import config``.

Defaults are tuned for a single Tesla V100 32 GB with FP32 (Gemma 3 overflows
FP16; V100 has no native BF16 tensor cores).
"""

import os

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _env(name, default):
    return os.environ.get(name, str(default))

def _env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v is not None else default

def _env_float(name, default):
    v = os.environ.get(name)
    return float(v) if v is not None else default

def _env_bool(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

IMAGE_ROOT       = os.path.expanduser(_env("IMAGE_ROOT", "~/DATASET/CUB_200_2011/images"))
TRAIN_JSON       = _env("TRAIN_JSON", os.path.join(os.path.dirname(__file__), "train3.json"))
TEST_JSON        = _env("TEST_JSON",  os.path.join(os.path.dirname(__file__), "test3.json"))
OUTPUT_DIR       = _env("OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "checkpoints"))

# ---------------------------------------------------------------------------
# model identifiers
# ---------------------------------------------------------------------------

GEMMA_MODEL      = _env("GEMMA_MODEL",  "google/gemma-3-4b-pt")
BIOCLIP_MODEL    = _env("BIOCLIP_MODEL", "hf-hub:imageomics/bioclip")

# ---------------------------------------------------------------------------
# training hyper-parameters
# ---------------------------------------------------------------------------

EPOCHS                    = _env_int("EPOCHS", 15)
BATCH_SIZE                = _env_int("BATCH_SIZE", 4)
GRADIENT_ACCUMULATION     = _env_int("GRADIENT_ACCUMULATION_STEPS", 1)

LEARNING_RATE             = _env_float("LEARNING_RATE", 2e-4)
WEIGHT_DECAY              = _env_float("WEIGHT_DECAY", 0.01)
WARMUP_RATIO              = _env_float("WARMUP_RATIO", 0.05)
GRAD_CLIP_NORM            = _env_float("GRAD_CLIP_NORM", 1.0)

# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

LORA_R                    = _env_int("LORA_R", 16)
LORA_ALPHA                = _env_int("LORA_ALPHA", 32)
LORA_DROPOUT              = _env_float("LORA_DROPOUT", 0.05)
LORA_TARGET_MODULES       = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ---------------------------------------------------------------------------
# tokenisation / generation
# ---------------------------------------------------------------------------

MAX_TEXT_TOKENS            = _env_int("MAX_TEXT_TOKENS", 512)
MAX_NEW_TOKENS             = _env_int("MAX_NEW_TOKENS", 512)

# ---------------------------------------------------------------------------
# precision & memory
# ---------------------------------------------------------------------------

# Gemma 3 overflows FP16 activations. V100 has no native BF16 tensor cores.
# Default is FP32 for model weights. AMP with FP16 is used only for the
# forward/backward pass via torch.cuda.amp (autocast), NOT for storing
# the model weights themselves.
DTYPE                     = _env("DTYPE", "fp32")
GRADIENT_CHECKPOINTING    = _env_bool("GRADIENT_CHECKPOINTING", True)
USE_AMP                   = _env_bool("USE_AMP", False)   # FP16 AMP — disabled by default for V100+Gemma3 safety

# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------

SEED                      = _env_int("SEED", 42)

# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------

LOG_EVERY_STEPS           = _env_int("LOG_EVERY_STEPS", 10)

# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------

STEERING_PROMPT = "Describe the visible attributes of this bird."

# ---------------------------------------------------------------------------
# safety
# ---------------------------------------------------------------------------

# If the trainable parameter count exceeds this, training is aborted.
# Gemma 3 4B has ~4 billion parameters; LoRA + projector should be < 100M.
MAX_TRAINABLE_PARAMS      = _env_int("MAX_TRAINABLE_PARAMS", 500_000_000)

# ---------------------------------------------------------------------------
# number of final generation samples
# ---------------------------------------------------------------------------

FINAL_GENERATION_SAMPLES  = _env_int("FINAL_GENERATION_SAMPLES", 10)
