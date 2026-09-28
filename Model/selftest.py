"""
Self-test for the TraitGen Gemma pipeline. Needs NO downloads, NO Hugging Face login and
NO real data: it builds tiny random Gemma 3 models on the fly and checks every code path
of a_gemma_TD.py and a_model.py (loss maths, LoRA + gradient checkpointing, generation,
text-tower extraction from a multimodal checkpoint, plots, checkpoints, resume).

Run it once on the server BEFORE the real training run:
    cd Model && python selftest.py
Expected last line: ALL CHECKS PASSED
"""
import os
import sys
import json
import math
import random
import shutil
import tempfile
import traceback
import zlib

os.environ["FORCE_CPU"] = "1"                       # keep the test independent of GPU load
TMP = tempfile.mkdtemp(prefix="traitgen_selftest_")
os.environ.update(
    CHECKPOINT_DIR=os.path.join(TMP, "ckpt"),
    CUB_IMAGES_ROOT=os.path.join(TMP, "images"),
    TEST_JSON=os.path.join(TMP, "test.json"),
    EPOCHS="3", BATCH_SIZE="2", TEST_EVAL_SUBSET="4",
    LEARNING_RATE="3e-3", LOG_EVERY="4", DECODER_ID="tiny-local",
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import Gemma3ForCausalLM, Gemma3TextConfig, PreTrainedTokenizerFast

import a_gemma_TD
import a_model

FAILED = []


def check(cond, message):
    print(("PASS  " if cond else "FAIL  ") + message)
    if not cond:
        FAILED.append(message)


# ----------------------------------------------------------------------- fake data
COLORS = ["brown", "black", "white", "grey", "red", "blue"]
PATTERNS = ["solid", "striped", "spotted", "plain"]
CLASSES = ["001.Alpha_Bird", "002.Beta_Bird", "003.Gamma_Bird", "004.Delta_Bird"]
rnd = random.Random(0)


def make_item(cls, i):
    name = cls.split(".", 1)[1].replace("_", " ")
    gt = (f" The color of wing is {rnd.choice(COLORS)}. The pattern of breast is {rnd.choice(PATTERNS)}. "
          f"The size is small. The color of bill is {rnd.choice(COLORS)}. This species is likely {name}.")
    return {"imagePath": f"{cls}/img_{i}.jpg", "gt": gt}


train = [make_item(c, i) for c in CLASSES for i in range(6)]
test = [make_item(c, 100 + i) for c in CLASSES for i in range(2)]
train_json = os.path.join(TMP, "train.json")
json.dump(train, open(train_json, "w"))
json.dump(test, open(os.environ["TEST_JSON"], "w"))

words = set()
for it in train + test:
    words.update(it["gt"].replace(".", " . ").split())
words.update(STEERING for STEERING in a_gemma_TD.STEERING_PROMPT.replace(".", " . ").split())
vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3}
for w in sorted(words):
    vocab.setdefault(w, len(vocab))
V = len(vocab) + 4

tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
tok.pre_tokenizer = pre_tokenizers.Whitespace()
fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<bos>", eos_token="<eos>",
                               pad_token="<pad>", unk_token="<unk>")


class StubVE(torch.nn.Module):
    """Stands in for BioCLIP: deterministic pseudo-features per image path."""
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    @torch.no_grad()
    def forward(self, paths):
        out = []
        for p in paths:
            g = torch.Generator().manual_seed(zlib.crc32(p.encode()))
            out.append(torch.randn(6, 24, generator=g))
        return torch.stack(out)


tcfg = Gemma3TextConfig(
    vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
    num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=512,
    sliding_window=64, pad_token_id=0, bos_token_id=2, eos_token_id=1, query_pre_attn_scalar=16,
)


def section(title):
    print(f"\n=== {title} ===")


# ------------------------------------------------- A. text-only Gemma 3 checkpoint path
try:
    section("A. decoder: loading, loss maths, gradients, generation")
    text_dir = os.path.join(TMP, "text_only")
    Gemma3ForCausalLM(tcfg).save_pretrained(text_dir)
    fast.save_pretrained(text_dir)

    dec = a_gemma_TD.GemmaDecoder(text_dir, vision_dim=24)
    a_gemma_TD.LOSS_CHUNK_TOKENS = 5                 # force several chunks even on tiny inputs

    caps = [train[0]["gt"], train[7]["gt"]]
    feats = torch.randn(2, 6, 24)

    a, s = dec._tokenize_caption(caps[0])
    check(len(a) > 0 and len(s) > 0, "caption splits into attribute tokens and species tokens")
    check(dec.caption_lengths(caps) == [len(dec._tokenize_caption(c)[0]) + len(dec._tokenize_caption(c)[1]) + 1 for c in caps],
          "caption_lengths matches the tokenised captions")

    dec.eval()
    loss = dec(feats, caps)
    check(torch.isfinite(loss).item(), f"forward loss is finite ({loss.item():.4f}; ln(vocab)={math.log(V):.2f})")

    ids, mask, weights = dec._encode_captions(caps, feats.device)
    prefix = dec._prefix(feats)
    P, T = prefix.size(1), ids.size(1)
    inputs = torch.cat([prefix, dec.embedding(ids)], dim=1)
    attn = torch.cat([torch.ones(2, P, dtype=torch.long), mask.long()], dim=1)
    with torch.no_grad():
        logits = dec.gemma(inputs_embeds=inputs, attention_mask=attn).logits
        pred = logits[:, P - 1: P + T - 1, :]
        ce = F.cross_entropy(pred.reshape(-1, V), ids.reshape(-1), reduction="none").view_as(ids)
        ref = (ce * weights).sum() / weights.sum()
    check(abs(loss.item() - ref.item()) < 1e-4,
          f"chunked loss equals full-logits reference ({loss.item():.6f} vs {ref.item():.6f})")
    check(bool((weights[~mask] == 0).all()), "padding tokens carry zero loss weight")

    dec.train()
    loss = dec(feats, caps)
    loss.backward()
    lora_b = [p for n, p in dec.named_parameters() if "lora_B" in n]
    check(len(lora_b) > 0 and any(p.grad is not None and p.grad.abs().sum() > 0 for p in lora_b),
          "LoRA adapters receive gradients (with gradient checkpointing on)")
    check(dec.image_projection[0].weight.grad is not None and dec.image_projection[0].weight.grad.abs().sum() > 0,
          "image projector receives gradients")
    frozen = [p for n, p in dec.named_parameters() if not p.requires_grad]
    check(len(frozen) > 0 and all(p.grad is None for p in frozen), "base weights are frozen (no gradients)")

    dec.eval()
    dec.max_new_tokens = 6
    out = dec.generate(feats)
    check(isinstance(out, list) and len(out) == 2 and all(isinstance(x, str) for x in out), "generate() returns one string per image")

    fits = dec.probe_loss(2, 6, 12, feats.device)
    check(torch.isfinite(fits).item(), "worst-case probe batch runs")
except Exception:
    traceback.print_exc()
    check(False, "section A raised an exception (see traceback above)")

# --------------------------------------------- B. full pipeline: train / eval / plots / resume
try:
    section("B. pipeline: training loop, evaluation, plots, samples, checkpoints")
    dec2 = a_gemma_TD.GemmaDecoder(text_dir, vision_dim=24)
    model = a_model.Model(StubVE(), dec2)
    hist = model.start_training(train_json)

    check(len(hist["train"]) == 3 and len(hist["val"]) == 3 and len(hist["test"]) == 3, "3 epochs of train/val/test losses recorded")
    check(all(math.isfinite(x) for x in hist["train"] + hist["val"] + hist["test"]), "all recorded losses are finite")
    check(hist["train"][-1] < hist["train"][0], f"training loss decreases ({hist['train'][0]:.3f} -> {hist['train'][-1]:.3f})")
    check("final_test_full" in hist and math.isfinite(hist["final_test_full"]), "full test loss computed with the best checkpoint")

    ckpt_dir = os.environ["CHECKPOINT_DIR"]
    files = os.listdir(ckpt_dir)
    check(any(f.startswith("loss_curves_") and f.endswith(".png") for f in files), "loss-curve PNG written")
    check(any(f.startswith("loss_history_") and f.endswith(".json") for f in files), "loss history JSON written")
    samples = [f for f in files if f.startswith("samples_")]
    check(len(samples) == 1, "samples file written")
    if samples:
        txt = open(os.path.join(ckpt_dir, samples[0])).read()
        check(txt.count("prediction:") >= 1 + 2 * 3, "a generated caption is logged before training and twice after every epoch")

    check(os.path.exists(a_model.CHECKPOINT_PATH), "trainable-weights checkpoint saved")

    name = next(n for n, p in model.named_parameters() if p.requires_grad)
    saved = dict(model.named_parameters())[name].detach().clone()
    with torch.no_grad():
        dict(model.named_parameters())[name].add_(1.0)
    model.load_trainable(a_model.CHECKPOINT_PATH)
    check(torch.allclose(dict(model.named_parameters())[name], saved), "checkpoint load restores the trained weights (resume works)")
except Exception:
    traceback.print_exc()
    check(False, "section B raised an exception (see traceback above)")

# ------------------------------- C. extraction of the text tower from a multimodal Gemma 3
section("C. multimodal Gemma 3 checkpoint (4B/12B/27B style): text-tower extraction")
vlm = None
try:
    from transformers import Gemma3Config, Gemma3ForConditionalGeneration, SiglipVisionConfig
    vcfg = SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                              num_attention_heads=2, image_size=28, patch_size=14)
    vlm = Gemma3ForConditionalGeneration(Gemma3Config(text_config=tcfg, vision_config=vcfg, mm_tokens_per_image=4))
    vlm_dir = os.path.join(TMP, "vlm")
    vlm.save_pretrained(vlm_dir)
    fast.save_pretrained(vlm_dir)
except Exception as e:
    print(f"SKIP  could not build a tiny multimodal checkpoint here ({type(e).__name__}: {e})")
    vlm = None

if vlm is not None:
    try:
        dec3 = a_gemma_TD.GemmaDecoder(vlm_dir, vision_dim=24)
        want = vlm.model.language_model.layers[0].mlp.up_proj.weight
        got = dec3.gemma.get_base_model().model.layers[0].mlp.up_proj.base_layer.weight
        check(torch.allclose(want, got), "extracted text tower carries the original pretrained weights")
        check(not any("vision" in n or "multi_modal" in n for n, _ in dec3.named_parameters()),
              "vision tower and projector are dropped")
        check(dec3.gemma.get_base_model().lm_head.weight.data_ptr() == dec3.embedding.weight.data_ptr(),
              "lm_head stays tied to the embeddings")
        f3 = torch.randn(2, 6, 24)
        check(torch.isfinite(dec3(f3, [train[0]["gt"], train[1]["gt"]])).item(), "extracted decoder produces a finite loss")
        dec3.eval()
        dec3.max_new_tokens = 4
        check(len(dec3.generate(f3)) == 2, "extracted decoder generates")
    except Exception:
        traceback.print_exc()
        check(False, "section C raised an exception in the extraction code (see traceback above)")

shutil.rmtree(TMP, ignore_errors=True)
print()
if FAILED:
    print(f"{len(FAILED)} CHECK(S) FAILED:")
    for m in FAILED:
        print("  -", m)
    sys.exit(1)
print("ALL CHECKS PASSED")
