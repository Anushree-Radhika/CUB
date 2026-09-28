"""
Gemma text decoder for TraitGen (BioCLIP patch features -> Gemma -> caption).

* Works with Gemma 3 (1B text-only, or the multimodal 4B/12B/27B checkpoints, of
  which only the text tower is kept) and with Gemma 2 / Gemma 1.
* Base weights are frozen; LoRA adapters + the image projector are trained.
* Default precision is fp32: Gemma 3's activations overflow float16 (V100/T4 have
  no bf16 tensor cores), so fp16 is deliberately not offered.
* The loss is computed only on caption tokens, in chunks, so the 262k-token
  vocabulary never materialises a (batch x sequence x vocab) logits tensor.
"""
import gc
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model


STEERING_PROMPT = "Describe this bird species correctly."
SPECIES_MARKER = "This species is likely"

MAX_TEXT_LEN = 800            # caption tokens (incl. EOS), longer captions are truncated

NORMAL_TOKEN_WEIGHT = 2.0
SPECIES_TOKEN_WEIGHT = 5.0
EOS_TOKEN_WEIGHT = 5.0

LOSS_CHUNK_TOKENS = 512       # tokens per lm_head/cross-entropy chunk

LORA_R = int(os.environ.get("LORA_R", 16))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", 32))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", 0.05))

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
_dtype_name = os.environ.get("DECODER_DTYPE", "fp32")
if _dtype_name not in _DTYPES:
    raise ValueError(
        f"DECODER_DTYPE={_dtype_name!r} is not supported. Use 'fp32' (default) or 'bf16'. "
        "fp16 is intentionally unavailable: Gemma 3 activations exceed float16's range."
    )
DECODER_DTYPE = _DTYPES[_dtype_name]


def load_text_decoder(model_id, dtype):
    """Return a *ForCausalLM holding only the text decoder of `model_id`."""
    cfg = AutoConfig.from_pretrained(model_id)
    model_type = getattr(cfg, "model_type", "")

    if model_type == "gemma3":
        # Multimodal Gemma 3 checkpoint (4B / 12B / 27B): the config nests the text
        # settings and AutoModelForCausalLM cannot load it. Load the full model on
        # CPU, then keep only the language tower + lm_head and drop the vision tower.
        from transformers import Gemma3ForCausalLM, Gemma3ForConditionalGeneration

        full = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        text_cfg = full.config.text_config
        with torch.device("meta"):                      # no memory for the throw-away skeleton
            lm = Gemma3ForCausalLM(text_cfg)
        lm.model = full.model.language_model            # Gemma3TextModel (embeddings + layers + norm)
        lm.lm_head = full.lm_head                       # tied to the embeddings
        lm.lm_head.weight = lm.model.embed_tokens.weight  # Gemma always ties input/output embeddings; never leave lm_head untied/random
        lm.generation_config = full.generation_config
        del full
        gc.collect()
        return lm

    if model_type in ("gemma3_text", "gemma2", "gemma"):
        return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, low_cpu_mem_usage=True)

    raise ValueError(
        f"Unsupported model_type {model_type!r} for {model_id}. Supported: gemma3, gemma3_text, gemma2, gemma."
    )


class GemmaDecoder(nn.Module):
    def __init__(self, model, vision_dim=768):
        """
        model:      HF repo id or local directory, e.g. "google/gemma-3-4b-pt".
                    Gemma repos are gated: accept the license and log in first.
        vision_dim: width of the vision encoder's patch tokens (768 for BioCLIP ViT-B/16).
        """
        super().__init__()

        self.vision_dim = vision_dim
        self.max_new_tokens = 600     # overwritten by Model from the real caption lengths

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if DECODER_DTYPE == torch.bfloat16 and torch.cuda.is_available() \
                and torch.cuda.get_device_capability(0)[0] < 8:
            print("WARNING: bf16 on a pre-Ampere GPU (V100/T4) is emulated: slow and not validated here.")

        base = load_text_decoder(model, DECODER_DTYPE)
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        self.embedding = base.get_input_embeddings()
        hidden = base.config.hidden_size
        self.hidden_size = hidden

        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.gemma = get_peft_model(base, lora_config)

        # Linear projection into Gemma's embedding space, followed by a LayerNorm whose
        # gain is set so image tokens start with the same per-dimension scale as real
        # token embeddings (Gemma 3's embedding lookup is already multiplied by sqrt(d)).
        self.image_projection = nn.Sequential(nn.Linear(vision_dim, hidden), nn.LayerNorm(hidden))
        with torch.no_grad():
            vocab = base.config.vocab_size
            lo, hi = (1000, min(20000, vocab)) if vocab > 1500 else (4, vocab)
            ids = torch.arange(lo, hi, device=self.embedding.weight.device)
            emb_std = self.embedding(ids).float().std().item()
            self.image_projection[1].weight.fill_(emb_std)
            self.image_projection[1].bias.zero_()

    # ------------------------------------------------------------------ text side
    def _tokenize_caption(self, caption):
        caption = caption.strip()
        if SPECIES_MARKER in caption:
            attributes, rest = caption.split(SPECIES_MARKER, 1)
            species_text = SPECIES_MARKER + rest
        else:
            attributes, species_text = caption, ""

        attr = self.tokenizer(attributes.strip(), add_special_tokens=False)["input_ids"]
        spec = self.tokenizer(species_text.strip(), add_special_tokens=False)["input_ids"] if species_text else []

        room = MAX_TEXT_LEN - 1                       # keep one slot for EOS
        spec = spec[:room]
        attr = attr[: room - len(spec)]
        return attr, spec

    def caption_lengths(self, captions):
        """Token length of each caption as it will be trained on (attributes + species + EOS)."""
        out = []
        for c in captions:
            a, s = self._tokenize_caption(c)
            out.append(len(a) + len(s) + 1)
        return out

    def _encode_captions(self, captions, device):
        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id

        rows = []
        for c in captions:
            a, s = self._tokenize_caption(c)
            rows.append((a + s + [eos], len(a), len(s)))

        T = max(len(r[0]) for r in rows)
        B = len(rows)

        ids = torch.full((B, T), pad, dtype=torch.long)
        mask = torch.zeros((B, T), dtype=torch.bool)
        weights = torch.zeros((B, T), dtype=torch.float32)

        for i, (seq, n_attr, n_spec) in enumerate(rows):
            n = len(seq)
            ids[i, :n] = torch.tensor(seq, dtype=torch.long)
            mask[i, :n] = True
            weights[i, :n_attr] = NORMAL_TOKEN_WEIGHT
            weights[i, n_attr:n_attr + n_spec] = SPECIES_TOKEN_WEIGHT
            weights[i, n - 1] = EOS_TOKEN_WEIGHT

        return ids.to(device), mask.to(device), weights.to(device)

    # ----------------------------------------------------------------- image side
    def _prefix(self, image_features):
        """[BOS] + projected image tokens + steering prompt, as embeddings (B, P, D)."""
        device = image_features.device
        emb_dtype = self.embedding.weight.dtype

        proj_dtype = self.image_projection[0].weight.dtype
        img = self.image_projection(image_features.to(proj_dtype)).to(emb_dtype)
        B = img.size(0)

        parts = []
        if self.tokenizer.bos_token_id is not None:
            bos = torch.full((B, 1), self.tokenizer.bos_token_id, dtype=torch.long, device=device)
            parts.append(self.embedding(bos))
        parts.append(img)

        prompt = self.tokenizer(STEERING_PROMPT, add_special_tokens=False, return_tensors="pt")["input_ids"]
        parts.append(self.embedding(prompt.to(device).expand(B, -1)))

        return torch.cat(parts, dim=1)

    # ------------------------------------------------------------------------ loss
    def _hidden(self, inputs_embeds, attention_mask):
        lm = self.gemma.get_base_model()
        out = lm.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False)
        return out.last_hidden_state

    def _ce_chunk(self, h, target, w):
        lm = self.gemma.get_base_model()
        logits = lm.lm_head(h)
        cap = getattr(lm.config, "final_logit_softcapping", None)
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return (F.cross_entropy(logits.float(), target, reduction="none") * w).sum()

    def _loss_sums(self, prefix, ids, mask, weights):
        B, T = ids.shape
        P = prefix.size(1)

        inputs = torch.cat([prefix, self.embedding(ids)], dim=1)
        attn = torch.cat(
            [torch.ones(B, P, dtype=torch.long, device=ids.device), mask.long()], dim=1
        )

        h = self._hidden(inputs, attn)                 # (B, P+T, D)
        h_pred = h[:, P - 1: P + T - 1, :]             # hidden state at position t predicts token t+1

        h_sel = h_pred[mask]                           # only real (non-pad) caption tokens
        target = ids[mask]
        w = weights[mask]

        total = h_sel.new_zeros((), dtype=torch.float32)
        for i in range(0, h_sel.size(0), LOSS_CHUNK_TOKENS):
            sl = slice(i, i + LOSS_CHUNK_TOKENS)
            if torch.is_grad_enabled() and h_sel.requires_grad:
                part = checkpoint(self._ce_chunk, h_sel[sl], target[sl], w[sl], use_reentrant=False)
            else:
                part = self._ce_chunk(h_sel[sl], target[sl], w[sl])
            total = total + part

        return total, w.sum()

    def forward_sums(self, image_features, captions):
        """(sum of weighted token losses, sum of weights): lets callers aggregate exactly."""
        ids, mask, weights = self._encode_captions(captions, image_features.device)
        prefix = self._prefix(image_features)
        return self._loss_sums(prefix, ids, mask, weights)

    def forward(self, image_features, captions):
        total, wsum = self.forward_sums(image_features, captions)
        return total / wsum

    def probe_loss(self, batch_size, num_patches, text_len, device):
        """Worst-case synthetic batch, used to measure peak memory before training."""
        vocab = self.gemma.get_base_model().config.vocab_size
        lo, hi = (1000, min(20000, vocab)) if vocab > 1500 else (4, vocab)

        feats = torch.randn(batch_size, num_patches, self.vision_dim, device=device)
        ids = torch.randint(lo, hi, (batch_size, text_len), device=device)
        mask = torch.ones(batch_size, text_len, dtype=torch.bool, device=device)
        weights = torch.ones(batch_size, text_len, dtype=torch.float32, device=device)

        total, wsum = self._loss_sums(self._prefix(feats), ids, mask, weights)
        return total / wsum

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, image_features):
        prefix = self._prefix(image_features)
        attn = torch.ones(prefix.shape[:2], dtype=torch.long, device=prefix.device)

        output_ids = self.gemma.generate(
            inputs_embeds=prefix,
            attention_mask=attn,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
