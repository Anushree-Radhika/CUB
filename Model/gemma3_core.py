"""
Standalone Gemma 3 text decoder in plain PyTorch that loads real Hugging Face weights.

Parameter names match Hugging Face's Gemma3ForCausalLM
(model.layers.N.self_attn.q_proj, ...mlp.gate_proj, ...), so
  * pretrained checkpoints load with load_state_dict, and
  * PEFT/LoRA target names (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj) work unchanged.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

@dataclass
class Gemma3Config:
    vocab_size: int = 262_144
    hidden_size: int = 640
    intermediate_size: int = 2048
    num_layers: int = 18
    num_heads: int = 4
    num_kv_heads: int = 1
    head_dim: int = 256
    sliding_window: int = 512
    query_pre_attn_scalar: float = 256.0
    rope_local_base: float = 10_000.0
    rope_global_base: float = 1_000_000.0
    rope_global_factor: float = 1.0          # linear RoPE scaling on global layers (8.0 for 4B+)
    rms_eps: float = 1e-6
    layer_types: Optional[List[str]] = None  # "sliding_attention" / "full_attention"

    def __post_init__(self):
        if self.layer_types is None:               # 5 local : 1 global
            self.layer_types = [
                "full_attention" if (i + 1) % 6 == 0 else "sliding_attention"
                for i in range(self.num_layers)
            ]

    @classmethod
    def from_hf(cls, hf_cfg):
        c = getattr(hf_cfg, "text_config", hf_cfg)

        rope_global, rope_local, factor = None, None, 1.0
        rp = getattr(c, "rope_parameters", None)               # newer transformers layout
        if isinstance(rp, dict) and "full_attention" in rp:
            g, l = rp["full_attention"], rp.get("sliding_attention", {})
            rope_global = g.get("rope_theta")
            rope_local = l.get("rope_theta")
            if g.get("rope_type", "default") == "linear":
                factor = g.get("factor", 1.0)
        if rope_global is None:
            rope_global = getattr(c, "rope_theta", 1_000_000.0)
        if rope_local is None:
            rope_local = getattr(c, "rope_local_base_freq", 10_000.0)
        rs = getattr(c, "rope_scaling", None)
        if isinstance(rs, dict) and rs.get("rope_type", rs.get("type")) == "linear":
            factor = rs.get("factor", 1.0)

        layer_types = getattr(c, "layer_types", None)
        if layer_types is None:
            pattern = getattr(c, "sliding_window_pattern", 6)
            layer_types = [
                "full_attention" if (i + 1) % pattern == 0 else "sliding_attention"
                for i in range(c.num_hidden_layers)
            ]

        return cls(
            vocab_size=c.vocab_size,
            hidden_size=c.hidden_size,
            intermediate_size=c.intermediate_size,
            num_layers=c.num_hidden_layers,
            num_heads=c.num_attention_heads,
            num_kv_heads=c.num_key_value_heads,
            head_dim=c.head_dim,
            sliding_window=c.sliding_window,
            query_pre_attn_scalar=float(c.query_pre_attn_scalar),
            rope_local_base=float(rope_local),
            rope_global_base=float(rope_global),
            rope_global_factor=float(factor),
            rms_eps=c.rms_norm_eps,
            layer_types=list(layer_types),
        )


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Zero-initialised weight, applied as (1 + weight), normalisation in fp32."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * (1.0 + self.weight.float())).to(dtype)


class ScaledEmbedding(nn.Embedding):
    """Token embedding multiplied by sqrt(hidden_size), as in Gemma."""

    def __init__(self, vocab, dim):
        super().__init__(vocab, dim)
        self.scale = dim ** 0.5

    def forward(self, ids):
        return super().forward(ids) * self.scale


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def _apply_rope(x, cos, sin):
    return x * cos + _rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, c: Gemma3Config):
        super().__init__()
        self.nh, self.nkv, self.hd = c.num_heads, c.num_kv_heads, c.head_dim
        self.q_proj = nn.Linear(c.hidden_size, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(c.hidden_size, self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(c.hidden_size, self.nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, c.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.hd, c.rms_eps)
        self.k_norm = RMSNorm(self.hd, c.rms_eps)
        self.scaling = c.query_pre_attn_scalar ** -0.5

    def forward(self, x, mask, cos, sin):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.nkv, self.hd).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)

        rep = self.nh // self.nkv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scaling)
        return self.o_proj(out.transpose(1, 2).reshape(B, S, self.nh * self.hd))


class MLP(nn.Module):
    def __init__(self, c: Gemma3Config):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, c: Gemma3Config, layer_type: str):
        super().__init__()
        self.is_local = layer_type == "sliding_attention"
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_eps)
        self.self_attn = Attention(c)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_eps)
        self.pre_feedforward_layernorm = RMSNorm(c.hidden_size, c.rms_eps)
        self.mlp = MLP(c)
        self.post_feedforward_layernorm = RMSNorm(c.hidden_size, c.rms_eps)

    def forward(self, x, mask_global, mask_local, rope_global, rope_local):
        mask = mask_local if self.is_local else mask_global
        cos, sin = rope_local if self.is_local else rope_global

        h = self.self_attn(self.input_layernorm(x), mask, cos, sin)
        x = x + self.post_attention_layernorm(h)

        h = self.mlp(self.pre_feedforward_layernorm(x))
        return x + self.post_feedforward_layernorm(h)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

class Gemma3Model(nn.Module):
    def __init__(self, c: Gemma3Config):
        super().__init__()
        self.c = c
        self.embed_tokens = ScaledEmbedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(c, t) for t in c.layer_types])
        self.norm = RMSNorm(c.hidden_size, c.rms_eps)

        def inv_freq(base, factor):
            f = 1.0 / (base ** (torch.arange(0, c.head_dim, 2).float() / c.head_dim))
            return f / factor

        self.register_buffer("inv_freq_local", inv_freq(c.rope_local_base, 1.0), persistent=False)
        self.register_buffer("inv_freq_global",
                             inv_freq(c.rope_global_base, c.rope_global_factor), persistent=False)

    @staticmethod
    def _rope(inv_freq, positions, dtype):
        freqs = positions[:, None].float() * inv_freq[None, :].to(positions.device)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[None, None].to(dtype), emb.sin()[None, None].to(dtype)

    def _masks(self, B, S, device, attention_mask):
        """Boolean masks, True = may attend. Shapes (B, 1, S, S)."""
        i = torch.arange(S, device=device)[:, None]
        j = torch.arange(S, device=device)[None, :]
        causal = j <= i
        local = causal & ((i - j) < self.c.sliding_window)
        if attention_mask is not None:
            keep = attention_mask.bool()[:, None, None, :]          # (B,1,1,S)
            causal = causal[None, None] & keep
            local = local[None, None] & keep
        else:
            causal, local = causal[None, None], local[None, None]
        eye = torch.eye(S, dtype=torch.bool, device=device)[None, None]   # avoid empty rows
        return (causal | eye).expand(B, 1, S, S), (local | eye).expand(B, 1, S, S)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None):
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        B, S, _ = x.shape
        pos = torch.arange(S, device=x.device)

        rope_g = self._rope(self.inv_freq_global, pos, x.dtype)
        rope_l = self._rope(self.inv_freq_local, pos, x.dtype)
        mask_g, mask_l = self._masks(B, S, x.device, attention_mask)

        for layer in self.layers:
            x = layer(x, mask_g, mask_l, rope_g, rope_l)
        return self.norm(x)

class DummyOutput:
    def __init__(self, logits):
        self.logits = logits

class Gemma3ForCausalLM(nn.Module):
    def __init__(self, c: Gemma3Config):
        super().__init__()
        self.config = c
        self.model = Gemma3Model(c)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight        # tied

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def enable_input_require_grads(self):
        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)
        self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, logits_to_keep=0, return_dict=False):
        h = self.model(input_ids=input_ids, inputs_embeds=inputs_embeds,
                       attention_mask=attention_mask)
        if logits_to_keep:
            h = h[:, -logits_to_keep:]
        logits = self.lm_head(h)
        if return_dict:
            return DummyOutput(logits)
        return (logits,)


# ----------------------------------------------------------------------------
# Loading Hugging Face weights
# ----------------------------------------------------------------------------

def _rename_hf_key(k: str):
    """Map Hugging Face keys (text-only or multimodal checkpoints) to this module's keys."""
    if k.startswith(("vision_tower", "multi_modal_projector",
                     "model.vision_tower", "model.multi_modal_projector")):
        return None
    for old, new in (("model.language_model.", "model."),
                     ("language_model.model.", "model."),
                     ("language_model.lm_head.", "lm_head.")):
        if k.startswith(old):
            return new + k[len(old):]
    return k


def load_gemma3(model_id: str, dtype=torch.float32, device="cpu") -> Gemma3ForCausalLM:
    """Download a Gemma 3 checkpoint with transformers and load it into this implementation."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = Gemma3Config.from_hf(AutoConfig.from_pretrained(model_id))
    hf = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)

    state = {}
    for k, v in hf.state_dict().items():
        nk = _rename_hf_key(k)
        if nk is not None:
            state[nk] = v
    del hf

    model = Gemma3ForCausalLM(cfg)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if k != "lm_head.weight"]     # lm_head is tied to the embeddings
    if missing or unexpected:
        raise RuntimeError(f"Weight mismatch. Missing: {missing[:5]} Unexpected: {unexpected[:5]}")
    model.lm_head.weight = model.model.embed_tokens.weight
    return model.to(device=device, dtype=dtype).eval()
