"""
TraitGen model: BioCLIP (frozen) -> projector (trainable) -> Gemma 3 + LoRA.

CUB bird image
    -> BioCLIP ViT (frozen) -> patch tokens  (B, num_patches, vision_dim)
    -> Projector: LayerNorm -> Linear -> GELU -> Linear -> (B, num_patches, gemma_hidden)
    -> [BOS] + projected tokens + steering-prompt embeddings (visual prefix)
    -> Gemma 3 decoder (base frozen, LoRA adapters trainable)
    -> generated attribute description

Precision (V100): Gemma 3 overflows float16 and V100 has no native BF16,
so the default is FP32 weights. Keep USE_AMP off unless you verify the loss stays finite.
"""

import gc
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict

import config as cfg


# ============================================================================
# BioCLIP vision encoder (frozen)
# ============================================================================

class BioCLIPEncoder(nn.Module):
    """Loads BioCLIP via OpenCLIP and returns patch-level features (no CLS)."""

    def __init__(self, model_name: str = cfg.BIOCLIP_MODEL, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        self.model = self.model.to(device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        self._output_dim = self.model.visual.ln_post.normalized_shape[0]

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @torch.no_grad()
    def forward(self, image_paths: list) -> torch.Tensor:
        images = []
        for p in image_paths:
            with Image.open(p) as im:
                images.append(self.preprocess(im.convert("RGB")))
        images = torch.stack(images).to(self.device)

        visual = self.model.visual

        x = visual.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        cls = visual.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)

        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.patch_dropout(x)
        x = visual.ln_pre(x)
        x = visual.transformer(x)
        x = visual.ln_post(x)

        return x[:, 1:, :]


# ============================================================================
# Image projector
# ============================================================================

class ImageProjector(nn.Module):
    """LayerNorm -> Linear -> GELU -> Linear."""

    def __init__(self, vision_dim: int, hidden_dim: int):
        super().__init__()
        intermediate = min(vision_dim * 4, hidden_dim * 2)
        self.projector = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, intermediate),
            nn.GELU(),
            nn.Linear(intermediate, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


# ============================================================================
# Gemma loading (text-only and multimodal checkpoints)
# ============================================================================

def _load_gemma_text_decoder(model_id: str, dtype: torch.dtype):
    model_cfg = AutoConfig.from_pretrained(model_id)
    model_type = getattr(model_cfg, "model_type", "")

    if model_type == "gemma3":
        from transformers import Gemma3ForCausalLM, Gemma3ForConditionalGeneration
        full = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, dtype=dtype, low_cpu_mem_usage=True
        )
        text_cfg = full.config.text_config
        with torch.device("meta"):
            lm = Gemma3ForCausalLM(text_cfg)
        lm.model = full.model.language_model
        lm.lm_head = full.lm_head
        lm.lm_head.weight = lm.model.embed_tokens.weight   # tied embeddings
        lm.generation_config = full.generation_config
        del full
        gc.collect()
        return lm

    if model_type in ("gemma3_text", "gemma2", "gemma"):
        return AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, low_cpu_mem_usage=True
        )

    raise ValueError(
        f"Unsupported model_type '{model_type}' for {model_id}. "
        "Supported: gemma3, gemma3_text, gemma2, gemma."
    )


# ============================================================================
# Full TraitGen model
# ============================================================================

class TraitGenModel(nn.Module):
    """BioCLIP (frozen) -> ImageProjector -> Gemma 3 + LoRA."""

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16}
        if cfg.DTYPE not in dtype_map:
            raise ValueError(
                f"DTYPE='{cfg.DTYPE}' is not supported. Use 'fp32' (default) or 'bf16'. "
                "fp16 model weights are intentionally unavailable: Gemma 3 overflows float16."
            )
        self.model_dtype = dtype_map[cfg.DTYPE]

        # ---- BioCLIP (frozen) ----
        print(f"Loading BioCLIP: {cfg.BIOCLIP_MODEL} ...")
        self.vision_encoder = BioCLIPEncoder(cfg.BIOCLIP_MODEL, device=device)
        self.vision_dim = self.vision_encoder.output_dim
        print(f"  BioCLIP output dimension: {self.vision_dim}")

        # ---- Gemma ----
        print(f"Loading Gemma: {cfg.GEMMA_MODEL} (dtype={cfg.DTYPE}) ...")
        base_lm = _load_gemma_text_decoder(cfg.GEMMA_MODEL, self.model_dtype)

        if cfg.GRADIENT_CHECKPOINTING:
            base_lm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        self.hidden_size = base_lm.config.hidden_size
        print(f"  Gemma hidden size: {self.hidden_size}")

        # ---- Tokenizer ----
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.GEMMA_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---- LoRA ----
        lora_config = LoraConfig(
            r=cfg.LORA_R,
            lora_alpha=cfg.LORA_ALPHA,
            target_modules=cfg.LORA_TARGET_MODULES,
            lora_dropout=cfg.LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.gemma = get_peft_model(base_lm, lora_config)
        self.embedding = self.gemma.get_base_model().get_input_embeddings()
        self.gemma = self.gemma.to(device)

        # ---- Projector ----
        self.projector = ImageProjector(self.vision_dim, self.hidden_size).to(device)

        self._verify_frozen_params()

    # ------------------------------------------------------------------
    def _verify_frozen_params(self):
        for name, p in self.vision_encoder.named_parameters():
            if p.requires_grad:
                raise RuntimeError(f"BioCLIP param should be frozen: {name}")

        gemma_base_trainable = sum(
            p.numel() for name, p in self.gemma.named_parameters()
            if p.requires_grad and "lora_" not in name
        )
        if gemma_base_trainable > 0:
            raise RuntimeError(
                f"Gemma has {gemma_base_trainable:,} trainable NON-LoRA parameters. "
                "Base weights should be frozen."
            )

    def count_parameters(self):
        bioclip_total = sum(p.numel() for p in self.vision_encoder.parameters())
        bioclip_trainable = sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)

        gemma_lora = sum(p.numel() for n, p in self.gemma.named_parameters() if "lora_" in n)
        gemma_base = sum(p.numel() for n, p in self.gemma.named_parameters() if "lora_" not in n)
        gemma_base_trainable = sum(
            p.numel() for n, p in self.gemma.named_parameters()
            if p.requires_grad and "lora_" not in n
        )

        projector_total = sum(p.numel() for p in self.projector.parameters())
        projector_trainable = sum(p.numel() for p in self.projector.parameters() if p.requires_grad)

        total = bioclip_total + gemma_base + gemma_lora + projector_total
        trainable = bioclip_trainable + gemma_lora + projector_trainable

        return {
            "total": total,
            "trainable": trainable,
            "trainable_pct": 100.0 * trainable / max(total, 1),
            "bioclip_total": bioclip_total,
            "bioclip_trainable": bioclip_trainable,
            "gemma_base": gemma_base,
            "gemma_base_trainable": gemma_base_trainable,
            "gemma_lora": gemma_lora,
            "projector_total": projector_total,
            "projector_trainable": projector_trainable,
        }

    def trainable_parameters(self):
        for p in self.gemma.parameters():
            if p.requires_grad:
                yield p
        for p in self.projector.parameters():
            if p.requires_grad:
                yield p

    # ------------------------------------------------------------------
    # Visual prefix: [BOS] + projected image tokens + steering prompt
    # ------------------------------------------------------------------
    def _build_prefix(self, image_features: torch.Tensor) -> torch.Tensor:
        B = image_features.size(0)
        emb_dtype = self.embedding.weight.dtype
        dev = image_features.device

        projected = self.projector(image_features).to(emb_dtype)

        parts = []
        if self.tokenizer.bos_token_id is not None:
            bos_ids = torch.full((B, 1), self.tokenizer.bos_token_id,
                                 dtype=torch.long, device=dev)
            parts.append(self.embedding(bos_ids))

        parts.append(projected)

        prompt_ids = self.tokenizer(
            cfg.STEERING_PROMPT, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(dev)
        parts.append(self.embedding(prompt_ids.expand(B, -1)))

        return torch.cat(parts, dim=1)

    # ------------------------------------------------------------------
    # Text batch: EOS is appended to EACH caption BEFORE padding
    # ------------------------------------------------------------------
    def _text_batch(self, captions):
        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id

        seqs = [
            self.tokenizer(
                c, add_special_tokens=False, truncation=True,
                max_length=cfg.MAX_TEXT_TOKENS - 1,
            )["input_ids"] + [eos]
            for c in captions
        ]
        B, T = len(seqs), max(len(s) for s in seqs)

        ids = torch.full((B, T), pad, dtype=torch.long, device=self.device)
        mask = torch.zeros((B, T), dtype=torch.long, device=self.device)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=self.device)
            mask[i, :len(s)] = 1
        return ids, mask

    # ------------------------------------------------------------------
    # Loss (shared by forward / forward_loss_components)
    # ------------------------------------------------------------------
    def _loss(self, image_paths: list, captions: list):
        with torch.no_grad():
            image_features = self.vision_encoder(image_paths)

        prefix = self._build_prefix(image_features)
        P = prefix.size(1)

        ids, mask = self._text_batch(captions)
        B, T = ids.shape

        inputs_embeds = torch.cat([prefix, self.embedding(ids)], dim=1)
        attention_mask = torch.cat(
            [torch.ones(B, P, dtype=torch.long, device=self.device), mask], dim=1
        )

        # Gemma's vocab is ~262k: only compute logits for the last T+1 positions
        # (the +1 is the final prefix position, which predicts caption token 0).
        out = self.gemma(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=T + 1,
        )
        logits = out.logits[:, :-1].float()                     # (B, T, V)

        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            ids.reshape(-1),
            reduction="none",
        ).view(B, T)

        n_tokens = mask.sum()
        loss = (ce * mask).sum() / n_tokens
        return loss, int(n_tokens.item())

    def forward(self, image_paths: list, captions: list) -> torch.Tensor:
        return self._loss(image_paths, captions)[0]

    def forward_loss_components(self, image_paths: list, captions: list):
        """Returns (mean_loss, num_target_tokens) for token-weighted averaging."""
        return self._loss(image_paths, captions)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, image_paths: list, max_new_tokens: int = None) -> list:
        if max_new_tokens is None:
            max_new_tokens = cfg.MAX_NEW_TOKENS

        image_features = self.vision_encoder(image_paths)
        prefix_embeds = self._build_prefix(image_features)
        attention_mask = torch.ones(
            prefix_embeds.shape[:2], dtype=torch.long, device=prefix_embeds.device
        )

        output_ids = self.gemma.generate(
            inputs_embeds=prefix_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------
    def save_checkpoint(self, checkpoint_dir: str, epoch: int, global_step: int,
                        optimizer, scheduler, scaler, loss_history: dict):
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.gemma.save_pretrained(os.path.join(checkpoint_dir, "lora_adapter"))
        self.tokenizer.save_pretrained(os.path.join(checkpoint_dir, "tokenizer"))
        torch.save(self.projector.state_dict(), os.path.join(checkpoint_dir, "projector.pt"))

        training_state = {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "loss_history": loss_history,
        }
        if scaler is not None:
            training_state["scaler"] = scaler.state_dict()
        torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))

        config_snapshot = {
            "gemma_model": cfg.GEMMA_MODEL,
            "bioclip_model": cfg.BIOCLIP_MODEL,
            "lora_r": cfg.LORA_R,
            "lora_alpha": cfg.LORA_ALPHA,
            "lora_dropout": cfg.LORA_DROPOUT,
            "lora_target_modules": cfg.LORA_TARGET_MODULES,
            "max_text_tokens": cfg.MAX_TEXT_TOKENS,
            "max_new_tokens": cfg.MAX_NEW_TOKENS,
            "vision_dim": self.vision_dim,
            "hidden_size": self.hidden_size,
            "dtype": cfg.DTYPE,
            "steering_prompt": cfg.STEERING_PROMPT,
        }
        with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
            json.dump(config_snapshot, f, indent=2)

        print(f"  Checkpoint saved: {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_dir: str, optimizer=None, scheduler=None, scaler=None):
        lora_dir = os.path.join(checkpoint_dir, "lora_adapter")
        if not os.path.isdir(lora_dir):
            raise FileNotFoundError(f"LoRA adapter dir not found: {lora_dir}")

        import safetensors.torch
        st_path = os.path.join(lora_dir, "adapter_model.safetensors")
        if os.path.exists(st_path):
            adapter_weights = safetensors.torch.load_file(st_path, device=str(self.device))
        else:
            bin_path = os.path.join(lora_dir, "adapter_model.bin")
            if not os.path.exists(bin_path):
                raise FileNotFoundError(
                    f"No adapter weights in {lora_dir} "
                    "(expected adapter_model.safetensors or adapter_model.bin)"
                )
            adapter_weights = torch.load(bin_path, map_location=self.device, weights_only=True)
        set_peft_model_state_dict(self.gemma, adapter_weights)

        projector_path = os.path.join(checkpoint_dir, "projector.pt")
        if not os.path.exists(projector_path):
            raise FileNotFoundError(f"Projector weights not found: {projector_path}")
        self.projector.load_state_dict(
            torch.load(projector_path, map_location=self.device, weights_only=True)
        )
        print(f"  Loaded projector from {projector_path}")

        state_path = os.path.join(checkpoint_dir, "training_state.pt")
        if not os.path.exists(state_path):
            print(f"  WARNING: No training_state.pt in {checkpoint_dir} - cannot resume training.")
            return {"epoch": 0, "global_step": 0, "loss_history": {}}

        state = torch.load(state_path, map_location=self.device, weights_only=False)

        if optimizer is not None and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
            print("  Restored optimizer state")
        if scheduler is not None and "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
            print("  Restored scheduler state")
        if scaler is not None and "scaler" in state:
            scaler.load_state_dict(state["scaler"])
            print("  Restored AMP scaler state")

        print(f"  Resumed from epoch {state['epoch']}, global_step {state['global_step']}")
        return {
            "epoch": state["epoch"],
            "global_step": state["global_step"],
            "loss_history": state.get("loss_history", {}),
        }
