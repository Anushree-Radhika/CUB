"""
TraitGen model: BioCLIP (frozen) → projector (trainable) → Gemma 3 + LoRA.

Architecture
------------
CUB bird image
    → BioCLIP ViT (frozen) → patch tokens  (B, num_patches, 768)
    → Projector: LayerNorm → Linear → GELU → Linear  →  (B, num_patches, gemma_hidden)
    → prepend [BOS] + projected tokens + steering-prompt embeddings as visual prefix
    → Gemma 3 decoder (base frozen, LoRA trainable)
    → generated attribute description

NOTE on precision (V100)
------------------------
Gemma 3 activations overflow float16.  V100 has no native BF16 tensor cores.
Default: FP32 model weights.  AMP (FP16 autocast) is available but disabled by
default for safety — enable via USE_AMP=1 only after verifying loss stays finite.
"""

import gc
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

import config as cfg


# ============================================================================
# BioCLIP vision encoder (frozen)
# ============================================================================

class BioCLIPEncoder(nn.Module):
    """
    Loads BioCLIP via OpenCLIP and extracts patch-level features (no CLS token).

    The encoder is always in eval mode and wrapped with @torch.no_grad so it
    never receives gradients.  Preprocessing is handled internally.
    """

    def __init__(self, model_name: str = cfg.BIOCLIP_MODEL, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        self.model = self.model.to(device)
        self.model.eval()

        # Freeze everything
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Determine output dimension from the visual backbone
        self._output_dim = self.model.visual.ln_post.normalized_shape[0]

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @torch.no_grad()
    def forward(self, image_paths: list) -> torch.Tensor:
        """
        Parameters
        ----------
        image_paths : list[str]
            Paths to images on disk.

        Returns
        -------
        torch.Tensor
            (batch, num_patches, embed_dim) — patch features without CLS.
        """
        images = [
            self.preprocess(Image.open(p).convert("RGB"))
            for p in image_paths
        ]
        images = torch.stack(images).to(self.device)

        visual = self.model.visual

        # Patch embeddings
        x = visual.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        # CLS token
        cls = visual.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)

        # Positional embeddings
        x = x + visual.positional_embedding.to(x.dtype)

        # Transformer
        x = visual.patch_dropout(x)
        x = visual.ln_pre(x)
        x = visual.transformer(x)
        x = visual.ln_post(x)

        # Remove CLS — return only patch tokens
        x = x[:, 1:, :]
        return x


# ============================================================================
# Image projector: LayerNorm → Linear → GELU → Linear
# ============================================================================

class ImageProjector(nn.Module):
    """
    Projects BioCLIP patch features into Gemma's embedding space.

    Architecture: LayerNorm → Linear → GELU → Linear
    """

    def __init__(self, vision_dim: int, hidden_dim: int):
        super().__init__()
        # Intermediate size: 4x vision_dim (capped to avoid waste)
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
# Gemma decoder loading (handles text-only and multimodal checkpoints)
# ============================================================================

def _load_gemma_text_decoder(model_id: str, dtype: torch.dtype):
    """
    Load a Gemma ForCausalLM model.  Handles:
      - gemma3_text / gemma2 / gemma  → AutoModelForCausalLM directly
      - gemma3 (multimodal 4B/12B/27B) → extract text tower + lm_head
    """
    model_cfg = AutoConfig.from_pretrained(model_id)
    model_type = getattr(model_cfg, "model_type", "")

    if model_type == "gemma3":
        # Multimodal checkpoint: extract text tower
        from transformers import Gemma3ForCausalLM, Gemma3ForConditionalGeneration
        full = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        text_cfg = full.config.text_config
        with torch.device("meta"):
            lm = Gemma3ForCausalLM(text_cfg)
        lm.model = full.model.language_model
        lm.lm_head = full.lm_head
        # Gemma ties input/output embeddings
        lm.lm_head.weight = lm.model.embed_tokens.weight
        lm.generation_config = full.generation_config
        del full
        gc.collect()
        return lm

    if model_type in ("gemma3_text", "gemma2", "gemma"):
        return AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, low_cpu_mem_usage=True
        )

    raise ValueError(
        f"Unsupported model_type '{model_type}' for {model_id}. "
        "Supported: gemma3, gemma3_text, gemma2, gemma."
    )


# ============================================================================
# Full TraitGen model
# ============================================================================

class TraitGenModel(nn.Module):
    """
    BioCLIP (frozen) → ImageProjector (trainable) → Gemma 3 + LoRA (trainable adapters).

    The loss is standard causal-LM cross entropy computed ONLY on target-description
    tokens (prompt and visual-prefix tokens are masked with -100).
    """

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

        # ---- Gemma 3 ----
        print(f"Loading Gemma: {cfg.GEMMA_MODEL} (dtype={cfg.DTYPE}) ...")
        base_lm = _load_gemma_text_decoder(cfg.GEMMA_MODEL, self.model_dtype)

        # Gradient checkpointing (before LoRA so LoRA modules also benefit)
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

        # Keep the embedding layer reference for building prefix embeddings
        self.embedding = self.gemma.get_base_model().get_input_embeddings()

        # Move Gemma to device
        self.gemma = self.gemma.to(device)

        # ---- Image projector (trainable) ----
        self.projector = ImageProjector(self.vision_dim, self.hidden_size).to(device)

        # ---- Freeze: BioCLIP is already frozen; freeze Gemma base weights ----
        # PEFT's get_peft_model already marks base params as requires_grad=False
        # and LoRA params as requires_grad=True. Verify this:
        self._verify_frozen_params()

    # ------------------------------------------------------------------
    # Freeze verification
    # ------------------------------------------------------------------
    def _verify_frozen_params(self):
        """Ensure only LoRA + projector params have requires_grad=True."""
        for name, p in self.vision_encoder.named_parameters():
            if p.requires_grad:
                raise RuntimeError(f"BioCLIP param should be frozen: {name}")

        # Count base vs LoRA params in Gemma
        gemma_base_trainable = 0
        for name, p in self.gemma.named_parameters():
            if p.requires_grad and "lora_" not in name:
                gemma_base_trainable += p.numel()
        if gemma_base_trainable > 0:
            raise RuntimeError(
                f"Gemma has {gemma_base_trainable:,} trainable NON-LoRA parameters. "
                "Base weights should be frozen."
            )

    # ------------------------------------------------------------------
    # Parameter counting
    # ------------------------------------------------------------------
    def count_parameters(self):
        """Return a dict with parameter breakdowns."""
        bioclip_total = sum(p.numel() for p in self.vision_encoder.parameters())
        bioclip_trainable = sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)

        gemma_lora = 0
        gemma_base = 0
        for name, p in self.gemma.named_parameters():
            if "lora_" in name:
                gemma_lora += p.numel()
            else:
                gemma_base += p.numel()
        gemma_base_trainable = sum(
            p.numel() for name, p in self.gemma.named_parameters()
            if p.requires_grad and "lora_" not in name
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
        """Yield all parameters that should be optimised."""
        for p in self.gemma.parameters():
            if p.requires_grad:
                yield p
        for p in self.projector.parameters():
            if p.requires_grad:
                yield p

    # ------------------------------------------------------------------
    # Visual prefix construction
    # ------------------------------------------------------------------
    def _build_prefix(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Build the visual prefix: [BOS] + projected_image_tokens + steering_prompt_embeddings.

        Returns: (batch, prefix_len, hidden_size)
        """
        B = image_features.size(0)
        emb_dtype = self.embedding.weight.dtype

        # Project image features
        projected = self.projector(image_features).to(emb_dtype)  # (B, num_patches, hidden)

        parts = []

        # BOS token
        if self.tokenizer.bos_token_id is not None:
            bos_ids = torch.full((B, 1), self.tokenizer.bos_token_id,
                                 dtype=torch.long, device=image_features.device)
            parts.append(self.embedding(bos_ids))

        # Projected image tokens
        parts.append(projected)

        # Steering prompt
        prompt_ids = self.tokenizer(
            cfg.STEERING_PROMPT, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(image_features.device)
        parts.append(self.embedding(prompt_ids.expand(B, -1)))

        return torch.cat(parts, dim=1)  # (B, prefix_len, hidden)

    # ------------------------------------------------------------------
    # Forward: compute loss
    # ------------------------------------------------------------------
    def forward(self, image_paths: list, captions: list) -> torch.Tensor:
        """
        Compute causal-LM loss on target-description tokens only.

        The sequence is: [visual_prefix] [caption_tokens]
        Labels are -100 for the visual prefix and prompt, and the actual
        token IDs only for the target description.
        """
        # 1. Vision encoding (frozen, no grad)
        with torch.no_grad():
            image_features = self.vision_encoder(image_paths)  # (B, patches, vision_dim)

        # 2. Build visual prefix
        prefix_embeds = self._build_prefix(image_features)  # (B, P, hidden)
        P = prefix_embeds.size(1)

        # 3. Tokenize captions
        tokenized = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=cfg.MAX_TEXT_TOKENS,
            return_tensors="pt",
            add_special_tokens=False,
        )
        caption_ids = tokenized["input_ids"].to(self.device)         # (B, T)
        caption_mask = tokenized["attention_mask"].to(self.device)   # (B, T)

        # Add EOS at the end of each caption
        B, T = caption_ids.shape
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        # Find the position after the last real token for each sample
        # and replace the first pad with EOS (or append if no padding)
        eos_ids = torch.full((B, 1), eos_id, dtype=torch.long, device=self.device)
        eos_mask = torch.ones((B, 1), dtype=torch.long, device=self.device)
        caption_ids = torch.cat([caption_ids, eos_ids], dim=1)       # (B, T+1)
        caption_mask = torch.cat([caption_mask, eos_mask], dim=1)    # (B, T+1)
        T = T + 1

        # 4. Build caption embeddings
        caption_embeds = self.embedding(caption_ids)  # (B, T, hidden)

        # 5. Concatenate: [prefix_embeds | caption_embeds]
        inputs_embeds = torch.cat([prefix_embeds, caption_embeds], dim=1)  # (B, P+T, hidden)
        attention_mask = torch.cat([
            torch.ones(B, P, dtype=torch.long, device=self.device),
            caption_mask,
        ], dim=1)  # (B, P+T)

        # 6. Build labels
        # For causal LM: labels[i] = the token that position i should predict (= token at i+1).
        # We want loss ONLY on caption tokens. The prefix predicts the first caption token,
        # the first caption token predicts the second, etc.
        #
        # Position layout: [prefix_0 ... prefix_{P-1}] [cap_0 ... cap_{T-1}]
        # For standard HF causal LM, labels are shifted internally:
        #   loss_at_position_i = CE(logits[i], labels[i])
        #   but HF shifts: loss = CE(logits[:-1], labels[1:])
        # So we need labels = [-100]*P + caption_ids
        # Then HF's internal shift means:
        #   logits[P-1] predicts labels[P] = cap_0  ← first caption token (this is correct)
        #   logits[P]   predicts labels[P+1] = cap_1
        #   ...
        # Mask padding in caption with -100
        labels = caption_ids.clone()
        labels[caption_mask == 0] = -100

        # Prepend -100 for the prefix positions
        prefix_labels = torch.full((B, P), -100, dtype=torch.long, device=self.device)
        labels = torch.cat([prefix_labels, labels], dim=1)  # (B, P+T)

        # 7. Forward through Gemma
        outputs = self.gemma(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )

        return outputs.loss

    # ------------------------------------------------------------------
    # Forward: compute loss returning sum & count for proper averaging
    # ------------------------------------------------------------------
    def forward_loss_components(self, image_paths: list, captions: list):
        """
        Like forward() but returns (loss_sum, num_tokens) so the caller
        can average across batches correctly.
        """
        with torch.no_grad():
            image_features = self.vision_encoder(image_paths)

        prefix_embeds = self._build_prefix(image_features)
        P = prefix_embeds.size(1)

        tokenized = self.tokenizer(
            captions, padding=True, truncation=True,
            max_length=cfg.MAX_TEXT_TOKENS, return_tensors="pt",
            add_special_tokens=False,
        )
        caption_ids = tokenized["input_ids"].to(self.device)
        caption_mask = tokenized["attention_mask"].to(self.device)

        B, T = caption_ids.shape
        eos_ids = torch.full((B, 1), self.tokenizer.eos_token_id, dtype=torch.long, device=self.device)
        eos_mask = torch.ones((B, 1), dtype=torch.long, device=self.device)
        caption_ids = torch.cat([caption_ids, eos_ids], dim=1)
        caption_mask = torch.cat([caption_mask, eos_mask], dim=1)
        T = T + 1

        caption_embeds = self.embedding(caption_ids)
        inputs_embeds = torch.cat([prefix_embeds, caption_embeds], dim=1)
        attention_mask = torch.cat([
            torch.ones(B, P, dtype=torch.long, device=self.device),
            caption_mask,
        ], dim=1)

        labels = caption_ids.clone()
        labels[caption_mask == 0] = -100
        prefix_labels = torch.full((B, P), -100, dtype=torch.long, device=self.device)
        labels = torch.cat([prefix_labels, labels], dim=1)

        outputs = self.gemma(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )

        # Count non-masked tokens for proper averaging
        # HF shifts labels internally: effective labels are labels[:, 1:]
        shifted_labels = labels[:, 1:]
        num_tokens = (shifted_labels != -100).sum().item()

        return outputs.loss, num_tokens

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, image_paths: list, max_new_tokens: int = None) -> list:
        """
        Generate attribute descriptions for the given images.

        Returns a list of strings, one per image.
        """
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
        )

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------
    def save_checkpoint(self, checkpoint_dir: str, epoch: int, global_step: int,
                        optimizer, scheduler, scaler, loss_history: dict):
        """Save a complete checkpoint that allows training to resume."""
        os.makedirs(checkpoint_dir, exist_ok=True)

        # LoRA adapter
        lora_dir = os.path.join(checkpoint_dir, "lora_adapter")
        self.gemma.save_pretrained(lora_dir)

        # Tokenizer (needed for inference)
        self.tokenizer.save_pretrained(os.path.join(checkpoint_dir, "tokenizer"))

        # Projector weights
        torch.save(self.projector.state_dict(),
                    os.path.join(checkpoint_dir, "projector.pt"))

        # Training state
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

        # Config snapshot
        import json
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
        """
        Load a checkpoint for resuming training or inference.

        Returns
        -------
        dict with 'epoch', 'global_step', 'loss_history' (and optionally restores
        optimizer/scheduler/scaler in-place).
        """
        from peft import PeftModel

        # Load LoRA adapter
        lora_dir = os.path.join(checkpoint_dir, "lora_adapter")
        if not os.path.isdir(lora_dir):
            raise FileNotFoundError(f"LoRA adapter dir not found: {lora_dir}")

        # PEFT: load adapter weights into the existing PeftModel
        # We need to load the adapter weights manually since the model is already
        # wrapped with PEFT.
        import safetensors.torch
        adapter_path = os.path.join(lora_dir, "adapter_model.safetensors")
        if os.path.exists(adapter_path):
            adapter_weights = safetensors.torch.load_file(adapter_path, device=str(self.device))
        else:
            bin_path = os.path.join(lora_dir, "adapter_model.bin")
            if not os.path.exists(bin_path):
                raise FileNotFoundError(
                    f"No adapter weights found in {lora_dir}. "
                    "Expected adapter_model.safetensors or adapter_model.bin"
                )
            adapter_weights = torch.load(bin_path, map_location=self.device, weights_only=True)

        # Load into model
        result = self.gemma.load_state_dict(adapter_weights, strict=False)
        # We expect missing keys (base model weights are not in the adapter file)
        unexpected = result.unexpected_keys
        if unexpected:
            raise RuntimeError(
                f"Unexpected keys when loading LoRA adapter: {unexpected[:5]}... "
                "Was this checkpoint created with a different model or LoRA config?"
            )

        # Load projector
        projector_path = os.path.join(checkpoint_dir, "projector.pt")
        if not os.path.exists(projector_path):
            raise FileNotFoundError(f"Projector weights not found: {projector_path}")
        proj_state = torch.load(projector_path, map_location=self.device, weights_only=True)
        self.projector.load_state_dict(proj_state)
        print(f"  Loaded projector from {projector_path}")

        # Load training state
        state_path = os.path.join(checkpoint_dir, "training_state.pt")
        if not os.path.exists(state_path):
            print(f"  WARNING: No training_state.pt found in {checkpoint_dir} — cannot resume training.")
            return {"epoch": 0, "global_step": 0, "loss_history": {}}

        training_state = torch.load(state_path, map_location=self.device, weights_only=False)

        if optimizer is not None and "optimizer" in training_state:
            optimizer.load_state_dict(training_state["optimizer"])
            print(f"  Restored optimizer state")

        if scheduler is not None and "scheduler" in training_state:
            scheduler.load_state_dict(training_state["scheduler"])
            print(f"  Restored scheduler state")

        if scaler is not None and "scaler" in training_state:
            scaler.load_state_dict(training_state["scaler"])
            print(f"  Restored AMP scaler state")

        print(f"  Resumed from epoch {training_state['epoch']}, "
              f"global_step {training_state['global_step']}")

        return {
            "epoch": training_state["epoch"],
            "global_step": training_state["global_step"],
            "loss_history": training_state.get("loss_history", {}),
        }
