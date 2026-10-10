import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from transformers import AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from peft import LoraConfig, get_peft_model

_ASCII_PRINTABLE_RE = re.compile(r'^[\x20-\x7E]*$')

class AllowedTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, device="cuda"):
        super().__init__()
        vocab = tokenizer.get_vocab()
        allowed_ids = set()
        for token_str, token_id in vocab.items():
            decoded = tokenizer.decode([token_id])
            if _ASCII_PRINTABLE_RE.match(decoded):
                allowed_ids.add(token_id)
        for special_id in [tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.pad_token_id]:
            if special_id is not None:
                allowed_ids.add(special_id)
        self._allowed_ids = torch.tensor(sorted(allowed_ids), dtype=torch.long, device=device)
        self._valid_ids = None

    def __call__(self, input_ids, scores):
        if self._valid_ids is None:
            vocab_size = scores.size(-1)
            self._valid_ids = self._allowed_ids[self._allowed_ids < vocab_size].to(scores.device)

        mask = torch.full_like(scores, -float('inf'))
        mask[:, self._valid_ids] = 0.0
        return scores + mask


class GemmaTraitGen(nn.Module):
    def __init__(self, clip_id="hf-hub:imageomics/bioclip", lm_id="google/gemma-3-1b-pt", device="cuda"):
        super().__init__()
        self.device = device
        
        # ------------------------------------------------------------------ encoder (BioCLIP + LoRA)
        clip, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(clip_id)
        self.visual = clip.visual
        self.visual.output_tokens = True                     # forward returns (pooled, patch_tokens)
        self.visual = self.visual.to(device, dtype=torch.float16)
        
        for p in self.visual.parameters():
            p.requires_grad = False

        # open_clip's attention is nn.MultiheadAttention (fused qkv), so q/v LoRA can't be targeted
        enc_cfg = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.1, bias="none",
            target_modules=r".*resblocks\.(6|7|8|9|10|11)\.mlp\.(c_fc|c_proj)",
        )
        self.visual = get_peft_model(self.visual, enc_cfg)
        
        # ------------------------------------------------------------------ decoder (Gemma 3 + LoRA)
        self.lm = AutoModelForCausalLM.from_pretrained(
            lm_id, torch_dtype=torch.float16, attn_implementation="sdpa"
        ).to(device)
        
        dec_cfg = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.lm = get_peft_model(self.lm, dec_cfg)
        
        # ------------------------------------------------------------------ projector
        vis_dim = 768             # BioCLIP ViT-B/16 token dim
        self.n_img_tokens = 49    # 14x14 patches pooled to 7x7
        
        hidden = self.lm.config.hidden_size
        self.projector = nn.Sequential(nn.Linear(vis_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden)).to(device)


    def encode_images(self, pixels):
        vis = self.visual.module if hasattr(self.visual, "module") else self.visual
        proj = self.projector.module if hasattr(self.projector, "module") else self.projector
        
        _, tokens = vis(pixels.to(self.device, dtype=torch.float16))               # (B, 196, 768)
        B, N, D = tokens.shape
        s = int(math.sqrt(N))
        grid = tokens.transpose(1, 2).reshape(B, D, s, s).float()
        side = int(math.sqrt(self.n_img_tokens))
        pooled = F.adaptive_avg_pool2d(grid, side).flatten(2).transpose(1, 2)  # (B, 49, 768)
        return proj(pooled).to(torch.float16)                                 # (B, 49, H)


    def forward(self, pixels, input_ids, attn, labels, species_mask, eos_token_id):
        img = self.encode_images(pixels)
        
        unwrapped_lm = self.lm.module if hasattr(self.lm, "module") else self.lm
        txt = unwrapped_lm.get_input_embeddings()(input_ids)
        emb = torch.cat([img, txt], dim=1)
        
        mask = torch.cat([torch.ones(img.shape[:2], device=self.device, dtype=attn.dtype), attn], 1)
        ignore = torch.full(img.shape[:2], -100, device=self.device, dtype=labels.dtype)
        full_labels = torch.cat([ignore, labels], 1)

        outputs = self.lm(inputs_embeds=emb, attention_mask=mask, return_dict=True)
        shift_logits = outputs.logits[:, :-1]
        shift_labels = full_labels[:, 1:]

        # Custom loss weights (normal=2.0, eos=5.0, species=5.0)
        weights = torch.full_like(shift_labels, 2.0, dtype=torch.float)
        weights[shift_labels == eos_token_id] = 5.0

        full_species_mask = torch.cat([torch.zeros(img.shape[:2], device=self.device, dtype=torch.bool), species_mask], 1)
        weights[full_species_mask[:, 1:]] = 5.0

        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)).float(),
            shift_labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)

        loss = loss * weights
        valid = shift_labels != -100
        return loss[valid].sum() / weights[valid].sum()


    @torch.no_grad()
    def generate_texts(self, pixels, tokenizer, max_new_tokens, logits_processor=None):
        img = self.encode_images(pixels)
        mask = torch.ones(img.shape[:2], device=self.device, dtype=torch.long)
        
        unwrapped_lm = self.lm.module if hasattr(self.lm, "module") else self.lm
        
        out = unwrapped_lm.generate(
            inputs_embeds=img, attention_mask=mask, max_new_tokens=max_new_tokens,
            do_sample=False, num_beams=1, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            logits_processor=LogitsProcessorList([logits_processor]) if logits_processor else None,
        )
        return tokenizer.batch_decode(out, skip_special_tokens=True)

    def print_trainable_parameters(self):
        self.visual.print_trainable_parameters()
        self.lm.print_trainable_parameters()
