import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor
from peft import LoraConfig, TaskType, get_peft_model

MAX_TEXT_LEN = 500
NORMAL_TOKEN_WEIGHT = 2.0
SPECIES_TOKEN_WEIGHT = 5.0
EOS_TOKEN_WEIGHT = 5.0

MAX_NEW_TOKENS = 300
GEN_DO_SAMPLE = False
GEN_TEMPERATURE = 0.0
GEN_TOP_P = 0.9
GEN_REPETITION_PENALTY = 1.0

# ─── ASCII-only vocabulary filter ──────────────────────────────────────────────
# Gemma3's 262K vocabulary includes CJK, emoji, code tokens, etc.  Our CUB trait
# dataset only ever produces ASCII-printable English text.  This processor masks
# every token whose decoded form contains non-ASCII bytes, preventing garbled
# punctuation (e.g. "Leg?brown", ";browncolor.") and non-English tokens.
# ───────────────────────────────────────────────────────────────────────────────

_ASCII_PRINTABLE_RE = re.compile(r'^[\x20-\x7E]*$')   # space through tilde

class AllowedTokensLogitsProcessor(LogitsProcessor):
    """
    At each decoding step, set logit = -inf for every token NOT in the
    pre-computed whitelist.  The whitelist is built once from the tokenizer
    vocabulary, keeping only tokens whose decoded string is purely ASCII
    printable (letters, digits, basic punctuation, space).
    """

    def __init__(self, tokenizer):
        super().__init__()
        vocab = tokenizer.get_vocab()           # str -> int
        allowed_ids = set()
        for token_str, token_id in vocab.items():
            decoded = tokenizer.decode([token_id])
            if _ASCII_PRINTABLE_RE.match(decoded):
                allowed_ids.add(token_id)

        # Always allow eos/bos/pad so the model can stop
        for special_id in [tokenizer.eos_token_id, tokenizer.bos_token_id,
                           tokenizer.pad_token_id]:
            if special_id is not None:
                allowed_ids.add(special_id)

        self._allowed_ids = torch.tensor(sorted(allowed_ids), dtype=torch.long)

    def __call__(self, input_ids, scores):
        # The tokenizer might contain token IDs larger than the model's actual LM head
        # output size. We filter them here to prevent CUDA index out of bounds errors.
        vocab_size = scores.size(-1)
        valid_ids = self._allowed_ids[self._allowed_ids < vocab_size].to(scores.device)
        
        # Build a mask over the full vocab dimension
        mask = torch.full_like(scores, -float('inf'))
        mask[:, valid_ids] = 0.0
        return scores + mask


class Gemma3Decoder(nn.Module):
    """
    Gemma3 Decoder with LoRA.
    Drop-in replacement following the same interface as Qwen3Decoder.
    
    args.decoder_model should be e.g. "google/gemma-3-1b-pt"
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        load_kwargs = dict(
            dtype=torch.float16,
            attn_implementation="sdpa",
        )
        if getattr(args, "load_in_4bit", False):
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            load_kwargs["device_map"] = {"": torch.cuda.current_device()}

        base_model = AutoModelForCausalLM.from_pretrained(
            args.decoder_model, **load_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(args.decoder_model)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.tokenizer.padding_side = "left"

        base_model.enable_input_require_grads()

        # LoRA — same target modules as Qwen3 (Gemma3 uses identical names)
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=32,
            target_modules=[
                "q_proj", "v_proj",
            ],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.llm = get_peft_model(base_model, peft_config)

        for p in self.llm.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        self.hidden_dim = self.llm.config.hidden_size  # e.g., 1152 for 1B
        self.embedding = self.llm.get_input_embeddings()

        # Build the ASCII-only vocabulary filter once (no trainable params)
        self.allowed_token_processor = AllowedTokensLogitsProcessor(self.tokenizer)
        print(f"[Gemma3Decoder] Allowed ASCII token IDs: "
              f"{len(self.allowed_token_processor._allowed_ids)} / "
              f"{len(self.tokenizer.get_vocab())}")

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        species_mask: torch.Tensor = None,
        image_len=196,
        prompt_len=8,
    ):
        device = self.device
        batch_size, text_len = species_mask.shape

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.logits

        shift_logits = logits[:, :-1]
        shift_labels = labels[:, 1:]

        image_weights = torch.zeros(batch_size, image_len, device=device)
        prompt_weights = torch.zeros(batch_size, prompt_len, device=device)
        text_weights = torch.full(
            (batch_size, text_len), NORMAL_TOKEN_WEIGHT, device=device,
        )

        eos_mask = (labels == self.tokenizer.eos_token_id)[:, image_len + prompt_len:]
        text_weights[species_mask.bool()] = SPECIES_TOKEN_WEIGHT
        text_weights[eos_mask] = EOS_TOKEN_WEIGHT

        weights = torch.cat([image_weights, prompt_weights, text_weights], dim=1)
        shift_weights = weights[:, 1:]

        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)).float(),
            shift_labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)

        loss = loss * shift_weights
        mask = shift_labels != -100

        return outputs, loss[mask].sum() / shift_weights[mask].sum()