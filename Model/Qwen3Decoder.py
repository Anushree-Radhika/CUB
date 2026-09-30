import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

MAX_TEXT_LEN = 800

NORMAL_TOKEN_WEIGHT = 2.0
SPECIES_TOKEN_WEIGHT = 5.0
EOS_TOKEN_WEIGHT = 5.0

MAX_NEW_TOKENS = 300
GEN_DO_SAMPLE = False
GEN_TEMPERATURE = 0.0
GEN_TOP_P = 0.9
GEN_REPETITION_PENALTY = 1.0


class Qwen3Decoder(nn.Module):
    """
    Qwen3-1.7B Decoder with LoRA.

    This module is responsible for:
        - Loading the pretrained Qwen3 language model.
        - Initializing the Qwen3 tokenizer.
        - Configuring padding for batch processing.
        - Applying LoRA for parameter-efficient fine-tuning.
        - Producing language model outputs.

    args.decoder_model should be e.g. "Qwen/Qwen3-1.7B" (or "Qwen/Qwen3-1.7B-Base").
    """

    def __init__(self, args):
        super().__init__()

        self.args = args

        # Tesla V100 (compute capability 7.0): no native bf16, no FlashAttention-2.
        # Use fp16 weights + SDPA attention. Qwen3-1.7B in fp16 is ~8 GB, which fits
        # easily on a 32 GB V100 with room for activations, so 4-bit is off by default
        # (set args.load_in_4bit = True only if you need extra headroom).
        load_kwargs = dict(
            torch_dtype=torch.float16,
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

        # Qwen3 already defines a pad token ("<|endoftext|>"), but keep a fallback.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base_model.config.pad_token_id = self.tokenizer.pad_token_id

        self.tokenizer.padding_side = "left"

        # Needed for gradient checkpointing + LoRA (inputs_embeds path)
        base_model.enable_input_require_grads()
        # Uncomment to save memory at the cost of speed:
        # base_model.gradient_checkpointing_enable()

        # LoRA configuration: Qwen3 uses separate q/k/v/o and gate/up/down projections
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.llm = get_peft_model(base_model, peft_config)

        # fp16 training: keep trainable (LoRA) weights in fp32 so GradScaler can
        # unscale their grads; the frozen base stays fp16 / 4-bit.
        for p in self.llm.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        self.hidden_dim = self.llm.config.hidden_size  # 2560 for Qwen3-1.7B
        self.embedding = self.llm.get_input_embeddings()

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

        # Labels are NOT passed to the model: we compute our own weighted loss below,
        # so letting HF compute its loss too would just waste memory (vocab is ~152k).
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
            (batch_size, text_len),
            NORMAL_TOKEN_WEIGHT,
            device=device,
        )

        eos_mask = (labels == self.tokenizer.eos_token_id)[:, image_len + prompt_len:]
        text_weights[species_mask.bool()] = SPECIES_TOKEN_WEIGHT
        text_weights[eos_mask] = EOS_TOKEN_WEIGHT

        weights = torch.cat([image_weights, prompt_weights, text_weights], dim=1)
        shift_weights = weights[:, 1:]

        # Upcast to fp32 for a numerically stable cross-entropy (model runs in bf16)
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)).float(),
            shift_labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)

        loss = loss * shift_weights

        mask = shift_labels != -100

        return outputs, loss[mask].sum() / shift_weights[mask].sum()