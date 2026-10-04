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

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.tokenizer.padding_side = "left"

        base_model.enable_input_require_grads()

        # LoRA — same target modules as Qwen3 (Gemma3 uses identical names)
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.llm = get_peft_model(base_model, peft_config)

        for p in self.llm.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        self.hidden_dim = self.llm.config.hidden_size  # e.g., 1152 for 1B
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
