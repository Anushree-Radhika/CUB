import torch
import torch.nn as nn

from transformers import (GPT2LMHeadModel,GPT2Tokenizer,)
import torch.nn.functional as F

from peft import (LoraConfig,TaskType,get_peft_model,)
MAX_TEXT_LEN = 800

NORMAL_TOKEN_WEIGHT = 2.0
SPECIES_TOKEN_WEIGHT = 5.0
EOS_TOKEN_WEIGHT = 5.0

MAX_NEW_TOKENS = 300
GEN_DO_SAMPLE = False
GEN_TEMPERATURE = 0.0
GEN_TOP_P = 0.9
GEN_REPETITION_PENALTY = 1.0


class GPT2Decoder(nn.Module):
    """
    GPT-2 Decoder with LoRA.

    This module is responsible for:
        - Loading the pretrained GPT-2 language model.
        - Initializing the GPT-2 tokenizer.
        - Configuring padding for batch processing.
        - Applying LoRA for parameter-efficient fine-tuning.
        - Producing language model outputs.
    """

    def __init__(self, args):
        super().__init__()

        self.args = args

        base_gpt2 = GPT2LMHeadModel.from_pretrained(args.decoder_model)
        self.tokenizer = GPT2Tokenizer.from_pretrained(args.decoder_model)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            base_gpt2.config.pad_token_id = self.tokenizer.pad_token_id

        self.tokenizer.padding_side = "left"

        # LoRA configuration
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "c_attn",
                "c_proj",
                "mlp.c_fc",
                "mlp.c_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.llm = get_peft_model(base_gpt2, peft_config)

        self.hidden_dim = self.llm.config.n_embd
        self.embedding = self.llm.get_input_embeddings()
    
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        species_mask : torch.Tensor = None,
        image_len = 196,
        prompt_len = 8):
        
        device = self.device
        batch_size, text_len = species_mask.shape
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,)

        logits = outputs.logits

        # calculate species mask        
        shift_logits = logits[:, :-1]
        shift_labels = labels[:, 1:]
        #shift_logits = logits[:, :-1, :].contiguous()
        #shift_labels = labels[:, 1:].contiguous()
        
        image_weights = torch.zeros(
            batch_size,
            image_len,
            device=device,
        )

        prompt_weights = torch.zeros(
            batch_size,
            prompt_len,
            device=device,
        )

        text_weights = torch.full(
            (batch_size, text_len),
            NORMAL_TOKEN_WEIGHT,
            device=device,
        )

        eos_mask = (labels == self.tokenizer.eos_token_id)[:,image_len+prompt_len:,]
        text_weights[species_mask] = SPECIES_TOKEN_WEIGHT
        

        text_weights[eos_mask] = EOS_TOKEN_WEIGHT

        weights = torch.cat(
            [
                image_weights,
                prompt_weights,
                text_weights,
            ],
            dim=1,
        )

        shift_weights = weights[:, 1:]

        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        ).view_as(shift_labels)
        #print(image_len,prompt_len,text_len)
        #print(labels.shape)
        #print(loss.shape)
        #print(shift_weights.shape)
        loss = loss * shift_weights

        mask = shift_labels != -100

        return outputs,loss[mask].sum() / shift_weights[mask].sum()
