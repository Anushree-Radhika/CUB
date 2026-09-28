import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model


STEERING_PROMPT = "Describe this bird species correctly."

MAX_TEXT_LEN = 800

NORMAL_TOKEN_WEIGHT = 2.0
SPECIES_TOKEN_WEIGHT = 5.0
EOS_TOKEN_WEIGHT = 5.0

MAX_NEW_TOKENS = 500
GEN_DO_SAMPLE = False
GEN_TEMPERATURE = 0.0
GEN_TOP_P = 0.9
GEN_REPETITION_PENALTY = 1.0


# fp16 by default: V100 / T4 / P100 have no native bf16. Gemma was trained in
# bf16, so if the first-batch loss comes back NaN/inf, rerun with
# DECODER_DTYPE=fp32 (needs ~4 bytes/param: fine for gemma-2-2b, not for 9b).
_DTYPES = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
DECODER_DTYPE = _DTYPES[os.environ.get("DECODER_DTYPE", "fp16")]


class GemmaDecoder(nn.Module):
    def __init__(self, model, vision_dim=768):
        """
        model: HF repo id, e.g. "google/gemma-2-2b" or "google/gemma-2-9b".
               Gemma checkpoints are gated on HF — you need `huggingface-cli login`
               with a token from an account that accepted the license, or pass
               token=... to from_pretrained below.
        vision_dim: BioCLIP's patch embedding dim (768 for the ViT-B/16 backbone
                    used in a_bioclip_VE.py). Only needs to change if you swap
                    the vision encoder too.
        """
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.gemma = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=DECODER_DTYPE,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # grab the embedding layer BEFORE wrapping in LoRA, same as the GPT2
        # version — this reference stays valid after get_peft_model since
        # embed_tokens isn't in target_modules below.
        self.embedding = self.gemma.get_input_embeddings()

        hidden_dim = self.gemma.config.hidden_size
        self.image_projection = nn.Linear(vision_dim, hidden_dim)

        # Gradient checkpointing keeps activation memory manageable (Gemma's
        # 256k-token vocab makes the logits large). It only runs in train mode,
        # so evaluation and generate() are unaffected. Base weights stay in the
        # dtype they were loaded in; get_peft_model freezes them and keeps the
        # LoRA adapter weights in fp32, which is what fp16 grad scaling needs.
        self.gemma.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        lora_config = LoraConfig(
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

        self.gemma = get_peft_model(self.gemma, lora_config)

    def _build_projection(self, image_features):
        if self.image_projection is None:
            vision_dim = image_features.shape[-1]
            text_dim = self.embedding.embedding_dim

            self.image_projection = nn.Linear(vision_dim, text_dim).to(image_features.device)

    def _prepare_captions(self, captions):
        device = self.embedding.weight.device
        eos_id = self.tokenizer.eos_token_id

        attribute_ids = []
        species_ids = []

        for caption in captions:
            marker = "This species is likely"

            if marker in caption:
                attributes, species = caption.split(marker, 1)
                species = marker + species
            else:
                attributes = caption
                species = ""

            attr = self.tokenizer(attributes, add_special_tokens=False, truncation=False)["input_ids"]
            spec = self.tokenizer(species, add_special_tokens=False, truncation=False)["input_ids"]

            available = MAX_TEXT_LEN - len(spec) - 1

            if available < 0:
                spec = spec[:MAX_TEXT_LEN - 1]
                available = 0

            attr = attr[:available]

            attribute_ids.append(attr)
            species_ids.append(spec)

        sequences = []
        species_masks = []
        eos_masks = []

        max_len = 0

        for attr, spec in zip(attribute_ids, species_ids):
            sequence = attr + spec + [eos_id]

            species_mask = [0] * len(attr) + [1] * len(spec) + [0]
            eos_mask = [0] * (len(attr) + len(spec)) + [1]

            sequences.append(sequence)
            species_masks.append(species_mask)
            eos_masks.append(eos_mask)

            max_len = max(max_len, len(sequence))

        batch_size = len(sequences)
        pad_id = self.tokenizer.pad_token_id

        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        species_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
        eos_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)

        for i, sequence in enumerate(sequences):
            n = len(sequence)

            input_ids[i, :n] = torch.tensor(sequence, dtype=torch.long, device=device)
            attention_mask[i, :n] = 1
            species_mask[i, :n] = torch.tensor(species_masks[i], dtype=torch.bool, device=device)
            eos_mask[i, :n] = torch.tensor(eos_masks[i], dtype=torch.bool, device=device)

        return input_ids, attention_mask, species_mask, eos_mask

    def _prompt_embeddings(self, batch_size, device):
        tokens = self.tokenizer(STEERING_PROMPT, add_special_tokens=False, return_tensors="pt")

        input_ids = tokens["input_ids"].to(device)
        input_ids = input_ids.expand(batch_size, -1)

        return self.embedding(input_ids)

    def forward(self, image_features, captions):
        device = image_features.device

        self._build_projection(image_features)

        # image_projection is a plain fp32 nn.Linear; the Gemma embedding table
        # is fp16 (see __init__). Match dtypes before concatenating or torch.cat
        # raises — this cast is the fix, not a no-op.
        image_features = self.image_projection(image_features).to(self.embedding.weight.dtype)

        batch_size = image_features.size(0)

        prompt_embeddings = self._prompt_embeddings(batch_size, device)

        input_ids, text_attention, species_mask, eos_mask = self._prepare_captions(captions)

        text_embeddings = self.embedding(input_ids)

        inputs_embeds = torch.cat([image_features, prompt_embeddings, text_embeddings], dim=1)

        image_len = image_features.size(1)
        prompt_len = prompt_embeddings.size(1)
        text_len = text_embeddings.size(1)

        image_attention = torch.ones(batch_size, image_len, dtype=torch.long, device=device)
        prompt_attention = torch.ones(batch_size, prompt_len, dtype=torch.long, device=device)

        attention_mask = torch.cat([image_attention, prompt_attention, text_attention], dim=1)

        labels = torch.full(attention_mask.shape, -100, dtype=torch.long, device=device)
        labels[:, image_len + prompt_len:] = input_ids

        outputs = self.gemma(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        logits = outputs.logits

        shift_logits = logits[:, :-1]
        shift_labels = labels[:, 1:]

        image_weights = torch.zeros(batch_size, image_len, device=device)
        prompt_weights = torch.zeros(batch_size, prompt_len, device=device)

        text_weights = torch.full((batch_size, text_len), NORMAL_TOKEN_WEIGHT, device=device)
        text_weights[species_mask] = SPECIES_TOKEN_WEIGHT
        text_weights[eos_mask] = EOS_TOKEN_WEIGHT

        weights = torch.cat([image_weights, prompt_weights, text_weights], dim=1)
        shift_weights = weights[:, 1:]

        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        ).view_as(shift_labels)

        loss = loss * shift_weights

        mask = shift_labels != -100

        return loss[mask].sum() / shift_weights[mask].sum()

    @torch.no_grad()
    def generate(self, image_features):
        device = image_features.device

        self._build_projection(image_features)

        image_features = self.image_projection(image_features).to(self.embedding.weight.dtype)

        batch_size = image_features.size(0)

        prompt_embeddings = self._prompt_embeddings(batch_size, device)

        inputs_embeds = torch.cat([image_features, prompt_embeddings], dim=1)

        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

        gen_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=GEN_DO_SAMPLE,
            repetition_penalty=GEN_REPETITION_PENALTY,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if GEN_DO_SAMPLE:  # temperature/top_p are only valid when sampling
            gen_kwargs.update(temperature=GEN_TEMPERATURE, top_p=GEN_TOP_P)

        output_ids = self.gemma.generate(**gen_kwargs)

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
