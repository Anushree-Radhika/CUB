import torch
from transformers import AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, TaskType, get_peft_model

model_id = "google/gemma-3-1b-pt"
config = AutoConfig.from_pretrained(model_id)

with torch.device("meta"):
    base_model = AutoModelForCausalLM.from_config(config)

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=[
        "q_proj", "v_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

llm = get_peft_model(base_model, peft_config)
llm.print_trainable_parameters()
