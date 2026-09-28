import os

import torch
from accelerate import Accelerator

import a_model
import a_bioclip_VE
import a_gemma_TD


def train_fn():
    # mixed_precision="fp16" turns on gradient scaling, which is what keeps
    # LoRA training numerically stable with fp16 base weights on V100
    # (no bf16 tensor cores here).
    accelerator = Accelerator(mixed_precision="fp16")

    model = a_model.Model(
        vision_encoder=a_bioclip_VE.BioCLIP(),
        text_decoder=a_gemma_TD.GemmaDecoder(a_model.DECODER_ID)
    )

    # First training pass — starts fresh from Gemma's pretrained weights +
    # newly initialized LoRA adapters + a fresh image_projection. Nothing to
    # resume from yet, so no checkpoint load here (unlike aa_play.py).

    for p in model.ve.parameters():
        p.requires_grad_(False)

    model.start_training('./train3.json', accelerator=accelerator)

    return accelerator


def run_inference():
    image_list = [
        os.path.join(a_model.IMAGES_ROOT, "002.Laysan_Albatross", "Laysan_Albatross_0085_564.jpg")
    ]

    model = a_model.Model(
        vision_encoder=a_bioclip_VE.BioCLIP(),
        text_decoder=a_gemma_TD.GemmaDecoder(a_model.DECODER_ID)
    )
    model.load_state_dict(torch.load(a_model.CHECKPOINT_PATH, map_location='cpu'))
    model = model.cuda()

    print(model.generate(image_list))


if __name__ == "__main__":
    accelerator = train_fn()
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        run_inference()
