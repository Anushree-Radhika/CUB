import os

import torch
from accelerate import Accelerator

import a_model
import a_bioclip_VE
import a_gemma_TD


def train_fn():
    accelerator = Accelerator(mixed_precision="fp16")

    model = a_model.Model(
        vision_encoder=a_bioclip_VE.BioCLIP(),
        text_decoder=a_gemma_TD.GemmaDecoder(a_model.DECODER_ID)
    )

    # Resumes from the checkpoint a_play.py's run produced. Only run this
    # script after a_play.py has completed at least one full pass —
    # a_model.CHECKPOINT_PATH won't exist before that.
    state_dict = torch.load(a_model.CHECKPOINT_PATH, map_location='cpu')
    model.load_state_dict(state_dict)

    for p in model.ve.parameters():
        p.requires_grad_(False)

    model.start_training('./train3.json', accelerator=accelerator)

    return accelerator


def run_inference():
    image_list = [
        os.path.join(a_model.IMAGES_ROOT, "200.Common_Yellowthroat", "Common_Yellowthroat_0055_190967.jpg")
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
