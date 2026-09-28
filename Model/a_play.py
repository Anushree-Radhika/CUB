import os

import a_model
import a_bioclip_VE
import a_gemma_TD


def build_model():
    return a_model.Model(
        vision_encoder=a_bioclip_VE.BioCLIP(),
        text_decoder=a_gemma_TD.GemmaDecoder(a_model.DECODER_ID),
    )


def train_fn():
    model = build_model()
    if os.environ.get("RESUME") == "1":
        meta = model.load_trainable(a_model.CHECKPOINT_PATH)
        print(f"Resumed trainable weights from {a_model.CHECKPOINT_PATH} {meta}")
    model.start_training("./train3.json")   # ends with the best checkpoint loaded
    return model


def run_inference(model):
    """Captions for a few images with the best checkpoint."""
    names = [
        "002.Laysan_Albatross/Laysan_Albatross_0085_564.jpg",
        "200.Common_Yellowthroat/Common_Yellowthroat_0055_190967.jpg",
    ]
    paths = [os.path.join(a_model.IMAGES_ROOT, n) for n in names]
    model.td.eval()
    for name, text in zip(names, model.generate(paths)):
        print(f"\n{name}\n  {text.strip()}")


if __name__ == "__main__":
    trained = train_fn()
    run_inference(trained)
