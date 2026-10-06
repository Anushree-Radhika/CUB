import os
import subprocess
import difflib
from pathlib import Path

# Configuration
IMAGE_DIR = Path("/home/paul/DATASET/CUB_200_2011/images/001.Black_footed_Albatross")
CHECKPOINT = "output-exp-gemma/best_model.pth"
OUTPUT_FILE = "generation_diff_gemma.txt"

# Supported image extensions
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Get all images recursively
images = sorted(
    p for p in IMAGE_DIR.rglob("*")
    if p.suffix.lower() in EXTENSIONS
)

previous_generation = None
previous_image = None

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for i, image_path in enumerate(images):

        print(f"[{i+1}/{len(images)}] {image_path}")

        command = [
            "python",
            "infer.py",
            "--input_image", str(image_path),
            "--load_path", CHECKPOINT,
            "--decoder_model", "google/gemma-3-1b-pt",
            "--max_seq_len", "300"
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            f.write(f"\nERROR: {image_path}\n")
            f.write(result.stderr + "\n")
            continue

        generated = result.stdout.strip()

        # First image: no previous generation to compare
        if previous_generation is None:

            f.write("=" * 80 + "\n")
            f.write(f"IMAGE: {image_path}\n")
            f.write("FIRST GENERATION\n")
            f.write(f"GENERATED:\n{generated}\n")

        else:

            # Word-level diff
            diff = list(difflib.ndiff(
                previous_generation.split(),
                generated.split()
            ))

            changes = [
                line for line in diff
                if line.startswith("+ ") or line.startswith("- ")
            ]

            f.write("=" * 80 + "\n")
            f.write(f"IMAGE: {image_path}\n")
            f.write(f"PREVIOUS IMAGE: {previous_image}\n")

            if not changes:

                f.write("STATUS: IDENTICAL TO PREVIOUS\n")

            else:

                f.write("STATUS: DIFFERENT\n\n")

                f.write("PREVIOUS GENERATION:\n")
                f.write(previous_generation + "\n\n")

                f.write("CURRENT GENERATION:\n")
                f.write(generated + "\n\n")

                f.write("CHANGES:\n")
                f.write("\n".join(changes) + "\n")

        previous_generation = generated
        previous_image = image_path

print(f"\nDone! Results saved to {OUTPUT_FILE}")