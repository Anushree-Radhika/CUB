import os
import subprocess
import difflib
import json
import re
from pathlib import Path
from utils import classification_accuracy, text_f1_score

# Assuming DATA_ROOT is the base directory containing the images as specified in the script
DATA_ROOT = Path("/home/paul/DATASET/CUB_200_2011/images")
CHECKPOINT = "output/best_model.pth"
OUTPUT_FILE = "generation_diff_gemma.txt"
TRAIN_JSON = "train.json"
TEST_JSON = "test.json"

# Load ground truths
data = []
for json_file in [TEST_JSON]:
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            data.extend(json.load(f))

total_samples = 0
correct_classifications = 0
total_f1 = 0.0

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for i, item in enumerate(data):
        rel_path = item["imagePath"]
        gt_text = item["gt"]
        image_path = DATA_ROOT / rel_path

        print(f"[{i+1}/{len(data)}] {image_path}")
        
        # Skip if image doesn't exist to prevent errors
        if not image_path.exists():
            print(f"Warning: {image_path} does not exist, skipping.")
            continue

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

        # Extract ground truth category
        # Assumes ground truth format: It is a species of "Category Name" as it has...
        match = re.search(r'\"([^\"]+)\"', gt_text)
        gt_category = match.group(1) if match else gt_text
        
        # Calculate classification accuracy for this image
        acc = classification_accuracy([generated], [gt_category])
        f1 = text_f1_score([generated], [gt_text])
        
        correct_classifications += acc
        total_f1 += f1
        total_samples += 1

        # Word-level diff between Ground Truth and Generated Text
        diff = list(difflib.ndiff(
            gt_text.split(),
            generated.split()
        ))

        changes = [
            line for line in diff
            if line.startswith("+ ") or line.startswith("- ")
        ]

        f.write("=" * 80 + "\n")
        f.write(f"IMAGE: {image_path}\n")
        
        if not changes:
            f.write("STATUS: IDENTICAL TO GROUND TRUTH\n")
        else:
            f.write("STATUS: DIFFERENT FROM GROUND TRUTH\n\n")
            f.write("GROUND TRUTH:\n")
            f.write(gt_text + "\n\n")
            f.write("GENERATED:\n")
            f.write(generated + "\n\n")
            f.write("CHANGES ( '-' means missing in generated, '+' means extra in generated ):\n")
            f.write("\n".join(changes) + "\n")

    if total_samples > 0:
        final_accuracy = correct_classifications / total_samples
        final_f1 = total_f1 / total_samples
        f.write("=" * 80 + "\n")
        f.write(f"FINAL SPECIES CLASSIFICATION ACCURACY: {final_accuracy:.4f} ({int(correct_classifications)}/{total_samples})\n")
        f.write(f"FINAL TEXT F1 SCORE (WORD OVERLAP): {final_f1:.4f}\n")
        print(f"\nFinal Species Classification Accuracy: {final_accuracy:.4f} ({int(correct_classifications)}/{total_samples})")
        print(f"Final Text F1 Score (Word Overlap): {final_f1:.4f}")

print(f"\nDone! Results saved to {OUTPUT_FILE}")
