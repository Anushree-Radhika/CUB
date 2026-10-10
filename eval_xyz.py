import json
import re
import torch
from tqdm import tqdm
from utils import classification_accuracy, text_f1_score

# Import the inference components from your script
from xyz import caption, DEVICE, visual, lm, projector, tok
from peft import PeftModel
import os

def evaluate_xyz():
    if not os.path.exists("enc_lora") or not os.path.exists("dec_lora"):
        print("Error: Model checkpoints (enc_lora, dec_lora) not found. Has xyz.py finished training at least one epoch?")
        return

    # Load the trained weights
    visual_eval = PeftModel.from_pretrained(visual, "enc_lora").to(DEVICE)
    lm_eval = PeftModel.from_pretrained(lm, "dec_lora").to(DEVICE)
    projector.load_state_dict(torch.load("projector.pt", map_location=DEVICE))
    
    with open("test.json", "r") as f:
        test_data = json.load(f)
    
    correct = 0
    total_f1 = 0.0
    total = len(test_data)
    
    print(f"Evaluating {total} images from test.json...")
    
    for item in tqdm(test_data, desc="Evaluating Accuracy"):
        image_path = f"/home/paul/DATASET/CUB_200_2011/images/{item['imagePath']}"
        
        # In case the path logic differs (handle missing prefixes)
        if not os.path.exists(image_path) and os.path.exists(item['imagePath']):
             image_path = item['imagePath']
             
        gt_text = item["gt"]
        
        # Get ground truth category (e.g. from 'species of "Laysan Albatross"')
        match = re.search(r'\"([^\"]+)\"', gt_text)
        gt_category = match.group(1) if match else gt_text
        
        # Generate caption (using 4 beams as defined in xyz.py)
        generated = caption(image_path, beams=4)
        
        # Calculate scores
        acc = classification_accuracy([generated], [gt_category])
        f1 = text_f1_score([generated], [gt_text])
        
        correct += acc
        total_f1 += f1

    print("=" * 50)
    print(f"Final Species Classification Accuracy: {correct/total:.4f} ({int(correct)}/{total})")
    print(f"Final Text F1 Score (Word Overlap): {total_f1/total:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    evaluate_xyz()
