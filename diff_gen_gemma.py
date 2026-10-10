import os
import re
import json
import difflib
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from PIL import Image
from transformers import AutoTokenizer

from Model.gemma_model import GemmaTraitGen, AllowedTokensLogitsProcessor
from dataset.gemma_dataset import load_records
from utils import classification_accuracy

def get_args_parser():
    parser = argparse.ArgumentParser('Gemma Diff Generator', add_help=False)
    
    parser.add_argument('--test_json', default='test.json')
    parser.add_argument('--img_root', default='/home/paul/DATASET/CUB_200_2011/images/')
    parser.add_argument('--load_path', default='scratch', help='Path to trained weights (e.g., output/)')
    parser.add_argument('--output_file', default='generation_diff_gemma.txt')
    
    parser.add_argument('--lm_id', default='google/gemma-3-1b-pt')
    parser.add_argument('--clip_id', default='hf-hub:imageomics/bioclip')
    parser.add_argument('--max_len', default=300, type=int)
    parser.add_argument('--batch_size', default=8, type=int)

    return parser

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ------------------------------------------------------------------ Setup Model
    tok = AutoTokenizer.from_pretrained(args.lm_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    allowed_processor = AllowedTokensLogitsProcessor(tok, device=device)
    
    print("Loading Gemma 3 model...")
    model = GemmaTraitGen(clip_id=args.clip_id, lm_id=args.lm_id, device=device)
    model.eval()

    if args.load_path != "scratch":
        print(f"Loading trained weights from {args.load_path}...")
        enc_path = os.path.join(args.load_path, "enc_lora")
        dec_path = os.path.join(args.load_path, "dec_lora")
        proj_path = os.path.join(args.load_path, "projector.pt")
        
        if os.path.exists(enc_path):
            model.visual.load_adapter(enc_path, "default")
        if os.path.exists(dec_path):
            model.lm.load_adapter(dec_path, "default")
        if os.path.exists(proj_path):
            model.projector.load_state_dict(torch.load(proj_path, map_location=device))

    # ------------------------------------------------------------------ Process Dataset
    recs = load_records(args.test_json)
    print(f"Found {len(recs)} records in {args.test_json}")
    
    total_samples = 0
    correct_classifications = 0

    with open(args.output_file, "w", encoding="utf-8") as f:
        
        # Batching loop
        for i in tqdm(range(0, len(recs), args.batch_size)):
            batch_recs = recs[i : i + args.batch_size]
            
            paths = []
            gt_texts = []
            gt_categories = []
            pixels = []
            
            for rel_path, gt_text, _ in batch_recs:
                img_path = os.path.join(args.img_root, rel_path)
                if not os.path.exists(img_path):
                    continue
                    
                infer_image = Image.open(img_path).convert("RGB")
                pix = model.preprocess_val(infer_image)
                
                paths.append(img_path)
                gt_texts.append(gt_text)
                pixels.append(pix)
                
                # Extract ground truth category
                match = re.search(r'\"([^\"]+)\"', gt_text)
                gt_category = match.group(1) if match else gt_text
                gt_categories.append(gt_category)
                
            if not pixels:
                continue
                
            pixels_tensor = torch.stack(pixels).to(device)
            generated_texts = model.generate_texts(pixels_tensor, tok, args.max_len, allowed_processor)
            
            for p, gt_text, gt_cat, gen_text in zip(paths, gt_texts, gt_categories, generated_texts):
                acc = classification_accuracy([gen_text], [gt_cat])
                
                correct_classifications += acc
                total_samples += 1
                
                diff = list(difflib.ndiff(gt_text.split(), gen_text.split()))
                changes = [line for line in diff if line.startswith("+ ") or line.startswith("- ")]
                
                f.write("=" * 80 + "\n")
                f.write(f"IMAGE: {p}\n")
                if not changes:
                    f.write("STATUS: IDENTICAL TO GROUND TRUTH\n")
                else:
                    f.write("STATUS: DIFFERENT FROM GROUND TRUTH\n\n")
                    f.write("GROUND TRUTH:\n")
                    f.write(gt_text + "\n\n")
                    f.write("GENERATED:\n")
                    f.write(gen_text + "\n\n")
                    f.write("CHANGES ( '-' means missing in generated, '+' means extra in generated ):\n")
                    f.write("\n".join(changes) + "\n")

        if total_samples > 0:
            final_accuracy = correct_classifications / total_samples
            f.write("=" * 80 + "\n")
            f.write(f"FINAL SPECIES CLASSIFICATION ACCURACY: {final_accuracy:.4f} ({int(correct_classifications)}/{total_samples})\n")
            
            print(f"\nFinal Species Classification Accuracy: {final_accuracy:.4f} ({int(correct_classifications)}/{total_samples})")

    print(f"\nDone! Results saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Gemma Diff Generator', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
