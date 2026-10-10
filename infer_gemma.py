import os
import argparse
import random
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoTokenizer

from Model.gemma_model import GemmaTraitGen, AllowedTokensLogitsProcessor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}

def get_args_parser():
    parser = argparse.ArgumentParser('Gemma Inference', add_help=False)
    
    parser.add_argument('--input_image', default='scratch', help='Path to a single image for testing')
    parser.add_argument('--load_path', default='scratch', help='Path to the trained model directory or weights')
    
    parser.add_argument('--pick_random', default=0, type=int, help='Number of random images to test')
    parser.add_argument('--random_dir_path', default='scratch', help='Folder to pick random images from')
    
    parser.add_argument('--lm_id', default='google/gemma-3-1b-pt')
    parser.add_argument('--clip_id', default='hf-hub:imageomics/bioclip')
    parser.add_argument('--max_len', default=300, type=int)

    return parser

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ------------------------------------------------------------------ Setup Tokenizer
    tok = AutoTokenizer.from_pretrained(args.lm_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    allowed_processor = AllowedTokensLogitsProcessor(tok, device=device)
    
    # ------------------------------------------------------------------ Setup Model
    print("Loading Gemma 3 model...")
    model = GemmaTraitGen(clip_id=args.clip_id, lm_id=args.lm_id, device=device)
    model.eval()

    if args.load_path != "scratch":
        print(f"Loading trained weights from {args.load_path}...")
        enc_path = os.path.join(args.load_path, "enc_lora")
        dec_path = os.path.join(args.load_path, "dec_lora")
        proj_path = os.path.join(args.load_path, "projector.pt")
        
        if os.path.exists(enc_path):
            from peft import set_peft_model_state_dict
            # Peft provides ways to load directly or we can just load the state dicts if we have them
            model.visual.load_adapter(enc_path, "default")
        if os.path.exists(dec_path):
            model.lm.load_adapter(dec_path, "default")
        if os.path.exists(proj_path):
            model.projector.load_state_dict(torch.load(proj_path, map_location=device))
    
    # ------------------------------------------------------------------ Mode 1: Random Images
    if args.pick_random > 0:
        target_dir = Path(args.random_dir_path)
        if not target_dir.is_dir():
            print(f"Error: Directory '{target_dir}' does not exist.")
            return

        all_images = [
            path for path in target_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if not all_images:
            print("No image files found.")
            return

        sample_size = min(args.pick_random, len(all_images))
        selected_images = random.sample(all_images, sample_size)

        print(f"\nFound {len(all_images)} images. Testing {sample_size} at random:\n")

        for img in selected_images:
            infer_image = Image.open(img).convert("RGB")
            pix = model.preprocess_val(infer_image).unsqueeze(0).to(device)
            
            text = model.generate_texts(pix, tok, args.max_len, allowed_processor)[0]
            
            print("="*80)
            print(f"IMAGE: {img}")
            print(f"GENERATED TEXT:\n{text}")
            print("="*80 + "\n")
            
    # ------------------------------------------------------------------ Mode 2: Single Image
    elif args.input_image != 'scratch':
        if not os.path.exists(args.input_image):
            print(f"Error: Image '{args.input_image}' does not exist.")
            return
            
        infer_image = Image.open(args.input_image).convert("RGB")
        pix = model.preprocess_val(infer_image).unsqueeze(0).to(device)
        
        text = model.generate_texts(pix, tok, args.max_len, allowed_processor)[0]
        
        print("\n" + "="*80)
        print(f"IMAGE: {args.input_image}")
        print(f"GENERATED TEXT:\n{text}")
        print("="*80 + "\n")
        
    else:
        print("Please provide either --input_image <path> or --pick_random <N> with --random_dir_path <path>")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Gemma Inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
