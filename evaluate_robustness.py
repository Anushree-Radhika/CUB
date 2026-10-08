import os
import argparse
import torch
from torch.utils.data import DataLoader
from dataset.dataset import CocoFormatDataset
from Model.modelv2 import TraitGen
from Model.VisionEncoderPooled import VisionEncoder
from utils import classification_accuracy, AverageMeter, load_checkpoint
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageFilter
import random
import numpy as np

def get_args_parser():
    parser = argparse.ArgumentParser('Robustness Evaluation', add_help=False)
    parser.add_argument('--data_root', default='/home/paul/DATASET/CUB_200_2011')
    parser.add_argument('--ann_dir', default='.')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--max_seq_len', default=256, type=int)
    parser.add_argument('--encoder_model', default="hf-hub:imageomics/bioclip")
    parser.add_argument('--encoder_op_dim', default=768, type=int)
    parser.add_argument('--decoder_model', default="google/gemma-3-1b-pt")
    parser.add_argument('--streeing_prompt', default="species identification and corresponding textual explanation task.")
    parser.add_argument('--load_path', required=True, help="Path to best_model.pth")
    parser.add_argument('--perturbation', default='none', choices=['none', 'blur', 'noise', 'brightness', 'contrast', 'occlusion'])
    parser.add_argument('--severity', default=1.0, type=float, help="Severity of the perturbation")
    return parser

def apply_perturbation(img, p_type, severity):
    if p_type == 'none':
        return img
    elif p_type == 'blur':
        return img.filter(ImageFilter.GaussianBlur(radius=severity))
    elif p_type == 'noise':
        img_arr = np.array(img)
        # Severity controls noise standard deviation
        noise = np.random.normal(0, severity * 255, img_arr.shape)
        noisy_img = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)
    elif p_type == 'brightness':
        # Severity < 1.0 makes it darker, > 1.0 makes it brighter (1.0 is original)
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(severity)
    elif p_type == 'contrast':
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(severity)
    elif p_type == 'occlusion':
        # Severity is the fraction of the image to occlude (e.g., 0.25)
        img_arr = np.array(img).copy()
        h, w, _ = img_arr.shape
        box_sz = int(min(h, w) * severity)
        y = random.randint(0, max(0, h - box_sz))
        x = random.randint(0, max(0, w - box_sz))
        img_arr[y:y+box_sz, x:x+box_sz, :] = 0
        return Image.fromarray(img_arr)
    return img

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("Loading vision encoder...")
    vision_encoder = VisionEncoder(args)
    base_preprocess = vision_encoder.preprocess
    
    # Custom preprocess function that adds perturbation before OpenCLIP preprocessing
    def robustness_preprocess(cv_image):
        perturbed_img = apply_perturbation(cv_image, args.perturbation, args.severity)
        return base_preprocess(perturbed_img)
    
    print("Loading dataset...")
    test_dataset = CocoFormatDataset(args, ann_file=f'{args.ann_dir}/test.json', img_prefix=f'{args.data_root}/images/', preprocess=robustness_preprocess)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True, shuffle=False)
    
    print("Initializing model...")
    model = TraitGen(args, vision_encoder=vision_encoder).to(device)
    model.eval()
    
    print(f"Loading checkpoint from: {args.load_path}")
    load_checkpoint(args.load_path, model, None, None)
    
    accuracy_meter = AverageMeter()
    
    print(f"\n--- Running evaluation with Perturbation: {args.perturbation}, Severity: {args.severity} ---")
    
    with torch.no_grad():
        batches = tqdm(test_loader, desc=f"Evaluating {args.perturbation} (s={args.severity})")
        for batch in batches:
            images = batch["image"].to(device)
            prompt_ids = batch["prompt_ids"].to(device)
            prompt_mask = batch["prompt_mask"].to(device)
            category = batch["category_name"]
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                generated_text = model.generate_caption(images, prompt_ids, prompt_mask)
                
            batch_accuracy = classification_accuracy(generated_text, category)
            accuracy_meter.update(batch_accuracy, images.size(0))
            
            batches.set_postfix(acc=f"{accuracy_meter.avg:.4f}")
            
    print(f"\n=> Final Accuracy ({args.perturbation}, severity={args.severity}): {accuracy_meter.avg:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Robustness Eval', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
