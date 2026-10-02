"""
Using pre-existing model,
1) generate all patches by passing image through vision encoder and bridge.
2) from the bridge output reconstruct image patches
3) get embeddings of each text using given llm's tokenizer. compare with every patch image
"""


import os
import re
import json
import torch
import random
import logging
import argparse

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset.dataset import CocoFormatDataset
from PIL import Image
from Model.modelv2 import TraitGen
from Model.VisionEncoderPooled import VisionEncoder
from engine import *

import os
# Force HF to use the standard, stable downloader instead of Xet chunks
os.environ["HF_HUB_DISABLE_XET"] = "1"
# Give the proxy a massive timeout cushion so it doesn't instantly die
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"

def get_args_parser():
    parser = argparse.ArgumentParser('XAI Training', add_help=False)
    
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--input_image',default='scratch')
    parser.add_argument('--max_seq_len', default=256, type=int)
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size per GPU')
    parser.add_argument('--encoder_model', default="hf-hub:imageomics/bioclip")
    parser.add_argument('--encoder_op_dim', default=768, type=int)
    parser.add_argument('--decoder_model', default="openai-community/gpt2-medium")
    parser.add_argument('--streeing_prompt', default="species identification and corresponding textual explanation task.")
    parser.add_argument('--load_path',default="scratch")
    return parser

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    vision_encoder = VisionEncoder(args)
    preprocess = vision_encoder.preprocess
    
    model = TraitGen(args, vision_encoder=vision_encoder).to(device)
    model_state_path = args.load_path
    
    model.eval()
            
            
    if not (model_state_path == "scratch"):
        _ = load_checkpoint(model_state_path,model,None,None)
    
    print("State image path")
    image_path = str(input()).strip()
    
    
    if not image_path == 'scratch':
        infer_image = Image.open(image_path)
        infer_image = preprocess(infer_image)
        infer_image = infer_image.unsqueeze(0).to(device)
        
        prompt_list = []
        print("No. of input text")
        n = int(input())
        
        for i in range(n):
            prompt_list.append(str(input()))
        
        for i in prompt_list:
            _ = model.visualize_trait_integrated_gradients(infer_image,i,f"attention_map_{i}.png")
    else:
        print("ERROR - Image path not specified")
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser('TraitGen Training', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)