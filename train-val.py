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
from Model.modelv2 import TraitGen
from Model.VisionEncoderPooled import VisionEncoder
from engine import *

def setup_ddp():
    """Initializes the DDP environment."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    """Destroys the process group upon completion."""
    dist.destroy_process_group()

def get_args_parser():
    parser = argparse.ArgumentParser('XAI Training', add_help=False)

    parser.add_argument('--data_root', default='/home/paul/Paul/DATASETS/cub')
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--optimizer', default='adam', choices=['sgd', 'adam', 'adamw'],
                        help='Optimizer (default: %(default)s)')
    parser.add_argument('--max_seq_len', default=256, type=int)
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size per GPU')
    parser.add_argument('--encoder_model', default="hf-hub:imageomics/bioclip")
    parser.add_argument('--encoder_op_dim', default=768, type=int)
    parser.add_argument('--decoder_model', default="openai-community/gpt2-medium")
    parser.add_argument('--streeing_prompt', default="species identification and corresponding textual explanation task.")
    parser.add_argument('--ann_dir', default='/kaggle/input/custom-ds', help='Path to custom JSON annotations')
    parser.add_argument('--validate_model',default=0,type=int)
    parser.add_argument('--load_path',default="scratch")
    parser.add_argument('--drop_parts',default=2,type=int)
    return parser

def main(args):
    local_rank = setup_ddp()
    global_rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")

    # Set up logging ONLY on the primary process (Rank 0)
    logger = None
    if global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        log_file = os.path.join(args.output_dir, "log.txt")
        
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(message)s')
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    vision_encoder = VisionEncoder(args)
    preprocess = vision_encoder.preprocess

    train_dataset = CocoFormatDataset(args, ann_file=f'{args.ann_dir}/train.json', img_prefix=f'{args.data_root}/images/', preprocess=preprocess)
    test_dataset = CocoFormatDataset(args, ann_file=f'{args.ann_dir}/test.json', img_prefix=f'{args.data_root}/images/', preprocess=preprocess)

    # Wrap Datasets with DistributedSampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler, 
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        sampler=test_sampler, 
        num_workers=2,
        pin_memory=True
    )
    
    if(args.validate_model == 1):
        model_state_path = args.load_path
        if not (model_state_path == "scratch"):
        
            model_test = TraitGen(args, vision_encoder=vision_encoder).to(device)
            ckpt_info = load_checkpoint(model_state_path,model_test,None,None)
            model_test = DDP(model_test,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=False)
            val_loss,val_acc = validate(args,model_test,test_loader,device)
            if global_rank == 0:
                logger.info(f"Epoch {ckpt_info}: Accuracy={val_acc:.4f} Validation Loss={val_loss:.4f}")
        else:
            print("ERROR. NO PATH MENTIONED FOR LOADING")
        
    else:
        # Initialize model and wrap in DDP
        model = TraitGen(args, vision_encoder=vision_encoder).to(device)
        model_state_path = args.load_path
        
        start_epoch = 0
        
        if not (model_state_path == "scratch"):
            start_epoch = load_checkpoint(model_state_path,model,None,None)
            start_epoch += 1
        
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )    
        
        scaler = torch.amp.GradScaler('cuda')
        
        cnter = 1
        train_losses_history = []
        val_losses_history = []
        
        for epoch in range(start_epoch,start_epoch + args.epochs):
            # Set epoch for sampler to ensure proper shuffling across GPUs
            train_sampler.set_epoch(epoch)

            train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch,scaler=scaler)

            # Log and save checkpoints only from rank 0
            if global_rank == 0:
                train_losses_history.append(train_loss)

                if epoch == start_epoch + args.epochs - 1:
                    # Save model.module to strip the 'module.' wrapper prefix
                    checkpoint_path = os.path.join(args.output_dir, "best_model.pth")
                    save_checkpoint(checkpoint_path, model.module, optimizer,None,epoch)
                elif cnter % 2 == 0:
                    checkpoint_path = os.path.join(args.output_dir, f"best_model_{epoch}.pth")
                    # Save model.module to strip the 'module.' wrapper prefix
                    save_checkpoint(checkpoint_path, model.module, optimizer,None,epoch)
            
            val_loss, val_acc, sample_gt, sample_gen = validate(args, model, test_loader, device)
            
            if global_rank == 0:
                val_losses_history.append(val_loss)
                logger.info(f"\n{'='*60}")
                logger.info(f"EPOCH {epoch} SUMMARY:")
                logger.info(f"Train Loss:      {train_loss:.4f}")
                logger.info(f"Test/Val Loss:   {val_loss:.4f}")
                logger.info(f"Accuracy:        {val_acc:.4f}")
                logger.info(f"Sample GT:       {sample_gt}")
                logger.info(f"Sample Gen:      {sample_gen}")
                logger.info(f"{'='*60}\n")
                
                # Progressively form and save the train vs test loss curve
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10,6))
                current_epochs = range(start_epoch, epoch + 1)
                plt.plot(current_epochs, train_losses_history, label='Train Loss', marker='o')
                plt.plot(current_epochs, val_losses_history, label='Test/Val Loss', marker='o')
                plt.title('Training and Validation/Test Loss Curve')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                plot_path = os.path.join(args.output_dir, "loss_curve.png")
                plt.savefig(plot_path)
                plt.close()
                    
            cnter += 1

    cleanup_ddp()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('TraitGen Training', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
