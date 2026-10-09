import time
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

from utils import AverageMeter, save_checkpoint, load_checkpoint, classification_accuracy, text_f1_score


def is_main_process():
    """Checks if current process is rank 0 or non-distributed."""
    return not dist.is_initialized() or dist.get_rank() == 0


def reduce_tensor(tensor):
    """Averages a tensor across all distributed processes."""
    if not dist.is_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def train_one_epoch(model, train_loader, optimizer, device, epoch, scaler=None, grad_accum_steps=8, scheduler=None):
    model.train()
    
    # Safely unwrap DDP model to access custom methods like generate_caption
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.vision_encoder.eval()
    
    loss_meter = AverageMeter()
    #accuracy_meter = AverageMeter()

    # Disable tqdm on secondary GPU processes to prevent UI glitching
    batches = tqdm(
        train_loader, 
        desc=f"Train Epoch {epoch}", 
        leave=False, 
        disable=not is_main_process()
    )

    optimizer.zero_grad()
    for i, batch in enumerate(batches):
        images = batch["image"].to(device)
        prompt_ids = batch["prompt_ids"].to(device)
        prompt_mask = batch["prompt_mask"].to(device)
        target_ids = batch["target_ids"].to(device)
        target_mask = batch["target_mask"].to(device)
        category = batch["category_name"]  
        category_mask = batch["category_mask"].to(device)
        
        # loss = model(images, prompt_ids, prompt_mask, target_ids, target_mask,category_mask)
        
        #with torch.no_grad():
        #    generated_text = raw_model.generate_caption(images, prompt_ids, prompt_mask)

        if scaler is not None:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = model(images, prompt_ids, prompt_mask, target_ids, target_mask, category_mask)
                loss = loss / grad_accum_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
            
        else:
            loss = model(images, prompt_ids, prompt_mask, target_ids, target_mask, category_mask)
            loss = loss / grad_accum_steps

            loss.backward()
            
            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
        # Compute accuracy locally
        #batch_accuracy = classification_accuracy(generated_text, category)

        # Sync loss and accuracy across all GPUs for accurate logging
        reduced_loss = reduce_tensor(loss.detach())
        #acc_tensor = torch.tensor(batch_accuracy, device=device)
        #reduced_acc = reduce_tensor(acc_tensor)

        loss_meter.update(reduced_loss.item(), images.size(0))
        #accuracy_meter.update(reduced_acc.item(), images.size(0))

        if is_main_process():
            batches.set_postfix(
                loss=f"{loss_meter.avg:.4f}"
            )

    return loss_meter.avg


@torch.no_grad()
def validate(args, model, val_loader, device, max_gen_batches=40):
    model.eval()
    
    # Safely unwrap DDP model
    raw_model = model.module if hasattr(model, 'module') else model

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    f1_meter = AverageMeter()

    batches = tqdm(
        val_loader, 
        desc="Validation", 
        leave=False, 
        disable=not is_main_process()
    )

    import random
    # Pick a random batch to print, up to max_gen_batches or len(val_loader)
    print_batch_idx = random.randint(0, max(0, min(len(val_loader) - 1, max_gen_batches - 1)))
    
    sample_gt = "N/A"
    sample_gen = "N/A"

    for i,batch in enumerate(batches):
        images = batch["image"].to(device)
        prompt_ids = batch["prompt_ids"].to(device)
        prompt_mask = batch["prompt_mask"].to(device)
        target_ids = batch["target_ids"].to(device)
        target_mask = batch["target_mask"].to(device)
        category_mask = batch["category_mask"].to(device)
        
        category = batch["category_name"]
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(images, prompt_ids, prompt_mask, target_ids, target_mask,category_mask)
        
        reduced_loss = reduce_tensor(loss.detach())
        loss_meter.update(reduced_loss.item(), images.size(0))
        #if torch.isnan(loss) or torch.isinf(loss):
        #    print("[Warning] NaN detected during validation loss calculation. Skipping generation for this batch.")
        #    continue
        if i < max_gen_batches:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                generated_text = raw_model.generate_caption(images, prompt_ids, prompt_mask)
            
            # Print a random generated text from a random batch to debug
            if i == print_batch_idx:
                print_sample_idx = random.randint(0, len(batch["caption"]) - 1)
                sample_gt = batch["caption"][print_sample_idx]
                sample_gen = generated_text[print_sample_idx]
                if is_main_process():
                    print(f"\n[DEBUG] Ground Truth: {sample_gt}")
                    print(f"[DEBUG] Generated   : {sample_gen}\n")

            batch_accuracy = classification_accuracy(generated_text, category)
            batch_f1 = text_f1_score(generated_text, batch["caption"])
            
            # Sync loss and accuracy across all GPUs    
            acc_tensor = torch.tensor(batch_accuracy, device=device)
            reduced_acc = reduce_tensor(acc_tensor)
            accuracy_meter.update(reduced_acc.item(), images.size(0))
            
            f1_tensor = torch.tensor(batch_f1, device=device)
            reduced_f1 = reduce_tensor(f1_tensor)
            f1_meter.update(reduced_f1.item(), images.size(0))
            
        if is_main_process():
            batches.set_postfix(
                loss=f"{loss_meter.avg:.4f}", 
                acc=f"{accuracy_meter.avg:.4f}",
                f1=f"{f1_meter.avg:.4f}"
            )

    return loss_meter.avg, accuracy_meter.avg, f1_meter.avg, sample_gt, sample_gen
