import torch
from utils import classification_accuracy
import torch.distributed as dist
from torch.cuda.amp import autocast
import contextlib

def allreduce_sum(vals, device, local_rank):
    t = torch.tensor(vals, dtype=torch.float64, device=device)
    if local_rank != -1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()

def train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, eos_token_id, local_rank, epoch, epochs, grad_accum_steps, tqdm_bar=None):
    model.train()
    train_loss_sum = 0

    pbar = tqdm_bar if tqdm_bar is not None else train_loader
    for step, (pix, ids, attn, lab, sp_mask) in enumerate(pbar):
        pix, ids, attn, lab, sp_mask = (x.to(device) for x in (pix, ids, attn, lab, sp_mask))
        
        is_accumulating = (step + 1) % grad_accum_steps != 0 and (step + 1) != len(train_loader)
        
        sync_context = contextlib.ExitStack()
        if is_accumulating and local_rank != -1:
            if hasattr(model, "no_sync"):
                sync_context.enter_context(model.no_sync())
                
        with sync_context:
            with autocast(dtype=torch.float16):
                loss = model(pix, ids, attn, lab, sp_mask, eos_token_id)
                
            scaled_loss = loss / grad_accum_steps
            scaler.scale(scaled_loss).backward()
        
        # Only step the optimizer when accumulation is done
        if not is_accumulating:
            scaler.unscale_(optimizer)
            trainable = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_([p for p in trainable if p.grad is not None], 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        train_loss_sum += loss.item()
        if local_rank in [-1, 0] and tqdm_bar is not None:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = train_loss_sum / len(train_loader)
    
    if local_rank != -1:
        tl_tensor = torch.tensor(avg_train_loss, device=device)
        dist.all_reduce(tl_tensor, op=dist.ReduceOp.SUM)
        avg_train_loss = tl_tensor.item() / dist.get_world_size()

    return avg_train_loss


@torch.no_grad()
def evaluate_loss(model, val_loader, device, eos_token_id, local_rank):
    model.eval()
    tot, n = 0.0, 0
    for pix, ids, attn, lab, sp_mask in val_loader:
        pix, ids, attn, lab, sp_mask = (x.to(device) for x in (pix, ids, attn, lab, sp_mask))
        loss = model(pix, ids, attn, lab, sp_mask, eos_token_id)
        tot += loss.item()
        n += 1
        
    tot, n = allreduce_sum([tot, n], device, local_rank)
    return tot / max(n, 1)


@torch.no_grad()
def evaluate_accuracy(model, gen_loader, tokenizer, max_new_tokens, logits_processor, device, local_rank, show=0):
    model.eval()
    local_texts = []
    local_gts = []
    shown = 0
    
    for pix, gts in gen_loader:
        gen = model.generate_texts(pix, tokenizer, max_new_tokens, logits_processor)
        local_texts.extend(gen)
        local_gts.extend(gts)
        
        if local_rank in [-1, 0] and shown < show:
            for text, gt in zip(gen, gts):
                if shown < show:
                    print(f"  gt: {gt!r}  |  generated: {text[:90]!r}")
                    shown += 1
                
    local_acc = classification_accuracy(local_texts, local_gts)
    local_hits = local_acc * len(local_gts)
    local_total = len(local_gts)
    
    hits, total = allreduce_sum([local_hits, local_total], device, local_rank)
    
    return hits / max(total, 1), 1.0  # Parsed rate is fixed to 1.0 for compatibility
