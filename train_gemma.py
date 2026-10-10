import os
import math
import time
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset.gemma_dataset import load_records, create_collate_fn, gen_collate, make_loader
from Model.gemma_model import GemmaTraitGen, AllowedTokensLogitsProcessor
from engine_gemma import train_one_epoch, evaluate_loss, evaluate_accuracy


def get_args_parser():
    parser = argparse.ArgumentParser('Gemma TraitGen Training', add_help=False)

    parser.add_argument('--train_json', default='train.json')
    parser.add_argument('--test_json', default='test.json')
    parser.add_argument('--img_root', default='/home/paul/DATASET/CUB_200_2011/images/')
    parser.add_argument('--output_dir', default='output')
    
    parser.add_argument('--lm_id', default='google/gemma-3-1b-pt')
    parser.add_argument('--clip_id', default='hf-hub:imageomics/bioclip')
    
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--warm_epochs', default=1, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--grad_accum_steps', default=8, type=int)
    parser.add_argument('--gen_batch_size', default=16, type=int)
    parser.add_argument('--max_len', default=300, type=int)
    parser.add_argument('--gen_tokens_acc', default=60, type=int)

    parser.add_argument('--lr_enc', default=5e-5, type=float)
    parser.add_argument('--lr_dec', default=1e-4, type=float)
    parser.add_argument('--lr_proj', default=1e-3, type=float)

    return parser

def main(args):
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ dataset
    train_recs = load_records(args.train_json)
    val_recs = load_records(args.test_json)
    if local_rank in [-1, 0]:
        print(f"train {len(train_recs)}  val(test.json) {len(val_recs)} (image-caption pairs)")

    # ------------------------------------------------------------------ model
    tok = AutoTokenizer.from_pretrained(args.lm_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    
    allowed_processor = AllowedTokensLogitsProcessor(tok, device=device)
    collate = create_collate_fn(tok, args.max_len)
    
    model = GemmaTraitGen(clip_id=args.clip_id, lm_id=args.lm_id, device=device)
    
    train_dl = make_loader(train_recs, model.preprocess_train, args.img_root, args.batch_size, collate, local_rank, shuffle=True)
    val_dl = make_loader(val_recs, model.preprocess_val, args.img_root, args.batch_size, collate, local_rank)
    val_gen_dl = make_loader(val_recs, model.preprocess_val, args.img_root, args.gen_batch_size, gen_collate, local_rank)
    
    train_eval_dl = make_loader(train_recs, model.preprocess_val, args.img_root, args.batch_size, collate, local_rank)
    train_gen_dl = make_loader(train_recs, model.preprocess_val, args.img_root, args.gen_batch_size, gen_collate, local_rank)

    if local_rank != -1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    base_model = model.module if hasattr(model, "module") else model

    # ------------------------------------------------------------------ optimizer
    enc_params = [p for n, p in base_model.visual.named_parameters() if "lora_" in n]
    dec_params = [p for n, p in base_model.lm.named_parameters() if "lora_" in n]
    for p in enc_params + dec_params:
        p.requires_grad = True
        
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": args.lr_enc},
         {"params": dec_params, "lr": args.lr_dec},
         {"params": base_model.projector.parameters(), "lr": args.lr_proj}],
        weight_decay=0.05,
    )
    
    scaler = torch.cuda.amp.GradScaler()
    
    total_steps_per_epoch = math.ceil(len(train_dl) / args.grad_accum_steps)
    total_steps = args.epochs * total_steps_per_epoch
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * total_steps), total_steps)

    def set_lora_trainable(flag):
        for p in enc_params + dec_params:
            p.requires_grad = flag

    def snapshot():
        return {
            "enc": {n: p.detach().cpu().clone() for n, p in base_model.visual.named_parameters() if "lora_" in n},
            "dec": {n: p.detach().cpu().clone() for n, p in base_model.lm.named_parameters() if "lora_" in n},
            "proj": {k: v.detach().cpu().clone() for k, v in base_model.projector.state_dict().items()}
        }

    def restore(s):
        with torch.no_grad():
            for n, p in base_model.visual.named_parameters():
                if n in s["enc"]: p.copy_(s["enc"][n])
            for n, p in base_model.lm.named_parameters():
                if n in s["dec"]: p.copy_(s["dec"][n])
        base_model.projector.load_state_dict(s["proj"])

    # ------------------------------------------------------------------ train
    if local_rank in [-1, 0]:
        os.makedirs(os.path.join(args.output_dir, "best_model"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "last_model"), exist_ok=True)

    best, best_epoch, best_state = math.inf, 0, None
    history_train_loss, history_val_loss = [], []

    for ep in range(args.epochs):
        t0 = time.time()
        if local_rank != -1:
            train_dl.sampler.set_epoch(ep)

        stage1 = ep < args.warm_epochs
        set_lora_trainable(not stage1)

        pbar = tqdm(train_dl, desc=f"Epoch {ep+1}/{args.epochs}") if local_rank in [-1, 0] else None
        
        avg_train_loss = train_one_epoch(
            model, train_dl, opt, sched, scaler, device, tok.eos_token_id, local_rank, ep, args.epochs, args.grad_accum_steps, pbar
        )
        
        vl = evaluate_loss(model, val_dl, device, tok.eos_token_id, local_rank)

        is_best = vl < best
        if is_best:
            best, best_epoch, best_state = vl, ep + 1, snapshot()

        if local_rank in [-1, 0]:
            print(f"epoch {ep+1}/{args.epochs} ({'projector only' if stage1 else 'LoRA + projector'})  "
                  f"train_loss {avg_train_loss:.4f}  val_loss {vl:.4f}  "
                  f"{(time.time()-t0)/60:.1f} min{'  <- best' if is_best else ''}", flush=True)

            history_train_loss.append(avg_train_loss)
            history_val_loss.append(vl)

            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(history_train_loss) + 1), history_train_loss, label='Train Loss', marker='o')
            plt.plot(range(1, len(history_val_loss) + 1), history_val_loss, label='Val (test.json) Loss', marker='o')
            plt.axvline(best_epoch, ls=":", color="green", label=f"best epoch ({best_epoch})")
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training vs Validation Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig('loss_curve.png')
            plt.close()

            if is_best:
                best_dir = os.path.join(args.output_dir, "best_model")
                base_model.visual.save_pretrained(os.path.join(best_dir, "enc_lora"))
                base_model.lm.save_pretrained(os.path.join(best_dir, "dec_lora"))
                torch.save(base_model.projector.state_dict(), os.path.join(best_dir, "projector.pt"))

    # ------------------------------------------------------------------ save last model
    if local_rank in [-1, 0]:
        last_dir = os.path.join(args.output_dir, "last_model")
        base_model.visual.save_pretrained(os.path.join(last_dir, "enc_lora"))
        base_model.lm.save_pretrained(os.path.join(last_dir, "dec_lora"))
        torch.save(base_model.projector.state_dict(), os.path.join(last_dir, "projector.pt"))

    # ------------------------------------------------------------------ final evaluation
    restore(best_state)
    
    if local_rank in [-1, 0]:
        print(f"\nEvaluating best checkpoint (epoch {best_epoch}) ...")
        
    final = {}
    for name, loss_dl, gen_dl in (("train", train_eval_dl, train_gen_dl), ("test", val_dl, val_gen_dl)):
        l = evaluate_loss(model, loss_dl, device, tok.eos_token_id, local_rank)
        a, parsed = evaluate_accuracy(
            model, gen_dl, tok, args.gen_tokens_acc, allowed_processor, device, local_rank, show=3 if name == "test" else 0
        )
        final[name] = (l, a, parsed, len(gen_dl.dataset))
        
    if local_rank in [-1, 0]:
        print(f"\nBest checkpoint: epoch {best_epoch}")
        print(f"{'split':<8}{'loss':>10}{'species_acc':>14}{'parsed':>10}   accuracy computed on N pairs")
        for name, (l, a, parsed, n) in final.items():
            print(f"{name:<8}{l:>10.4f}{a:>14.2%}{parsed:>10.1%}   N={n}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Gemma TraitGen', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
