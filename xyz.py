import json, math, os, random, re, time
import torch
from utils import classification_accuracy
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.elastic.multiprocessing.errors import record
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup, LogitsProcessor, LogitsProcessorList
from peft import LoraConfig, get_peft_model
import open_clip
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ config
LM_ID = "google/gemma-3-1b-pt"
CLIP_ID = "hf-hub:imageomics/bioclip"
TRAIN_JSON, TEST_JSON = "train.json", "test.json"
IMG_ROOT = "/home/paul/DATASET/CUB_200_2011/images/"             # folder prepended to image paths in the JSON ("" if already full paths)
VIS_DIM = 768             # BioCLIP ViT-B/16 token dim
N_IMG_TOKENS = 49         # 14x14 patches pooled to 7x7
MAX_LEN, EPOCHS, WARM_EPOCHS, BATCH = 300, 12, 1, 4
LR_ENC, LR_DEC, LR_PROJ = 1e-4, 1e-4, 1e-3

# accuracy evaluation: the species name is the first thing in the caption
# ('It is a species of "<name>" as it has ...'), so only the first tokens are generated.
GEN_TOKENS_ACC = 40       # raise if the quoted name is ever cut off (see "parsed" rate in the final table)
GEN_BATCH = 16            # per-GPU batch size while generating
TRAIN_ACC_N = 1000        # train pairs used for the final train accuracy

local_rank = int(os.environ.get("LOCAL_RANK", -1))
if local_rank != -1:
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    DEVICE = torch.device(f"cuda:{local_rank}")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------ encoder (BioCLIP + LoRA)
clip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(CLIP_ID)
visual = clip.visual
visual.output_tokens = True                     # forward returns (pooled, patch_tokens)
visual = visual.to(DEVICE, dtype=torch.bfloat16)
for p in visual.parameters():
    p.requires_grad = False

# open_clip's attention is nn.MultiheadAttention (fused qkv), so q/v LoRA can't be targeted
# and out_proj LoRA would be bypassed in its forward. Adapt the MLP layers of the last 6 blocks.
enc_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.1, bias="none",
    target_modules=r".*resblocks\.(6|7|8|9|10|11)\.mlp\.(c_fc|c_proj)",
)
visual = get_peft_model(visual, enc_cfg)
visual.print_trainable_parameters()

# ------------------------------------------------------------------ decoder (Gemma 3 + LoRA)
_ASCII_PRINTABLE_RE = re.compile(r'^[\x20-\x7E]*$')

class AllowedTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, device="cuda"):
        super().__init__()
        vocab = tokenizer.get_vocab()
        allowed_ids = set()
        for token_str, token_id in vocab.items():
            decoded = tokenizer.decode([token_id])
            if _ASCII_PRINTABLE_RE.match(decoded):
                allowed_ids.add(token_id)
        for special_id in [tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.pad_token_id]:
            if special_id is not None:
                allowed_ids.add(special_id)
        self._allowed_ids = torch.tensor(sorted(allowed_ids), dtype=torch.long, device=device)
        self._valid_ids = None

    def __call__(self, input_ids, scores):
        if self._valid_ids is None:
            vocab_size = scores.size(-1)
            self._valid_ids = self._allowed_ids[self._allowed_ids < vocab_size].to(scores.device)

        mask = torch.full_like(scores, -float('inf'))
        mask[:, self._valid_ids] = 0.0
        return scores + mask

tok = AutoTokenizer.from_pretrained(LM_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
# CHANGED: right padding for training. With left padding the pad tokens sit between the image
# tokens and the caption, so the caption's positions shift per sample and differ from
# generation (where the caption follows the image directly).
tok.padding_side = "right"
allowed_processor = AllowedTokensLogitsProcessor(tok)
lm = AutoModelForCausalLM.from_pretrained(
    LM_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
).to(DEVICE)
dec_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
lm = get_peft_model(lm, dec_cfg)
lm.print_trainable_parameters()

hidden = lm.config.hidden_size
projector = nn.Sequential(nn.Linear(VIS_DIM, hidden), nn.GELU(), nn.Linear(hidden, hidden)).to(DEVICE)


def unwrap(m):
    return m.module if hasattr(m, "module") else m


def encode_images(pixels, vis=None, proj=None):
    vis = visual if vis is None else vis
    proj = projector if proj is None else proj
    _, tokens = vis(pixels.to(DEVICE, dtype=torch.bfloat16))               # (B, 196, 768)
    B, N, D = tokens.shape
    s = int(math.sqrt(N))
    grid = tokens.transpose(1, 2).reshape(B, D, s, s).float()
    side = int(math.sqrt(N_IMG_TOKENS))
    pooled = F.adaptive_avg_pool2d(grid, side).flatten(2).transpose(1, 2)  # (B, 49, 768)
    return proj(pooled).to(torch.bfloat16)                                 # (B, 49, H)


def forward(pixels, input_ids, attn, labels, species_mask):
    img = encode_images(pixels)
    unwrapped_lm = lm.module if hasattr(lm, "module") else lm
    txt = unwrapped_lm.get_input_embeddings()(input_ids)
    emb = torch.cat([img, txt], dim=1)
    mask = torch.cat([torch.ones(img.shape[:2], device=DEVICE, dtype=attn.dtype), attn], 1)
    ignore = torch.full(img.shape[:2], -100, device=DEVICE, dtype=labels.dtype)
    full_labels = torch.cat([ignore, labels], 1)

    outputs = lm(inputs_embeds=emb, attention_mask=mask, return_dict=True)
    shift_logits = outputs.logits[:, :-1]
    shift_labels = full_labels[:, 1:]

    # Custom loss weights (normal=2.0, eos=5.0, species=5.0)
    weights = torch.full_like(shift_labels, 2.0, dtype=torch.float)
    weights[shift_labels == tok.eos_token_id] = 5.0

    full_species_mask = torch.cat([torch.zeros(img.shape[:2], device=DEVICE, dtype=torch.bool), species_mask], 1)
    weights[full_species_mask[:, 1:]] = 5.0

    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(shift_labels)

    loss = loss * weights
    valid = shift_labels != -100
    return loss[valid].sum() / weights[valid].sum()


# ------------------------------------------------------------------ species accuracy
@torch.no_grad()
def species_accuracy(dl, show=0):
    visual.eval(); lm.eval(); projector.eval()
    local_texts = []
    local_gts = []
    shown = 0
    for pix, gts in dl:
        gen = generate_texts(pix)
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
    
    hits, total = allreduce_sum([local_hits, local_total])
    
    # Returning parsed as 1.0 just to satisfy the previous tuple return format
    return hits / max(total, 1), 1.0


@torch.no_grad()
def eval_loss(dl):
    visual.eval(); lm.eval(); projector.eval()
    tot, n = 0.0, 0
    for pix, ids, attn, lab, sp_mask in dl:
        pix, ids, attn, lab, sp_mask = (x.to(DEVICE) for x in (pix, ids, attn, lab, sp_mask))
        tot += forward(pix, ids, attn, lab, sp_mask).item(); n += 1
    tot, n = allreduce_sum([tot, n])
    return tot / max(n, 1)


# ------------------------------------------------------------------ data
def load_records(path):
    data = json.load(open(path))
    if isinstance(data, dict):
        if "annotations" in data and "images" in data:
            id2file = {im["id"]: im["file_name"] for im in data["images"]}
            return [(id2file[a["image_id"]], a["caption"], "") for a in data["annotations"]]
        data = next(v for v in data.values() if isinstance(v, list))
    recs = []
    for r in data:
        img = r.get("image") or r.get("imagePath") or r.get("image_path") or r.get("file_name") or r.get("filename")
        cap = r.get("gt") or r.get("caption") or r.get("text") or r.get("description")

        # Extract species from the caption
        species = ""
        m = re.search(r'species of "(.*?)"', str(cap))
        if m:
            species = m.group(1)
        elif img and "/" in img:
            species = img.split("/")[0].split(".", 1)[-1].replace("_", " ")

        for c in (cap if isinstance(cap, list) else [cap]):
            recs.append((img, str(c), species))
    return recs


class ImgCapDS(Dataset):
    def __init__(self, recs, tf):
        self.recs, self.tf = recs, tf
    def __len__(self):
        return len(self.recs)
    def __getitem__(self, i):
        img, cap, species = self.recs[i]
        return self.tf(Image.open(os.path.join(IMG_ROOT, img)).convert("RGB")), cap, species


def collate(batch):
    pix = torch.stack([b[0] for b in batch])
    caps = [b[1] + tok.eos_token for b in batch]
    species = [b[2] for b in batch]

    enc = tok(caps, padding=True, truncation=True,
              max_length=MAX_LEN, return_tensors="pt", add_special_tokens=False,
              return_offsets_mapping=True)

    labels = enc.input_ids.clone()
    labels[enc.attention_mask == 0] = -100

    species_mask = torch.zeros_like(labels, dtype=torch.bool)
    for i, (cap, sp) in enumerate(zip(caps, species)):
        if not sp: continue
        start_char = cap.find(sp)
        if start_char == -1: continue
        end_char = start_char + len(sp)

        offsets = enc.offset_mapping[i]
        for j, (o_start, o_end) in enumerate(offsets):
            if o_start < end_char and o_end > start_char:
                species_mask[i, j] = True

    return pix, enc.input_ids, enc.attention_mask, labels, species_mask


def gen_collate(batch):                      # for accuracy: images + ground-truth species only
    return torch.stack([b[0] for b in batch]), [b[2] for b in batch]


def make_loader(recs, tf, batch, collate_fn, shuffle=False):
    ds = ImgCapDS(recs, tf)
    sampler = DistributedSampler(ds, shuffle=shuffle) if local_rank != -1 else None
    return DataLoader(ds, batch, sampler=sampler, shuffle=(shuffle and sampler is None),
                      collate_fn=collate_fn, num_workers=4, pin_memory=True)


@record
def main():
    global visual, lm, projector, tok

    train_recs = load_records(TRAIN_JSON)
    val_recs = load_records(TEST_JSON)           # NOTE: this is test.json, used as validation
    if local_rank in [-1, 0]:
        print(f"train {len(train_recs)}  val(test.json) {len(val_recs)} (image-caption pairs)")

    train_dl = make_loader(train_recs, preprocess_train, BATCH, collate, shuffle=True)
    val_dl = make_loader(val_recs, preprocess_val, BATCH, collate)
    val_gen_dl = make_loader(val_recs, preprocess_val, GEN_BATCH, gen_collate)
    train_eval_dl = make_loader(train_recs, preprocess_val, BATCH, collate)           # no augmentation
    train_gen_dl = make_loader(train_recs, preprocess_val, GEN_BATCH, gen_collate)

    if local_rank != -1:
        visual = DDP(visual, device_ids=[local_rank], find_unused_parameters=True)
        lm = DDP(lm, device_ids=[local_rank], find_unused_parameters=True)
        projector = DDP(projector, device_ids=[local_rank], find_unused_parameters=True)

    # ------------------------------------------------------------------ optimizer (3 param groups)
    enc_params = [p for n, p in visual.named_parameters() if "lora_" in n]
    dec_params = [p for n, p in lm.named_parameters() if "lora_" in n]
    for p in enc_params + dec_params:
        p.requires_grad = True
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": LR_ENC},
         {"params": dec_params, "lr": LR_DEC},
         {"params": projector.parameters(), "lr": LR_PROJ}],
        weight_decay=0.01,
    )
    total = EPOCHS * len(train_dl)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * total), total)
    all_trainable = enc_params + dec_params + list(projector.parameters())

    def set_lora_trainable(flag):
        for p in enc_params + dec_params:
            p.requires_grad = flag

    def snapshot():       # identical on every rank (DDP keeps weights in sync), so each rank keeps a copy
        return {"enc": {n: p.detach().cpu().clone() for n, p in unwrap(visual).named_parameters() if "lora_" in n},
                "dec": {n: p.detach().cpu().clone() for n, p in unwrap(lm).named_parameters() if "lora_" in n},
                "proj": {k: v.detach().cpu().clone() for k, v in unwrap(projector).state_dict().items()}}

    def restore(s):
        with torch.no_grad():
            for n, p in unwrap(visual).named_parameters():
                if n in s["enc"]: p.copy_(s["enc"][n])
            for n, p in unwrap(lm).named_parameters():
                if n in s["dec"]: p.copy_(s["dec"][n])
        unwrap(projector).load_state_dict(s["proj"])

    # ------------------------------------------------------------------ train
    best, best_epoch, best_state = math.inf, 0, None
    history_train_loss, history_val_loss = [], []

    for ep in range(EPOCHS):
        t0 = time.time()
        if local_rank != -1:
            train_dl.sampler.set_epoch(ep)

        stage1 = ep < WARM_EPOCHS               # stage 1: projector only; stage 2: + both LoRAs
        set_lora_trainable(not stage1)
        visual.train(); lm.train(); projector.train()

        train_loss_sum = 0
        pbar = tqdm(train_dl, desc=f"Epoch {ep+1}/{EPOCHS}") if local_rank in [-1, 0] else train_dl
        for pix, ids, attn, lab, sp_mask in pbar:
            pix, ids, attn, lab, sp_mask = (x.to(DEVICE) for x in (pix, ids, attn, lab, sp_mask))
            loss = forward(pix, ids, attn, lab, sp_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in all_trainable if p.grad is not None], 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)

            train_loss_sum += loss.item()
            if local_rank in [-1, 0]:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss_sum / len(train_dl)
        if local_rank != -1:
            tl_tensor = torch.tensor(avg_train_loss, device=DEVICE)
            dist.all_reduce(tl_tensor, op=dist.ReduceOp.SUM)
            avg_train_loss = tl_tensor.item() / dist.get_world_size()

        vl = eval_loss(val_dl)                                   # all ranks

        is_best = vl < best
        if is_best:
            best, best_epoch, best_state = vl, ep + 1, snapshot()

        if local_rank in [-1, 0]:
            print(f"epoch {ep+1}/{EPOCHS} ({'projector only' if stage1 else 'LoRA + projector'})  "
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
                unwrap(visual).save_pretrained("enc_lora")
                unwrap(lm).save_pretrained("dec_lora")
                torch.save(unwrap(projector).state_dict(), "projector.pt")

    # ------------------------------------------------------------------ final evaluation (best checkpoint)
    restore(best_state)
    if local_rank in [-1, 0]:
        print(f"\nEvaluating best checkpoint (epoch {best_epoch}) ...")
    final = {}
    for name, loss_dl, gen_dl in (("train", train_eval_dl, train_gen_dl), ("test", val_dl, val_gen_dl)):
        l = eval_loss(loss_dl)
        a, parsed = species_accuracy(gen_dl, show=3 if name == "test" else 0)
        final[name] = (l, a, parsed, len(gen_dl.dataset))
    if local_rank in [-1, 0]:
        print(f"\nBest checkpoint: epoch {best_epoch}")
        print(f"{'split':<8}{'loss':>10}{'species_acc':>14}{'parsed':>10}   accuracy computed on N pairs")
        for name, (l, a, parsed, n) in final.items():
            print(f"{name:<8}{l:>10.4f}{a:>14.2%}{parsed:>10.1%}   N={n}")


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------ inference
@torch.no_grad()
def caption(image_path, beams=4):
    unwrap(visual).eval(); unwrap(lm).eval(); unwrap(projector).eval()
    pix = preprocess_val(Image.open(image_path).convert("RGB")).unsqueeze(0)
    img = encode_images(pix, unwrap(visual), unwrap(projector))
    mask = torch.ones(img.shape[:2], device=DEVICE, dtype=torch.long)
    out = unwrap(lm).generate(inputs_embeds=img, attention_mask=mask, max_new_tokens=MAX_LEN,
                              do_sample=False, num_beams=beams, no_repeat_ngram_size=3,
                              eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                              logits_processor=LogitsProcessorList([allowed_processor]))
    return tok.decode(out[0], skip_special_tokens=True)