#!/bin/bash

# Force HF to use the standard, stable downloader instead of Xet chunks
export HF_HUB_DISABLE_XET="1"
export HF_HUB_DOWNLOAD_TIMEOUT="120"
export HF_HUB_ETAG_TIMEOUT="30"

# Using torchrun for DDP (Distributed Data Parallel) across 1 GPU
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train_gemma.py \
    --train_json "train.json" \
    --test_json "test.json" \
    --img_root "/home/paul/DATASET/CUB_200_2011/images/" \
    --output_dir "output_lr1e-4" \
    --epochs 10 \
    --warm_epochs 1 \
    --batch_size 4 \
    --gen_batch_size 16 \
    --lr_enc 1e-6 \
    --lr_dec 1e-6 \
    --lr_proj 1e-5
