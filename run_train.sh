#!/bin/bash
# run_train.sh
# Training script using Distributed Data Parallel (DDP)
# Needs torchrun even for a single GPU.

torchrun --nproc_per_node=2 main.py \
    --data_root "/home/paul/DATASET/CUB_200_2011" \
    --output_dir "output" \
    --lr 1e-4 \
    --epochs 20 \
    --optimizer "adamw" \
    --max_seq_len 256 \
    --batch_size 2 \
    --encoder_model "hf-hub:imageomics/bioclip" \
    --encoder_op_dim 768 \
    --decoder_model "google/gemma-3-1b-pt" \
    --streeing_prompt "species identification and corresponding textual explanation task." \
    --ann_dir "." \
    --validate_model 0 \
    --load_path "scratch" \