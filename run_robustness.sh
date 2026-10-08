#!/bin/bash
# run_robustness.sh
# Script to evaluate total overall accuracy and model robustness under different perturbations

# Make sure to replace this path with the actual path to your trained model checkpoint
LOAD_PATH="output/best_model.pth"
DATA_ROOT="/home/paul/DATASET/CUB_200_2011/images"
ANN_DIR="."

echo "=========================================================="
echo "1. Evaluating Total Overall Accuracy (No Perturbations)"
echo "=========================================================="
python evaluate_robustness.py \
    --data_root $DATA_ROOT \
    --ann_dir $ANN_DIR \
    --load_path $LOAD_PATH \
    --perturbation none

echo ""
echo "=========================================================="
echo "2. Evaluating Robustness against Gaussian Blur (radius=2.0)"
echo "=========================================================="
python evaluate_robustness.py \
    --data_root $DATA_ROOT \
    --ann_dir $ANN_DIR \
    --load_path $LOAD_PATH \
    --perturbation blur \
    --severity 2.0

echo ""
echo "=========================================================="
echo "3. Evaluating Robustness against Additive Noise (std=0.1)"
echo "=========================================================="
python evaluate_robustness.py \
    --data_root $DATA_ROOT \
    --ann_dir $ANN_DIR \
    --load_path $LOAD_PATH \
    --perturbation noise \
    --severity 0.1

echo ""
echo "=========================================================="
echo "4. Evaluating Robustness against Missing Parts (25% Occlusion)"
echo "=========================================================="
python evaluate_robustness.py \
    --data_root $DATA_ROOT \
    --ann_dir $ANN_DIR \
    --load_path $LOAD_PATH \
    --perturbation occlusion \
    --severity 0.25
