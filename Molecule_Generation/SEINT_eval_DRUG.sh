#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Evaluate UniGEM model on DRUG dataset
cd UniGEM
python -u eval_analyze.py \
  --model_path outputs/Drug_SEINT-0.3_resume \
  --n_samples 10000 \
  --save_to_xyz 0 \
  --checkpoint_epoch 15