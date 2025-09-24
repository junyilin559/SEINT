#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Evaluate EDM model on QM9 dataset
cd e3_diffusion_for_molecules
python -u eval_analyze.py \
  --model_path SEINT_outputs/SEINT-0.3 \
  --n_samples 10_000 \
  --save_to_xyz 1 \
  --checkpoint_epoch 3000

# Evaluate UniGEM model on QM9 dataset
cd ../UniGEM
python -u eval_analyze.py \
   --model_path outputs/SEINT-0.3 \
   --n_samples 10_000 \
   --save_to_xyz 1 \
   --checkpoint_epoch 3000