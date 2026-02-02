#!/bin/bash

# Fine-tune UniGEM model on QM9 dataset
cd UniGEM
export CUDA_VISIBLE_DEVICES=0
python -u main_qm9.py --n_epochs 1001 \
  --exp_name QM9_SEINT-0.3 \
  --n_stability_samples 1000 \
  --diffusion_noise_schedule polynomial_2 \
  --diffusion_noise_precision 1e-6 \
  --diffusion_steps 1000 \
  --diffusion_loss_type l2 \
  --batch_size 64 \
  --nf 256 \
  --n_layers 9 \
  --lr 1e-5 \
  --normalize_factors [1,4,10] \
  --test_epochs 20 \
  --ema_decay 0.9999 \
  --property_pred 1 \
  --prediction_threshold_t 10 \
  --model DGAP \
  --sep_noisy_node 1 \
  --target_property lumo \
  --atom_type_pred 1 \
  --branch_layers_num 8 \
  --use_prop_pred 1 \
  --resume outputs/UniGEM_QM9_ckpt \
  --start_epoch 2000 \
  --no_wandb


# Fine-tune UniGEM model on DRUG dataset
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -u main_geom_drugs.py \
  --n_epochs 3 --exp_name Drug_SEINT-0.3 \
  --n_stability_samples 500 \
  --diffusion_noise_schedule polynomial_2 \
  --diffusion_noise_precision 1e-5 \
  --diffusion_steps 1000 \
  --diffusion_loss_type l2 \
  --batch_size 32 --nf 256 \
  --n_layers 4 --lr 1e-5 \
  --normalize_factors [1,4,10] \
  --test_epochs 1 \
  --ema_decay 0.9999 \
  --prediction_threshold_t 10 \
  --model DGAP \
  --sep_noisy_node 1 \
  --target_property lumo \
  --atom_type_pred 1 \
  --branch_layers_num 3 \
  --normalization_factor 1 \
  --resume outputs/UniGEM_DRUG_ckpt \
  --start_epoch 13 \
  --no_wandb
