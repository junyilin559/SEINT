#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Training EDM model on QM9 dataset
cd e3_diffusion_for_molecules
python main_qm9.py --n_epochs 3001 \
  --exp_name SEINT-0.3 \
  --n_stability_samples 1000 \
  --diffusion_noise_schedule polynomial_2 \
  --diffusion_noise_precision 1e-5 \
  --diffusion_steps 1000 \
  --diffusion_loss_type l2 \
  --batch_size 64 \
  --nf 256 \
  --n_layers 9 \
  --lr 1e-4 \
  --normalize_factors [1,4,10] \
  --test_epochs 20 \
  --ema_decay 0.9999 \
  --no_wandb \
  --gpu_id 0 \
  --output_dir SEINT_outputs

# Training UniGEM model on QM9 dataset
cd ../UniGEM
python -u main_qm9.py --n_epochs 3001 \
   --exp_name SEINT-0.3 \
   --n_stability_samples 1000 \
   --diffusion_noise_schedule polynomial_2 \
   --diffusion_noise_precision 1e-5 \
   --diffusion_steps 1000 \
   --diffusion_loss_type l2 \
   --batch_size 64 \
   --nf 256 \
   --n_layers 9 \
   --lr 1e-4 \
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
   --no_wandb