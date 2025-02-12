#!/bin/bash

python train.py -s example_datasets/cubemap/hilbert_largefov/ -m output/hilbert \
  --r_t_noise 0.0 0.0 1. --test_iterations 7000 15000 20000 --save_iterations 7000 15000 20000 \
  --checkpoint_iterations 7000 15000 20000 --iterations 20000 --r_t_lr 0.002 0.002 --cubemap \
  --no_init_iresnet --wandb_project_name release_code --wandb_mode online --opacity_reset_interval 20000 \
  --densify_until_iter 20000 --port 21112 --eval --iresnet_opt_duration 0 7000 --control_point_sample_scale 8 \
  --iresnet_lr 1e-9 --mask_radius 512

python train.py -s example_datasets/cubemap/living_largefov/ -m output/living \
  --r_t_noise 0.0 0.0 1. --test_iterations 7000 15000 20000 --save_iterations 7000 15000 20000 \
  --checkpoint_iterations 7000 15000 20000 --iterations 20000 --r_t_lr 0.002 0.002 --cubemap \
  --no_init_iresnet --wandb_project_name release_code --wandb_mode online --opacity_reset_interval 20000 \
  --densify_until_iter 20000 --port 21111 --eval --iresnet_opt_duration 0 7000 --control_point_sample_scale 8 \
  --iresnet_lr 1e-9 --mask_radius 400

python train.py -s example_datasets/cubemap/kitchen_largefov/ -m output/kitchen \
  --r_t_noise 0.0 0.0 1. --test_iterations 7000 15000 20000 --save_iterations 7000 15000 20000 \
  --checkpoint_iterations 7000 15000 20000 --iterations 20000 --r_t_lr 0.002 0.002 --cubemap \
  --no_init_iresnet --wandb_project_name release_code --wandb_mode online --opacity_reset_interval 20000 \
  --densify_until_iter 20000 --port 21110 --eval --iresnet_opt_duration 0 7000 --control_point_sample_scale 8 \
  --iresnet_lr 1e-9 --mask_radius 400
