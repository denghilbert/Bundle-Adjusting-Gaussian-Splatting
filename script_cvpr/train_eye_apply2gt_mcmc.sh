#!/bin/bash

# Define the path where you want to list directories
path_to_directories="dataset_cvpr/eyeful"

# Iterate over each directory in the specified path
for dir in "$path_to_directories"/*; do
    # Check if it's a directory
    if [ -d "$dir" ]; then
        name=$(basename "$dir")
        echo python train_outside.py -s $dir -m "eyeful/${name}_lr7_optcam_scale1.5_mcmc" --r_t_noise 0.0 0.0 1. --test_iterations 1 7000 20000 30000 --save_iterations 7000 20000 30000 40000 --checkpoint_iterations 7000 20000 30000 40000 --iterations 40000 --r_t_lr 0.002 0.002 --control_point_sample_scale 4 --extend_scale 10000 --opt_distortion --outside_rasterizer --opt_cam --flow_scale 1.5 1.5 --iresnet_lr 1e-7 --apply2gt --wandb_project_name eyeful --wandb_group_name ${name}_lr7_optcam_scale1.5_mcmc --wandb_mode online --port 11114 --cap_max 3000000 --no_distortion_mask --densification_interval 200 --mcmc
        python train_outside.py -s $dir -m "eyeful/${name}_lr7_optcam_scale1.5_mcmc" --r_t_noise 0.0 0.0 1. --test_iterations 1 7000 20000 30000 --save_iterations 7000 20000 30000 40000 --checkpoint_iterations 7000 20000 30000 40000 --iterations 40000 --r_t_lr 0.002 0.002 --control_point_sample_scale 4 --extend_scale 10000 --opt_distortion --outside_rasterizer --opt_cam --flow_scale 1.5 1.5 --iresnet_lr 1e-7 --apply2gt --wandb_project_name eyeful --wandb_group_name ${name}_lr7_optcam_scale1.5_mcmc --wandb_mode online --port 11114 --cap_max 3000000 --no_distortion_mask --densification_interval 200 --mcmc
    fi
done
