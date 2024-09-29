#!/bin/bash

# Define the path where you want to list directories
path_to_directories="dataset_cvpr/fisheyenerf"

# Iterate over each directory in the specified path
for dir in "$path_to_directories"/*; do
    # Check if it's a directory
    if [ -d "$dir" ]; then
        name=$(basename "$dir")
        if [ -f "cvpr/${name}_lr7_apply2render_res2/chkpnt30000.pth" ]; then
            echo "All ready trained under cvpr/${name}_lr7_apply2render_res2"
            continue
        fi
        echo python train_outside.py -s $dir -m "cvpr/${name}_lr7_apply2render_res2" --r_t_noise 0.0 0.0 1. --test_iterations 1 7000 20000 30000 --save_iterations 7000 20000 30000 --checkpoint_iterations 7000 20000 30000 --iterations 30000 --eval --r_t_lr 0.002 0.002 --control_point_sample_scale 32 --extend_scale 10000 --opt_distortion --outside_rasterizer --flow_scale 2. 2. --iresnet_lr 1e-7 --wandb_project_name cvpr --wandb_group_name "cvpr/${name}_lr7_apply2render_res2" --wandb_mode online --render_resolution 2
        python train_outside.py -s $dir -m "cvpr/${name}_lr7_apply2render_res2" --r_t_noise 0.0 0.0 1. --test_iterations 1 7000 20000 30000 --save_iterations 7000 20000 30000 --checkpoint_iterations 7000 20000 30000 --iterations 30000 --eval --r_t_lr 0.002 0.002 --control_point_sample_scale 32 --extend_scale 10000 --opt_distortion --outside_rasterizer --flow_scale 2. 2. --iresnet_lr 1e-7 --wandb_project_name cvpr --wandb_group_name "cvpr/${name}_lr7_apply2render_res2" --wandb_mode online --render_resolution 2
    fi
done
