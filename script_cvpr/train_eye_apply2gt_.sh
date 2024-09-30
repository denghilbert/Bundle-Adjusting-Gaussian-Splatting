#!/bin/bash

# Define the path where you want to list directories
path_to_directories="dataset_cvpr/eyeful"

# Iterate over each directory in the specified path

for flow_scale in 1.5 1. 2.; do
    for lr in 1e-7 1e-8 1e-9; do
        for dir in "$path_to_directories"/*; do
            # Check if it's a directory
            if [ -d "$dir" ]; then
                name=$(basename "$dir")
                if [[ "$name" == *"office"* ]]; then
                    echo "Skipping directory $name as it contains 'office'."
                    continue
                fi
                echo python train_outside.py -s $dir -m "eyeful/${name}_lr${lr}_optcam_scale${flow_scale}_denfrom1_noreset" --r_t_noise 0.0 0.0 1. --test_iterations 1 7000 20000 30000 40000 --save_iterations 7000 20000 30000 40000 --checkpoint_iterations 7000 20000 30000 40000 --iterations 40000 --r_t_lr 0.002 0.002 --control_point_sample_scale 4 --extend_scale 10000 --opt_distortion --outside_rasterizer --opt_cam --flow_scale $flow_scale $flow_scale --iresnet_lr $lr --apply2gt --wandb_project_name eyeful --wandb_group_name ${name}_lr${lr}_optcam_scale${flow_scale} --wandb_mode online --port 12413 --opacity_reset_interval 50000 --densify_from_iter 1 --densify_until_iter 40000
                python train_outside.py -s $dir -m "eyeful/${name}_lr${lr}_optcam_scale${flow_scale}_denfrom1_noreset" --r_t_noise 0.0 0.0 1. --test_iterations 1 7000 20000 30000 40000 --save_iterations 7000 20000 30000 40000 --checkpoint_iterations 7000 20000 30000 40000 --iterations 40000 --r_t_lr 0.002 0.002 --control_point_sample_scale 4 --extend_scale 10000 --opt_distortion --outside_rasterizer --opt_cam --flow_scale $flow_scale $flow_scale --iresnet_lr $lr --apply2gt --wandb_project_name eyeful --wandb_group_name ${name}_lr${lr}_optcam_scale${flow_scale} --wandb_mode online --port 12413 --opacity_reset_interval 50000 --densify_from_iter 1 --densify_until_iter 40000
            fi
        done
    done
done
