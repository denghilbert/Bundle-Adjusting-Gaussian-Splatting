#!/bin/bash

# Define the path where you want to list directories
path_to_directories="dataset_cvpr/dataset_mitsuba"

# Iterate over each directory in the specified path

for flow_scale in 2.; do
    for lr in 1e-7 1e-8 1e-9 1e-10; do
        for dir in "$path_to_directories"/*; do
            # Check if it's a directory
            if [ -d "$dir" ]; then
                name=$(basename "$dir")
                #if [[ "$name" == *"office"* ]]; then
                #    echo "Skipping directory $name as it contains 'office'."
                #    continue
                #fi
                #if [ -f "cvpr/mitsuba/${name}_woinit_$lr/chkpnt30000.pth" ]; then
                #    echo "skipping cvpr/mitsuba/${name}_woinit_$lr"
                #    continue
                #fi
                if [[ "$name" == *"rover"* ]]; then
                    echo python train_outside.py -s $dir -m "cvpr/mitsuba/${name}_woinit_$lr" --r_t_noise 0.0 0.0 1. --test_iterations 1 7000 20000 30000 --save_iterations 7000 20000 30000 --checkpoint_iterations 7000 20000 30000 --iterations 30000 --r_t_lr 0.002 0.002 --control_point_sample_scale 32 --extend_scale 10000 --opt_distortion --outside_rasterizer --flow_scale $flow_scale $flow_scale --iresnet_lr $lr --port 11433 --wandb_project_name mitsuba --wandb_mode online --no_init_iresnet -w
                    python train_outside.py -s $dir -m "cvpr/mitsuba/${name}_woinit_$lr" --r_t_noise 0.0 0.0 1. --test_iterations 1 7000 20000 30000 --save_iterations 7000 20000 30000 --checkpoint_iterations 7000 20000 30000 --iterations 30000 --r_t_lr 0.002 0.002 --control_point_sample_scale 32 --extend_scale 10000 --opt_distortion --outside_rasterizer --flow_scale $flow_scale $flow_scale --iresnet_lr $lr --port 11433 --wandb_project_name mitsuba --wandb_mode online --no_init_iresnet -w
                else
                    echo python train_outside.py -s $dir -m "cvpr/mitsuba/${name}_woinit_$lr" --r_t_noise 0.0 0.0 1. --test_iterations 1 7000 20000 30000 --save_iterations 7000 20000 30000 --checkpoint_iterations 7000 20000 30000 --iterations 30000 --r_t_lr 0.002 0.002 --control_point_sample_scale 32 --extend_scale 10000 --opt_distortion --outside_rasterizer --flow_scale $flow_scale $flow_scale --iresnet_lr $lr --port 11433 --wandb_project_name mitsuba --wandb_mode online --no_init_iresnet
                    python train_outside.py -s $dir -m "cvpr/mitsuba/${name}_woinit_$lr" --r_t_noise 0.0 0.0 1. --test_iterations 1 7000 20000 30000 --save_iterations 7000 20000 30000 --checkpoint_iterations 7000 20000 30000 --iterations 30000 --r_t_lr 0.002 0.002 --control_point_sample_scale 32 --extend_scale 10000 --opt_distortion --outside_rasterizer --flow_scale $flow_scale $flow_scale --iresnet_lr $lr --port 11433 --wandb_project_name mitsuba --wandb_mode online --no_init_iresnet
                fi
            fi
        done
    done
done
