#!/bin/bash

# Define the path where you want to list directories
path_to_directories="dataset_cvpr/smerf"

# Iterate over each directory in the specified path
for lr in 1e-8; do
    for dir in "$path_to_directories"/*; do
        # Check if it's a directory
        if [ -d "$dir" ]; then
            name=$(basename "$dir")
            flow_scale_x=2.
            flow_scale_y=2.
            if echo "$name" | grep -qE "berlin"; then
                flow_scale_x=4.
                flow_scale_y=4.
            fi
            #if [ -f "smerf/${name}_${lr}_flowx${flow_scale_x}_flowy${flow_scale_y}_allimgs/chkpnt30000.pth" ]; then
            #    echo "All ready trained under smerf/${name}_${lr}_flowx${flow_scale_x}_flowy${flow_scale_y}_allimgs"
            #    continue
            #fi
            echo python train_outside.py -s $dir -m "smerf/${name}_${lr}_flowx${flow_scale_x}_flowy${flow_scale_y}_denfrom500_reset3000" --r_t_noise 0.0 0.0 1. --test_iterations 1 10000 20000 30000 40000 --save_iterations 10000 20000 30000 40000 --checkpoint_iterations 10000 20000 30000 40000 --iterations 40000 --r_t_lr 0.002 0.002 --control_point_sample_scale 8 --extend_scale 10000 --opt_distortion --outside_rasterizer --flow_scale $flow_scale_x $flow_scale_y --apply2gt --iresnet_lr $lr --wandb_project_name smerf --wandb_mode online --port 13112
            python train_outside.py -s $dir -m "smerf/${name}_${lr}_flowx${flow_scale_x}_flowy${flow_scale_y}_denfrom500_reset3000" --r_t_noise 0.0 0.0 1. --test_iterations 1 10000 20000 30000 40000 --save_iterations 10000 20000 30000 40000 --checkpoint_iterations 10000 20000 30000 40000 --iterations 40000 --r_t_lr 0.002 0.002 --control_point_sample_scale 8 --extend_scale 10000 --opt_distortion --outside_rasterizer --flow_scale $flow_scale_x $flow_scale_y --apply2gt --iresnet_lr $lr --wandb_project_name smerf --wandb_mode online --port 13112
        fi
    done
done
