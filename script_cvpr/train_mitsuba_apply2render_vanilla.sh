#!/bin/bash

# Define the path where you want to list directories
path_to_directories="dataset_cvpr/dataset_mitsuba"

# Iterate over each directory in the specified path

for dir in "$path_to_directories"/*; do
    # Check if it's a directory
    if [ -d "$dir" ]; then
        name=$(basename "$dir")
        #if [[ "$name" == *"office"* ]]; then
        #    echo "Skipping directory $name as it contains 'office'."
        #    continue
        #fi
        #if [ -f "cvpr/mitsuba/${name}_init_$lr/chkpnt30000.pth" ]; then
        #    echo "skipping cvpr/mitsuba/${name}_init_$lr"
        #    continue
        #fi
        if [[ "$name" == *"rover"* ]]; then
            echo python train_outside.py -s $dir -m "cvpr/mitsuba/${name}_vanilla" --r_t_noise 0.0 0.0 1. --test_iterations 1 7000 20000 30000 --save_iterations 7000 20000 30000 --checkpoint_iterations 7000 20000 30000 --iterations 30000 --port 11424 --wandb_project_name mitsuba --wandb_mode online -w
            #python train_outside.py -s $dir -m "cvpr/mitsuba/${name}_vanilla" --r_t_noise 0.0 0.0 1. --test_iterations 1 7000 20000 30000 --save_iterations 7000 20000 30000 --checkpoint_iterations 7000 20000 30000 --iterations 30000 --port 11424 --wandb_project_name mitsuba --wandb_mode online -w
        else
            echo python train_outside.py -s $dir -m "cvpr/mitsuba/${name}_vanilla" --r_t_noise 0.0 0.0 1. --test_iterations 1 7000 20000 30000 --save_iterations 7000 20000 30000 --checkpoint_iterations 7000 20000 30000 --iterations 30000 --port 11424 --wandb_project_name mitsuba --wandb_mode online
            #python train_outside.py -s $dir -m "cvpr/mitsuba/${name}_vanilla" --r_t_noise 0.0 0.0 1. --test_iterations 1 7000 20000 30000 --save_iterations 7000 20000 30000 --checkpoint_iterations 7000 20000 30000 --iterations 30000 --port 11424 --wandb_project_name mitsuba --wandb_mode online
        fi
    fi
done
