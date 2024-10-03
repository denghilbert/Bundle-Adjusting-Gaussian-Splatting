


#!/bin/bash

# Define the path where you want to list directories
path_to_directories="dataset_cvpr/zip_fish"

# Iterate over each directory in the specified path

for dir in "$path_to_directories"/*; do
    # Check if it's a directory
    if [ -d "$dir" ]; then
        name=$(basename "$dir")
        if [[ "$name" == *"office"* ]]; then
            echo "Skipping directory $name as it contains 'office'."
            continue
        fi
        echo python train_outside.py -s $dir -m "cvpr/mitsuba/undis_$name" --r_t_noise 0. 0. 1.0 --test_iterations 1 7000 15000 30000 --save_iterations 7000 15000 30000 --iterations 30000 --wandb_project_name mitsuba --wandb_mode online --port 11110
        python train_outside.py -s $dir -m "cvpr/mitsuba/undis_$name" --r_t_noise 0. 0. 1.0 --test_iterations 1 7000 15000 30000 --save_iterations 7000 15000 30000 --iterations 30000 --wandb_project_name mitsuba --wandb_mode online --port 11110
    fi
done
