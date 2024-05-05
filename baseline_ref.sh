#!/bin/bash

# Define the path where you want to list directories
path_to_directories="dataset/FisheyeNeRF"

# Iterate over each directory in the specified path
for dir in "$path_to_directories"/*; do
    # Check if it's a directory
    if [ -d "$dir" ]; then
        name=$(basename "$dir")
        #echo python train_ref.py -s $dir -m "baseline/ref_$name" --r_t_noise 0.0 0.0 --test_iterations 1 7000 10000 15000 20000 25000 30000 40000 50000 60000 --save_iterations 7000 10000 20000 30000 40000 50000 60000 --iterations 30000 --eval --r_t_lr 0.01 0.05 --extend_scale 4.
        #python train_ref.py -s $dir -m "baseline/ref_$name" --r_t_noise 0.0 0.0 --test_iterations 1 7000 10000 15000 20000 25000 30000 40000 50000 60000 --save_iterations 7000 10000 20000 30000 40000 50000 60000 --iterations 30000 --eval --r_t_lr 0.01 0.05 --extend_scale 4.
        echo python render.py -s $dir -m "baseline/ref_$name" --iteration 30000 --skip_train --extend_scale 4.
        python render.py -s $dir -m "baseline/ref_$name" --iteration 30000 --skip_train --extend_scale 4.
        echo python metrics.py -m "baseline/ref_$name" --eval_set 'test'
        python metrics.py -m "baseline/ref_$name" --eval_set 'test'
    fi
done
