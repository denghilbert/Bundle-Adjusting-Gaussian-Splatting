#!/bin/bash

# Define the path where you want to list directories
path_to_directories="/home/yd428/gaussian-splatting/dataset/fisheye_perspective"

# Iterate over each directory in the specified path
for dir in "$path_to_directories"/*; do
    # Check if it's a directory
    if [ -d "$dir" ]; then
        name=$(basename "$dir")
        echo python train_vanilla.py -s $dir -m "fisheye_perspective_perturb/$name" --r_t_noise 0.05 0.05 --test_iterations 7000 15000 30000 --save_iterations 7000 15000 30000 --iterations 30000 --eval --r_t_lr 0.01 0.025 --port 10001
        python train_vanilla.py -s $dir -m "fisheye_perspective_perturb/$name" --r_t_noise 0.05 0.05 --test_iterations 7000 15000 30000 --save_iterations 7000 15000 30000 --iterations 30000 --eval --r_t_lr 0.01 0.025 --port 10001
    fi
done



path_to_directories="dataset/FisheyeNeRF"
for dir in "$path_to_directories"/*; do
    # Check if it's a directory
    if [ -d "$dir" ]; then
        name=$(basename "$dir")
        echo python render_undist.py -s $dir -m "fisheye_perspective_perturb/$name" --iteration 30000 --skip_test --render_fisheyefov_from_colmap_perspective
        python render_undist.py -s $dir -m "fisheye_perspective_perturb/$name" --iteration 30000 --skip_test --render_fisheyefov_from_colmap_perspective
    fi
done
