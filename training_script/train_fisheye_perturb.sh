#!/bin/bash

# Define the path where you want to list directories
path_to_directories="dataset/FisheyeNeRF"

# Iterate over each directory in the specified path
for dir in "$path_to_directories"/*; do
    # Check if it's a directory
    if [ -d "$dir" ]; then
        name=$(basename "$dir")
        echo python train_neural.py -s $dir -m "fisheye_perturb/01noise_opt0005_$name" --r_t_noise 0.1 0.1 --test_iterations 7000 10000 20000 30000 --save_iterations 7000 10000 20000 30000 --iterations 30000 --eval --r_t_lr 0.005 0.01 --port 11122 --opt_cam --start_opt_lens 7000 --opt_distortion
        python train_neural.py -s $dir -m "fisheye_perturb/01noise_opt0005_$name" --r_t_noise 0.1 0.1 --test_iterations 7000 10000 20000 30000 --save_iterations 7000 10000 20000 30000 --iterations 30000 --eval --r_t_lr 0.005 0.01 --port 11122 --opt_cam --start_opt_lens 7000 --opt_distortion
        echo python train_neural.py -s $dir -m "fisheye_perturb/01noise_wo_opt_$name" --r_t_noise 0.1 0.1 --test_iterations 7000 10000 20000 30000 --save_iterations 7000 10000 20000 30000 --iterations 30000 --eval --r_t_lr 0.01 0.05 --port 11122 --start_opt_lens 7000 --opt_distortion
        python train_neural.py -s $dir -m "fisheye_perturb/01noise_wo_opt_$name" --r_t_noise 0.1 0.1 --test_iterations 7000 10000 20000 30000 --save_iterations 7000 10000 20000 30000 --iterations 30000 --eval --r_t_lr 0.01 0.05 --port 11122 --start_opt_lens 7000 --opt_distortion
    fi
done
