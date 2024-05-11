#!/bin/bash

# Define the path where you want to list directories
path_to_directories="dataset/record3d1"

# Iterate over each directory in the specified path
for dir in "$path_to_directories"/*; do
    # Check if it's a directory
    if [ -d "$dir" ]; then
        name=$(basename "$dir")
        #echo python train_vanilla.py -s $dir -m "arkit_perturb/opt001_$name" --r_t_noise 0. 0. --test_iterations 7000 10000 20000 30000 --save_iterations 7000 10000 20000 30000 --iterations 30000 --eval --r_t_lr 0.01 0.025 --port 10002 --opt_cam --vis_pose
        #python train_vanilla.py -s $dir -m "arkit_perturb/opt001_$name" --r_t_noise 0. 0. --test_iterations 7000 10000 20000 30000 --save_iterations 7000 10000 20000 30000 --iterations 30000 --eval --r_t_lr 0.01 0.025 --port 10002 --opt_cam --vis_pose
        #echo python train_vanilla.py -s $dir -m "arkit_perturb/opt0005$name" --r_t_noise 0. 0. --test_iterations 7000 10000 20000 30000 --save_iterations 7000 10000 20000 30000 --iterations 30000 --eval --r_t_lr 0.005 0.01 --port 10002 --opt_cam --vis_pose
        #python train_vanilla.py -s $dir -m "arkit_perturb/opt0005$name" --r_t_noise 0. 0. --test_iterations 7000 10000 20000 30000 --save_iterations 7000 10000 20000 30000 --iterations 30000 --eval --r_t_lr 0.005 0.01 --port 10002 --opt_cam --vis_pose
        #echo python train_vanilla.py -s $dir -m "arkit_perturb/wo_opt_$name" --r_t_noise 0. 0. --test_iterations 7000 10000 20000 30000 --save_iterations 7000 10000 20000 30000 --iterations 30000 --eval --r_t_lr 0.01 0.05 --port 10002 --vis_pose
        #python train_vanilla.py -s $dir -m "arkit_perturb/wo_opt_$name" --r_t_noise 0. 0. --test_iterations 7000 10000 20000 30000 --save_iterations 7000 10000 20000 30000 --iterations 30000 --eval --r_t_lr 0.01 0.05 --port 10002 --vis_pose
    fi
done
