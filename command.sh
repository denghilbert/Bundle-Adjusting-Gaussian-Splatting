#!/bin/bash

python convert_wild.py --source_path dataset/nerf_synthetic/mic --colmap_executable /share/davis/ruyu/colmap/build/__install__/bin/colmap

python train.py -s dataset/lego -m output/lego_noise --r_t_noise 0.005 0.005 --opt_cam

python train.py -s dataset/lego -m output/lego_noise --r_t_noise 0.005 0.005 --wandb --wandb_project_name lego_noise_opt --wandb_group_name with_optimization --wandb_mode "online" --opt_cam
python train.py -s dataset/nerf_synthetic/lego/ -m output/lego --r_t_noise 0.2 0.2 --test_iterations 7000 20000 30000 --iterations 30000 --eval --wandb_project_name global_align --wandb_group_name find_outlier --wandb_mode "online" --r_t_lr 0.01 0.05 --opt_cam --global_alignment_lr 0.00 --white_background --vis_pose

python render.py -m output/drum -s dataset/drum --iteration 7000

python metrics.py -m output/gates_colmap_opt --eval_set "train"
