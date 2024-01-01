#!/bin/bash

python convert_wild.py --source_path dataset/nerf_synthetic/mic --colmap_executable /share/davis/ruyu/colmap/build/__install__/bin/colmap

python train.py -s dataset/lego -m output/lego_noise --r_t_noise 0.005 0.005 --opt_cam

python train.py -s dataset/lego -m output/lego_noise --r_t_noise 0.005 0.005 --wandb --wandb_project_name lego_noise_opt --wandb_group_name with_optimization --wandb_mode "online" --opt_cam

python render.py -m output/drum -s dataset/drum --iteration 7000

python metrics.py -m output/gates_colmap_opt --eval_set "train"
