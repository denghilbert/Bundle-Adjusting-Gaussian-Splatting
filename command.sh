#!/bin/bash

python convert_wild.py --source_path dataset/nerf_synthetic/mic --colmap_executable /usr/bin/colmap

python train.py -s dataset/lego -m output/lego_noise --r_t_noise 0.005 0.005 --opt_cam

python render.py -m output/drum -s dataset/drum --iteration 7000

python metrics.py -m output/gates_colmap_opt --eval_set "train"
