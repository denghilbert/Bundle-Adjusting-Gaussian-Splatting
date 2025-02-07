#!/bin/bash

DATASET_ROOT="example_datasets/single_planar"

for scene in "$DATASET_ROOT"/*; do
  if [ -d "$scene" ]; then
    scene_name=$(basename "$scene")

    CMD="python train.py -s example_datasets/single_planar/${scene_name}/ \
        -m output/${scene_name} \
        --r_t_noise 0.0 0.0 1. --test_iterations 3000 10000 20000 30000 \
        --save_iterations 3000 10000 20000 30000 --checkpoint_iterations 3000 10000 20000 30000 \
        --iterations 30000 --eval --r_t_lr 0.002 0.002 --control_point_sample_scale 16 \
        --extend_scale 10000 --opt_distortion --outside_rasterizer --flow_scale 2. 2. \
        --iresnet_lr 1e-7 --wandb_project_name release_code --wandb_mode online --port 11112 \
        --opacity_reset_interval 100000 --densify_until_iter 100000 --iresnet_opt_duration 0 7000"

    echo "Running: $CMD"
    eval $CMD
  fi
done
