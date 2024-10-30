#!/bin/bash

scenes=("garden" "living_room" "other_room")
render_resolutions=(1 2 4)
flow_scales=(1.5 2 2.5)

gpu_id=0

wait_for_free_gpu_slot() {
    while [ "$(jobs | wc -l)" -ge 8 ]; do
        sleep 1
    done
}

trap 'echo "Stopping..."; kill $(jobs -p); exit' SIGINT

for scene in "${scenes[@]}"; do
    for render_resolution in "${render_resolutions[@]}"; do
        for flow_scale in "${flow_scales[@]}"; do

            wait_for_free_gpu_slot

            output_dir="netflix/${scene}_highres_scale${flow_scale}_res${render_resolution}_lr8"
            port=$((RANDOM % 10000 + 10000))

            CUDA_VISIBLE_DEVICES=$gpu_id WANDB_API="74669df238a8478603b81ad5cbf58414ec0af742" python train_outside.py -s dataset_cvpr/netflix_high_resolution/$scene/ \
                -m $output_dir \
                --r_t_noise 0.0 0.0 1. \
                --test_iterations 1 7000 20000 30000 40000 \
                --save_iterations 7000 20000 30000 40000 \
                --checkpoint_iterations 7000 20000 30000 40000 \
                --iterations 30000 \
                --r_t_lr 0.002 0.002 \
                --control_point_sample_scale 32 \
                --extend_scale 10000 \
                --opt_distortion \
                --outside_rasterizer \
                --flow_scale $flow_scale $flow_scale \
                --iresnet_lr 1e-8 \
                --wandb_project_name netflix_high_resolution \
                --wandb_mode online \
                --port $port \
                --opacity_reset_interval 50000 \
                --densify_until_iter 50000 \
                --render_resolution $render_resolution &

            gpu_id=$(( (gpu_id + 1) % 8 ))

        done
    done
done
wait

