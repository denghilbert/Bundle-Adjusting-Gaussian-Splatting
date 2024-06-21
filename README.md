# playaround_gaussian_platting

```shell
# create a director and put all rgb images into the directory
mkdir cube
mkdir datasets/cube/input
mv *.jpg datasets/cube/input
# run colmap
bash training_script/preprocess.sh
# vanilla gs
python train_outside.py -s dataset/nerf_synthetic/lego -m output/test --r_t_noise 0.0 0.0 1. --test_iterations 1 7000 10000 20000 30000 --save_iterations 7000 10000 20000 30000 --iterations 30000 --eval
# opt pose and intrinsic
python train_outside.py -s dataset/nerf_synthetic/lego -m output/test --r_t_noise 0.1 0.1 1.02 --test_iterations 1 7000 10000 20000 30000 --save_iterations 7000 10000 20000 30000 --iterations 30000 --eval --r_t_lr 0.005 0.01 --opt_cam --opt_intrinsic
# init with colmap w/o distortion opt (which means you believe in colmap)
python train_outside.py -s cube_for_outside -m output/test --r_t_noise 0.0 0.0 1. --test_iterations 1 7000 10000 20000 30000 --save_iterations 7000 10000 20000 30000 --iterations 30000 --eval --r_t_lr 0.005 0.01 --control_point_sample_scale 32 --extend_scale 1000000000 --outside_rasterizer
# opt distortion with colmap init
python train_outside.py -s cube_for_outside -m output/test --r_t_noise 0.0 0.0 1. --test_iterations 1 7000 10000 20000 30000 --save_iterations 7000 10000 20000 30000 --iterations 30000 --eval --r_t_lr 0.005 0.01 --control_point_sample_scale 32 --extend_scale 1000000000 --opt_distortion --outside_rasterizer
```

