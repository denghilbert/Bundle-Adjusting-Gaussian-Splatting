# playaround_gaussian_platting

```shell
# create a director and put all rgb images into the directory
mkdir cube
mkdir datasets/cube/input
mv *.jpg datasets/cube/input
# run colmap
bash training_script/preprocess.sh

# original reconstruction on lego
python train_outside.py -s lego -m output/lego --r_t_noise 0.0 0.0 1. --test_iterations 1 7000 10000 20000 30000 --save_iterations 7000 10000 20000 30000 --iterations 30000 --eval --r_t_lr 0.005 0.01 --control_point_sample_scale 32 --extend_scale 1000000
# perturbing pose and intrinsic, and optimizing them
python train_outside.py -s lego -m output/lego_perturb_opt --r_t_noise 0.1 0.1 1.02 --test_iterations 1 7000 10000 20000 30000 --save_iterations 7000 10000 20000 30000 --iterations 30000 --eval --r_t_lr 0.005 0.01 --control_point_sample_scale 32 --extend_scale 1000000 --opt_cam --opt_intrinsic

# reconstruct of undistorted images from colmap
python train_outside.py -s cube_for_outside -m output/cube_colmap_undist --r_t_noise 0.0 0.0 1. --test_iterations 1 7000 10000 20000 30000 --save_iterations 7000 10000 20000 30000 --iterations 30000 --eval --r_t_lr 0.005 0.01 --control_point_sample_scale 32 --extend_scale 1000000
# init distortion field with colmap and fix it during the reconstruction
python train_outside.py -s cube_for_outside -m output/cube_fixed_colmap_init --r_t_noise 0.0 0.0 1. --test_iterations 1 7000 10000 20000 30000 --save_iterations 7000 10000 20000 30000 --iterations 30000 --eval --r_t_lr 0.005 0.01 --control_point_sample_scale 32 --extend_scale 1000000 --outside_rasterizer
# opt distortion
python train_outside.py -s cube_for_outside -m output/cube_opt_distortion --r_t_noise 0.0 0.0 1. --test_iterations 1 7000 10000 20000 30000 --save_iterations 7000 10000 20000 30000 --iterations 30000 --eval --r_t_lr 0.005 0.01 --control_point_sample_scale 32 --extend_scale 1000000 --opt_distortion --outside_rasterizer
```

