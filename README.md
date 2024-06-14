# playaround_gaussian_platting

```shell
# create a director and put all rgb images into the directory
mkdir cube
mkdir datasets/cube/input
mv *.jpg datasets/cube/input
# run colmap
bash training_script/preprocess.sh
# training
python train_outside.py -s datasets/cube -m output/slow_training --r_t_noise 0.0 0.0 1. --test_iterations 7000 10000 20000 30000 --save_iterations 7000 10000 20000 30000 --iterations 30000 --eval --r_t_lr 0.005 0.01 --control_point_sample_scale 8 --extend_scale 1000000000
```

