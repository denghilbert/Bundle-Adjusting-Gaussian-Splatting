# Bundle-Adjusting-Gaussian-Splatting

***

## Cloning the Repository

The repository contains submodules. Please clone with:

```bash
# SSH
git clone --depth 1 git@github.com:denghilbert/Bundle-Adjusting-Gaussian-Splatting.git --recursive
```

***

## Setup Environment

Our test environment:

* Ubuntu 22.04 with NVCC 11.8 and g++ 11.4.0
* RTX 4090 (sm_89 architecture, CUDA 11.8 and later)
* RTX 3090, A5000, A6000 (sm_86 architecture, CUDA 11.1 and later)

We recommend to use NVCC 11.8 or later than that.

The installation of SIBR viewers can follow [original 3dgs repo](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#interactive-viewers).

Our method is based on Conda package and environment management:

```shell
conda create -n bags python=3.10
conda activate bags
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install tqdm plyfile imageio easydict FrEIA wandb matplotlib ipdb termcolor visdom selenium pywavefront

cd 3dgs-pose
pip install .
cd ..
cd simple-knn
pip install .
cd ..
```

***

For quick start, we provide [example datasets](https://drive.google.com/file/d/118bDUMFfdths00UWQL-Xs5q-BpdVVsxA/view?usp=sharing).

## Running

### Intrinsic and Extrinsic Optimization

To run pose optimization with visualization, we first need to run visdom and then train 3D Guassians, the visualization will run global pose alignment:

```shell
# run visdom with a port
screen -S vis
visdom -port 8600
```

```shell
# perturb poses without optimization
python train_outside.py -s example_datasets/lego -m output/vis_pose --r_t_noise 0.15 0.15 1.0 --test_iterations 7000 15000 30000 --save_iterations 7000 15000 30000 --iterations 30000 --eval --r_t_lr 0.01 0.02 --vis_pose
# optimize poses
python train_outside.py -s example_datasets/lego -m output/vis_pose --r_t_noise 0.15 0.15 1.0 --test_iterations 7000 15000 30000 --save_iterations 7000 15000 30000 --iterations 30000 --eval --r_t_lr 0.01 0.02 --opt_cam --vis_pose
```

We can also jointly optimize extrinsic and intrinsic:

```shell
python train_outside.py -s example_datasets/lego -m output/opt_extrinsic_intrinsic --r_t_noise 0.1 0.1 1.02 --test_iterations 7000 15000 30000 --save_iterations 7000 15000 30000 --iterations 30000 --eval --r_t_lr 0.01 0.02 --opt_cam --opt_intrinsic
```

### Distortion Modeling

To run vanilla-GS reconstruction without any optimization:

```shell
python train_outside.py -s example_datasets/cube_for_outside/ -m output/cube_woopt_distortion --r_t_noise 0.0 0.0 1. --test_iterations 7000 15000 30000 --save_iterations 7000 15000 30000 --iterations 30000 --eval --r_t_lr 0.005 0.01 --control_point_sample_scale 32
```

We can also tantalize distortion network with the prediction from COLMAP and fixed the network:

```shell
python train_outside.py -s example_datasets/cube_for_outside/ -m output/cube_w_init --r_t_noise 0.0 0.0 1. --test_iterations 7000 15000 30000 --save_iterations 7000 15000 30000 --iterations 30000 --eval --r_t_lr 0.005 0.01 --control_point_sample_scale 32 --outside_rasterizer --flow_scale 1.2
```

Finally, our method optimize from inaccurate COLMAP prediction:

```shell
python train_outside.py -s example_datasets/cube_for_outside/ -m output/cube_opt_distortion --r_t_noise 0.0 0.0 1. --test_iterations 7000 15000 30000 --save_iterations 7000 15000 30000 --iterations 30000 --eval --r_t_lr 0.005 0.01 --control_point_sample_scale 32 --opt_distortion --outside_rasterizer --flow_scale 1.2
```

***

## Data Preparation

After taking several images, all you need to do is to put images like this:

```
<location>
|---input
    |---<image 0>
    |---<image 1>
    |---...
```

And run:

```shell
# run colmap
python convert.py --source_path path_to_your_data/<location> --colmap_executable /opt/homebrew/bin/colmap --camera OPENCV_FISHEYE
cd path_to_your_data/<location>
mkdir fish
cp -r input fish
cp -r distorted/sparse fish
mv fish/input fish/images
```

The final directory of a single scene should looks like this where `<location>/fish/images` contains distorted images and `<location>/images` contains perspective ones.

```
<location>
├── fish
│   ├── images
│   │   ├── 000.jpg
│   │   ├── 001.jpg
│   │   ├── ...
│   └── sparse
│       └── 0
│           ├── cameras.bin
│           ├── images.bin
│           ├── points3D.bin
├── images
│   ├── 000.jpg
│   ├── 001.jpg
│   ├── ...
└── sparse
    └── 0
        ├── cameras.bin
        ├── images.bin
        ├── points3D.bin
        └── points3D.ply
```

