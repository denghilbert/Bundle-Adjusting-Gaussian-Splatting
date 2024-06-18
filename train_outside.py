#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from os import makedirs
import torchvision
from random import randint
import random
from utils.loss_utils import l1_loss, ssim, kl_divergence, l2_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, SpecularModel, iResNet
from scene.dataset_readers import read_intrinsics_binary
from utils.general_utils import safe_state, get_linear_noise_func, linear_to_srgb
from projection_test import image_pair_candidates, light_glue_simple, projection_loss, dist_point_point, dist_point_line, correspondence_projection
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.graphics_utils import fov2focal, focal2fov, getProjectionMatrix
from utils.visualization import wandb_image
from utils.util_vis import vis_cameras
from utils.util import check_socket_open
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import wandb
import visdom
from easydict import EasyDict
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from PIL import Image
import time
from io import BytesIO
from torch import nn
import torch.nn.functional as F

# set random seeds
import numpy as np
import random
seed_value = 100  # Replace this with your desired seed value

torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

np.random.seed(seed_value)
random.seed(seed_value)

def homogenize(X: torch.Tensor):
    assert X.ndim == 2
    assert X.shape[1] in (2, 3)
    return torch.cat(
        (X, torch.ones((X.shape[0], 1), dtype=X.dtype, device=X.device)), dim=1
    )
def dehomogenize(X: torch.Tensor):
    assert X.ndim == 2
    assert X.shape[1] in (3, 4)
    return X[:, :-1] / X[:, -1:]

def colorize(uv_im, max_mag=None):
    hsv = np.zeros((uv_im.shape[0], uv_im.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(uv_im[..., 0], uv_im[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    # print(mag.max())
    if max_mag is None:
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    else:
        mag = np.clip(mag, 0.0, max_mag)
        mag = mag / max_mag * 255.0
        hsv[..., 2] = mag.astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def pass_neuralens(lens_net, min_w, max_w, min_h, max_h, sample_width, sample_height, K, pass_neural=True):
    i, j = np.meshgrid(
        np.linspace(min_w, max_w, sample_width),
        np.linspace(min_h, max_h, sample_height),
        indexing="ij",
    )
    i = i.T
    j = j.T
    P_sensor = (
        torch.from_numpy(np.stack((i, j), axis=-1))
        .to(torch.float32)
        .cuda()
    )
    P_sensor_hom = homogenize(P_sensor.reshape((-1, 2)))
    P_view_insidelens_direction_hom = (torch.inverse(K) @ P_sensor_hom.T).T
    P_view_insidelens_direction = dehomogenize(P_view_insidelens_direction_hom)

    if pass_neural:
        P_view_outsidelens_direction = lens_net.forward(P_view_insidelens_direction)
    else:
        P_view_outsidelens_direction = P_view_insidelens_direction

    camera_directions_w_lens = homogenize(P_view_outsidelens_direction)
    camera_directions_w_lens = camera_directions_w_lens.reshape((P_sensor.shape[0], P_sensor.shape[1], 3))[:, :, :2]

    return camera_directions_w_lens, P_view_insidelens_direction[-1]

def plot_points(ref_points, path):
    p1 = ref_points.clone().reshape(-1, 2)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(int(ref_points.shape[1]/4), int(ref_points.shape[0]/4)))
    x = p1[:, 0].detach().cpu().numpy()  # Convert tensor to numpy for plotting
    y = p1[:, 1].detach().cpu().numpy()
    plt.scatter(x, y)
    plt.title('2D Points Plot')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.xlim(p1[:, 0].min().item() - 0.1, p1[:, 0].max().item() + 0.1)
    plt.ylim(p1[:, 1].min().item() - 0.1, p1[:, 1].max().item() + 0.1)
    plt.grid(True)
    #plt.show()
    plt.savefig(path)

def center_crop(tensor, target_height, target_width):
    _, _, height, width = tensor.size()

    # Calculate the starting coordinates for the crop
    start_y = (height - target_height) // 2
    start_x = (width - target_width) // 2

    # Create a grid for the interpolation
    grid_y, grid_x = torch.meshgrid(torch.linspace(start_y, start_y + target_height - 1, target_height),
                                    torch.linspace(start_x, start_x + target_width - 1, target_width))
    grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0).to(tensor.device)

    # Normalize grid to [-1, 1]
    grid = 2.0 * grid / torch.tensor([width - 1, height - 1]).cuda() - 1.0
    grid = grid.permute(0, 1, 2, 3).expand(tensor.size(0), target_height, target_width, 2)

    # Perform the interpolation
    cropped_tensor = F.grid_sample(tensor, grid, align_corners=True)

    return cropped_tensor

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, use_wandb=False, random_init=False, hybrid=False, opt_cam=False, opt_distortion=False, opt_intrinsic=False, r_t_noise=[0., 0.], r_t_lr=[0.001, 0.001], global_alignment_lr=0.001, extra_loss=False, start_opt_lens=1, extend_scale=2., control_point_sample_scale=8., outside_rasterizer=False):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset.asg_degree)
    if hybrid:
        specular_mlp = SpecularModel()
        specular_mlp.train_setting(opt)
    lens_net = iResNet().cuda()
    l_lens_net = [{'params': lens_net.parameters(), 'lr': 1e-5}]
    optimizer_lens_net = torch.optim.Adam(l_lens_net, eps=1e-15)
    scheduler_lens_net = torch.optim.lr_scheduler.MultiStepLR(optimizer_lens_net, milestones=[3000, 5000], gamma=0.5)
    def zero_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 0.)
            nn.init.constant_(m.bias, 0.)
    #lens_net.apply(zero_weights)
    #for param in lens_net.parameters():
    #    print(param)

    scene = Scene(dataset, gaussians, random_init=random_init, r_t_noise=r_t_noise, r_t_lr=r_t_lr, global_alignment_lr=global_alignment_lr, outside_rasterizer=outside_rasterizer)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        scene.train_cameras = torch.load(os.path.join(path, 'opt_cams.pt'))
        scene.unnoisy_train_cameras = torch.load(os.path.join(path, 'gt_cams.pt'))
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    viewpoint_stack_constant = scene.getTrainCameras()
    camera_id = [camera.uid for camera in viewpoint_stack_constant]
    extrinsic_list = [camera.get_w2c for camera in viewpoint_stack_constant]
    camera_pairs = image_pair_candidates(extrinsic_list, args.angle_threshold, camera_id)
    camera_matching_points = {}
    projection_loss_count = 0
    if args.vis_pose:
        opt_vis = EasyDict({'group': 'exp_synthetic', 'name': 'l2g_lego', 'model': 'l2g_nerf', 'yaml': 'l2g_nerf_blender', 'seed': 0, 'gpu': 0, 'cpu': False, 'load': None, 'arch': {'layers_feat': [None, 256, 256, 256, 256, 256, 256, 256, 256], 'layers_rgb': [None, 128, 3], 'skip': [4], 'posenc': {'L_3D': 10, 'L_view': 4}, 'density_activ': 'softplus', 'tf_init': True, 'layers_warp': [None, 256, 256, 256, 256, 256, 256, 6], 'skip_warp': [4], 'embedding_dim': 128}, 'data': {'root': '/the/data/path/of/nerf_synthetic/', 'dataset': 'blender', 'image_size': [400, 400], 'num_workers': 4, 'preload': True, 'augment': {}, 'center_crop': None, 'val_on_test': False, 'train_sub': None, 'val_sub': 4, 'scene': 'lego', 'bgcolor': 1}, 'loss_weight': {'render': 0, 'render_fine': None, 'global_alignment': 2}, 'optim': {'lr': 0.0005, 'lr_end': 0.0001, 'algo': 'Adam', 'sched': {'type': 'ExponentialLR', 'gamma': None}, 'lr_pose': 0.001, 'lr_pose_end': 1e-08, 'sched_pose': {'type': 'ExponentialLR', 'gamma': None}, 'warmup_pose': None, 'test_photo': True, 'test_iter': 100}, 'batch_size': None, 'max_epoch': None, 'resume': False, 'output_root': 'output', 'tb': {'num_images': [4, 8]}, 'visdom': {'server': 'localhost', 'port': 8600, 'cam_depth': 0.5}, 'freq': {'scalar': 200, 'vis': 1000, 'val': 2000, 'ckpt': 5000}, 'nerf': {'view_dep': True, 'depth': {'param': 'metric', 'range': [2, 6]}, 'sample_intvs': 128, 'sample_stratified': True, 'fine_sampling': False, 'sample_intvs_fine': None, 'rand_rays': 1024, 'density_noise_reg': None, 'setbg_opaque': False}, 'camera': {'model': 'perspective', 'ndc': False, 'noise': True, 'noise_r': 0.07, 'noise_t': 0.5}, 'max_iter': 200000, 'trimesh': {'res': 128, 'range': [-1.2, 1.2], 'thres': 25.0, 'chunk_size': 16384}, 'barf_c2f': [0.1, 0.5], 'error_map_size': None, 'output_path': 'output/exp_synthetic/l2g_lego', 'device': 'cuda:0', 'H': 400, 'W': 400})
        if opt_vis.visdom and args.vis_pose:
            # check if visdom server is runninng
            is_open = check_socket_open(opt_vis.visdom.server,opt_vis.visdom.port)
            retry = None
            #while not is_open:
            #    retry = input("visdom port ({}) not open, retry? (y/n) ".format(opt_vis.visdom.port))
            #    if retry not in ["y","n"]: continue
            #    if retry=="y":
            #        is_open = check_socket_open(opt_vis.visdom.server,opt_vis.visdom.port)
            #    else: break
            vis = visdom.Visdom(server=opt_vis.visdom.server,port=opt_vis.visdom.port,env=opt_vis.group)
            pose_GT, pose_aligned = scene.loadAlignCameras(if_vis_train=True, path=scene.model_path)
            vis_cameras(opt_vis, vis, step=0, poses=[pose_aligned, pose_GT])
            os.makedirs(os.path.join(args.model_path, 'plot'), exist_ok=True)
            #download_pose_vis(os.path.join(args.model_path, 'plot'), 0)

    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)

    height, width = 1024, 684
    mask = torch.zeros((height, width), dtype=torch.bool)
    center_x, center_y = 342, 512
    radius = 512  # Example radius
    for y in range(height):
        for x in range(width):
            if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                mask[y, x] = 1.
    mask = mask.unsqueeze(0)  # For channel-first, adds a new dimension at the beginning
    mask = mask.repeat(3, 1, 1).cuda()  # Repeat the mask across the channel dimension

    distortion_params = nn.Parameter(torch.zeros(8).cuda().requires_grad_(True))
    optimizer_distortion = torch.optim.Adam([{'params': distortion_params, 'lr': 0.001}])

    #u_init = nn.Parameter(torch.zeros(200, 200).cuda().requires_grad_(True))
    # check index for bilinear interpolation
    #u_init[:, 0] = 1
    #u_init[86, 80] = 1
    #u_init[86, 81] = 2
    #u_init[87, 80] = 3
    #u_init[87, 81] = 4
    u_distortion = nn.Parameter(torch.zeros(400, 400).cuda().requires_grad_(True))
    v_distortion = nn.Parameter(torch.zeros(400, 400).cuda().requires_grad_(True))
    u_radial = nn.Parameter(torch.ones(400, 400).cuda().requires_grad_(True))
    v_radial = nn.Parameter(torch.ones(400, 400).cuda().requires_grad_(True))
    radial = nn.Parameter(torch.ones(1000).cuda().requires_grad_(True))
    #radial = nn.Parameter(torch.zeros(1000, 4).cuda().requires_grad_(True))
    optimizer_u_distortion = torch.optim.Adam([{'params': u_distortion, 'lr': 0.0001}])
    optimizer_v_distortion = torch.optim.Adam([{'params': v_distortion, 'lr': 0.0001}])
    optimizer_u_radial = torch.optim.Adam([{'params': u_radial, 'lr': 0.0001}])
    optimizer_v_radial = torch.optim.Adam([{'params': v_radial, 'lr': 0.0001}])
    optimizer_radial = torch.optim.Adam([{'params': radial, 'lr': 0.0001}])

    # colmap init
    if outside_rasterizer:
        viewpoint_cam = scene.getTrainCameras().copy()[0]
        width = viewpoint_cam.image_width
        height = viewpoint_cam.image_height
        sample_width = int(width / 50)
        sample_height = int(height / 50)
        K = viewpoint_cam.get_K
        width = viewpoint_cam.fish_gt_image.shape[2]
        height = viewpoint_cam.fish_gt_image.shape[1]
        width = int(width * 2)
        height = int(height * 2)
        K[0, 2] = width / 2
        K[1, 2] = height / 2
        def sigmoid(x):
            #return (1 / (1 + np.exp(-x)) - 0.5) * 1.5 + 0.5
            return (1 / (1 + np.exp(-x)) - 0.5) * 1.2 + 0.5
        i, j = np.meshgrid(
            #sigmoid(np.linspace(-2.398, 2.398, sample_width))*width,
            #sigmoid(np.linspace(-2.398, 2.398, sample_height))*height,
            #sigmoid(np.linspace(-1.61, 1.61, sample_width))*width,#1.5
            #sigmoid(np.linspace(-1.61, 1.61, sample_height))*height,
            np.linspace(0, width, sample_width),
            np.linspace(0, height, sample_height),
            #np.linspace(0 - width/2, width + width/2, sample_width),
            #np.linspace(0 - height/2, height + height/2, sample_height),
            #np.linspace(0 - width/1, width + width/1, sample_width),
            #np.linspace(0 - height/1, height + height/1, sample_height),
            indexing="ij",
        )
        i = i.T
        j = j.T
        P_sensor = (
            torch.from_numpy(np.stack((i, j), axis=-1))
            .to(torch.float32)
            .cuda()
        )
        #plot_points(P_sensor, os.path.join(scene.model_path, f"ref.png"))
        #import pdb;pdb.set_trace()
        P_sensor_hom = homogenize(P_sensor.reshape((-1, 2)))
        P_view_insidelens_direction_hom = (torch.inverse(K) @ P_sensor_hom.T).T
        P_view_insidelens_direction = dehomogenize(P_view_insidelens_direction_hom)
        P_view_outsidelens_direction = P_view_insidelens_direction
        camera_directions_w_lens = homogenize(P_view_outsidelens_direction)
        ref_points = camera_directions_w_lens.reshape((P_sensor.shape[0], P_sensor.shape[1], 3))[:, :, :2]
        r = torch.sqrt(torch.sum(ref_points**2, dim=-1, keepdim=True))
        inv_r = 1 / r
        theta = torch.atan(r)
        coeff = [0, 0, 0, 0]
        if os.path.exists(os.path.join(dataset.source_path, 'fish/sparse/0/cameras.bin')):
            cam_intrinsics = read_intrinsics_binary(os.path.join(dataset.source_path, 'fish/sparse/0/cameras.bin'))
            for idx, key in enumerate(cam_intrinsics):
                if 'RADIAL' in cam_intrinsics[key].model:
                    coeff = cam_intrinsics[key].params[-2:].tolist()
                if 'FISHEYE' in cam_intrinsics[key].model:
                    coeff = cam_intrinsics[key].params[-4:].tolist()
                    break
        if len(coeff) == 4:
            ref_points = ref_points * (inv_r * (theta + coeff[0] * theta**3 + coeff[1] * theta**5 + coeff[2] * theta**7 + coeff[3] * theta**9))
        elif len(coeff) == 2:
            ref_points = ref_points * (1 + coeff[0] * r**2 + coeff[1] * r**4)
        elif len(coeff) == 3:
            ref_points = ref_points * (1 + coeff[0] * r**2 + coeff[1] * r**4 + coeff[2] * r**6)
        else:
            ref_points = ref_points
        inf_mask = torch.isinf(ref_points)
        nan_mask = torch.isnan(ref_points)
        ref_points[inf_mask] = 0
        ref_points[nan_mask] = 0
        boundary_original_points = P_view_insidelens_direction[-1]
        print(boundary_original_points)
        #ref_points = nn.Parameter(ref_points.cuda().requires_grad_(True))
        #optimizer_ref_points = torch.optim.Adam([{'params': ref_points, 'lr': 0.0001}])

        width = viewpoint_cam.image_width
        height = viewpoint_cam.image_height
        sample_width = int(width / 50)
        sample_height = int(height / 50)
        K = viewpoint_cam.get_K
        width = viewpoint_cam.fish_gt_image.shape[2]
        height = viewpoint_cam.fish_gt_image.shape[1]
        width = int(width * 2)
        height = int(height * 2)
        K[0, 2] = width / 2
        K[1, 2] = height / 2
        i, j = np.meshgrid(
            #sigmoid(np.linspace(-2.398, 2.398, sample_width))*width,
            #sigmoid(np.linspace(-2.398, 2.398, sample_height))*height,
            #sigmoid(np.linspace(-1.61, 1.61, sample_width))*width,
            #sigmoid(np.linspace(-1.61, 1.61, sample_height))*height,
            np.linspace(0, width, sample_width),
            np.linspace(0, height, sample_height),
            #np.linspace(0 - width/2, width + width/2, sample_width),
            #np.linspace(0 - height/2, height + height/2, sample_height),
            #np.linspace(0 - width/1, width + width/1, sample_width),
            #np.linspace(0 - height/1, height + height/1, sample_height),
            indexing="ij",
        )
        i = i.T
        j = j.T
        P_sensor = (
            torch.from_numpy(np.stack((i, j), axis=-1))
            .to(torch.float32)
            .cuda()
        )
        P_sensor_hom = homogenize(P_sensor.reshape((-1, 2)))
        P_view_insidelens_direction_hom = (torch.inverse(K) @ P_sensor_hom.T).T
        P_view_insidelens_direction = dehomogenize(P_view_insidelens_direction_hom)


        progress_bar_ires = tqdm(range(0, 2000), desc="Init Iresnet")
        for i in range(2000):
            P_view_outsidelens_direction = lens_net.forward(P_view_insidelens_direction, sensor_to_frustum=True)
            control_points = homogenize(P_view_outsidelens_direction)
            control_points = control_points.reshape((P_sensor.shape[0], P_sensor.shape[1], 3))[:, :, :2]
            inf_mask = torch.isinf(control_points)
            nan_mask = torch.isnan(control_points)
            control_points[inf_mask] = 0
            control_points[nan_mask] = 0
            loss = ((control_points - ref_points)**2).mean()
            progress_bar_ires.set_postfix(loss=loss.item())
            progress_bar_ires.update(1)
            loss.backward()
            optimizer_lens_net.step()
            optimizer_lens_net.zero_grad(set_to_none = True)
            scheduler_lens_net.step()
        progress_bar_ires.close()

        for param_group in optimizer_lens_net.param_groups:
            param_group['lr'] = 1e-6

        width = viewpoint_cam.image_width
        height = viewpoint_cam.image_height
        sample_width = int(width / control_point_sample_scale)
        sample_height = int(height/ control_point_sample_scale)
        K = viewpoint_cam.get_K
        width = viewpoint_cam.fish_gt_image.shape[2]
        height = viewpoint_cam.fish_gt_image.shape[1]
        width = int(width * 2)
        height = int(height * 2)
        K[0, 2] = width / 2
        K[1, 2] = height / 2
        i, j = np.meshgrid(
            #sigmoid(np.linspace(-2.398, 2.398, sample_width))*width,
            #sigmoid(np.linspace(-2.398, 2.398, sample_height))*height,
            #sigmoid(np.linspace(-1.61, 1.61, sample_width))*width,
            #sigmoid(np.linspace(-1.61, 1.61, sample_height))*height,
            np.linspace(0, width, sample_width),
            np.linspace(0, height, sample_height),
            #np.linspace(0 - width/1, width + width/1, sample_width),
            #np.linspace(0 - height/1, height + height/1, sample_height),
            #np.linspace(0 - width/2, width + width/2, sample_width),
            #np.linspace(0 - height/2, height + height/2, sample_height),
            indexing="ij",
        )
        i = i.T
        j = j.T
        P_sensor = (
            torch.from_numpy(np.stack((i, j), axis=-1))
            .to(torch.float32)
            .cuda()
        )
        P_sensor_hom = homogenize(P_sensor.reshape((-1, 2)))
        P_view_insidelens_direction_hom = (torch.inverse(K) @ P_sensor_hom.T).T
        P_view_insidelens_direction = dehomogenize(P_view_insidelens_direction_hom)
        P_view_outsidelens_direction = P_view_insidelens_direction
        camera_directions_w_lens = homogenize(P_view_outsidelens_direction)
        rectangle_points = camera_directions_w_lens.reshape((P_sensor.shape[0], P_sensor.shape[1], 3))[:, :, :2]

    # |1 e c_x|
    # |d c c_y|
    # |0 0 1  |
    # init as e=d=0, c=1, c_x=c_y=0, order [1, e, d, c, c_x, c_y]
    affine_coeff = nn.Parameter(torch.tensor([1., 0., 0., 1., 0., 0.]).cuda().requires_grad_(True))
    poly_coeff = nn.Parameter(torch.tensor([0, 0, 0, 0.]).cuda().requires_grad_(True))
    optimizer_affine = torch.optim.Adam([{'params': affine_coeff, 'lr': 0.0001}])
    optimizer_poly = torch.optim.Adam([{'params': poly_coeff, 'lr': 0.0001}])

    count_57 = 0
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        if hybrid:
            specular_mlp.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        # input type
        N = gaussians.get_xyz.shape[0]
        mlp_color = 0

        render_pkg = render(viewpoint_cam, gaussians, pipe, background, mlp_color, iteration=iteration, hybrid=hybrid, global_alignment=scene.getGlobalAlignment())
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if outside_rasterizer:
            P_view_outsidelens_direction = lens_net.forward(P_view_insidelens_direction, sensor_to_frustum=False)
            camera_directions_w_lens = homogenize(P_view_outsidelens_direction)
            control_points = camera_directions_w_lens.reshape((P_sensor.shape[0], P_sensor.shape[1], 3))[:, :, :2]

            #plot_points(control_points, os.path.join(scene.model_path, f"control_points_beforeK.png"))
            projection_matrix = viewpoint_cam.projection_matrix
            flow = control_points @ projection_matrix[:2, :2]
            #plot_points(flow, os.path.join(scene.model_path, f"control_points_applyK.png"))
            flow = nn.functional.interpolate(flow.permute(2, 0, 1).unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).squeeze(0)
            #torchvision.utils.save_image(image, os.path.join(scene.model_path, f"rendered.png"))
            image = F.grid_sample(
                image.unsqueeze(0),
                flow.unsqueeze(0),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            #torchvision.utils.save_image(image, os.path.join(scene.model_path, f"rendered_fish.png"))
            image = center_crop(image, int(height/2), int(width/2)).squeeze(0)
            #torchvision.utils.save_image(image, os.path.join(scene.model_path, f"rendered_fish_crop.png"))
            #torchvision.utils.save_image(viewpoint_cam.fish_gt_image, os.path.join(scene.model_path, f"gt.png"))

            mask = (~((image[0]==0) & (image[1]==0)).unsqueeze(0)).float()
            #torchvision.utils.save_image(mask*viewpoint_cam.fish_gt_image.cuda(), os.path.join(scene.model_path, f"mask.png"))

        # Loss
        if outside_rasterizer:
            gt_image = viewpoint_cam.fish_gt_image.cuda()
            Ll1 = l1_loss(image, gt_image*mask)
            ssim_loss = ssim(image, gt_image*mask)
        else:
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            ssim_loss = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_loss)# + 0.1 * (loss_projection / len(camera_pairs[viewpoint_cam.uid]))
        loss.backward(retain_graph=True)


        if iteration % 10 == 0:
            scalars = {
                f"loss/l1_loss": Ll1,
                f"loss/ssim": ssim_loss,
                f"loss/overall_loss": loss,
            }
            if projection_loss_count > 0:
                scalars["loss/projection_loss"] = (loss_projection / len(camera_pairs[viewpoint_cam.uid]))
            if use_wandb:
                wandb.log(scalars, step=iteration)

        if iteration % 3000 == 0 or iteration == 1:
            wandb_img = image.unsqueeze(0).detach()
            #wandb_img = tmp_img.unsqueeze(0).detach()
            wandb_img_gt = gt_image.unsqueeze(0).detach()
            images_error = (wandb_img_gt - wandb_img).abs()
            images = {
                f"vis/rgb_target": wandb_image(gt_image),
                f"vis/rgb_render": wandb_image(wandb_img),
                f"vis/rgb_error": wandb_image(images_error),
            }
            if use_wandb:
                wandb.log(images, step=iteration)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                #progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            #if iteration in testing_iterations:
            if iteration % 500 == 0 and args.vis_pose:
                pose_GT, pose_aligned = scene.loadAlignCameras(if_vis_train=True, iteration=iteration, path=scene.model_path)
                vis_cameras(opt_vis, vis, step=iteration, poses=[pose_aligned, pose_GT])

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, mlp_color), lens_net, opt_distortion, P_view_insidelens_direction, P_sensor, outside_rasterizer)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                torch.save(lens_net, os.path.join(scene.model_path, f'lens_net{iteration}.pth'))
                torch.save(scene.train_cameras, os.path.join(scene.model_path, 'opt_cams.pt'))
                torch.save(scene.unnoisy_train_cameras, os.path.join(scene.model_path, 'gt_cams.pt'))
                torch.save(distortion_params, os.path.join(scene.model_path, 'distortion_params.pt'))
                torch.save(u_distortion, os.path.join(scene.model_path, f'u_distortion{iteration}.pt'))
                torch.save(v_distortion, os.path.join(scene.model_path, f'v_distortion{iteration}.pt'))
                torch.save(u_radial, os.path.join(scene.model_path, f'u_radial{iteration}.pt'))
                torch.save(v_radial, os.path.join(scene.model_path, f'v_radial{iteration}.pt'))
                torch.save(affine_coeff, os.path.join(scene.model_path, f'affine_coeff{iteration}.pt'))
                torch.save(poly_coeff, os.path.join(scene.model_path, f'poly_coeff{iteration}.pt'))
                torch.save(radial, os.path.join(scene.model_path, f'radial{iteration}.pt'))
                torch.save(scene.train_cameras, os.path.join(scene.model_path, f'cams_train{iteration}.pt'))
                if hybrid:
                    specular_mlp.save_weights(args.model_path, iteration)


            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

                viewspace_point_tensor_densify = render_pkg["viewspace_points_densify"]
                gaussians.add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_densify, visibility_filter)
                if iteration % 10 == 0:
                    scalars = {
                        f"gradient/2d_gradient": viewspace_point_tensor.grad.mean(),
                    }
                    if use_wandb:
                        wandb.log(scalars, step=iteration)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if opt_distortion:
                    optimizer_lens_net.step() #lens_net.i_resnet_linear.module_list[0].residual[0].weight
                    optimizer_lens_net.zero_grad(set_to_none=True)
                # do not update camera pose when densify or prune gaussians
                if opt_cam:
                    scene.optimizer_rotation.step()
                    scene.optimizer_translation.step()
                    scene.optimizer_rotation.zero_grad(set_to_none=True)
                    scene.optimizer_translation.zero_grad(set_to_none=True)
                    scene.scheduler_rotation.step()
                    scene.scheduler_translation.step()
                if opt_intrinsic:
                    scene.optimizer_fovx.step()
                    scene.optimizer_fovy.step()
                    scene.optimizer_fovx.zero_grad(set_to_none=True)
                    scene.optimizer_fovy.zero_grad(set_to_none=True)
                    scene.scheduler_fovx.step()
                    scene.scheduler_fovy.step()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                torch.save(scene.train_cameras, os.path.join(scene.model_path, 'opt_cams.pt'))
                torch.save(scene.unnoisy_train_cameras, os.path.join(scene.model_path, 'gt_cams.pt'))

    torch.save(scene.train_cameras, os.path.join(scene.model_path, 'opt_cams.pt'))
    torch.save(scene.unnoisy_train_cameras, os.path.join(scene.model_path, 'gt_cams.pt'))

def download_pose_vis(path, iteration):
        # download image
        chrome_binary_path = '/home/youming/Downloads/chrome-linux64/chrome'
        chrome_options = Options()
        chrome_options.binary_location = chrome_binary_path
        webdriver_path = '/home/youming/Downloads/chromedriver-linux64/chromedriver'
        service = Service(webdriver_path)
        driver = webdriver.Chrome(service=service, options=chrome_options)
        visdom_url = 'http://localhost:8600'
        driver.get(visdom_url)
        time.sleep(0.2)
        element = driver.find_element(By.ID, 'scene')
        png = driver.get_screenshot_as_png()
        location = element.location
        size = element.size
        left = location['x']
        top = location['y'] + 100
        right = location['x'] + 2 * size['width']
        bottom = location['y'] + 2 * size['height']
        im = Image.open(BytesIO(png))
        im = im.crop((left, top, right, bottom))
        im.save(f'{path}/visdom_plot{iteration}.png')
        driver.quit()

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, lens_net, opt_distortion, P_view_insidelens_direction, P_sensor, outside_rasterizer):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                makedirs(os.path.join(scene.model_path, 'training_val'), exist_ok=True)
                for idx, viewpoint in enumerate(config['cameras']):
                    gaussians_xyz = scene.gaussians.get_xyz.detach()
                    gaussians_xyz_homo = torch.cat((gaussians_xyz, torch.ones(gaussians_xyz.size(0), 1).cuda()), dim=1)
                    # glm use the transpose of w2c
                    w2c = viewpoint.get_world_view_transform().t().detach()
                    p_w2c = (w2c @ gaussians_xyz_homo.T).T.cuda().detach()
                    undistorted_p_w2c_homo = p_w2c
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, global_alignment=scene.getGlobalAlignment())["render"], 0.0, 1.0)

                    if outside_rasterizer:
                        P_view_outsidelens_direction = lens_net.forward(P_view_insidelens_direction, sensor_to_frustum=False)
                        camera_directions_w_lens = homogenize(P_view_outsidelens_direction)
                        flow = camera_directions_w_lens.reshape((P_sensor.shape[0], P_sensor.shape[1], 3))[:, :, :2]
                        projection_matrix = viewpoint.projection_matrix
                        flow = flow @ projection_matrix[:2, :2]
                        flow = nn.functional.interpolate(flow.permute(2, 0, 1).unsqueeze(0), size=(viewpoint.fish_gt_image.shape[1]*2, viewpoint.fish_gt_image.shape[2]*2), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).squeeze(0)
                        image = F.grid_sample(
                            image.unsqueeze(0),
                            flow.unsqueeze(0),
                            mode="bilinear",
                            padding_mode="zeros",
                            align_corners=True,
                        )
                    name = config['name']
                    torchvision.utils.save_image(image, os.path.join(scene.model_path, 'training_val/{0:05d}'.format(idx) + "_" + name + "_before_crop.png"))
                    if outside_rasterizer:
                        image = center_crop(image, viewpoint.fish_gt_image.shape[1], viewpoint.fish_gt_image.shape[2]).squeeze(0)
                        gt_image = torch.clamp(viewpoint.fish_gt_image.to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    torchvision.utils.save_image(image, os.path.join(scene.model_path, 'training_val/{0:05d}'.format(idx) + "_" + name + ".png"))
                    torchvision.utils.save_image(gt_image, os.path.join(scene.model_path, 'training_val/gt_{0:05d}'.format(idx) + "_" + name + ".png"))
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


def init_wandb(cfg, wandb_id=None, project="", run_name=None, mode="online", resume=False, use_group=False, set_group=None):
    r"""Initialize Weights & Biases (wandb) logger.

    Args:
        cfg (obj): Global configuration.
        wandb_id (str): A unique ID for this run, used for resuming.
        project (str): The name of the project where you're sending the new run.
            If the project is not specified, the run is put in an "Uncategorized" project.
        run_name (str): name for each wandb run (useful for logging changes)
        mode (str): online/offline/disabled
    """
    print('Initialize wandb')
    if not wandb_id:
        wandb_path = os.path.join(cfg.model_path, "wandb_id.txt")
        if resume and os.path.exists(wandb_path):
            with open(wandb_path, "r") as f:
                wandb_id = f.read()
        else:
            wandb_id = wandb.util.generate_id()
            with open(wandb_path, "w+") as f:
                f.write(wandb_id)
    if use_group:
        group, name = cfg.model_path.split("/")[-2:]
        group = set_group
    else:
        group, name = None, os.path.basename(cfg.model_path)
        group = set_group

    if run_name is not None:
        name = run_name
    wandb.init(id=wandb_id,
               project=project,
               config=vars(cfg),
               group=group,
               name=name,
               dir=cfg.model_path,
               resume=resume,
               settings=wandb.Settings(start_method="fork"),
               mode=mode)
    wandb.config.update({'dataset': cfg.source_path})

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # wandb setting
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project_name", type=str, default = None)
    parser.add_argument("--wandb_group_name", type=str, default = None)
    parser.add_argument("--wandb_mode", type=str, default = "online")
    parser.add_argument("--resume", action="store_true", default=False)
    # random init point cloud
    parser.add_argument("--random_init_pc", action="store_true", default=False)

    # use hybrid for specular
    parser.add_argument("--hybrid", action="store_true", default=False)
    # if optimize camera poses
    parser.add_argument("--opt_cam", action="store_true", default=False)
    # if opt camera intrinsic
    parser.add_argument("--opt_intrinsic", action="store_true", default=False)
    parser.add_argument("--r_t_lr", nargs="+", type=float, default=[0.01, 0.01])
    # learning rate for global alignment
    parser.add_argument('--global_alignment_lr', type=float, default=0.01)
    # noise for rotation and translation
    parser.add_argument("--r_t_noise", nargs="+", type=float, default=[0., 0.])
    # rotation filter for light_glue
    parser.add_argument('--angle_threshold', type=float, default=30.)
    # if optimize camera poses with projection_loss
    parser.add_argument("--projection_loss", action="store_true", default=False)
    # if visualize camera pose
    parser.add_argument("--vis_pose", action="store_true", default=False)
    # optimize distortion
    parser.add_argument("--opt_distortion", action="store_true", default=False)
    parser.add_argument("--extra_loss", action="store_true", default=False)
    parser.add_argument('--start_opt_lens', type=int, default=1)
    parser.add_argument('--extend_scale', type=float, default=2.)
    parser.add_argument('--control_point_sample_scale', type=float, default=8.)
    parser.add_argument("--outside_rasterizer", action="store_true", default=False)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)

    # Initialize wandb
    os.makedirs(args.model_path, exist_ok=True)
    if args.wandb:
        wandb.login()
        wandb_run = init_wandb(args,
                               project=args.wandb_project_name,
                               mode=args.wandb_mode,
                               resume=args.resume,
                               use_group=True,
                               set_group=args.wandb_group_name
                               )

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, use_wandb=args.wandb, random_init=args.random_init_pc, hybrid=args.hybrid, opt_cam=args.opt_cam, opt_distortion=args.opt_distortion, opt_intrinsic=args.opt_intrinsic, r_t_lr=args.r_t_lr, r_t_noise=args.r_t_noise, global_alignment_lr=args.global_alignment_lr, extra_loss=args.extra_loss, start_opt_lens=args.start_opt_lens, extend_scale=args.extend_scale, control_point_sample_scale=args.control_point_sample_scale, outside_rasterizer=args.outside_rasterizer)

    # All done
    print("\nTraining complete.")
