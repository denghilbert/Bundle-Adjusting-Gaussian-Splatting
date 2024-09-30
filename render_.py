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
import sys
import torch
import torchvision
import random
from random import randint
from utils.loss_utils import l1_loss, ssim, kl_divergence, l2_loss
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel, SpecularModel, iResNet
from utils.general_utils import safe_state, get_linear_noise_func, linear_to_srgb
from tqdm import tqdm
from utils.image_utils import psnr
from utils.loss_utils import ssim
from utils.lpipsPyTorch import lpips
from utils.graphics_utils import fov2focal, focal2fov, getProjectionMatrix, cubemap_to_perspective
from utils.visualization import wandb_image
from utils.util_vis import vis_cameras
from utils.util import check_socket_open, prepare_output_and_logger, init_wandb
from utils.camera_utils import rotate_camera
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
import wandb
import visdom
from easydict import EasyDict
from PIL import Image
import time
from io import BytesIO
from torch import nn
import torch.nn.functional as F
from utils.util_distortion import homogenize, dehomogenize, colorize, plot_points, center_crop, init_from_colmap, apply_distortion, generate_control_pts
import copy

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

def create_differentiable_vignetting_mask(image_tensor, scaling_factors):
    # Step 1: Get the height and width of the image
    _, width, height = image_tensor.shape  # Assume image_tensor is of shape (C, H, W)

    # Step 2: Create a grid of distances from the center
    x = torch.arange(width).float().to(image_tensor.device) - (width - 1) / 2
    y = torch.arange(height).float().to(image_tensor.device) - (height - 1) / 2
    X, Y = torch.meshgrid(x, y, indexing="ij")

    # Calculate Euclidean distance from the center
    distances = torch.sqrt(X**2 + Y**2)

    # Step 3: Normalize distances to the range [0, N-1]
    max_distance = distances.max()
    distances_normalized = distances / max_distance * (scaling_factors.size(0) - 1)

    # Step 4: Perform linear interpolation for differentiable mask creation
    lower_indices = distances_normalized.floor().long().clamp(0, scaling_factors.size(0) - 2)
    upper_indices = lower_indices + 1

    lower_weights = (upper_indices - distances_normalized).cuda()  # weight for lower index
    upper_weights = (distances_normalized - lower_indices).cuda()  # weight for upper index

    # Get the scaling factors for the lower and upper indices
    scaling_mask = lower_weights * scaling_factors[lower_indices] + upper_weights * scaling_factors[upper_indices]

    return scaling_mask

class VignettingModel(torch.nn.Module):
    def __init__(self, n_terms=4, device='cuda'):
        super(VignettingModel, self).__init__()

        # Number of terms in the summation (higher order model)
        self.n_terms = n_terms

        # Initialize learnable parameters (requires_grad=True for optimization)
        # Initialize a_k close to zero for minimal vignetting effect at the start
        self.a_k = torch.nn.Parameter(torch.full((n_terms,), 0.01, device=device))  # Coefficients a_k initialized near 0

        # Initialize beta_k to a small value but ensuring it gives a smooth falloff
        self.beta_k = torch.nn.Parameter(torch.linspace(2.0, 8.0, n_terms).to(device))

        # Initialize gamma to 1 for neutral global exponentiation
        #self.gamma = torch.nn.Parameter(torch.tensor(1.0).to(device))
        self.gamma = 1.

        # Store the device (e.g., 'cuda' or 'cpu')
        self.device = device

    def bak_forward(self, image_size):
        # Get image dimensions
        height, width = image_size

        # Compute the image center
        x_c, y_c = width / 2, height / 2

        # Compute the maximum possible distance from the center (diagonal distance)
        r_max = torch.sqrt(torch.tensor((x_c ** 2 + y_c ** 2), dtype=torch.float32).to(self.device))

        # Create a meshgrid of pixel coordinates on the specified device
        y, x = torch.meshgrid(torch.arange(height, dtype=torch.float32).to(self.device),
                              torch.arange(width, dtype=torch.float32).to(self.device))

        # Calculate the distance of each pixel from the image center
        d = torch.sqrt((x - x_c) ** 2 + (y - y_c) ** 2)

        # Normalize the distance by r_max
        d_normalized = d / r_max

        # Initialize the vignetting mask
        mask = torch.zeros_like(d_normalized)

        # Sum over the higher-order terms to compute the vignetting mask
        for k in range(self.n_terms):
            mask += self.a_k[k] * d_normalized ** self.beta_k[k]

        # Apply the global exponent gamma
        mask = 1 - mask.clamp(0, 1) ** self.gamma

        return mask

    def forward(self, image_size):
        # Get image dimensions
        height, width = image_size

        # Compute the image center
        x_c, y_c = width / 2, height / 2

        # Create a meshgrid of pixel coordinates on the specified device
        y, x = torch.meshgrid(torch.arange(height, dtype=torch.float32).to(self.device),
                              torch.arange(width, dtype=torch.float32).to(self.device))

        # Calculate the distance of each pixel from the image center
        r = torch.sqrt((x - x_c) ** 2 + (y - y_c) ** 2)

        # Apply the arctan(r)/r normalization
        #r_normalized = torch.arctan(r) / r
        r_normalized = torch.arctan(r)

        # Replace NaNs from division by zero at the center (r=0) with 1
        r_normalized[r == 0] = 1

        # Initialize the vignetting mask
        mask = torch.zeros_like(r_normalized)

        # Sum over the higher-order terms to compute the vignetting mask
        for k in range(self.n_terms):
            mask += self.a_k[k] * r_normalized ** self.beta_k[k]

        # Apply the global exponent gamma
        mask = 1 - mask.clamp(0, 1) ** self.gamma

        return mask

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, use_wandb=False, random_init=False, hybrid=False, opt_cam=False, opt_shift=False, no_distortion_mask=False, opt_distortion=False, start_vignetting=10000000000, opt_intrinsic=False, r_t_noise=[0., 0.], r_t_lr=[0.001, 0.001], global_alignment_lr=0.001, extra_loss=False, start_opt_lens=1, extend_scale=2., control_point_sample_scale=8., outside_rasterizer=False, abs_grad=False, densi_num=0.0002, if_circular_mask=False, flow_scale=[1., 1.], render_resolution=1., apply2gt=False, iresnet_lr=1e-7, opacity_threshold=0.005):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset.asg_degree)
    lens_net = iResNet().cuda()
    l_lens_net = [{'params': lens_net.parameters(), 'lr': 1e-5}]
    optimizer_lens_net = torch.optim.Adam(l_lens_net, eps=1e-15)
    scheduler_lens_net = torch.optim.lr_scheduler.MultiStepLR(optimizer_lens_net, milestones=[7000], gamma=0.5)
    vignetting_model = VignettingModel(n_terms=4, device='cuda')
    vignetting_optimizer = torch.optim.Adam(vignetting_model.parameters(), lr=0.01)
    vignetting_scheduler = torch.optim.lr_scheduler.MultiStepLR(vignetting_optimizer, milestones=[1000], gamma=10)
    shift_factors = nn.Parameter(torch.tensor([-0., -0., -0.], requires_grad=True, device='cuda'))
    shift_optimizer = torch.optim.Adam([shift_factors], lr=1e-3)

    scene = Scene(dataset, gaussians, random_init=random_init, r_t_noise=r_t_noise, r_t_lr=r_t_lr, global_alignment_lr=global_alignment_lr, outside_rasterizer=outside_rasterizer, flow_scale=flow_scale, render_resolution=render_resolution, apply2gt=apply2gt, vis_pose=args.vis_pose)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        print(f'loading iteration {first_iter}')
        lens_net = torch.load(os.path.join(scene.model_path, f'lens_net{first_iter}.pth'))
        gaussians.restore(model_params, opt, validation=True)
        if opt_cam:
            scene.train_cameras = torch.load(os.path.join(scene.model_path, 'opt_cams.pt'))
            scene.unnoisy_train_cameras = torch.load(os.path.join(scene.model_path, 'gt_cams.pt'))
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = None
    init_from_colmap(scene, dataset, optimizer_lens_net, lens_net, scheduler_lens_net, resume_training=checkpoint, iresnet_lr=iresnet_lr)
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

    iteration = 30000
    mlp_color = 0
    render_pkg = render(viewpoint_cam, gaussians, pipe, background, mlp_color, shift_factors, iteration=iteration, hybrid=hybrid, global_alignment=scene.getGlobalAlignment())
    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

    flow_apply2_gt_or_img = None
    P_sensor, P_view_insidelens_direction = generate_control_pts(viewpoint_cam, control_point_sample_scale, flow_scale)
    if not apply2gt:
        image, mask, flow_apply2_gt_or_img = apply_distortion(lens_net, P_view_insidelens_direction, P_sensor, viewpoint_cam, image, apply2gt=apply2gt, flow_scale=flow_scale)
        if if_circular_mask:
            mask = mask * circular_mask
    if apply2gt:
        gt_image, mask, flow_apply2_gt_or_img = apply_distortion(lens_net, P_view_insidelens_direction, P_sensor, viewpoint_cam, image, apply2gt=apply2gt)

    with torch.no_grad():
        training_report(iteration, scene, render, (pipe, background, mlp_color, shift_factors), lens_net, opt_distortion, no_distortion_mask, outside_rasterizer, flow_scale, control_point_sample_scale, flow_apply2_gt_or_img, apply2gt)


def training_report(iteration, scene : Scene, renderFunc, renderArgs, lens_net, opt_distortion, no_distortion_mask, outside_rasterizer, flow_scale, control_point_sample_scale, flow_apply2_gt_or_img, apply2gt):
    torch.cuda.empty_cache()
    validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                          {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

    file_path = os.path.join(scene.model_path, 'validation.txt')
    with open(file_path, 'a') as f:
        for config in validation_configs:
            name = config['name']
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssims = []
                lpipss = []
                os.makedirs(os.path.join(scene.model_path, 'validation_{}').format(name), exist_ok=True)
                os.makedirs(os.path.join(scene.model_path, 'validation_{}/gt').format(name), exist_ok=True)
                os.makedirs(os.path.join(scene.model_path, 'validation_{}/renderred').format(name), exist_ok=True)
                for idx, viewpoint in enumerate(config['cameras']):
                    viewpoint.reset_intrinsic(
                        #focal2fov(viewpoint.focal_x, int(flow_scale[0] * viewpoint.           orig_fov_w)),
                        #focal2fov(viewpoint.focal_y, int(flow_scale[1] * viewpoint.           orig_fov_h)),
                        viewpoint.FoVx,
                        viewpoint.FoVy,
                        viewpoint.focal_x,
                        viewpoint.focal_y,
                        2. * viewpoint.image_width,
                        2. * viewpoint.image_height
                    )

                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, global_alignment=scene.getGlobalAlignment())["render"], 0.0, 1.0)
                    torchvision.utils.save_image(image, os.path.join(scene.model_path, 'validation_{}/renderred/{}'.format(name, viewpoint.image_name) + "_" + name + ".png"))
                    if outside_rasterizer:
                        if not apply2gt:
                            image = F.grid_sample(
                                image.unsqueeze(0),
                                flow_apply2_gt_or_img.unsqueeze(0),
                                mode="bilinear",
                                padding_mode="zeros",
                                align_corners=True,
                            )
                            image = center_crop(image, viewpoint.fish_gt_image_resolution[1], viewpoint.fish_gt_image_resolution[2]).squeeze(0)
                            mask = (~((image.squeeze(0)[0]==0.) & (image.squeeze(0)[1]==0.)).unsqueeze(0)).float()
                            gt_image = viewpoint.fish_gt_image.cuda() * mask
                            torchvision.utils.save_image(gt_image.cpu(), os.path.join(scene.model_path, 'validation_{}/gt/masked_{}'.format(name, viewpoint.image_name) + "_" + name + ".png"))
                            torchvision.utils.save_image(image, os.path.join(scene.model_path, 'validation_{}/renderred/distorted_{}'.format(name, viewpoint.image_name) + "_" + name + ".png"))
                        elif apply2gt:
                            P_sensor, P_view_insidelens_direction = generate_control_pts(viewpoint, control_point_sample_scale, flow_scale)
                            gt_image, mask, flow_apply2_gt_or_img = apply_distortion(lens_net, P_view_insidelens_direction, P_sensor, viewpoint, image, apply2gt=apply2gt)
                            if no_distortion_mask:
                                image = image
                            else:
                                image = image * mask
                            torchvision.utils.save_image(gt_image, os.path.join(scene.model_path, 'validation_{}/gt/{}_perspective'.format(name, viewpoint.image_name) + "_" + name + ".png"))
                            torchvision.utils.save_image(viewpoint.fish_gt_image, os.path.join(scene.model_path, 'validation_{}/gt/{}_fish'.format(name, viewpoint.image_name) + "_" + name + ".png"))
                            torchvision.utils.save_image(viewpoint.original_image, os.path.join(scene.model_path, 'validation_{}/gt/{}_undis'.format(name, viewpoint.image_name) + "_" + name + ".png"))
                            torchvision.utils.save_image(image*mask, os.path.join(scene.model_path, 'validation_{}/renderred/{}_masked'.format(name, viewpoint.image_name) + "_" + name + ".png"))
                    else:
                        gt_image = viewpoint.original_image.cuda()
                        torchvision.utils.save_image(gt_image, os.path.join(scene.model_path, 'validation_{}/gt/{}_perspective'.format(name, viewpoint.image_name) + "_" + name + ".png"))

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssims.append(ssim(image, gt_image))
                    lpipss.append(lpips(image, gt_image))

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                print("\nSSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("\nLPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                f.write("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                f.write("\nSSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                f.write("\nLPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))


    torch.cuda.empty_cache()


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
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # wandb setting
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
    parser.add_argument('--start_vignetting', type=int, default=10000000000)
    parser.add_argument("--extra_loss", action="store_true", default=False)
    parser.add_argument('--start_opt_lens', type=int, default=1)
    parser.add_argument('--extend_scale', type=float, default=2.)
    parser.add_argument('--control_point_sample_scale', type=float, default=8.)
    parser.add_argument("--outside_rasterizer", action="store_true", default=False)
    parser.add_argument("--apply2gt", action="store_true", default=False)
    parser.add_argument("--abs_grad", action="store_true", default=False)
    parser.add_argument('--densi_num', type=float, default=0.0002)
    parser.add_argument("--if_circular_mask", action="store_true", default=False)
    # flow_scale[0] is width and flow_scale[1] is height
    parser.add_argument("--flow_scale", nargs="+", type=float, default=[1., 1.])
    parser.add_argument("--render_resolution", type=float, default=1.)
    parser.add_argument('--iresnet_lr', type=float, default=1e-7)
    parser.add_argument('--opacity_threshold', type=float, default=0.005)

    parser.add_argument("--opt_shift", action="store_true", default=False)
    parser.add_argument("--no_distortion_mask", action="store_true", default=False)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)

    # Initialize wandb
    os.makedirs(args.model_path, exist_ok=True)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, use_wandb=(args.wandb_project_name!=None), random_init=args.random_init_pc, hybrid=args.hybrid, opt_cam=args.opt_cam, opt_shift=args.opt_shift, no_distortion_mask=args.no_distortion_mask, opt_distortion=args.opt_distortion, start_vignetting=args.start_vignetting, opt_intrinsic=args.opt_intrinsic, r_t_lr=args.r_t_lr, r_t_noise=args.r_t_noise, global_alignment_lr=args.global_alignment_lr, extra_loss=args.extra_loss, start_opt_lens=args.start_opt_lens, extend_scale=args.extend_scale, control_point_sample_scale=args.control_point_sample_scale, outside_rasterizer=args.outside_rasterizer, abs_grad=args.abs_grad, densi_num=args.densi_num, if_circular_mask=args.if_circular_mask, flow_scale=args.flow_scale, render_resolution=args.render_resolution, apply2gt=args.apply2gt, iresnet_lr=args.iresnet_lr, opacity_threshold=args.opacity_threshold)

    # All done
    print("\nTraining complete.")
