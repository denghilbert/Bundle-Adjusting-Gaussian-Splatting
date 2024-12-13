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
import io
import sys
import torch
import torchvision
import random
import math
from random import randint
from utils.loss_utils import l1_loss, ssim, kl_divergence, l2_loss
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel, SpecularModel, iResNet
from scene.gaussian_model import build_scaling_rotation
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
from utils.util_distortion import homogenize, dehomogenize, colorize, plot_points, center_crop, apply_distortion, generate_control_pts
from utils.cubemap_utils import apply_flow_up_down_left_right, generate_pts_up_down_left_right, mask_half
import copy
from scene.cameras import Camera, quaternion_to_rotation_matrix
from scipy.ndimage import binary_erosion
import matplotlib.pyplot as plt

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

def mask_img(img):
    black_mask = (img == 0).all(dim=0)  # Shape: [684, 228], True where all RGB values are 0
    alpha_channel = (~black_mask).float()  # Shape: [684, 228], 1 for non-black, 0 for black
    new_img_with_alpha = torch.cat([img, alpha_channel.unsqueeze(0)], dim=0)  # Shape: [4, 684, 228]
    new_img_with_alpha = new_img_with_alpha.clamp(0, 1)  # Ensure the values are in [0, 1]
    new_img_with_alpha = (new_img_with_alpha * 255).byte()  # Convert to 0-255 range
    img_pil = torchvision.transforms.functional.to_pil_image(new_img_with_alpha)
    return img_pil


def generate_circular_mask(image_shape: torch.Size, radius: int) -> torch.Tensor:
    """
    Generates a circular mask based on the provided radius.

    Args:
    - image_shape (torch.Size): The shape of the image (C, H, W).
    - radius (int): The radius of the circular mask.

    Returns:
    - torch.Tensor: A circular mask with shape (C, H, W) where the area inside the
      radius from the center is 1 and outside is 0.
    """
    _, h, w = image_shape

    # Create a coordinate grid centered at the image center
    y_center, x_center = h // 2, w // 2
    y_grid, x_grid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")

    # Calculate the distance of each point from the center
    dist_from_center = torch.sqrt((x_grid - x_center) ** 2 + (y_grid - y_center) ** 2)

    # Create the circular mask
    mask = (dist_from_center <= radius).float()

    # Expand the mask to match the shape of the image (C, H, W)
    mask = mask.unsqueeze(0).repeat(3, 1, 1)

    return mask


def render_cubemap(viewpoint_cam, lens_net, mask_fov90, shift_width, shift_height, gaussians, pipe, background, mlp_color, shift_factors, iteration, hybrid, scene, validation=False):
    img_list, viewspace_point_tensor_list, visibility_filter_list, radii_list = [], [], [], []
    if validation:
        img_perspective_list = []

    rays_forward = generate_pts_up_down_left_right(viewpoint_cam, shift_width=0, shift_height=0)
    rays_residual = generate_pts_up_down_left_right(viewpoint_cam, shift_width=0, shift_height=0, sample_rate=8)
    render_pkg = render(viewpoint_cam, gaussians, pipe, background, mlp_color, shift_factors, iteration=iteration, hybrid=hybrid, global_alignment=scene.getGlobalAlignment())
    img_forward, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"] * mask_fov90, render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
    img_distorted, img_perspective, residual = apply_flow_up_down_left_right(viewpoint_cam, lens_net, rays_forward, rays_residual, img_forward, types="forward", is_fisheye=True, iteration=iteration)
    img_list.append(img_distorted)
    if validation:
        img_perspective_list.append(img_perspective)
    viewspace_point_tensor_list.append(viewspace_point_tensor)
    visibility_filter_list.append(visibility_filter)
    radii_list.append(radii)

    name = ['up', 'down', 'left', 'right']
    for i, sub_camera in enumerate(viewpoint_cam.sub_cameras):
        if i == 4: break
        render_pkg = render(sub_camera, gaussians, pipe, background, mlp_color, shift_factors, iteration=iteration, hybrid=hybrid, global_alignment=scene.getGlobalAlignment())
        img_up, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"] * mask_fov90, render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        if name[i] == 'up':
            rays_up = generate_pts_up_down_left_right(sub_camera, shift_width=0, shift_height=-shift_height)
            rays_residual = generate_pts_up_down_left_right(sub_camera, shift_width=0, shift_height=-shift_height, sample_rate=8)
        elif name[i] == 'down':
            rays_up = generate_pts_up_down_left_right(sub_camera, shift_width=0, shift_height=shift_height)
            rays_residual = generate_pts_up_down_left_right(sub_camera, shift_width=0, shift_height=-shift_height, sample_rate=8)
        elif name[i] == 'left':
            rays_up = generate_pts_up_down_left_right(sub_camera, shift_width=shift_width, shift_height=0)
            rays_residual = generate_pts_up_down_left_right(sub_camera, shift_width=0, shift_height=-shift_height, sample_rate=8)
        elif name[i] == 'right':
            rays_up = generate_pts_up_down_left_right(sub_camera, shift_width=-shift_width, shift_height=0)
            rays_residual = generate_pts_up_down_left_right(sub_camera, shift_width=0, shift_height=-shift_height, sample_rate=8)
        img_distorted, img_perspective = apply_flow_up_down_left_right(sub_camera, lens_net, rays_up, rays_residual, img_up, types=name[i], is_fisheye=True)
        img_distorted_masked, half_mask = mask_half(img_distorted, name[i])

        img_list.append(img_distorted_masked)
        viewspace_point_tensor_list.append(viewspace_point_tensor)
        visibility_filter_list.append(visibility_filter)
        radii_list.append(radii)
        if validation:
            img_perspective_list.append(img_perspective)

    if not validation:
        return img_list, viewspace_point_tensor_list, visibility_filter_list, radii_list, residual
    else:
        return img_list, img_perspective_list


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, use_wandb=False, random_init=False, hybrid=False, opt_cam=False, opt_shift=False, no_distortion_mask=False, opt_distortion=False, start_vignetting=10000000000, opt_intrinsic=False, r_t_noise=[0., 0.], r_t_lr=[0.001, 0.001], global_alignment_lr=0.001, extra_loss=False, start_opt_lens=1, extend_scale=2., control_point_sample_scale=8., outside_rasterizer=False, abs_grad=False, densi_num=0.0002, if_circular_mask=False, flow_scale=[1., 1.], render_resolution=1., apply2gt=False, iresnet_lr=1e-7, no_init_iresnet=False, opacity_threshold=0.005, mcmc=False, cubemap=False, table1=False):
    if dataset.cap_max == -1 and mcmc:
        print("Please specify the maximum number of Gaussians using --cap_max.")
        exit()
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset.asg_degree)
    if hybrid:
        specular_mlp = SpecularModel()
        specular_mlp.train_setting(opt)
    lens_net = iResNet().cuda()
    l_lens_net = [{'params': lens_net.parameters(), 'lr': 1e-5}]
    optimizer_lens_net = torch.optim.Adam(l_lens_net, eps=1e-15)
    scheduler_lens_net = torch.optim.lr_scheduler.MultiStepLR(optimizer_lens_net, milestones=[7000], gamma=0.5)
    def zero_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 0.)
            nn.init.constant_(m.bias, 0.)
    #lens_net.apply(zero_weights)
    #for param in lens_net.parameters():
    #    print(param)

    #vignetting_factors = nn.Parameter(torch.ones(100, requires_grad=True, device='cuda'))
    #vignetting_optimizer = torch.optim.Adam([vignetting_factors], lr=1e-3)
    vignetting_model = VignettingModel(n_terms=4, device='cuda')
    vignetting_optimizer = torch.optim.Adam(vignetting_model.parameters(), lr=0.01)
    vignetting_scheduler = torch.optim.lr_scheduler.MultiStepLR(vignetting_optimizer, milestones=[1000], gamma=10)

    shift_factors = nn.Parameter(torch.tensor([-0., -0., -0.], requires_grad=True, device='cuda'))
    shift_optimizer = torch.optim.Adam([shift_factors], lr=1e-5)
    shift_scheduler = torch.optim.lr_scheduler.MultiStepLR(shift_optimizer, milestones=[30000], gamma=0.1)

    scene = Scene(dataset, gaussians, random_init=random_init, r_t_noise=r_t_noise, r_t_lr=r_t_lr, global_alignment_lr=global_alignment_lr, outside_rasterizer=outside_rasterizer, flow_scale=flow_scale, render_resolution=render_resolution, apply2gt=apply2gt, vis_pose=args.vis_pose, cubemap=cubemap, table1=table1)

    viewpoint = scene.getTestCameras()[0]
    #pose_GT, pose_aligned = scene.loadAlignCameras(if_vis_train=True, path=scene.model_path)
    #torch.save(pose_GT, os.path.join(scene.model_path, 'gt.pt'))
    #torch.save(pose_aligned, os.path.join(scene.model_path, 'align.pt'))
    #import pdb;pdb.set_trace()
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        print(f'loading iteraton {first_iter}')
        lens_net = torch.load(os.path.join(scene.model_path, f'lens_net{first_iter}.pth'))
        gaussians.restore(model_params, opt, validation=True)
        if opt_cam:
            scene.train_cameras = torch.load(os.path.join(scene.model_path, 'opt_cams.pt'))
            scene.unnoisy_train_cameras = torch.load(os.path.join(scene.model_path, 'gt_cams.pt'))
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    mlp_color = 0
    if 'garden' in scene.model_path:
        gaussians.load_ply("netflix/netflix_garden_lr8_apply2render_res2_scale2.5_filtersky_/point_cloud/iteration_20000/point_cloud.ply")

    os.makedirs(os.path.join(scene.model_path, 'trajectory'), exist_ok=True)

    with torch.no_grad():
        viewpoint_cam = scene.getTestCameras()[0]
        P_sensor, P_view_insidelens_direction = generate_control_pts(viewpoint_cam, control_point_sample_scale, flow_scale)
        P_view_outsidelens_direction = lens_net.forward(P_view_insidelens_direction, sensor_to_frustum=apply2gt)
        camera_directions_w_lens = homogenize(P_view_outsidelens_direction)
        control_points = camera_directions_w_lens.reshape((P_sensor.shape[0], P_sensor.shape[1], 3))[:, :, :2]
        projection_matrix = viewpoint_cam.projection_matrix
        flow = control_points @ projection_matrix[:2, :2]
        flow = nn.functional.interpolate(flow.permute(2, 0, 1).unsqueeze(0), size=(int(viewpoint_cam.fish_gt_image_resolution[1]*flow_scale[0]), int(viewpoint_cam.fish_gt_image_resolution[2]*flow_scale[1])), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).squeeze(0)

        with open(os.path.join(scene.model_path, 'images.txt'), 'r') as f:
            lines = f.readlines()
        count = 0
        for line in lines:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            camera_id = int(parts[8])
            image_name = parts[9]
            quaternion_tensor = torch.tensor([qw, qx, qy, qz], dtype=torch.float32)
            R = quaternion_to_rotation_matrix(quaternion_tensor).t().cpu().numpy()
            T = torch.tensor([tx, ty, tz]).cpu().numpy()
            viewpoint_cam.reset_extrinsic(R, T)

            if outside_rasterizer:
                if count == 0:
                    viewpoint_cam.reset_intrinsic(
                        focal2fov(viewpoint_cam.focal_x, 1.5 * viewpoint_cam.image_width),
                        focal2fov(viewpoint_cam.focal_y, 1.5 * viewpoint_cam.image_height),
                        viewpoint_cam.focal_x, viewpoint_cam.focal_y,
                        1.5* viewpoint_cam.image_width,
                        1.5* viewpoint_cam.image_height)
                render_pkg = render(viewpoint_cam, gaussians, pipe, background, mlp_color, shift_factors, iteration=first_iter, hybrid=hybrid, global_alignment=scene.getGlobalAlignment())
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

                torchvision.utils.save_image(image, os.path.join(scene.model_path, f'trajectory/perspective_{count}.png'))
                #torchvision.utils.save_image(viewpoint_cam.fish_gt_image, os.path.join(scene.model_path, f'trajectory/gt_{viewpoint_cam.image_name}.png'))

                image = F.grid_sample(
                    image.unsqueeze(0),
                    flow.unsqueeze(0),
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )
                image = center_crop(image, viewpoint_cam.fish_gt_image_resolution[1], viewpoint_cam.fish_gt_image_resolution[2]).squeeze(0)
                mask = (~((image[0]==0.0000) & (image[1]==0.0000)).unsqueeze(0)).float()
                torchvision.utils.save_image(image*mask, os.path.join(scene.model_path, f'trajectory/fisheye_{count}.png'))
                print(line)
                count += 1
            else:
                if count == 0:
                    viewpoint_cam.reset_intrinsic(
                        focal2fov(viewpoint_cam.focal_x, viewpoint_cam.fish_gt_image_resolution[2]),
                        focal2fov(viewpoint_cam.focal_y, viewpoint_cam.fish_gt_image_resolution[1]),
                        viewpoint_cam.focal_x, viewpoint_cam.focal_y,
                        viewpoint_cam.fish_gt_image_resolution[2],
                        viewpoint_cam.fish_gt_image_resolution[1])
                render_pkg = render(viewpoint_cam, gaussians, pipe, background, mlp_color, shift_factors, iteration=first_iter, hybrid=hybrid, global_alignment=scene.getGlobalAlignment())
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                torchvision.utils.save_image(image, os.path.join(scene.model_path, f'trajectory/perspective_{count}.png'))
                print(line)
                count += 1


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

    parser.add_argument("--mcmc", action="store_true", default=False)
    parser.add_argument("--no_init_iresnet", action="store_true", default=False)
    parser.add_argument("--cubemap", action="store_true", default=False)
    parser.add_argument("--table1", action="store_true", default=False)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)

    # Initialize wandb
    os.makedirs(args.model_path, exist_ok=True)
    if args.wandb_project_name != None:
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
    training(
        lp.extract(args), op.extract(args), pp.extract(args),
        args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,
        use_wandb=(args.wandb_project_name!=None), random_init=args.random_init_pc, hybrid=args.hybrid, opt_cam=args.opt_cam,
        opt_shift=args.opt_shift, no_distortion_mask=args.no_distortion_mask, opt_distortion=args.opt_distortion,
        start_vignetting=args.start_vignetting, opt_intrinsic=args.opt_intrinsic, r_t_lr=args.r_t_lr, r_t_noise=args.r_t_noise,
        global_alignment_lr=args.global_alignment_lr, extra_loss=args.extra_loss, start_opt_lens=args.start_opt_lens,
        extend_scale=args.extend_scale, control_point_sample_scale=args.control_point_sample_scale, outside_rasterizer=args.outside_rasterizer,
        abs_grad=args.abs_grad, densi_num=args.densi_num, if_circular_mask=args.if_circular_mask, flow_scale=args.flow_scale,
        render_resolution=args.render_resolution, apply2gt=args.apply2gt, iresnet_lr=args.iresnet_lr, no_init_iresnet=args.no_init_iresnet, opacity_threshold=args.opacity_threshold, mcmc=args.mcmc, cubemap=args.cubemap, table1=args.table1)

    # All done
    print("\nTraining complete.")
