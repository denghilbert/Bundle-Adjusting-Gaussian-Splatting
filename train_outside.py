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
from scipy.ndimage import zoom
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
from utils.util_distortion import homogenize, dehomogenize, colorize, plot_points, center_crop, init_from_colmap, apply_distortion, generate_control_pts, read_colmap_coeff
from utils.cubemap_utils import apply_flow_up_down_left_right, generate_pts_up_down_left_right, mask_half
import copy
from scene.cameras import Camera
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


def render_cubemap(viewpoint_cam, cubemap_net, mask_fov90, shift_width, shift_height, gaussians, pipe, background, mlp_color, shift_factors, iteration, hybrid, scene, validation=False, vis_cubemap_net=False):
    control_r = torch.linspace(0, 1.392, 10000).unsqueeze(1).cuda()
    control_theta = cubemap_net.forward(control_r, sensor_to_frustum=True)
    if vis_cubemap_net:
        control_r_cpu = control_r.squeeze(1).detach().cpu().numpy()
        control_theta_cpu = control_theta.squeeze(1).detach().cpu().numpy()
        plt.figure()
        plt.plot(control_r_cpu, control_theta_cpu)
        plt.xlabel("control_r")
        plt.ylabel("control_theta")
        plt.title("Control R vs. Control Theta")
        visualize = plt.gcf()  # Get the current figure as `visualize`
        wandb.log({f"cubemap_net/function": wandb.Image(visualize)})
        plt.close()
        wandb.log({"cubemap_net/status": control_theta.mean().item()})

    img_list, viewspace_point_tensor_list, visibility_filter_list, radii_list = [], [], [], []
    if validation:
        img_perspective_list = []

    rays_forward = generate_pts_up_down_left_right(viewpoint_cam, shift_width=0, shift_height=0)
    rays_residual = generate_pts_up_down_left_right(viewpoint_cam, shift_width=0, shift_height=0, sample_rate=8)
    render_pkg = render(viewpoint_cam, gaussians, pipe, background, mlp_color, shift_factors, iteration=iteration, hybrid=hybrid, global_alignment=scene.getGlobalAlignment())
    img_forward, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"] * mask_fov90, render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
    img_distorted, img_perspective, residual, control_theta = apply_flow_up_down_left_right(viewpoint_cam, rays_forward, rays_residual, img_forward, types="forward", is_fisheye=True, iteration=iteration, control_theta=control_theta, control_r=control_r)
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
        img_distorted, img_perspective = apply_flow_up_down_left_right(sub_camera, rays_up, rays_residual, img_up, types=name[i], is_fisheye=True, control_theta=control_theta, control_r=control_r)
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

def log_vector_field_to_wandb(residual, magnification_factor=100000, step=None):
    """
    Logs a magnified vector field visualization to Weights & Biases.

    Parameters:
        residual (torch.Tensor): A tensor of shape [1, 2, 100, 100], storing (x, y) values.
        magnification_factor (float): Factor to magnify the flow vectors.
        step (int, optional): The current step or iteration for wandb logging.
    """
    # Assuming residual has the shape [1, 2, 128, 85]
    U = residual.squeeze(0)[0].detach().cpu().numpy()  # X component
    V = residual.squeeze(0)[1].detach().cpu().numpy()  # Y component

# Resize U and V to (10, 10) using interpolation
    U_small = zoom(U, (10 / U.shape[0], 10 / U.shape[1]), order=1)
    V_small = zoom(V, (10 / V.shape[0], 10 / V.shape[1]), order=1)

# Apply magnification for better visualization
    U_small *= magnification_factor
    V_small *= magnification_factor

# Create a grid for the vector field
    x = np.arange(U_small.shape[1])
    y = np.arange(U_small.shape[0])
    X, Y = np.meshgrid(x, y)

# Plotting the vector field
    plt.figure(figsize=(10, 10))
    plt.quiver(X, Y, U_small, V_small, angles='xy', scale_units='xy', scale=1, color='b')
    plt.title("Magnified Vector Field Visualization (10x10)")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.gca().invert_yaxis()  # Optional: invert Y-axis to match image coordinates

# Save the plot to an in-memory file object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    plt.close()

# Convert the in-memory file to a PIL Image and upload to wandb
    image = Image.open(buf)
    image_array = np.array(image)
    wandb.log({"vector_field/fig": wandb.Image(image_array, caption="Magnified Vector Field Visualization (10x10)")}, step=step)
    buf.close()

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
    scheduler_lens_net = torch.optim.lr_scheduler.MultiStepLR(optimizer_lens_net, milestones=[4000], gamma=0.5)
    def zero_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 0.00001)
            nn.init.constant_(m.bias, 0.00001)
    #lens_net.apply(zero_weights)
    #for param in lens_net.parameters():
    #    print(param)


    #coeff = read_colmap_coeff(dataset)
    y_train = torch.linspace(0, 15., 5000).unsqueeze(1).cuda()
    x_train = torch.atan(y_train) - 0.03936274483986258 * torch.atan(y_train)**3 + 0.005866545097262303 * torch.atan(y_train)**5 - 0.0012988220238146179 * torch.atan(y_train)**7
    cubemap_net = iResNet(input_num=1).cuda()
    l_cubemap_net = [{'params': cubemap_net.parameters(), 'lr': 1e-5}]
    optimizer_cubemap_net = torch.optim.Adam(l_cubemap_net, eps=1e-15)
    scheduler_cubemap_net = torch.optim.lr_scheduler.MultiStepLR(optimizer_cubemap_net, milestones=[2000, 7000, 9000], gamma=0.5)
    #x_train = torch.linspace(0, 1.5, 1000).unsqueeze(1).cuda()
    #y_train = torch.tan(x_train).cuda()
    criterion = nn.MSELoss()
    progress_bar_ires = tqdm(range(0, 0), desc="Init cubemapnet")
    for epoch in range(0):
        y_pred = cubemap_net.forward(x_train, sensor_to_frustum=True) + torch.tan(x_train)
        loss = criterion(y_pred, y_train)
        optimizer_cubemap_net.zero_grad()
        loss.backward()
        optimizer_cubemap_net.step()
        scheduler_cubemap_net.step()
        progress_bar_ires.set_postfix(loss=loss.item())
        progress_bar_ires.update(1)
    for param_group in optimizer_cubemap_net.param_groups:
        param_group['lr'] = iresnet_lr
    print(f"The learning rate is reset to {param_group['lr']}")
    #cubemap_net.forward(torch.tensor([[0.5], [1.2]]).cuda(), sensor_to_frustum=True)
    #cubemap_net.forward(torch.tensor([[0.5], [1.2]]).cuda(), sensor_to_frustum=False)
    #import pdb;pdb.set_trace()

    #vignetting_factors = nn.Parameter(torch.ones(100, requires_grad=True, device='cuda'))
    #vignetting_optimizer = torch.optim.Adam([vignetting_factors], lr=1e-3)
    vignetting_model = VignettingModel(n_terms=4, device='cuda')
    vignetting_optimizer = torch.optim.Adam(vignetting_model.parameters(), lr=0.01)
    vignetting_scheduler = torch.optim.lr_scheduler.MultiStepLR(vignetting_optimizer, milestones=[1000], gamma=10)

    shift_factors = nn.Parameter(torch.tensor([-0., -0., -0.], requires_grad=True, device='cuda'))
    shift_optimizer = torch.optim.Adam([shift_factors], lr=1e-5)
    shift_scheduler = torch.optim.lr_scheduler.MultiStepLR(shift_optimizer, milestones=[30000], gamma=0.1)

    #shift_outside_factors = nn.Parameter(torch.tensor([0.002, 0.002, 0.002], requires_grad=True, device='cuda'))
    shift_outside_factors = nn.Parameter(torch.tensor([0.002, 0.002, 0.002], requires_grad=True, device='cuda').repeat(1000000, 1))
    shift_outside_optimizer = torch.optim.Adam([shift_outside_factors], lr=1e-5)

    scene = Scene(dataset, gaussians, random_init=random_init, r_t_noise=r_t_noise, r_t_lr=r_t_lr, global_alignment_lr=global_alignment_lr, outside_rasterizer=outside_rasterizer, flow_scale=flow_scale, render_resolution=render_resolution, apply2gt=apply2gt, vis_pose=args.vis_pose, cubemap=cubemap, table1=table1)

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

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    if args.vis_pose:
        opt_vis = EasyDict({'group': 'opt_pose', 'name': 'opt_pose', 'visdom': {'server': 'localhost', 'port': 8600, 'cam_depth': 0.5}})
        if opt_vis.visdom and args.vis_pose:
            is_open = check_socket_open(opt_vis.visdom.server,opt_vis.visdom.port)
            retry = None
            vis = visdom.Visdom(server=opt_vis.visdom.server,port=opt_vis.visdom.port,env=opt_vis.group)
            pose_GT, pose_aligned = scene.loadAlignCameras(if_vis_train=True, path=scene.model_path)
            vis_cameras(opt_vis, vis, step=0, poses=[pose_aligned, pose_GT])
            os.makedirs(os.path.join(args.model_path, 'plot'), exist_ok=True)

    # colmap init
    if outside_rasterizer and not no_init_iresnet:
        init_from_colmap(scene, dataset, optimizer_lens_net, lens_net, scheduler_lens_net, resume_training=checkpoint, iresnet_lr=1e-5)
    if cubemap:
        #init_from_tan(scene, dataset, optimizer_lens_net, lens_net, scheduler_lens_net, resume_training=checkpoint, iresnet_lr=iresnet_lr)
        #import pdb;pdb.set_trace()
        for param_group in optimizer_lens_net.param_groups:
            param_group['lr'] = iresnet_lr
        print(f"The learning rate is reset to {param_group['lr']}")

    # circular mask
    if if_circular_mask:
        height, width = scene.getTrainCameras().copy()[0].image_height, scene.getTrainCameras().copy()[0].image_width
        circular_mask = torch.zeros((height, width), dtype=torch.bool)
        center_x, center_y = width/2, height/2
        radius = height/2 if height >= width else width/2
        for y in range(height):
            for x in range(width):
                if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                    circular_mask[y, x] = 1.
        circular_mask = circular_mask.unsqueeze(0)
        circular_mask = circular_mask.repeat(3, 1, 1).cuda().float()


    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
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

        xyz_lr = gaussians.update_learning_rate(iteration)
        if hybrid:
            specular_mlp.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if opt_shift:
            c2w = viewpoint_cam.get_c2w()
            R = c2w[:3, :3]
            cam_pos = viewpoint_cam.get_camera_center()
            look_at_direction_camera = torch.tensor([0, 0, -1.], device=cam_pos.device)
            look_at_direction_world = R @ look_at_direction_camera
            direction_vectors = gaussians._xyz - cam_pos
            look_at_direction = -look_at_direction_world
            direction_vectors_normalized = direction_vectors / direction_vectors.norm(dim=1, keepdim=True)
            look_at_direction_normalized = look_at_direction / look_at_direction.norm()
            dot_products = torch.sum(direction_vectors_normalized * look_at_direction_normalized, dim=1)
            angles = torch.acos(dot_products)
            #shift = shift_outside_factors[0] * angles**3 + shift_outside_factors[1] * angles**5 + shift_outside_factors[2] * angles**7
            shift = shift_outside_factors[:, 0] * angles**3 + shift_outside_factors[:, 1] * angles**5 + shift_outside_factors[:, 2] * angles**7
            #gaussians._xyz = gaussians._xyz + shift.unsqueeze(1) * look_at_direction_world.detach()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        # input type
        N = gaussians.get_xyz.shape[0]
        mlp_color = 0

        if cubemap:
            mask_fov90 = torch.zeros((1, viewpoint_cam.image_height, viewpoint_cam.image_width), dtype=torch.float32).cuda()
            #mask_fov90[:, viewpoint_cam.image_height//2 - int(viewpoint_cam.focal_y) - 1:viewpoint_cam.image_height//2 + int(viewpoint_cam.focal_y) + 1, viewpoint_cam.image_width//2 - int(viewpoint_cam.focal_x) - 1:viewpoint_cam.image_width//2 + int(viewpoint_cam.focal_x) + 1] = 1
            mask_fov90[:, viewpoint_cam.image_height//2 - int(viewpoint_cam.focal_y) - 2:viewpoint_cam.image_height//2 + int(viewpoint_cam.focal_y) + 2, viewpoint_cam.image_width//2 - int(viewpoint_cam.focal_x) - 2:viewpoint_cam.image_width//2 + int(viewpoint_cam.focal_x) + 2] = 1

            if iteration % 3000 == 1:
                vis_cubemap_net = True
            img_list, viewspace_point_tensor_list, visibility_filter_list, radii_list, residual = render_cubemap(viewpoint_cam, cubemap_net, mask_fov90, 0., 0., gaussians, pipe, background, mlp_color, shift_factors, iteration, hybrid, scene, vis_cubemap_net=vis_cubemap_net)
            vis_cubemap_net = False

            img_mask_list = []
            for img in img_list:
                mask = (img[0] > 0.001) | (img[1] > 0.001) | (img[2] > 0.001).cuda()
                img_mask_list.append(mask)

            if residual != None:
                if iteration % 1000 == 2:
                    log_vector_field_to_wandb(residual, magnification_factor=5, step=iteration)
                if iteration % 100 == 2:
                    wandb.log({"vector_field/status": residual.mean().item()}, step=iteration)

        else:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, mlp_color, shift_factors, iteration=iteration, hybrid=hybrid, global_alignment=scene.getGlobalAlignment())
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            #P_sensor, P_view_insidelens_direction = generate_control_pts(viewpoint_cam, control_point_sample_scale, flow_scale)
            #image, mask, flow_apply2_gt_or_img = apply_distortion(lens_net, P_view_insidelens_direction, P_sensor, viewpoint_cam, image, apply2gt=False, flow_scale=flow_scale)
            #torchvision.utils.save_image(image, os.path.join(scene.model_path, f'before.png'))

            #c2w = viewpoint_cam.get_c2w()
            #R = c2w[:3, :3]
            #cam_pos = viewpoint_cam.get_camera_center()
            #look_at_direction_camera = torch.tensor([0, 0, -1.], device=cam_pos.device)
            #look_at_direction_world = R @ look_at_direction_camera
            #direction_vectors = gaussians._xyz - cam_pos
            #look_at_direction = -look_at_direction_world
            #direction_vectors_normalized = direction_vectors / direction_vectors.norm(dim=1, keepdim=True)
            #look_at_direction_normalized = look_at_direction / look_at_direction.norm()
            #dot_products = torch.sum(direction_vectors_normalized * look_at_direction_normalized, dim=1)
            #angles = torch.acos(dot_products)
            #shift = shift_outside_factors[0] * angles**3 + shift_outside_factors[1] * angles**5 + shift_outside_factors[2] * angles**7
            #gaussians._xyz = gaussians._xyz + look_at_direction_world.detach().unsqueeze(0).expand(shift.shape[0], -1) * shift.unsqueeze(1)

            #render_pkg = render(viewpoint_cam, gaussians, pipe, background, mlp_color, shift_factors, iteration=iteration, hybrid=hybrid, global_alignment=scene.getGlobalAlignment())
            #image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            #P_sensor, P_view_insidelens_direction = generate_control_pts(viewpoint_cam, control_point_sample_scale, flow_scale)
            #image, mask, flow_apply2_gt_or_img = apply_distortion(lens_net, P_view_insidelens_direction, P_sensor, viewpoint_cam, image, apply2gt=False, flow_scale=flow_scale)
            #torchvision.utils.save_image(image, os.path.join(scene.model_path, f'after.png'))
            #import pdb;pdb.set_trace()

        flow_apply2_gt_or_img = None
        if outside_rasterizer and not cubemap:
            P_sensor, P_view_insidelens_direction = generate_control_pts(viewpoint_cam, control_point_sample_scale, flow_scale)
            if not apply2gt:
                image, mask, flow_apply2_gt_or_img = apply_distortion(lens_net, P_view_insidelens_direction, P_sensor, viewpoint_cam, image, apply2gt=apply2gt, flow_scale=flow_scale)
                if if_circular_mask:
                    mask = mask * circular_mask

            if apply2gt:
                gt_image, mask, flow_apply2_gt_or_img = apply_distortion(lens_net, P_view_insidelens_direction, P_sensor, viewpoint_cam, image, apply2gt=apply2gt)

        if start_vignetting < iteration:
            vignetting_mask = vignetting_model((image.shape[1], image.shape[2]))
            mask = mask * vignetting_mask

            if iteration % 1000 == 1:
                mask_ = vignetting_model((image.shape[1], image.shape[2])).cpu().detach().unsqueeze(0)
                mask_ = mask_.permute(1, 2, 0)
                mask_ = mask_.cpu().numpy()
                wandb.log({f"vignetting_model/vis": wandb.Image(mask_)})

        # Loss
        if outside_rasterizer and not apply2gt:
            gt_image = viewpoint_cam.fish_gt_image.cuda()
            if not no_distortion_mask:
                gt_image = gt_image * mask
            Ll1 = l1_loss(image, gt_image)
            ssim_loss = ssim(image, gt_image)
        elif outside_rasterizer and apply2gt:
            if not no_distortion_mask:
                image = image * mask
            Ll1 = l1_loss(image, gt_image)
            ssim_loss = ssim(image, gt_image)
        elif cubemap:
            gt_image = viewpoint_cam.original_image.cuda()
            #mask_gt_image = generate_circular_mask(gt_image.shape, min(gt_image.shape[-2:])//2).cuda()
            mask_gt_image = generate_circular_mask(gt_image.shape, 450).cuda()

            Ll1 = (
                l1_loss(img_list[0]*mask_gt_image*img_mask_list[0], gt_image*mask_gt_image*img_mask_list[0]) +
                l1_loss(img_list[1]*mask_gt_image*img_mask_list[1], gt_image*mask_gt_image*img_mask_list[1]) +
                l1_loss(img_list[2]*mask_gt_image*img_mask_list[2], gt_image*mask_gt_image*img_mask_list[2]) +
                l1_loss(img_list[3]*mask_gt_image*img_mask_list[3], gt_image*mask_gt_image*img_mask_list[3]) +
                l1_loss(img_list[4]*mask_gt_image*img_mask_list[4], gt_image*mask_gt_image*img_mask_list[4])
            )
            ssim_loss = (
                ssim(img_list[0]*mask_gt_image*img_mask_list[0], gt_image*mask_gt_image*img_mask_list[0]) +
                ssim(img_list[1]*mask_gt_image*img_mask_list[1], gt_image*mask_gt_image*img_mask_list[1]) +
                ssim(img_list[2]*mask_gt_image*img_mask_list[2], gt_image*mask_gt_image*img_mask_list[2]) +
                ssim(img_list[3]*mask_gt_image*img_mask_list[3], gt_image*mask_gt_image*img_mask_list[3]) +
                ssim(img_list[4]*mask_gt_image*img_mask_list[4], gt_image*mask_gt_image*img_mask_list[4])
            )

            #torchvision.utils.save_image(gt_image * mask_gt_image * img_mask_list[0], os.path.join(scene.model_path, f'gt_0.png'))
            #torchvision.utils.save_image(gt_image * mask_gt_image * img_mask_list[1], os.path.join(scene.model_path, f'gt_1.png'))
            #torchvision.utils.save_image(gt_image * mask_gt_image * img_mask_list[2], os.path.join(scene.model_path, f'gt_2.png'))
            #torchvision.utils.save_image(gt_image * mask_gt_image * img_mask_list[3], os.path.join(scene.model_path, f'gt_3.png'))
            #torchvision.utils.save_image(gt_image * mask_gt_image * img_mask_list[4], os.path.join(scene.model_path, f'gt_4.png'))
            #torchvision.utils.save_image(img_list[0] * mask_gt_image * img_mask_list[0], os.path.join(scene.model_path, f'img_0.png'))
            #torchvision.utils.save_image(img_list[1] * mask_gt_image * img_mask_list[1], os.path.join(scene.model_path, f'img_1.png'))
            #torchvision.utils.save_image(img_list[2] * mask_gt_image * img_mask_list[2], os.path.join(scene.model_path, f'img_2.png'))
            #torchvision.utils.save_image(img_list[3] * mask_gt_image * img_mask_list[3], os.path.join(scene.model_path, f'img_3.png'))
            #torchvision.utils.save_image(img_list[4] * mask_gt_image * img_mask_list[4], os.path.join(scene.model_path, f'img_4.png'))

            #torchvision.utils.save_image(mask_gt_image, os.path.join(scene.model_path, f'mask_gt.png'))
            #torchvision.utils.save_image(img_mask_list[0].float(), os.path.join(scene.model_path, f'mask_0.png'))
            #torchvision.utils.save_image(img_mask_list[1].float(), os.path.join(scene.model_path, f'mask_1.png'))
            #torchvision.utils.save_image(img_mask_list[2].float(), os.path.join(scene.model_path, f'mask_2.png'))
            #torchvision.utils.save_image(img_mask_list[3].float(), os.path.join(scene.model_path, f'mask_3.png'))
            #torchvision.utils.save_image(img_mask_list[4].float(), os.path.join(scene.model_path, f'mask_4.png'))
            #import pdb;pdb.set_trace()
        else:
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            ssim_loss = ssim(image, gt_image)
        if iteration % 10000 == 1:
            if cubemap:
                torchvision.utils.save_image(img_list[0] * mask_gt_image * img_mask_list[0], os.path.join(scene.model_path, f'render_{iteration}.png'))
                torchvision.utils.save_image(gt_image * mask_gt_image * img_mask_list[0], os.path.join(scene.model_path, f'gt_{iteration}.png'))
            else:
                torchvision.utils.save_image(image, os.path.join(scene.model_path, f"render_{iteration}.png"))
                torchvision.utils.save_image(gt_image, os.path.join(scene.model_path, f"gt_fish2perspective_{iteration}.png"))

        if cubemap:
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (5.0 - ssim_loss)# + 0.1 * (loss_projection / len(camera_pairs[viewpoint_cam.uid]))
        else:
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_loss)# + 0.1 * (loss_projection / len(camera_pairs[viewpoint_cam.uid]))

        if mcmc:
            loss = loss + args.opacity_reg * torch.abs(gaussians.get_opacity).mean()
            loss = loss + args.scale_reg * torch.abs(gaussians.get_scaling).mean()

        #residual.retain_grad()
        #control_theta.retain_grad()
        #gaussians._xyz.retain_grad()
        loss.backward(retain_graph=True)
        #print(residual.grad)
        #for name, param in lens_net.named_parameters():
        #    if param.grad is not None:
        #        print(f"Layer: {name} | Gradient: {param.grad}")
        #    else:
        #        print(f"Layer: {name} | Gradient: None")
        #print(control_theta.grad)
        #if iteration > 2000 and loss > 0.1:
        #    import pdb;pdb.set_trace()
        #    print(viewpoint_cam.image_name)

        #if iteration % 100 == 1 and use_wandb:
        #    scalars = {
        #        f"cubemap_net/grad_layer0": cubemap_net.i_resnet_linear.module_list[0].residual[10].weight.grad.mean(),
        #        f"cubemap_net/grad_layer1": cubemap_net.i_resnet_linear.module_list[1].residual[10].weight.grad.mean(),
        #        f"cubemap_net/grad_layer2": cubemap_net.i_resnet_linear.module_list[2].residual[10].weight.grad.mean(),
        #        f"cubemap_net/grad_layer3": cubemap_net.i_resnet_linear.module_list[3].residual[10].weight.grad.mean(),
        #    }
        #    wandb.log(scalars, step=iteration)

        #for name, param in cubemap_net.named_parameters():
        #    if param.grad is not None:
        #        print(f"Layer: {name} | Gradient: {param.grad}")
        #    else:
        #        print(f"Layer: {name} | Gradient: None")
        #import pdb;pdb.set_trace()

        #if use_wandb:
        #    scalars = {
        #        f"gradient/3d_gradient": gaussians._xyz.grad.mean(),
        #        f"gradient/3d_gradient_max": gaussians._xyz.grad.max(),
        #        f"gradient/3d_gradient_min": gaussians._xyz.grad.min(),
        #        f"gradient/nan": torch.isnan(gaussians._xyz.grad).sum(),
        #        f"gradient/inf": torch.isinf(gaussians._xyz.grad).sum(),
        #    }
        #    wandb.log(scalars, step=iteration)

        #last_linear_layer = lens_net.i_resnet_linear.module_list[-1].residual[-1]
        #print(last_linear_layer.weight.grad)
        #print(shift_factors.grad)
        #import pdb;pdb.set_trace()
        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            # Progress bar
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            #if iteration in testing_iterations:
            if iteration % 500 == 0 and args.vis_pose:
                pose_GT, pose_aligned = scene.loadAlignCameras(if_vis_train=True, iteration=iteration, path=scene.model_path)
                vis_cameras(opt_vis, vis, step=iteration, poses=[pose_aligned, pose_GT])

            # Log and save
            if not outside_rasterizer:
                P_view_insidelens_direction = None
                P_sensor = None

            training_report(use_wandb, iteration, Ll1, ssim_loss, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, mlp_color, shift_factors), lens_net, cubemap_net, opt_distortion, no_distortion_mask, outside_rasterizer, flow_scale, control_point_sample_scale, flow_apply2_gt_or_img, apply2gt, cubemap, table1, opt_shift, shift_outside_factors)

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if hybrid:
                    specular_mlp.save_weights(args.model_path, iteration)


            # Densification
            if mcmc:
                if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
                    gaussians.relocate_gs(dead_mask=dead_mask)
                    gaussians.add_new_gs(cap_max=args.cap_max)
                    if use_wandb and iteration % 10 == 0:
                        scalars = {
                            f"gradient/2d_gradient": viewspace_point_tensor.grad.mean(),
                        }
                        wandb.log(scalars, step=iteration)
            else:
                if iteration < opt.densify_until_iter:
                    if not cubemap:
                        # Keep track of max radii in image-space for pruning
                        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                        viewspace_point_tensor_densify = render_pkg["viewspace_points_densify"]
                        gaussians.add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_densify, visibility_filter, abs_grad)

                        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                            if abs_grad:
                                gaussians.densify_and_prune(opt.abs_densify_grad_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                            else:
                                gaussians.densify_and_prune(opt.densify_grad_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                                #gaussians.densify_and_prune(densi_num, 0.005, scene.cameras_extent, size_threshold)

                        if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                            gaussians.reset_opacity()

                        if use_wandb and iteration % 1 == 0:
                            scalars = {
                                f"gradient/2d_gradient": viewspace_point_tensor.grad.mean(),
                            }
                            wandb.log(scalars, step=iteration)
                    elif cubemap:
                        for i in range(len(viewspace_point_tensor_list)):
                            gaussians.max_radii2D[visibility_filter_list[i]] = torch.max(gaussians.max_radii2D[visibility_filter_list[i]], radii_list[i][visibility_filter_list[i]])
                            gaussians.add_densification_stats(viewspace_point_tensor_list[i], None, visibility_filter_list[i], abs_grad)
                            if use_wandb and iteration % 10 == 0:
                                scalars = {
                                    f"gradient/2d_gradient_{i}": viewspace_point_tensor_list[i].grad.mean(),
                                }
                                wandb.log(scalars, step=iteration)

                        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                            if abs_grad:
                                gaussians.densify_and_prune(opt.abs_densify_grad_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                            else:
                                gaussians.densify_and_prune(opt.densify_grad_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                                #gaussians.densify_and_prune(densi_num, 0.005, scene.cameras_extent, size_threshold)

                        if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                            gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                optimizer_cubemap_net.step()
                optimizer_cubemap_net.zero_grad(set_to_none = True)

                if mcmc:
                    L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
                    actual_covariance = L @ L.transpose(1, 2)
                    def op_sigmoid(x, k=100, x0=0.995):
                        return 1 / (1 + torch.exp(-k * (x - x0)))
                    noise = torch.randn_like(gaussians._xyz) * (op_sigmoid(1- gaussians.get_opacity))*args.noise_lr*xyz_lr
                    noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                    gaussians._xyz.add_(noise)

                if start_vignetting < iteration:
                    vignetting_optimizer.step()
                    vignetting_optimizer.zero_grad(set_to_none = True)
                    if use_wandb and iteration % 10 == 0:
                        scalars = {
                            f"vignetting_model/a_k0": vignetting_model.a_k[0].cpu().item(),
                            f"vignetting_model/a_k1": vignetting_model.a_k[1].cpu().item(),
                            f"vignetting_model/a_k2": vignetting_model.a_k[2].cpu().item(),
                            f"vignetting_model/a_k3": vignetting_model.a_k[3].cpu().item(),
                            f"vignetting_model/beta_k0": vignetting_model.beta_k[0].cpu().item(),
                            f"vignetting_model/beta_k1": vignetting_model.beta_k[1].cpu().item(),
                            f"vignetting_model/beta_k2": vignetting_model.beta_k[2].cpu().item(),
                            f"vignetting_model/beta_k3": vignetting_model.beta_k[3].cpu().item(),
                            #f"vignetting_model/gamma": vignetting_model.gamma.cpu().item()
                        }
                        wandb.log(scalars, step=iteration)
                if opt_distortion:
                    optimizer_lens_net.step()
                    optimizer_lens_net.zero_grad(set_to_none=True)
                if opt_shift:
                    if iteration % 100 == 1:
                        wandb.log({"shift/0": shift_factors[0].item()}, step=iteration)
                        wandb.log({"shift/1": shift_factors[1].item()}, step=iteration)
                        wandb.log({"shift/2": shift_factors[2].item()}, step=iteration)
                        #wandb.log({"shift_outside/0": shift_outside_factors[0].item()}, step=iteration)
                        #wandb.log({"shift_outside/1": shift_outside_factors[1].item()}, step=iteration)
                        #wandb.log({"shift_outside/2": shift_outside_factors[2].item()}, step=iteration)
                        wandb.log({"shift_outside/0": shift_outside_factors[0, 0].item()}, step=iteration)
                        wandb.log({"shift_outside/1": shift_outside_factors[0, 1].item()}, step=iteration)
                        wandb.log({"shift_outside/2": shift_outside_factors[0, 2].item()}, step=iteration)
                    #shift_optimizer.step()
                    #shift_optimizer.zero_grad(set_to_none=True)
                    #shift_scheduler.step()
                    shift_outside_optimizer.step()
                    shift_outside_optimizer.zero_grad(set_to_none=True)
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
                torch.save(lens_net, os.path.join(scene.model_path, f'lens_net{iteration}.pth'))
                if opt_cam:
                    torch.save(scene.train_cameras, os.path.join(scene.model_path, 'opt_cams.pt'))
                    torch.save(scene.unnoisy_train_cameras, os.path.join(scene.model_path, 'gt_cams.pt'))
                    torch.save(scene.train_cameras, os.path.join(scene.model_path, f'cams_train{iteration}.pt'))

def training_report(use_wandb, iteration, Ll1, ssim_loss, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, lens_net, cubemap_net, opt_distortion, no_distortion_mask, outside_rasterizer, flow_scale, control_point_sample_scale, flow_apply2_gt_or_img, apply2gt, cubemap, table1, opt_shift, shift_outside_factors):
    if use_wandb and iteration % 10 == 0:
        scalars = {
            f"loss/l1_loss": Ll1,
            f"loss/ssim": ssim_loss,
            f"loss/overall_loss": loss,
        }
        wandb.log(scalars, step=iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : []},
                              {'name': 'train', 'cameras' : []})

        for camera in scene.getTestCameras()[:]:
            validation_configs[0]['cameras'].append(
                Camera(camera.colmap_id, camera.R, camera.T, camera.intrinsic_matrix_numpy, camera.FoVx, camera.FoVy, camera.focal_x, camera.focal_y, camera.original_image_pil, None, camera.fish_gt_image_pil, camera.image_name, camera.uid, depth=None, ori_path=camera.ori_path, outside_rasterizer=camera.outside_rasterizer, test_outside_rasterizer=camera.test_outside_rasterizer, orig_fov_w=camera.orig_fov_w, orig_fov_h=camera.orig_fov_h, original_image_resolution=camera.original_image_resolution, fish_gt_image_resolution=camera.fish_gt_image_resolution, flow_scale=camera.flow_scale, apply2gt=camera.apply2gt, render_resolution=camera.render_resolution, cubemap=cubemap, table1=table1)
            )

        for camera in scene.getTrainCameras()[:5]:
            validation_configs[1]['cameras'].append(
                Camera(camera.colmap_id, camera.R, camera.T, camera.intrinsic_matrix_numpy, camera.FoVx, camera.FoVy, camera.focal_x, camera.focal_y, camera.original_image_pil, None, camera.fish_gt_image_pil, camera.image_name, camera.uid, depth=None, ori_path=camera.ori_path, outside_rasterizer=camera.outside_rasterizer, test_outside_rasterizer=camera.test_outside_rasterizer, orig_fov_w=camera.orig_fov_w, orig_fov_h=camera.orig_fov_h, original_image_resolution=camera.original_image_resolution, fish_gt_image_resolution=camera.fish_gt_image_resolution, flow_scale=camera.flow_scale, apply2gt=camera.apply2gt, render_resolution=camera.render_resolution, cubemap=cubemap)
            )

        file_path = os.path.join(scene.model_path, 'evaluation_results.txt')
        with open(file_path, 'a') as f:
            for config in validation_configs:
                name = config['name']
                if config['cameras'] and len(config['cameras']) > 0:
                    l1_test = 0.0
                    psnr_test = 0.0
                    ssims = []
                    lpipss = []
                    os.makedirs(os.path.join(scene.model_path, 'training_val_{}').format(iteration), exist_ok=True)
                    os.makedirs(os.path.join(scene.model_path, 'training_val_{}/gt').format(iteration), exist_ok=True)
                    os.makedirs(os.path.join(scene.model_path, 'training_val_{}/renderred').format(iteration), exist_ok=True)
                    if cubemap:
                        os.makedirs(os.path.join(scene.model_path, 'training_val_{}/renderred/forward').format(iteration), exist_ok=True)
                        os.makedirs(os.path.join(scene.model_path, 'training_val_{}/renderred/up').format(iteration), exist_ok=True)
                        os.makedirs(os.path.join(scene.model_path, 'training_val_{}/renderred/down').format(iteration), exist_ok=True)
                        os.makedirs(os.path.join(scene.model_path, 'training_val_{}/renderred/left').format(iteration), exist_ok=True)
                        os.makedirs(os.path.join(scene.model_path, 'training_val_{}/renderred/right').format(iteration), exist_ok=True)
                    if table1:
                        os.makedirs(os.path.join(scene.model_path, 'training_val_{}/table1').format(iteration), exist_ok=True)
                    for idx, viewpoint in enumerate(config['cameras']):
                        #if outside_rasterizer:
                        #    viewpoint.reset_intrinsic(
                        #        viewpoint.FoVx,
                        #        viewpoint.FoVy,
                        #        viewpoint.focal_x,
                        #        viewpoint.focal_y,
                        #        int(2. * viewpoint.image_width),
                        #        int(2. * viewpoint.image_height)
                        #        #int(flow_scale[0] * viewpoint.image_width),
                        #        #int(flow_scale[1] * viewpoint.image_height)
                        #    )
                        if opt_shift:
                            c2w = viewpoint.get_c2w()
                            R = c2w[:3, :3]
                            cam_pos = viewpoint.get_camera_center()
                            look_at_direction_camera = torch.tensor([0, 0, -1.], device=cam_pos.device)
                            look_at_direction_world = R @ look_at_direction_camera
                            direction_vectors = scene.gaussians._xyz - cam_pos
                            look_at_direction = -look_at_direction_world
                            direction_vectors_normalized = direction_vectors / direction_vectors.norm(dim=1, keepdim=True)
                            look_at_direction_normalized = look_at_direction / look_at_direction.norm()
                            dot_products = torch.sum(direction_vectors_normalized * look_at_direction_normalized, dim=1)
                            angles = torch.acos(dot_products)
                            #shift = shift_outside_factors[0] * angles**3 + shift_outside_factors[1] * angles**5 + shift_outside_factors[2] * angles**7
                            shift = shift_outside_factors[:, 0] * angles**3 + shift_outside_factors[:, 1] * angles**5 + shift_outside_factors[:, 2] * angles**7
                            scene.gaussians._xyz = scene.gaussians._xyz + shift.unsqueeze(1) * look_at_direction_world.detach()
                        if table1 and name == 'test':
                            gt_image = viewpoint.original_image.cuda()
                            torchvision.utils.save_image(gt_image, os.path.join(scene.model_path, 'training_val_{}/table1/{}_gt'.format(iteration, viewpoint.image_name) + "_" + name + ".png"))
                            image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, global_alignment=scene.getGlobalAlignment())["render"], 0.0, 1.0)
                            torchvision.utils.save_image(image, os.path.join(scene.model_path, 'training_val_{}/table1/{}_rendering'.format(iteration, viewpoint.image_name) + "_" + name + ".png"))
                            l1_test += l1_loss(image, gt_image).mean().double()
                            psnr_test += psnr(image, gt_image).mean().double()
                            ssims.append(ssim(image, gt_image))
                            lpipss.append(lpips(image, gt_image))
                            continue
                        if not cubemap:
                            image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, global_alignment=scene.getGlobalAlignment())["render"], 0.0, 1.0)
                            torchvision.utils.save_image(image, os.path.join(scene.model_path, 'training_val_{}/renderred/{}'.format(iteration, viewpoint.image_name) + "_" + name + ".png"))
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
                                if iteration == 1:
                                    gt_image = viewpoint.fish_gt_image.cuda()
                                else:
                                    gt_image = viewpoint.fish_gt_image.cuda() * mask
                                torchvision.utils.save_image(gt_image.cpu(), os.path.join(scene.model_path, 'training_val_{}/gt/masked_{}'.format(iteration, viewpoint.image_name) + "_" + name + ".png"))
                                torchvision.utils.save_image(viewpoint.fish_gt_image, os.path.join(scene.model_path, 'training_val_{}/gt/{}'.format(iteration, viewpoint.image_name) + "_" + name + ".png"))
                                torchvision.utils.save_image(image, os.path.join(scene.model_path, 'training_val_{}/renderred/distorted_{}'.format(iteration, viewpoint.image_name) + "_" + name + ".png"))
                                if use_wandb and name == 'train':
                                    img_tensor = torch.cat((image.cpu(), gt_image.cpu()), dim=2)
                                    img_tensor = img_tensor.permute(1, 2, 0)
                                    img_numpy = img_tensor.cpu().numpy()
                                    wandb.log({f"images/gt_rendering_{viewpoint.image_name}": wandb.Image(img_numpy)})
                            elif apply2gt:
                                P_sensor, P_view_insidelens_direction = generate_control_pts(viewpoint, control_point_sample_scale, flow_scale)
                                gt_image, mask, flow_apply2_gt_or_img = apply_distortion(lens_net, P_view_insidelens_direction, P_sensor, viewpoint, image, apply2gt=apply2gt)
                                if iteration == 1:
                                    image = image
                                else:
                                    image = image * mask
                                torchvision.utils.save_image(gt_image, os.path.join(scene.model_path, 'training_val_{}/gt/{}_perspective'.format(iteration, viewpoint.image_name) + "_" + name + ".png"))
                                torchvision.utils.save_image(viewpoint.fish_gt_image, os.path.join(scene.model_path, 'training_val_{}/gt/{}_fish'.format(iteration, viewpoint.image_name) + "_" + name + ".png"))
                                torchvision.utils.save_image(viewpoint.original_image, os.path.join(scene.model_path, 'training_val_{}/gt/{}_undis'.format(iteration, viewpoint.image_name) + "_" + name + ".png"))
                                torchvision.utils.save_image(image*mask, os.path.join(scene.model_path, 'training_val_{}/renderred/{}_masked'.format(iteration, viewpoint.image_name) + "_" + name + ".png"))
                                if use_wandb and name == 'train':
                                    img_tensor = torch.cat(((image*mask).cpu(), gt_image.cpu()), dim=2)
                                    img_tensor = img_tensor.permute(1, 2, 0)
                                    img_numpy = img_tensor.cpu().numpy()
                                    wandb.log({f"images/gt_rendering_{viewpoint.image_name}": wandb.Image(img_numpy)})
                        elif cubemap:
                            mask_fov90 = torch.zeros((1, viewpoint.image_height, viewpoint.image_width), dtype=torch.float32).cuda()
                            #mask_fov90[:, viewpoint.image_height//2 - int(viewpoint.focal_y) - 1:viewpoint.image_height//2 + int(viewpoint.focal_y) + 1, viewpoint.image_width//2 - int(viewpoint.focal_x) - 1:viewpoint.image_width//2 + int(viewpoint.focal_x) + 1] = 1
                            mask_fov90[:, viewpoint.image_height//2 - int(viewpoint.focal_y) - 2:viewpoint.image_height//2 + int(viewpoint.focal_y) + 2, viewpoint.image_width//2 - int(viewpoint.focal_x) - 2:viewpoint.image_width//2 + int(viewpoint.focal_x) + 2] = 1
                            torchvision.utils.save_image(mask_fov90.float(), os.path.join(scene.model_path, 'mask1.png'))
                            img_list, img_perspective_list = render_cubemap(viewpoint, cubemap_net, mask_fov90, 0., 0., scene.gaussians, *renderArgs, iteration, False, scene, validation=True)
                            direction_name = ['forward', 'up', 'down', 'left', 'right']
                            for i in range(5):
                                torchvision.utils.save_image(img_perspective_list[i], os.path.join(scene.model_path, 'training_val_{}/renderred/{}/{}_perspective'.format(iteration, direction_name[i], viewpoint.image_name) + "_" + name + ".png"))
                                #torchvision.utils.save_image(img_list[i], os.path.join(scene.model_path, 'training_val_{}/renderred/{}'.format(iteration, i) + "_" + name + ".png"))
                            final_image = torch.zeros_like(img_list[0])
                            intensity_final = final_image.sum(dim=0, keepdim=True)  # Track the current intensities of the final image

                            for img in img_list:
                                intensity_img = img.sum(dim=0, keepdim=True)  # Calculate the intensity for the current image
                                mask = intensity_img > intensity_final  # Find pixels where the new image has a larger intensity
                                final_image = torch.where(mask, img, final_image)  # Update final image where intensity is larger
                                intensity_final = torch.where(mask, intensity_img, intensity_final)  # Update intensity tracker

                            torchvision.utils.save_image(final_image, os.path.join(scene.model_path, 'training_val_{}/renderred/{}_distorted_stitch'.format(iteration, viewpoint.image_name) + "_" + name + ".png"))
                            #mask_gt_image = generate_circular_mask(viewpoint.original_image.shape, min(viewpoint.original_image.shape[-2:])//2).cuda()
                            mask_gt_image = generate_circular_mask(viewpoint.original_image.shape, 450).cuda()
                            torchvision.utils.save_image(final_image*mask_gt_image, os.path.join(scene.model_path, 'training_val_{}/renderred/{}_distorted_stitch_masked'.format(iteration, viewpoint.image_name) + "_" + name + ".png"))
                            gt_image = viewpoint.original_image.cuda()
                            torchvision.utils.save_image(gt_image, os.path.join(scene.model_path, 'training_val_{}/gt/{}_perspective'.format(iteration, viewpoint.image_name) + "_" + name + ".png"))
                            if use_wandb and name == 'train':
                                img_tensor = torch.cat((final_image.clamp(0, 1).cpu(), gt_image.cpu()), dim=2)
                                img_tensor = img_tensor.permute(1, 2, 0)
                                img_numpy = img_tensor.cpu().numpy()
                                wandb.log({f"images/gt_rendering_{viewpoint.image_name}": wandb.Image(img_numpy)})
                        else:
                            gt_image = viewpoint.original_image.cuda()
                            torchvision.utils.save_image(gt_image, os.path.join(scene.model_path, 'training_val_{}/gt/{}_perspective'.format(iteration, viewpoint.image_name) + "_" + name + ".png"))
                            if use_wandb and name == 'train':
                                img_tensor = torch.cat((image.cpu(), gt_image.cpu()), dim=2)
                                img_tensor = img_tensor.permute(1, 2, 0)
                                img_numpy = img_tensor.cpu().numpy()
                                wandb.log({f"images/gt_rendering_{viewpoint.image_name}": wandb.Image(img_numpy)})

                        if not cubemap:
                            l1_test += l1_loss(image, gt_image).mean().double()
                            psnr_test += psnr(image, gt_image).mean().double()
                            ssims.append(ssim(image, gt_image))
                            lpipss.append(lpips(image, gt_image))
                        elif cubemap:
                            #mask_gt_image = generate_circular_mask(gt_image.shape, min(gt_image.shape[-2:])//2).cuda()
                            mask_gt_image = generate_circular_mask(gt_image.shape, 450).cuda()
                            l1_test += l1_loss(final_image*mask_gt_image, gt_image*mask_gt_image).mean().double()
                            psnr_test += psnr(final_image*mask_gt_image, gt_image*mask_gt_image).mean().double()
                            ssims.append(ssim(final_image*mask_gt_image, gt_image*mask_gt_image))
                            lpipss.append(lpips(final_image*mask_gt_image, gt_image*mask_gt_image))


                    psnr_test /= len(config['cameras'])
                    l1_test /= len(config['cameras'])
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                    print("\nSSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                    print("\nLPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                    f.write("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                    f.write("\nSSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                    f.write("\nLPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))


                    if use_wandb and name == 'test':
                        scalars = {
                            f"validation/l1_loss": l1_test,
                            f"validation/psnr": psnr_test,
                            f"validation/ssim": torch.tensor(ssims).mean().item(),
                            f"validation/lpips": torch.tensor(lpipss).mean().item(),
                        }
                        wandb.log(scalars, step=iteration)
        torch.cuda.empty_cache()

    if use_wandb and iteration % 10 == 0:
        wandb.log({"stats/gs_num": scene.gaussians.get_xyz.shape[0]}, step=iteration)
        #opacity_numpy = scene.gaussians.get_opacity.view(-1).cpu().numpy()
        #wandb.log({"stats/opacity_histogram": wandb.Histogram(opacity_numpy)})


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
