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

import torch
from torch import nn
from torch.nn import functional as F
from scene import Scene, SpecularModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.pose_utils import pose_spherical, render_wander_path
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import iResNet
import imageio
import numpy as np
import cv2
from pose_math import *
import visdom
from easydict import EasyDict
from copy import deepcopy

from utils.util import check_socket_open
from utils.util_vis import vis_cameras
from random import randint
from utils.loss_utils import l1_loss, ssim, kl_divergence, l2_loss
import matplotlib.pyplot as plt
from utils.camera import Lie
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from utils.camera import Lie
from scene.cameras import Camera


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

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, specular=None, hybrid=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "mask")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(mask_path, exist_ok=True)


    lens_net = torch.load(os.path.join(model_path, f'lens_net{iteration}.pth'))
    width = views[0].image_width
    height = views[0].image_height
    sample_width = int(width / 4)
    sample_height = int(height/ 4)
    K = views[0].get_K
    width = views[0].fish_gt_image.shape[2]
    height = views[0].fish_gt_image.shape[1]
    width = int(width * 2)
    height = int(height * 2)
    K[0, 2] = width / 2
    K[1, 2] = height / 2
    i, j = np.meshgrid(
        np.linspace(0, width, sample_width),
        np.linspace(0, height, sample_height),
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
    P_view_outsidelens_direction = lens_net.forward(P_view_insidelens_direction, sensor_to_frustum=False)
    camera_directions_w_lens = homogenize(P_view_outsidelens_direction)
    control_points = camera_directions_w_lens.reshape((P_sensor.shape[0], P_sensor.shape[1], 3))[:, :, :2]
    projection_matrix = views[0].projection_matrix
    flow = control_points @ projection_matrix[:2, :2]
    flow = nn.functional.interpolate(flow.permute(2, 0, 1).unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).squeeze(0)


    lie = Lie()
    pi = 3.1415926
    for idx, view in enumerate(views):
        if idx == 139: break
    angle = pi / 80

    cam_center_fixed = view.get_camera_center().clone()

    so3_noise = torch.tensor([-angle*0, angle*0, angle*0])
    t_noise = torch.tensor([angle*0, 0., 0.]).numpy()# gene
    so3 = lie.so3_to_SO3(so3_noise).cpu().detach().numpy()
    tmp_R = so3 @ view.R
    tmp_T = view.T + t_noise
    view.reset_extrinsic(tmp_R, tmp_T)
    #view.rotate_at_current_location(tmp_R, cam_center_fixed)


    look_around_scale = 5#cube
    view1 = Camera(view.colmap_id, view.R, view.T, view.intrinsic_matrix.cpu().numpy(), view.FoVx, view.FoVy, view.focal_x, view.focal_y, view.original_image.cpu(), None, view.image_name, view.uid, test_outside_rasterizer=True, orig_fov_w=view.original_image.shape[2], orig_fov_h=view.original_image.shape[1])
    so3_noise = torch.tensor([-angle*look_around_scale, -angle*0, angle*0])
    so3 = lie.so3_to_SO3(so3_noise).cpu().detach().numpy()
    tmp_R = so3 @ view1.R
    view1.reset_extrinsic(tmp_R, tmp_T)
    view2 = Camera(view.colmap_id, view.R, view.T, view.intrinsic_matrix.cpu().numpy(), view.FoVx, view.FoVy, view.focal_x, view.focal_y, view.original_image.cpu(), None, view.image_name, view.uid, test_outside_rasterizer=True, orig_fov_w=view.original_image.shape[2], orig_fov_h=view.original_image.shape[1])
    so3_noise = torch.tensor([angle*look_around_scale, -angle*0, angle*0])
    so3 = lie.so3_to_SO3(so3_noise).cpu().detach().numpy()
    tmp_R = so3 @ view2.R
    view2.reset_extrinsic(tmp_R, tmp_T)
    view3 = Camera(view.colmap_id, view.R, view.T, view.intrinsic_matrix.cpu().numpy(), view.FoVx, view.FoVy, view.focal_x, view.focal_y, view.original_image.cpu(), None, view.image_name, view.uid, test_outside_rasterizer=True, orig_fov_w=view.original_image.shape[2], orig_fov_h=view.original_image.shape[1])
    so3_noise = torch.tensor([angle*0, -angle*look_around_scale, angle*0])
    so3 = lie.so3_to_SO3(so3_noise).cpu().detach().numpy()
    tmp_R = so3 @ view3.R
    view3.reset_extrinsic(tmp_R, tmp_T)
    view4 = Camera(view.colmap_id, view.R, view.T, view.intrinsic_matrix.cpu().numpy(), view.FoVx, view.FoVy, view.focal_x, view.focal_y, view.original_image.cpu(), None, view.image_name, view.uid, test_outside_rasterizer=True, orig_fov_w=view.original_image.shape[2], orig_fov_h=view.original_image.shape[1])
    so3_noise = torch.tensor([angle*0, angle*look_around_scale, angle*0])
    so3 = lie.so3_to_SO3(so3_noise).cpu().detach().numpy()
    tmp_R = so3 @ view4.R
    view4.reset_extrinsic(tmp_R, tmp_T)
    view_boundary = np.stack((view.get_c2w().cpu().detach().numpy(), view1.get_c2w().cpu().detach().numpy(), view2.get_c2w().cpu().detach().numpy(), view3.get_c2w().cpu().detach().numpy(), view4.get_c2w().cpu().detach().numpy()), axis=2)
    comps = [False, False, False, True, False]
    render_poses = generate_render_path(view_boundary, focal=5, comps=comps, N=30)
    render_poses = render_poses[:, :, :4]
    camera_list = []
    view.get_w2c
    view.get_c2w()[:3, :3].t()
    -view.get_c2w()[:3, :3].t() @ view.get_c2w()[:3, 3].t()
    for pose in render_poses:
        so3_noise = torch.tensor([angle*5, -angle*15, angle*0]) # cube
        t_noise = torch.tensor([-angle*100, angle*0, angle*0]).numpy()
        so3 = lie.so3_to_SO3(so3_noise).cpu().detach().numpy()
        view_tmp = Camera(view.colmap_id, so3@pose[:3, :3].transpose(), -pose[:3, :3].transpose()@pose[:, 3] + t_noise, view.intrinsic_matrix.cpu().numpy(), view.FoVx, view.FoVy, view.focal_x, view.focal_y, view.original_image.cpu(), None, view.image_name, view.uid, test_outside_rasterizer=True, orig_fov_w=view.original_image.shape[2], orig_fov_h=view.original_image.shape[1])
        camera_list.append(view_tmp)

    #for view in [view, view1, view2, view3, view4]:
    #    camera_list.append(view)

    #for idx, view in enumerate(views):
    #    camera_list.append(view)

    for idx, view in enumerate(tqdm(camera_list, desc="Rendering progress")):
        so3_noise = torch.tensor([angle*0, angle*0, angle*0])
        t_noise = torch.tensor([angle*0, angle*0, angle*0]).numpy()
        so3 = lie.so3_to_SO3(so3_noise).cpu().detach().numpy()
        tmp_R = so3 @ view.R
        tmp_T = view.T + t_noise

        gt = view.original_image[0:3, :, :]
        mask = gt[:1, :, :].bool()
        mlp_color = 0
        global_alignment = [torch.tensor([[1., 0, 0], [0, 1., 0], [0, 0, 1.]], device='cuda'), torch.tensor([1.], device='cuda')]

        results = render(view, gaussians, pipeline, background, mlp_color, global_alignment=global_alignment)
        rendering, depth_tensor, weight_mask = results["render"], results["depth"], results["weights"]
        output = F.grid_sample(
            rendering.unsqueeze(0),
            flow.unsqueeze(0),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        output = center_crop(output, int(height/2), int(width/2)).squeeze(0)

        torchvision.utils.save_image(output.squeeze(0), os.path.join(render_path, f'perspective_frame{idx}' + ".png"))
        torchvision.utils.save_image(rendering, os.path.join(render_path, f'frame{idx}_fish' + ".png"))

        mask = F.interpolate(mask.unsqueeze(0).float(), size=depth_tensor.shape[1:], mode='bilinear', align_corners=False).squeeze(0).bool()
        depth_tensor_normalized = (depth_tensor - depth_tensor[mask].min()) / (depth_tensor[mask].max() - depth_tensor[mask].min())
        depth_array = depth_tensor_normalized.squeeze().cpu().detach().numpy()
        torchvision.utils.save_image(depth_tensor_normalized, os.path.join(depth_path, f'grey_frame{idx}_perspective' + ".png"))
        depth_colored = plt.get_cmap('viridis')(depth_array)[:, :, :3]  # Drop the alpha channel
        depth_colored_tensor = torch.from_numpy(depth_colored).permute(2, 0, 1).float()  # Rearrange dimensions to CxHxW
        torchvision.utils.save_image(depth_colored_tensor, os.path.join(depth_path, f'colored_frame{idx}_perspective' + ".png"))
        output = F.grid_sample(
            depth_tensor_normalized.unsqueeze(0),
            flow.unsqueeze(0),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        output = center_crop(output, int(height/2), int(width/2)).squeeze(0)
        torchvision.utils.save_image(output, os.path.join(depth_path, f'grey_frame{idx}_fisheye' + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, mode: str, hybrid: bool, opt_test_cam: bool, opt_intrinsic: bool, opt_extrinsic: bool):
    gaussians = GaussianModel(dataset.sh_degree, dataset.asg_degree)
    lens_net = iResNet()
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, outside_rasterizer=True)

    scene.train_cameras = torch.load(os.path.join(scene.model_path, f'cams_train{iteration}.pt'))
    specular = None
    if hybrid:
        specular = SpecularModel()
        specular.load_weights(dataset.model_path)

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    render_func = render_set

    if not skip_train:
         render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, specular, hybrid)

    if not skip_test:
         render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, specular, hybrid)

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
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--hybrid", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'view', 'all', 'pose', 'original'])
    parser.add_argument("--opt_test_cam", action="store_true", default=False)
    # if opt camera intrinsic
    parser.add_argument("--opt_intrinsic", action="store_true", default=False)
    parser.add_argument("--opt_extrinsic", action="store_true", default=False)
    # wandb setting
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project_name", type=str, default = None)
    parser.add_argument("--wandb_group_name", type=str, default = None)
    parser.add_argument("--wandb_mode", type=str, default = "online")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    # Initialize wandb
    if args.wandb:
        wandb.login()
        wandb_run = init_wandb(args,
                               project=args.wandb_project_name,
                               mode=args.wandb_mode,
                               resume=args.resume,
                               use_group=True,
                               set_group=args.wandb_group_name
                               )

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.mode, args.hybrid, args.opt_test_cam, args.opt_intrinsic, args.opt_extrinsic)
