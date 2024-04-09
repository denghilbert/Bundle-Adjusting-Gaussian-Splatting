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
import visdom
from easydict import EasyDict

from utils.util import check_socket_open
from utils.util_vis import vis_cameras
from random import randint
from utils.loss_utils import l1_loss, ssim, kl_divergence, l2_loss
import matplotlib.pyplot as plt
from utils.camera import Lie
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

opt_vis = EasyDict({'group': 'exp_synthetic', 'name': 'l2g_lego', 'model': 'l2g_nerf', 'yaml': 'l2g_nerf_blender', 'seed': 0, 'gpu': 0, 'cpu': False, 'load': None, 'arch': {'layers_feat': [None, 256, 256, 256, 256, 256, 256, 256, 256], 'layers_rgb': [None, 128, 3], 'skip': [4], 'posenc': {'L_3D': 10, 'L_view': 4}, 'density_activ': 'softplus', 'tf_init': True, 'layers_warp': [None, 256, 256, 256, 256, 256, 256, 6], 'skip_warp': [4], 'embedding_dim': 128}, 'data': {'root': '/the/data/path/of/nerf_synthetic/', 'dataset': 'blender', 'image_size': [400, 400], 'num_workers': 4, 'preload': True, 'augment': {}, 'center_crop': None, 'val_on_test': False, 'train_sub': None, 'val_sub': 4, 'scene': 'lego', 'bgcolor': 1}, 'loss_weight': {'render': 0, 'render_fine': None, 'global_alignment': 2}, 'optim': {'lr': 0.0005, 'lr_end': 0.0001, 'algo': 'Adam', 'sched': {'type': 'ExponentialLR', 'gamma': None}, 'lr_pose': 0.001, 'lr_pose_end': 1e-08, 'sched_pose': {'type': 'ExponentialLR', 'gamma': None}, 'warmup_pose': None, 'test_photo': True, 'test_iter': 100}, 'batch_size': None, 'max_epoch': None, 'resume': False, 'output_root': 'output', 'tb': {'num_images': [4, 8]}, 'visdom': {'server': 'localhost', 'port': 8600, 'cam_depth': 0.5}, 'freq': {'scalar': 200, 'vis': 1000, 'val': 2000, 'ckpt': 5000}, 'nerf': {'view_dep': True, 'depth': {'param': 'metric', 'range': [2, 6]}, 'sample_intvs': 128, 'sample_stratified': True, 'fine_sampling': False, 'sample_intvs_fine': None, 'rand_rays': 1024, 'density_noise_reg': None, 'setbg_opaque': False}, 'camera': {'model': 'perspective', 'ndc': False, 'noise': True, 'noise_r': 0.07, 'noise_t': 0.5}, 'max_iter': 200000, 'trimesh': {'res': 128, 'range': [-1.2, 1.2], 'thres': 25.0, 'chunk_size': 16384}, 'barf_c2f': [0.1, 0.5], 'error_map_size': None, 'output_path': 'output/exp_synthetic/l2g_lego', 'device': 'cuda:0', 'H': 400, 'W': 400})
is_open = check_socket_open(opt_vis.visdom.server,opt_vis.visdom.port)
vis = visdom.Visdom(server=opt_vis.visdom.server,port=opt_vis.visdom.port,env=opt_vis.group)



def render_set(model_path, name, iteration, views, gaussians, pipeline, background, specular=None, hybrid=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "mask")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(mask_path, exist_ok=True)


    if os.path.exists(os.path.join(model_path, 'distortion_params.pt')):
        distortion_params = torch.load(os.path.join(model_path, 'distortion_params.pt'))
        u_distortion = torch.load(os.path.join(model_path, f'u_distortion{iteration}.pt'))
        v_distortion = torch.load(os.path.join(model_path, f'v_distortion{iteration}.pt'))
        u_radial = torch.load(os.path.join(model_path, f'u_radial{iteration}.pt'))
        v_radial = torch.load(os.path.join(model_path, f'v_radial{iteration}.pt'))
        v_radial = torch.load(os.path.join(model_path, f'v_radial{iteration}.pt'))
    else:
        distortion_params = torch.nn.Parameter(torch.zeros(8).cuda())
        u_distortion = nn.Parameter(torch.zeros(399, 399).cuda().requires_grad_(True))
        v_distortion = nn.Parameter(torch.zeros(399, 399).cuda().requires_grad_(True))
        u_radial = nn.Parameter(torch.ones(399, 399).cuda().requires_grad_(True))
        v_radial = nn.Parameter(torch.ones(399, 399).cuda().requires_grad_(True))
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt = view.original_image[0:3, :, :]
        mask = gt[:1, :, :].bool()

        if hybrid:
            dir_pp = (gaussians.get_xyz - view.camera_center.repeat(gaussians.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            mlp_color = specular.step(gaussians.get_asg_features, dir_pp_normalized)
            results = render(view, gaussians, pipeline, background, mlp_color)
            rendering = results["render"]
            #depth = results["depth"]
            #depth = depth / (depth.max() + 1e-5)
        else:
            mlp_color = 0
            global_alignment = [torch.tensor([[1., 0, 0], [0, 1., 0], [0, 0, 1.]], device='cuda'), torch.tensor([1.], device='cuda')]

            # render current view
            gaussians_xyz = gaussians.get_xyz.detach()
            gaussians_xyz_homo = torch.cat((gaussians_xyz, torch.ones(gaussians_xyz.size(0), 1).cuda()), dim=1)
            #gaussians_xyz_homo.retain_grad()
            # glm use the transpose of w2c
            w2c = view.get_world_view_transform().t().detach()
            p_w2c = (w2c @ gaussians_xyz_homo.T).T.cuda().detach()
            intrinsic = view.get_intrinsic().t().detach()
            proj_mat = view.get_full_proj_transform().t().detach()
            p_proj = (proj_mat @ gaussians_xyz_homo.T).T.cuda().detach()
            p_2d = p_proj[:, :2] / p_proj[:, -1:]
            #if opt_distortion and iteration > 3000:
            if False:
                undistorted_p_w2c = lens_net.forward(p_w2c[:, :3])
                undistorted_p_w2c_homo = torch.cat((undistorted_p_w2c, torch.ones(undistorted_p_w2c.size(0), 1).cuda()), dim=1)
            else:
                undistorted_p_w2c_homo = p_w2c

            distortion_params = torch.nn.Parameter(torch.zeros(8).cuda())
            results = render(view, gaussians, pipeline, background, mlp_color, undistorted_p_w2c_homo, distortion_params, u_distortion, v_distortion, u_radial, v_radial, global_alignment=global_alignment)
            rendering, depth_tensor, weight_mask = results["render"], results["depth"], results["weights"]
            depth_tensor_normalized = (depth_tensor - depth_tensor[mask].min()) / (depth_tensor[mask].max() - depth_tensor[mask].min())
            depth_tensor_grey = depth_tensor_normalized.repeat(3, 1, 1)
            depth_array = depth_tensor_normalized.squeeze().cpu().detach().numpy()
            depth_colored = plt.get_cmap('viridis')(depth_array)[:, :, :3]  # Drop the alpha channel
            depth_colored_tensor = torch.from_numpy(depth_colored).permute(2, 0, 1).float()  # Rearrange dimensions to CxHxW



        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth_colored_tensor, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth_colored_tensor, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth_tensor_grey, os.path.join(depth_path, 'grey_{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(weight_mask, os.path.join(mask_path, 'mask_{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, mode: str, hybrid: bool, opt_test_cam: bool, opt_intrinsic: bool):
    gaussians = GaussianModel(dataset.sh_degree, dataset.asg_degree)
    lens_net = iResNet()
    scene = Scene(dataset, gaussians, lens_net, load_iteration=iteration, shuffle=False)
    scene.train_cameras = torch.load(os.path.join(scene.model_path, f'cams_train{iteration}.pt'))
    specular = None
    if hybrid:
        specular = SpecularModel()
        specular.load_weights(dataset.model_path)

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    render_func = render_set

    #scene.loadAlignCameras(if_vis_test=True, path=scene.model_path)

    viewpoint_stack = scene.getTestCameras().copy()

    if opt_test_cam:
        if os.path.exists(os.path.join(scene.model_path, 'opt_test_cam.pt')):
            scene.test_cameras = torch.load(os.path.join(scene.model_path, 'opt_test_cam.pt'))
        progress_bar = tqdm(range(0, 50000), desc="Training progress")
        for iteration in range(50000):
            if iteration % 1000 == 0:
                pose_gt, pose_aligned = scene.visTestCameras()
                vis_cameras(opt_vis, vis, iteration, poses=[pose_aligned, pose_gt])
            if not viewpoint_stack:
                viewpoint_stack = scene.getTestCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            mlp_color = 0
            render_pkg = render(viewpoint_cam, gaussians, pipeline, background, mlp_color, iteration=iteration, hybrid=hybrid, global_alignment=scene.getGlobalAlignment())
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            ssim_loss = ssim(image, gt_image)
            loss = 0.8 * Ll1 + 0.2 * (1.0 - ssim_loss)
            loss.backward(retain_graph=True)


            with torch.no_grad():
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}"})
                    progress_bar.update(10)
                if iteration == 50000:
                    progress_bar.close()

                scene.optimizer_rotation_test.step()
                scene.optimizer_translation_test.step()
                scene.optimizer_rotation_test.zero_grad(set_to_none=True)
                scene.optimizer_translation_test.zero_grad(set_to_none=True)
                scene.scheduler_rotation_test.step()
                scene.scheduler_translation_test.step()

                if opt_intrinsic:
                    scene.optimizer_fovx.step()
                    scene.optimizer_fovy.step()
                    scene.optimizer_fovx.zero_grad(set_to_none=True)
                    scene.optimizer_fovy.zero_grad(set_to_none=True)
                    scene.scheduler_fovx.step()
                    scene.scheduler_fovy.step()

        torch.save(scene.test_cameras, os.path.join(scene.model_path, 'opt_test_cam.pt'))

    torch.save(scene.test_cameras, os.path.join(scene.model_path, 'opt_test_cam.pt'))

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

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.mode, args.hybrid, args.opt_test_cam, args.opt_intrinsic)
