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
from utils.general_utils import safe_state, get_linear_noise_func, linear_to_srgb
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
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
from utils.util_distortion import homogenize, dehomogenize, colorize, plot_points, center_crop, init_from_colmap

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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, use_wandb=False, random_init=False, hybrid=False, opt_cam=False, opt_distortion=False, opt_intrinsic=False, r_t_noise=[0., 0.], r_t_lr=[0.001, 0.001], global_alignment_lr=0.001, extra_loss=False, start_opt_lens=1, extend_scale=2., control_point_sample_scale=8., outside_rasterizer=False, abs_grad=False, densi_num=0.0002, if_circular_mask=False, flow_scale=1.):
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
    #lens_net.apply(zero_weights)
    #for param in lens_net.parameters():
    #    print(param)

    scene = Scene(dataset, gaussians, random_init=random_init, r_t_noise=r_t_noise, r_t_lr=r_t_lr, global_alignment_lr=global_alignment_lr, outside_rasterizer=outside_rasterizer, flow_scale=flow_scale)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        lens_net = torch.load(os.path.join(scene.model_path, f'lens_net{first_iter}.pth'))
        gaussians.restore(model_params, opt)
        scene.train_cameras = torch.load(os.path.join(scene.model_path, 'opt_cams.pt'))
        scene.unnoisy_train_cameras = torch.load(os.path.join(scene.model_path, 'gt_cams.pt'))
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    viewpoint_stack_constant = scene.getTrainCameras()
    camera_id = [camera.uid for camera in viewpoint_stack_constant]
    extrinsic_list = [camera.get_w2c for camera in viewpoint_stack_constant]
    camera_matching_points = {}
    projection_loss_count = 0
    if args.vis_pose:
        opt_vis = EasyDict({'group': 'opt_pose', 'name': 'opt_pose', 'visdom': {'server': 'localhost', 'port': 8600, 'cam_depth': 0.5}})
        if opt_vis.visdom and args.vis_pose:
            is_open = check_socket_open(opt_vis.visdom.server,opt_vis.visdom.port)
            retry = None
            vis = visdom.Visdom(server=opt_vis.visdom.server,port=opt_vis.visdom.port,env=opt_vis.group)
            pose_GT, pose_aligned = scene.loadAlignCameras(if_vis_train=True, path=scene.model_path)
            vis_cameras(opt_vis, vis, step=0, poses=[pose_aligned, pose_GT])
            os.makedirs(os.path.join(args.model_path, 'plot'), exist_ok=True)

    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # colmap init
    if outside_rasterizer:
        P_sensor, P_view_insidelens_direction = init_from_colmap(scene, dataset, optimizer_lens_net, lens_net, scheduler_lens_net, control_point_sample_scale, flow_scale)

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
            P_view_outsidelens_direction = lens_net.forward(P_view_insidelens_direction, sensor_to_frustum=True)
            camera_directions_w_lens = homogenize(P_view_outsidelens_direction)
            control_points = camera_directions_w_lens.reshape((P_sensor.shape[0], P_sensor.shape[1], 3))[:, :, :2]

            projection_matrix = viewpoint_cam.save4flow
            flow = control_points @ projection_matrix[:2, :2]
            flow = nn.functional.interpolate(flow.permute(2, 0, 1).unsqueeze(0), size=(viewpoint_cam.image_height, viewpoint_cam.image_width), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).squeeze(0)
            gt_image = F.grid_sample(
                viewpoint_cam.fish_gt_image.cuda().unsqueeze(0),
                flow.unsqueeze(0),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )

            # apply distortion on gt version debugging
            #torchvision.utils.save_image(viewpoint_cam.fish_gt_image, os.path.join(scene.model_path, f"gt_fish.png"))
            #torchvision.utils.save_image(viewpoint_cam.original_image, os.path.join(scene.model_path, f"gt_perspective.png"))
            #torchvision.utils.save_image(image, os.path.join(scene.model_path, f"rendered.png"))
            #torchvision.utils.save_image(gt_image, os.path.join(scene.model_path, f"gt_fish2perspective.png"))
            #import pdb;pdb.set_trace()

            mask = (~((image[0]==0) & (image[1]==0)).unsqueeze(0)).float()

            #if iteration % 1000 == 1:
            #    P_view_outsidelens_direction = lens_net.forward(P_view_insidelens_direction, sensor_to_frustum=True)
            #    camera_directions_w_lens = homogenize(P_view_outsidelens_direction)
            #    control_points = camera_directions_w_lens.reshape((P_sensor.shape[0], P_sensor.shape[1], 3))[:, :, :2]
            #    projection_matrix = viewpoint_cam.save4flow
            #    vis_flow = control_points @ projection_matrix[:2, :2]
            #    plot_points(vis_flow, os.path.join(scene.model_path, f"flow_{iteration}.png"))

        # Loss
        if outside_rasterizer:
            gt_image = gt_image.squeeze(0)
            Ll1 = l1_loss(image, gt_image*mask)
            ssim_loss = ssim(image, gt_image*mask)
        else:
            gt_image = viewpoint_cam.original_image.cuda()
            mask = (~((gt_image[0]==0) & (gt_image[1]==0)).unsqueeze(0)).float()
            Ll1 = l1_loss(image*mask, gt_image)
            ssim_loss = ssim(image*mask, gt_image)
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
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, mlp_color), lens_net, opt_distortion, P_view_insidelens_direction, P_sensor, outside_rasterizer, flow_scale)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if hybrid:
                    specular_mlp.save_weights(args.model_path, iteration)


            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

                viewspace_point_tensor_densify = render_pkg["viewspace_points_densify"]
                gaussians.add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_densify, visibility_filter, abs_grad)
                if iteration % 10 == 0:
                    scalars = {
                        f"gradient/2d_gradient": viewspace_point_tensor.grad.mean(),
                    }
                    if use_wandb:
                        wandb.log(scalars, step=iteration)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    if abs_grad:
                        gaussians.densify_and_prune(opt.abs_densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    else:
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                        #gaussians.densify_and_prune(densi_num, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if opt_distortion:
                    optimizer_lens_net.step()
                    optimizer_lens_net.zero_grad(set_to_none=True)
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
                torch.save(scene.train_cameras, os.path.join(scene.model_path, 'opt_cams.pt'))
                torch.save(scene.unnoisy_train_cameras, os.path.join(scene.model_path, 'gt_cams.pt'))
                torch.save(lens_net, os.path.join(scene.model_path, f'lens_net{iteration}.pth'))
                torch.save(scene.train_cameras, os.path.join(scene.model_path, f'cams_train{iteration}.pt'))

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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, lens_net, opt_distortion, P_view_insidelens_direction, P_sensor, outside_rasterizer, flow_scale):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # we wish to change the fov during the validation. Remember to copy cameras
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras().copy()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras().copy()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            name = config['name']
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssims = []
                lpipss = []
                makedirs(os.path.join(scene.model_path, 'training_val'), exist_ok=True)
                makedirs(os.path.join(scene.model_path, 'training_val/gt'), exist_ok=True)
                makedirs(os.path.join(scene.model_path, 'training_val/distorted'), exist_ok=True)
                makedirs(os.path.join(scene.model_path, 'training_val/perspective'), exist_ok=True)
                if outside_rasterizer:
                    width = config['cameras'][0].image_width
                    height = config['cameras'][0].image_height
                    sample_width = int(width / 4)
                    sample_height = int(height/ 4)
                    K = config['cameras'][0].get_K
                    width = config['cameras'][0].fish_gt_image.shape[2]
                    height = config['cameras'][0].fish_gt_image.shape[1]
                    width = int(width * flow_scale)
                    height = int(height * flow_scale)
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
                    flow = camera_directions_w_lens.reshape((P_sensor.shape[0], P_sensor.shape[1], 3))[:, :, :2]
                    projection_matrix = config['cameras'][0].save4flow
                    flow = flow @ projection_matrix[:2, :2]
                    # the same
                    #flow = nn.functional.interpolate(flow.permute(2, 0, 1).unsqueeze(0), size=(int(config['cameras'][0].fish_gt_image.shape[1]*flow_scale), int(config['cameras'][0].fish_gt_image.shape[2]*flow_scale)), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).squeeze(0)
                    flow = nn.functional.interpolate(flow.permute(2, 0, 1).unsqueeze(0), size=(config['cameras'][0].image_height, config['cameras'][0].image_width), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).squeeze(0)
                for idx, viewpoint in enumerate(config['cameras']):
                    gaussians_xyz = scene.gaussians.get_xyz.detach()
                    gaussians_xyz_homo = torch.cat((gaussians_xyz, torch.ones(gaussians_xyz.size(0), 1).cuda()), dim=1)
                    # glm use the transpose of w2c
                    w2c = viewpoint.get_world_view_transform().t().detach()
                    p_w2c = (w2c @ gaussians_xyz_homo.T).T.cuda().detach()
                    undistorted_p_w2c_homo = p_w2c
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, global_alignment=scene.getGlobalAlignment())["render"], 0.0, 1.0)

                    torchvision.utils.save_image(image, os.path.join(scene.model_path, 'training_val/perspective/{0:05d}'.format(idx) + "_" + name + "_perspective.png"))
                    if outside_rasterizer:
                        image = F.grid_sample(
                            image.unsqueeze(0),
                            flow.unsqueeze(0),
                            mode="bilinear",
                            padding_mode="zeros",
                            align_corners=True,
                        )
                        image2fish = center_crop(image, viewpoint.fish_gt_image.shape[1], viewpoint.fish_gt_image.shape[2]).squeeze(0)
                        torchvision.utils.save_image(image2fish, os.path.join(scene.model_path, 'training_val/distorted/{0:05d}'.format(idx) + "_" + name + "_fish.png"))
                    torchvision.utils.save_image(viewpoint.fish_gt_image, os.path.join(scene.model_path, 'training_val/gt/gt_{0:05d}_fish'.format(idx) + "_" + name + ".png"))
                    torchvision.utils.save_image(viewpoint.original_image, os.path.join(scene.model_path, 'training_val/gt/gt_{0:05d}_perspective'.format(idx) + "_" + name + ".png"))
                    import torchvision.transforms as transforms; from PIL import Image; image_tensor = transforms.ToTensor()(Image.open("dataset/gene_table/frame_00433.png").convert("RGB"))
                    #torchvision.utils.save_image(image_tensor, os.path.join(scene.model_path, 'training_val/gt/gt_{0:05d}_perspective'.format(idx) + "_" + name + ".png"))
                    #image_tensor = F.grid_sample(
                    #    image_tensor.cuda().unsqueeze(0),
                    #    flow.unsqueeze(0),
                    #    mode="bilinear",
                    #    padding_mode="zeros",
                    #    align_corners=True,
                    #)
                    #image_tensor = center_crop(image_tensor, viewpoint.fish_gt_image.shape[1], viewpoint.fish_gt_image.shape[2]).squeeze(0)
                    #torchvision.utils.save_image(image_tensor, os.path.join(scene.model_path, 'training_val/gt/gt_{0:05d}_distorted'.format(idx) + "_" + name + ".png"))

                    if outside_rasterizer:
                        gt_image = F.grid_sample(
                            image_tensor.cuda().unsqueeze(0),
                            flow.unsqueeze(0),
                            mode="bilinear",
                            padding_mode="zeros",
                            align_corners=True,
                        )
                    else:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssims.append(ssim(image, gt_image))
                    lpipss.append(lpips(image, gt_image))
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                print("\nSSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("\nLPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
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
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
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
    parser.add_argument("--abs_grad", action="store_true", default=False)
    parser.add_argument('--densi_num', type=float, default=0.0002)
    parser.add_argument("--if_circular_mask", action="store_true", default=False)
    parser.add_argument('--flow_scale', type=float, default=1.)

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
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, use_wandb=args.wandb, random_init=args.random_init_pc, hybrid=args.hybrid, opt_cam=args.opt_cam, opt_distortion=args.opt_distortion, opt_intrinsic=args.opt_intrinsic, r_t_lr=args.r_t_lr, r_t_noise=args.r_t_noise, global_alignment_lr=args.global_alignment_lr, extra_loss=args.extra_loss,
             start_opt_lens=args.start_opt_lens, extend_scale=args.extend_scale, control_point_sample_scale=args.control_point_sample_scale, outside_rasterizer=args.outside_rasterizer, abs_grad=args.abs_grad, densi_num=args.densi_num, if_circular_mask=args.if_circular_mask, flow_scale=args.flow_scale)

    # All done
    print("\nTraining complete.")
