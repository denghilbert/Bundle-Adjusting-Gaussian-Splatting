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
from utils.graphics_utils import fov2focal, focal2fov, getProjectionMatrix
from utils.visualization import wandb_image
from utils.util_vis import vis_cameras
from utils.util import check_socket_open, prepare_output_and_logger, init_wandb
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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, use_wandb=False, random_init=False, hybrid=False, opt_cam=False, opt_distortion=False, opt_intrinsic=False, r_t_noise=[0., 0.], r_t_lr=[0.001, 0.001], global_alignment_lr=0.001, extra_loss=False, start_opt_lens=1, extend_scale=2., control_point_sample_scale=8., outside_rasterizer=False, abs_grad=False, densi_num=0.0002, if_circular_mask=False, flow_scale=[1., 1.], apply2gt=False, iresnet_lr=1e-7, opacity_threshold=0.005):
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
    #lens_net.apply(zero_weights)
    #for param in lens_net.parameters():
    #    print(param)

    scene = Scene(dataset, gaussians, random_init=random_init, r_t_noise=r_t_noise, r_t_lr=r_t_lr, global_alignment_lr=global_alignment_lr, outside_rasterizer=outside_rasterizer, flow_scale=flow_scale)

    #pose_GT, pose_aligned = scene.loadAlignCameras(if_vis_train=True, path=scene.model_path)
    #torch.save(pose_GT, os.path.join(scene.model_path, 'gt.pt'))
    #torch.save(pose_aligned, os.path.join(scene.model_path, 'align.pt'))
    #import pdb;pdb.set_trace()
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        print(f'loading iteraton {first_iter}')
        lens_net = torch.load(os.path.join(scene.model_path, f'lens_net{first_iter}.pth'))
        gaussians.restore(model_params, opt)
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
    if outside_rasterizer:
        init_from_colmap(scene, dataset, optimizer_lens_net, lens_net, scheduler_lens_net, resume_training=checkpoint, iresnet_lr=iresnet_lr)

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

        flow_apply2_gt_or_img = None
        if outside_rasterizer:
            P_sensor, P_view_insidelens_direction = generate_control_pts(viewpoint_cam, control_point_sample_scale, flow_scale)
            if not apply2gt:
                #torchvision.utils.save_image(image, os.path.join(scene.model_path, f"render_perspective.png"))
                image, mask, flow_apply2_gt_or_img = apply_distortion(lens_net, P_view_insidelens_direction, P_sensor, viewpoint_cam, image, apply2gt=apply2gt, flow_scale=flow_scale)
                #torchvision.utils.save_image(image, os.path.join(scene.model_path, f"render_distorted.png"))
                #torchvision.utils.save_image(viewpoint_cam.fish_gt_image, os.path.join(scene.model_path, f"gt_fish.png"))
                #torchvision.utils.save_image(viewpoint_cam.original_image, os.path.join(scene.model_path, f"gt_pers.png"))
                #torchvision.utils.save_image(mask, os.path.join(scene.model_path, f"mask.png"))
                #import pdb;pdb.set_trace()
                if if_circular_mask:
                    mask = mask * circular_mask

            if apply2gt:
                gt_image, mask, flow_apply2_gt_or_img = apply_distortion(lens_net, P_view_insidelens_direction, P_sensor, viewpoint_cam, image, apply2gt=apply2gt)
                #plot_points(control_points, os.path.join(scene.model_path, f"flow.png"))
                #torchvision.utils.save_image(viewpoint_cam.fish_gt_image, os.path.join(scene.model_path, f"gt_fish.png"))
                #torchvision.utils.save_image(viewpoint_cam.original_image, os.path.join(scene.model_path, f"gt_perspective.png"))
                #torchvision.utils.save_image(image, os.path.join(scene.model_path, f"render_{iteration}.png"))
                #torchvision.utils.save_image(gt_image, os.path.join(scene.model_path, f"gt_fish2perspective_{iteration}.png"))
                #import pdb;pdb.set_trace()

            #if iteration % 1000 == 1:
            #    P_view_outsidelens_direction = lens_net.forward(P_view_insidelens_direction, sensor_to_frustum=True)
            #    camera_directions_w_lens = homogenize(P_view_outsidelens_direction)
            #    control_points = camera_directions_w_lens.reshape((P_sensor.shape[0], P_sensor.shape[1], 3))[:, :, :2]
            #    projection_matrix = viewpoint_cam.projection_matrix
            #    flow = control_points @ projection_matrix[:2, :2]
            #    plot_points(flow, os.path.join(scene.model_path, f"flow_{iteration}.png"))

        # Loss
        if outside_rasterizer and not apply2gt:
            gt_image = viewpoint_cam.fish_gt_image.cuda()
            Ll1 = l1_loss(image, gt_image*mask)
            ssim_loss = ssim(image, gt_image*mask)
        elif outside_rasterizer and apply2gt:
            Ll1 = l1_loss(image*mask, gt_image)
            ssim_loss = ssim(image*mask, gt_image)
            if iteration % 1000 == 1:
                torchvision.utils.save_image(image, os.path.join(scene.model_path, f"render_{iteration}.png"))
                torchvision.utils.save_image(gt_image, os.path.join(scene.model_path, f"gt_fish2perspective_{iteration}.png"))
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
            if use_wandb:
                wandb.log(scalars, step=iteration)

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
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, mlp_color), lens_net, opt_distortion, outside_rasterizer, flow_scale, control_point_sample_scale, flow_apply2_gt_or_img, apply2gt)
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
                torch.save(lens_net, os.path.join(scene.model_path, f'lens_net{iteration}.pth'))
                if opt_cam:
                    torch.save(scene.train_cameras, os.path.join(scene.model_path, 'opt_cams.pt'))
                    torch.save(scene.unnoisy_train_cameras, os.path.join(scene.model_path, 'gt_cams.pt'))
                    torch.save(scene.train_cameras, os.path.join(scene.model_path, f'cams_train{iteration}.pt'))

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, lens_net, opt_distortion, outside_rasterizer, flow_scale, control_point_sample_scale, flow_apply2_gt_or_img, apply2gt):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

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
                    for idx, viewpoint in enumerate(config['cameras']):
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
                                image = center_crop(image, viewpoint.fish_gt_image.shape[1], viewpoint.fish_gt_image.shape[2]).squeeze(0)
                                mask = (~((image.squeeze(0)[0]==0.) & (image.squeeze(0)[1]==0.)).unsqueeze(0)).float()
                                gt_image = mask * viewpoint.fish_gt_image.cuda()
                                torchvision.utils.save_image(gt_image.cpu(), os.path.join(scene.model_path, 'training_val_{}/gt/masked_{}'.format(iteration, idx) + "_" + name + ".png"))
                                torchvision.utils.save_image(image, os.path.join(scene.model_path, 'training_val_{}/renderred/distorted_{}'.format(iteration, idx) + "_" + name + ".png"))
                            elif apply2gt:
                                P_sensor, P_view_insidelens_direction = generate_control_pts(viewpoint, control_point_sample_scale, flow_scale)
                                gt_image, mask, flow_apply2_gt_or_img = apply_distortion(lens_net, P_view_insidelens_direction, P_sensor, viewpoint, image, apply2gt=apply2gt)
                                #gt_image = F.grid_sample(
                                #    viewpoint.fish_gt_image.cuda().unsqueeze(0),
                                #    flow_apply2_gt_or_img.unsqueeze(0),
                                #    mode="bilinear",
                                #    padding_mode="zeros",
                                #    align_corners=True,
                                #).squeeze(0)
                                torchvision.utils.save_image(gt_image, os.path.join(scene.model_path, 'training_val_{}/gt/{}_perspective'.format(iteration, viewpoint.image_name) + "_" + name + ".png"))
                                torchvision.utils.save_image(viewpoint.fish_gt_image, os.path.join(scene.model_path, 'training_val_{}/gt/{}_fish'.format(iteration, viewpoint.image_name) + "_" + name + ".png"))
                                torchvision.utils.save_image(viewpoint.original_image, os.path.join(scene.model_path, 'training_val_{}/gt/{}_undis'.format(iteration, viewpoint.image_name) + "_" + name + ".png"))
                                torchvision.utils.save_image(image*mask, os.path.join(scene.model_path, 'training_val_{}/renderred/{}_masked'.format(iteration, viewpoint.image_name) + "_" + name + ".png"))
                        #torchvision.utils.save_image(viewpoint.fish_gt_image, os.path.join(scene.model_path, 'training_val_{}/gt/{}'.format(iteration, viewpoint.image_name) + "_" + name + ".png"))
                        else:
                            gt_image = viewpoint.original_image.cuda()
                            torchvision.utils.save_image(gt_image, os.path.join(scene.model_path, 'training_val_{}/gt/{}_perspective'.format(iteration, viewpoint.image_name) + "_" + name + ".png"))

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
                    f.write("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                    f.write("\nSSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                    f.write("\nLPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                    if tb_writer:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
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
    parser.add_argument("--apply2gt", action="store_true", default=False)
    parser.add_argument("--abs_grad", action="store_true", default=False)
    parser.add_argument('--densi_num', type=float, default=0.0002)
    parser.add_argument("--if_circular_mask", action="store_true", default=False)
    # flow_scale[0] is width and flow_scale[1] is height
    parser.add_argument("--flow_scale", nargs="+", type=float, default=[1., 1.])
    parser.add_argument('--iresnet_lr', type=float, default=1e-7)
    parser.add_argument('--opacity_threshold', type=float, default=0.005)

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
             start_opt_lens=args.start_opt_lens, extend_scale=args.extend_scale, control_point_sample_scale=args.control_point_sample_scale, outside_rasterizer=args.outside_rasterizer, abs_grad=args.abs_grad, densi_num=args.densi_num, if_circular_mask=args.if_circular_mask, flow_scale=args.flow_scale, apply2gt=args.apply2gt, iresnet_lr=args.iresnet_lr, opacity_threshold=args.opacity_threshold)

    # All done
    print("\nTraining complete.")
