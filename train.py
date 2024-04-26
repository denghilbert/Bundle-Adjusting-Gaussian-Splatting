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
from random import randint
import random
from utils.loss_utils import l1_loss, ssim, kl_divergence, l2_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, SpecularModel, iResNet
from utils.general_utils import safe_state, get_linear_noise_func, linear_to_srgb
from projection_test import image_pair_candidates, light_glue_simple, projection_loss, dist_point_point, dist_point_line, correspondence_projection
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
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

def pass_neuralens(lens_net, width, height, sample_width, sample_height, K, pass_neural=True):
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
    #K[0][2] -= 0.5
    #K[1][2] -= 0.5
    P_view_insidelens_direction_hom = (torch.inverse(K) @ P_sensor_hom.T).T
    P_view_insidelens_direction = dehomogenize(P_view_insidelens_direction_hom)

    if pass_neural:
        P_view_outsidelens_direction = lens_net.forward(P_view_insidelens_direction)
    else:
        P_view_outsidelens_direction = P_view_insidelens_direction

    camera_directions_w_lens = homogenize(P_view_outsidelens_direction)
    camera_directions_w_lens = camera_directions_w_lens.reshape((P_sensor.shape[0], P_sensor.shape[1], 3))[:, :, :2]

    return camera_directions_w_lens, P_view_insidelens_direction[-1]


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, use_wandb=False, random_init=False, hybrid=False, opt_cam=False, opt_distortion=False, opt_intrinsic=False, r_t_noise=[0., 0.], r_t_lr=[0.001, 0.001], global_alignment_lr=0.001):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset.asg_degree)
    if hybrid:
        specular_mlp = SpecularModel()
        specular_mlp.train_setting(opt)
    lens_net = iResNet().cuda()
    l_lens_net = [{'params': lens_net.parameters(), 'lr': 1e-6}]
    optimizer_lens_net = torch.optim.Adam(l_lens_net, eps=1e-15)
    scheduler_lens_net = torch.optim.lr_scheduler.MultiStepLR(optimizer_lens_net, milestones=[7000, 50000], gamma=1)
    def zero_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 0.)
            nn.init.constant_(m.bias, 0.)
    lens_net.apply(zero_weights)
    #for param in lens_net.parameters():
    #    print(param)

    scene = Scene(dataset, gaussians, random_init=random_init, r_t_noise=r_t_noise, r_t_lr=r_t_lr, global_alignment_lr=global_alignment_lr)
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

    height, width = 512, 512
    mask = torch.zeros((height, width), dtype=torch.bool)
    center_x, center_y = 256, 256
    radius = 256  # Example radius
    for y in range(height):
        for x in range(width):
            if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                mask[y, x] = True
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
    x = torch.linspace(-1, 1, 100)
    y = torch.linspace(1, -1, 100)  # Note the reversed order for y to get top to bottom
    x_grid, y_grid = torch.meshgrid(x, y, indexing='xy')
    control_points = nn.Parameter(torch.stack((x_grid, y_grid), dim=-1).cuda().requires_grad_(True))
    optimizer_control_points = torch.optim.Adam([{'params': control_points, 'lr': 0.0001}])
    print("Top-left corner:", control_points[0, 0])
    print("Top-right corner:", control_points[0, -1])
    print("Bottom-left corner:", control_points[-1, 0])
    print("Bottom-right corner:", control_points[-1, -1])

    # |1 e c_x|
    # |d c c_y|
    # |0 0 1  |
    # init as e=d=0, c=1, c_x=c_y=0, order [1, e, d, c, c_x, c_y]
    affine_coeff = nn.Parameter(torch.tensor([1., 0., 0., 1., 0., 0.]).cuda().requires_grad_(True))
    poly_coeff = nn.Parameter(torch.tensor([0, 0, 0, 0.]).cuda().requires_grad_(True))
    optimizer_affine = torch.optim.Adam([{'params': affine_coeff, 'lr': 0.0001}])
    optimizer_poly = torch.optim.Adam([{'params': poly_coeff, 'lr': 0.0001}])

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
        if iteration > 3000 and hybrid:
            dir_pp = (gaussians.get_xyz - viewpoint_cam.get_camera_center.repeat(gaussians.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            mlp_color = specular_mlp.step(gaussians.get_asg_features, dir_pp_normalized)
        else:
            mlp_color = 0

        # other views rendering
        if (iteration == 10001 or iteration == 20001) and args.projection_loss:
            progress_bar_matching = tqdm(range(0, len(viewpoint_stack_constant)), desc="Matching pairs")
            for viewpoint_cam_0 in viewpoint_stack_constant:
                progress_bar_matching.update(1)

                render_pkg = render(viewpoint_cam_0, gaussians, pipe, background, mlp_color, iteration=iteration, hybrid=hybrid)
                image = render_pkg["render"]
                matching_point_dic = {}
                for camera_id in camera_pairs[viewpoint_cam_0.uid]:
                    viewpoint_cam_i = viewpoint_stack_constant[camera_id]
                    render_pkg_i = render(viewpoint_cam_i, gaussians, pipe, background, mlp_color, iteration=iteration, hybrid=hybrid)
                    image_i = render_pkg_i["render"]
                    matching_point_dic[camera_id] = light_glue_simple(image, image_i, 'disk')
                camera_matching_points[viewpoint_cam_0.uid] = matching_point_dic

            progress_bar_matching.close()
            projection_loss_count = 10000

        # render current view
        gaussians_xyz = gaussians.get_xyz.detach()
        gaussians_xyz_homo = torch.cat((gaussians_xyz, torch.ones(gaussians_xyz.size(0), 1).cuda()), dim=1)
        #gaussians_xyz_homo.retain_grad()
        # glm use the transpose of w2c
        w2c = viewpoint_cam.get_world_view_transform().t().detach()
        p_w2c = (w2c @ gaussians_xyz_homo.T).T.cuda().detach()
        intrinsic = viewpoint_cam.get_intrinsic().t().detach()
        proj_mat = viewpoint_cam.get_full_proj_transform().t().detach()
        p_proj = (proj_mat @ gaussians_xyz_homo.T).T.cuda().detach()
        p_2d = p_proj[:, :2] / p_proj[:, -1:]

        if opt_distortion and False:
            undistorted_p_w2c = lens_net.forward(p_w2c[:, :3])
            undistorted_p_w2c_homo = torch.cat((undistorted_p_w2c, torch.ones(undistorted_p_w2c.size(0), 1).cuda()), dim=1)
        else:
            undistorted_p_w2c_homo = p_w2c
        #distorted_points = lens_net.forward(control_points)#lens_net.i_resnet_linear.module_list[4].residual[10].weight
        #directions = distorted_points - control_points
        #distorted_points.retain_grad()
        #u_distortion = directions[:, :, 0]
        #v_distortion = directions[:, :, 1]
        #if iteration % 1000 == 1:
        #    print((directions[:, :, 0]).mean())
        #    print((directions[:, :, 1]).mean())

        #if iteration == 6000:
        #    import matplotlib.pyplot as plt
        #    u = directions[:, :, 0].detach().cpu().numpy()  # X components of vectors
        #    v = directions[:, :, 1].detach().cpu().numpy()  # Y components of vectors
        #    x_positions, y_positions = np.meshgrid(np.arange(100), np.arange(100), indexing='ij')
        #    plt.figure(figsize=(10, 10))
        #    plt.quiver(x_positions, y_positions, u, v, scale=20)
        #    plt.title('Vector Field Plot')
        #    plt.xlabel('X coordinate')
        #    plt.ylabel('Y coordinate')
        #    plt.axis('equal')  # Ensure the aspect ratio is equal to better represent direction
        #    plt.show()
        #    import pdb;pdb.set_trace()


        #control_points, boundary_original_points = pass_neuralens(lens_net, viewpoint_cam.image_width, viewpoint_cam.image_height, int(viewpoint_cam.image_width / 8), int(viewpoint_cam.image_height / 8), viewpoint_cam.get_K)
        control_points, boundary_original_points = pass_neuralens(lens_net, viewpoint_cam.image_width, viewpoint_cam.image_height, 3, 3, viewpoint_cam.get_K)
        print(control_points)
        import pdb;pdb.set_trace()
        #control_points = nn.functional.interpolate(control_points.permute(2, 0, 1).unsqueeze(0), size=(viewpoint_cam.image_height, viewpoint_cam.image_width), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).squeeze(0)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, mlp_color, control_points, boundary_original_points, undistorted_p_w2c_homo, distortion_params, u_distortion, v_distortion, u_radial, v_radial, affine_coeff, poly_coeff, radial, iteration=iteration, hybrid=hybrid, global_alignment=scene.getGlobalAlignment())
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        import pdb;pdb.set_trace()

        if iteration == 7001:
            p1 = render_pkg["means2D"]
            p_w_lens = pass_neuralens(lens_net, viewpoint_cam.image_width, viewpoint_cam.image_height, int(viewpoint_cam.image_width / 8), int(viewpoint_cam.image_height / 8), viewpoint_cam.get_K)
            import matplotlib.pyplot as plt
            x = p1[:, 0].detach().cpu().numpy()  # Convert tensor to numpy for plotting
            y = p1[:, 1].detach().cpu().numpy()
            plt.scatter(x, y)
            plt.title('2D Points Plot')
            plt.xlabel('X axis')
            plt.ylabel('Y axis')
            plt.grid(True)
            plt.show()
            import pdb;pdb.set_trace()


        loss_projection = 0.
        if camera_matching_points != {} and projection_loss_count > 0:
            projection_loss_count = projection_loss_count - 1
            for matched_camera_id in camera_pairs[viewpoint_cam.uid]:
                m_kpts0, m_kpts1 = camera_matching_points[viewpoint_cam.uid][matched_camera_id]
                points_img0 = m_kpts0
                points_img1 = m_kpts1
                img0_row_col = m_kpts0.t()[[1, 0], :]
                img1_row_col = m_kpts1.t()[[1, 0], :]
                points_proj_img0, points_proj_img1, valid = correspondence_projection(img0_row_col, img1_row_col, viewpoint_cam, viewpoint_stack_constant[matched_camera_id], projection_type='average') # average, self, separate

                point_dists_0 = dist_point_point(points_img0[valid], points_proj_img0[valid])
                point_dists_1 = dist_point_point(points_img1[valid], points_proj_img1[valid])
                proj_ray_dist_threshold = 5.0

                loss_projection += projection_loss(point_dists_0, point_dists_1, proj_ray_dist_threshold)


        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        #Ll1 = l1_loss(image * mask, gt_image * mask)
        #ssim_loss = ssim(image * mask, gt_image * mask)
        Ll1 = l1_loss(image, gt_image)
        ssim_loss = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_loss)# + 0.1 * (loss_projection / len(camera_pairs[viewpoint_cam.uid]))

        #if iteration > 3000:
        #    residual_color = render(viewpoint_cam, gaussians, pipe, background, mlp_color, hybrid=hybrid)["render"]
        #    reflect_loss = l1_loss(gt_image - image, residual_color)
        #    loss = loss + reflect_loss

        loss.backward(retain_graph=True)

        if iteration == 7001 and False:
            outlier_stack = scene.getTrainCameras().copy()
            outliers = []
            uid_list = []
            best_ssim = []
            for idx, viewpoint_cam in enumerate(outlier_stack):
                render_pkg = render(viewpoint_cam, gaussians, pipe, background, mlp_color, iteration=iteration, hybrid=hybrid, global_alignment=scene.getGlobalAlignment())
                image, _, _, _ = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                gt_image = viewpoint_cam.original_image.cuda()
                Ll1 = l1_loss(image, gt_image)
                ssim_loss = ssim(image, gt_image)
                if Ll1 > 0.02 and ssim_loss < 0.8:
                #if ssim_loss < 0.9:
                #if Ll1 > 0.02:
                    outliers.append(viewpoint_cam)
                    uid_list.append(viewpoint_cam.uid)
                    best_ssim.append(ssim_loss)
                    wandb_img = image.unsqueeze(0).detach()
                    wandb_img_gt = gt_image.unsqueeze(0).detach()
                    images_error = (wandb_img_gt - wandb_img).abs()
                    cat_imgs = torch.cat((gt_image, image), dim=2).detach()
                    images = {
                        f"failure/gt_rendered": wandb_image(cat_imgs),
                        f"failure/gt_img": wandb_image(gt_image),
                        f"failure/rendered_img": wandb_image(wandb_img),
                        f"failure/rgb_error": wandb_image(images_error),
                        f"failure/loss": Ll1,
                        f"failure/uid": viewpoint_cam.uid,
                    }
                    if use_wandb:
                        wandb.log(images, step=iteration + idx)
            if len(outliers) > 0:
                for i in range(2000):
                    viewpoint_cam = outliers[i % len(outliers)]
                    render_pkg = render(viewpoint_cam, gaussians, pipe, background, mlp_color, iteration=iteration, hybrid=hybrid, global_alignment=scene.getGlobalAlignment())
                    image, _, _, _ = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                    gt_image = viewpoint_cam.original_image.cuda()

                    Ll1 = l1_loss(image, gt_image)
                    ssim_loss = ssim(image, gt_image)
                    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_loss)# + 0.1 * (loss_projection / len(camera_pairs[viewpoint_cam.uid]))
                    loss.backward(retain_graph=True)

                    if opt_cam:
                        if i <= 700:
                            scene.optimizer_translation.param_groups[viewpoint_cam.uid]['lr'] = scene.optimizer_translation.param_groups[viewpoint_cam.uid]['lr'] * 10
                            scene.optimizer_rotation.param_groups[viewpoint_cam.uid]['lr'] = scene.optimizer_rotation.param_groups[viewpoint_cam.uid]['lr'] * 10
                        elif 700 < i < 1500:
                            scene.optimizer_translation.param_groups[viewpoint_cam.uid]['lr'] = scene.optimizer_translation.param_groups[viewpoint_cam.uid]['lr'] * 5
                            scene.optimizer_rotation.param_groups[viewpoint_cam.uid]['lr'] = scene.optimizer_rotation.param_groups[viewpoint_cam.uid]['lr'] * 5
                        else:
                            scene.optimizer_translation.param_groups[viewpoint_cam.uid]['lr'] = scene.optimizer_translation.param_groups[viewpoint_cam.uid]['lr'] * 2
                            scene.optimizer_rotation.param_groups[viewpoint_cam.uid]['lr'] = scene.optimizer_rotation.param_groups[viewpoint_cam.uid]['lr'] * 2

                        scene.optimizer_rotation.step()
                        scene.optimizer_translation.step()
                        scene.optimizer_rotation.zero_grad(set_to_none=True)
                        scene.optimizer_translation.zero_grad(set_to_none=True)
                        scene.scheduler_rotation.step()
                        scene.scheduler_translation.step()
                        if i <= 700:
                            scene.optimizer_translation.param_groups[viewpoint_cam.uid]['lr'] = scene.optimizer_translation.param_groups[viewpoint_cam.uid]['lr'] / 10
                            scene.optimizer_rotation.param_groups[viewpoint_cam.uid]['lr'] = scene.optimizer_rotation.param_groups[viewpoint_cam.uid]['lr'] / 10
                        elif 700 < i < 1500:
                            scene.optimizer_translation.param_groups[viewpoint_cam.uid]['lr'] = scene.optimizer_translation.param_groups[viewpoint_cam.uid]['lr'] / 5
                            scene.optimizer_rotation.param_groups[viewpoint_cam.uid]['lr'] = scene.optimizer_rotation.param_groups[viewpoint_cam.uid]['lr'] / 5
                        else:
                            scene.optimizer_translation.param_groups[viewpoint_cam.uid]['lr'] = scene.optimizer_translation.param_groups[viewpoint_cam.uid]['lr'] / 2
                            scene.optimizer_rotation.param_groups[viewpoint_cam.uid]['lr'] = scene.optimizer_rotation.param_groups[viewpoint_cam.uid]['lr'] / 2

                    if args.vis_pose and i % 100 == 0:
                        pose_GT, pose_aligned = scene.loadAlignCameras(if_vis_train=True, camera_uid_list=uid_list, path=scene.model_path)
                        vis_cameras(opt_vis, vis, step=i, poses=[pose_aligned, pose_GT])


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
                #progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            #if iteration in testing_iterations:
            if iteration % 500 == 0 and args.vis_pose:
                pose_GT, pose_aligned = scene.loadAlignCameras(if_vis_train=True, iteration=iteration, path=scene.model_path)
                vis_cameras(opt_vis, vis, step=iteration, poses=[pose_aligned, pose_GT])
                #if iteration == 20000 or iteration == 30000:
                #    download_pose_vis(os.path.join(args.model_path, 'plot'), iteration)

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, mlp_color), lens_net, opt_distortion, distortion_params, u_distortion, v_distortion, u_radial, v_radial, affine_coeff, poly_coeff, radial)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
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

                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                #if opt_distortion and iteration > 3000:
                if opt_distortion:
                    # 8 params
                    optimizer_distortion.step()
                    optimizer_distortion.zero_grad(set_to_none=True)

                    # control points
                    #optimizer_control_points.step()
                    #optimizer_control_points.zero_grad(set_to_none=True)

                    #optimizer_lens_net.step() #lens_net.i_resnet_linear.module_list[0].residual[0].weight
                    #optimizer_lens_net.zero_grad(set_to_none=True)

                    # feature grid
                    #optimizer_u_distortion.step()
                    #optimizer_v_distortion.step()
                    #optimizer_u_distortion.zero_grad(set_to_none=True)
                    #optimizer_v_distortion.zero_grad(set_to_none=True)
                    #optimizer_u_radial.step()
                    #optimizer_v_radial.step()
                    #optimizer_u_radial.zero_grad(set_to_none=True)
                    #optimizer_v_radial.zero_grad(set_to_none=True)

                    # omindirectional
                    #optimizer_affine.step()
                    #optimizer_affine.zero_grad(set_to_none=True)
                    #optimizer_poly.step()
                    #optimizer_poly.zero_grad(set_to_none=True)

                    # radial table
                    #optimizer_radial.step()
                    #optimizer_radial.zero_grad(set_to_none=True)

                    # optimize fov
                    #scene.optimizer_fovx.step()
                    #scene.optimizer_fovy.step()
                    #scene.optimizer_fovx.zero_grad(set_to_none=True)
                    #scene.optimizer_fovy.zero_grad(set_to_none=True)

                    # neuralens
                    #scene.optimizer_lens_net.param_groups[0]['lr']
                    #scene.optimizer_lens_net.step()
                    #scene.optimizer_lens_net.zero_grad(set_to_none=True)
                    #scene.scheduler_lens_net.step()

                # do not update camera pose when densify or prune gaussians
                if opt_cam:
                    if iteration % opt.densification_interval != 0:# and iteration > opt.densify_from_iter:
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

                        if iteration > 100000 and iteration % 100 == 0:
                            scene.optimizer_global_alignment.step()
                            scene.optimizer_global_alignment.zero_grad(set_to_none=True)
                            scene.scheduler_global_aligment.step()

                if hybrid:
                    specular_mlp.optimizer.step()
                    specular_mlp.optimizer.zero_grad()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, lens_net, opt_distortion, distortion_params, u_distortion, v_distortion, u_radial, v_radial, affine_coeff, poly_coeff, radial):
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
                for idx, viewpoint in enumerate(config['cameras']):
                    gaussians_xyz = scene.gaussians.get_xyz.detach()
                    gaussians_xyz_homo = torch.cat((gaussians_xyz, torch.ones(gaussians_xyz.size(0), 1).cuda()), dim=1)
                    # glm use the transpose of w2c
                    w2c = viewpoint.get_world_view_transform().t().detach()
                    p_w2c = (w2c @ gaussians_xyz_homo.T).T.cuda().detach()
                    if opt_distortion and False:
                        undistorted_p_w2c = lens_net.forward(p_w2c[:, :3])
                        undistorted_p_w2c_homo = torch.cat((undistorted_p_w2c, torch.ones(undistorted_p_w2c.size(0), 1).cuda()), dim=1)
                    else:
                        undistorted_p_w2c_homo = p_w2c
                    control_points, boundary_original_points = pass_neuralens(lens_net, viewpoint.image_width, viewpoint.image_height, int(viewpoint.image_width / 8), int(viewpoint.image_height / 8), viewpoint.get_K)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, control_points, boundary_original_points, undistorted_p_w2c_homo, distortion_params, u_distortion, v_distortion, u_radial, v_radial, affine_coeff, poly_coeff, radial, global_alignment=scene.getGlobalAlignment())["render"], 0.0, 1.0)
                    #image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, p_w2c, global_alignment=scene.getGlobalAlignment())["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
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
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, use_wandb=args.wandb, random_init=args.random_init_pc, hybrid=args.hybrid, opt_cam=args.opt_cam, opt_distortion=args.opt_distortion, opt_intrinsic=args.opt_intrinsic, r_t_lr=args.r_t_lr, r_t_noise=args.r_t_noise, global_alignment_lr=args.global_alignment_lr)

    # All done
    print("\nTraining complete.")
