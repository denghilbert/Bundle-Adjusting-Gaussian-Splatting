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
import visdom
from easydict import EasyDict
from copy import deepcopy
from PIL import Image

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


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, specular=None, hybrid=False, distortion_params=None, u_distortion=None, v_distortion=None, u_radial=None, v_radial=None, affine_coeff=None, poly_coeff=None, save_distortion_map=None):
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
        affine_coeff = torch.load(os.path.join(model_path, f'affine_coeff{iteration}.pt'))
        poly_coeff = torch.load(os.path.join(model_path, f'poly_coeff{iteration}.pt'))
        radial = torch.load(os.path.join(model_path, f'radial{iteration}.pt'))
        lens_net = torch.load(os.path.join(model_path, f'lens_net{iteration}.pth'))
        if os.path.exists(os.path.join(model_path, f'ref_points_{iteration}.pt')):
            ref_points = torch.load(os.path.join(model_path, f'ref_points_{iteration}.pt'))
    else:
        distortion_params = torch.nn.Parameter(torch.zeros(8).cuda())
        u_distortion = nn.Parameter(torch.zeros(400, 400).cuda().requires_grad_(True))
        v_distortion = nn.Parameter(torch.zeros(400, 400).cuda().requires_grad_(True))
        u_radial = nn.Parameter(torch.ones(400, 400).cuda().requires_grad_(True))
        v_radial = nn.Parameter(torch.ones(400, 400).cuda().requires_grad_(True))
        affine_coeff = nn.Parameter(torch.tensor([1., 0., 0., 1., 0., 0.]).cuda().requires_grad_(True))
        #poly_coeff = nn.Parameter(torch.tensor([0.017343506884212139, -0.020094679982101907, -0.019892937295193619, 0.0085534590404976324]).cuda().requires_grad_(True))
        poly_coeff = nn.Parameter(torch.tensor([0., 0., 0., 0.]).cuda().requires_grad_(True))


    #cam = deepcopy(views[0])
    #cam.reset_intrinsic(cam.FoVx + 1, cam.FoVy + 0.5)
    with torch.no_grad():
        width = views[0].image_width
        height = views[0].image_height
        sample_width = int(width / 1)
        sample_height = int(height / 1)
        K = views[0].get_K
        i, j = np.meshgrid(
            np.linspace(0 - width/2, width + width/2, sample_width),
            np.linspace(0 - height/2, height + height/2, sample_height),
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
        P_view_outsidelens_direction = lens_net.forward(P_view_insidelens_direction)
        #P_view_outsidelens_direction = P_view_insidelens_direction
        control_points = homogenize(P_view_outsidelens_direction)
        control_points = control_points.reshape((P_sensor.shape[0], P_sensor.shape[1], 3))[:, :, :2]
        boundary_original_points = P_view_insidelens_direction[-1]
        P_view_outsidelens_direction = P_view_insidelens_direction
        tes_p = homogenize(P_view_outsidelens_direction)
        tes_p = tes_p.reshape((P_sensor.shape[0], P_sensor.shape[1], 3))[:, :, :2]
        tensor = control_points-tes_p
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 2, 596, 1060)
        downsampled_tensor = F.interpolate(tensor, size=(74, 132), mode='area')
        downsampled_tensor = downsampled_tensor.squeeze(0).permute(1, 2, 0)  # Shape: (74, 132, 2)
        torch.save(downsampled_tensor.cpu().detach(), './flow.pt')
        import pdb;pdb.set_trace()
        print(9)

    with torch.no_grad():
        width = views[0].image_width
        height = views[0].image_height
        sample_width = int(width / 1)
        sample_height = int(height / 1)
        K = views[0].get_K
        projection_matrix = views[0].projection_matrix
        i, j = np.meshgrid(
            #np.linspace(0 - width/2, width + width/2, sample_width),
            #np.linspace(0 - height/2, height + height/2, sample_height),
            #np.linspace(0 - width/2.5, width + width/2.5, sample_width),
            #np.linspace(0 - height/2.5, height + height/2.5, sample_height),
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
        P_view_outsidelens_direction = lens_net.forward(P_view_insidelens_direction)
        flow = homogenize(P_view_outsidelens_direction)
        flow = flow.reshape((P_sensor.shape[0], P_sensor.shape[1], 3))[:, :, :2]
        if os.path.exists(os.path.join(model_path, f'ref_points_{iteration}.pt')):
            flow = ref_points.clone() # vanillar gs, grid gs, and ref gs
            flow = nn.functional.interpolate(flow.permute(2, 0, 1).unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).squeeze(0)
            #import pdb;pdb.set_trace()
        flow[:, :, 0] = flow[:, :, 0] * projection_matrix[0][0]
        flow[:, :, 1] = flow[:, :, 1] * projection_matrix[1][1]


    with torch.no_grad():
        width = views[0].image_width
        height = views[0].image_height
        sample_width = int(width / 40)
        sample_height = int(height / 40)
        K = views[0].get_K
        projection_matrix = views[0].projection_matrix
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
        P_view_outsidelens_direction = lens_net.forward(P_view_insidelens_direction)
        dist = homogenize(P_view_outsidelens_direction)
        dist = dist.reshape((P_sensor.shape[0], P_sensor.shape[1], 3))[:, :, :2]
        P_view_outsidelens_direction = P_view_insidelens_direction
        wo_dist = homogenize(P_view_outsidelens_direction)
        wo_dist = wo_dist.reshape((P_sensor.shape[0], P_sensor.shape[1], 3))[:, :, :2]

    save_distortion_list = [dist, wo_dist]
    names = ['distorted', 'wo_dist']
    rainbow = colorize((nn.functional.interpolate(dist.permute(2, 0, 1).unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).squeeze(0)).detach().cpu().numpy())
    image = Image.fromarray(rainbow)
    image.save(os.path.join(model_path, f'final_rainbow.png'))


    image = views[0].original_image.permute(1, 2, 0).cpu().detach().numpy()
    height, width, _ = image.shape
    x, y = np.meshgrid(np.arange(0, width, 4), np.arange(0, height, 4))
    x = x.flatten()
    y = y.flatten()
    plt.figure(figsize=(int(save_distortion_list[0].shape[1]), int(save_distortion_list[0].shape[0])))
    plt.imshow(image)  # Display the image
    plt.scatter(x, y, color='orange', s=4, marker='o')  # Plot points on each pixel
    plt.axis('off')
    plt.savefig(os.path.join(model_path, f'final_each_pix.png'))
    import pdb;pdb.set_trace()


    #color = rainbow
    if save_distortion_map:
        for points, name in zip(save_distortion_list, names):
            p1 = points.reshape(-1, 2)
            plt.figure(figsize=(int(points.shape[1]), int(points.shape[0])))
            x = (p1[:, 0].detach().cpu().numpy() + p1[:, 0].max().item()) * color.shape[1] / (2 * p1[:, 0].max().item())
            y = (p1[:, 1].detach().cpu().numpy() + p1[:, 1].max().item()) * color.shape[0] / (2 * p1[:, 1].max().item())
            if 'wo' in name:
                plt.imshow(color)
                plt.scatter(x, y, s=60, c='orange')
            else:
                plt.scatter(x, y)
            #plt.xlim(p1[:, 0].min().item() - 0.1, p1[:, 0].max().item() + 0.1)
            #plt.ylim(p1[:, 1].min().item() - 0.1, p1[:, 1].max().item() + 0.1)
            plt.xlim(-1, color.shape[1] + 1)
            plt.ylim(-1, color.shape[0] + 1)
            #plt.grid(True)
            plt.gca().invert_yaxis()
            plt.savefig(os.path.join(model_path, f"final_visual_{name}.png"))
    import pdb;pdb.set_trace()


    if os.path.exists(os.path.join(model_path, f'ref_points_{iteration}.pt')):
        control_points = ref_points # vanillar gs, grid gs, and ref gs

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

            results = render(view, gaussians, pipeline, background, mlp_color, control_points, boundary_original_points, undistorted_p_w2c_homo, distortion_params, u_distortion, v_distortion, u_radial, v_radial, affine_coeff, poly_coeff, radial, global_alignment=global_alignment)
            rendering, depth_tensor, weight_mask = results["render"], results["depth"], results["weights"]
            output = F.grid_sample(
                rendering.unsqueeze(0),
                flow.unsqueeze(0),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            depth_tensor_normalized = (depth_tensor - depth_tensor[mask].min()) / (depth_tensor[mask].max() - depth_tensor[mask].min())
            depth_tensor_grey = depth_tensor_normalized.repeat(3, 1, 1)
            depth_array = depth_tensor_normalized.squeeze().cpu().detach().numpy()
            depth_colored = plt.get_cmap('viridis')(depth_array)[:, :, :3]  # Drop the alpha channel
            depth_colored_tensor = torch.from_numpy(depth_colored).permute(2, 0, 1).float()  # Rearrange dimensions to CxHxW



        torchvision.utils.save_image(output.squeeze(0), os.path.join(render_path, 'perspective_{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, 'perspective_{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, mode: str, hybrid: bool, opt_test_cam: bool, opt_intrinsic: bool, opt_extrinsic: bool, render_fisheyefov_from_colmap_perspective: bool, save_distortion_map: bool):
    gaussians = GaussianModel(dataset.sh_degree, dataset.asg_degree)
    lens_net = iResNet()
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    distortion_params = torch.nn.Parameter(torch.zeros(8).cuda())
    u_distortion = nn.Parameter(torch.zeros(400, 400).cuda().requires_grad_(True))
    v_distortion = nn.Parameter(torch.zeros(400, 400).cuda().requires_grad_(True))
    u_radial = nn.Parameter(torch.ones(400, 400).cuda().requires_grad_(True))
    v_radial = nn.Parameter(torch.ones(400, 400).cuda().requires_grad_(True))
    optimizer_u_distortion = torch.optim.Adam([{'params': u_distortion, 'lr': 0.0001}])
    optimizer_v_distortion = torch.optim.Adam([{'params': v_distortion, 'lr': 0.0001}])
    optimizer_u_radial = torch.optim.Adam([{'params': u_radial, 'lr': 0.0001}])
    optimizer_v_radial = torch.optim.Adam([{'params': v_radial, 'lr': 0.0001}])
    affine_coeff = nn.Parameter(torch.tensor([1., 0., 0., 1., 0., 0.]).cuda().requires_grad_(True))
    #poly_coeff = nn.Parameter(torch.tensor([0.017343506884212139, -0.020094679982101907, -0.019892937295193619, 0.0085534590404976324]).cuda().requires_grad_(True))
    poly_coeff = nn.Parameter(torch.tensor([0., 0., 0., 0.]).cuda().requires_grad_(True))

    if not render_fisheyefov_from_colmap_perspective:
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
        progress_bar = tqdm(range(0, 7000), desc="Training progress")
        for iteration in range(7000):
            if iteration % 1000 == 0:
                pose_gt, pose_aligned = scene.visTestCameras()
                vis_cameras(opt_vis, vis, iteration, poses=[pose_aligned, pose_gt])
            if not viewpoint_stack:
                viewpoint_stack = scene.getTestCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            mlp_color = 0
            global_alignment = [torch.tensor([[1., 0, 0], [0, 1., 0], [0, 0, 1.]], device='cuda'), torch.tensor([1.], device='cuda')]
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
            #if opt_distortion and iteration > 3000:
            if False:
                undistorted_p_w2c = lens_net.forward(p_w2c[:, :3])
                undistorted_p_w2c_homo = torch.cat((undistorted_p_w2c, torch.ones(undistorted_p_w2c.size(0), 1).cuda()), dim=1)
            else:
                undistorted_p_w2c_homo = p_w2c
            render_pkg = render(viewpoint_cam, gaussians, pipeline, background, mlp_color, undistorted_p_w2c_homo, distortion_params, u_distortion, v_distortion, u_radial, v_radial, affine_coeff, poly_coeff, global_alignment=global_alignment)
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

                if opt_extrinsic:
                    scene.optimizer_rotation_test.step()
                    scene.optimizer_translation_test.step()
                    scene.optimizer_rotation_test.zero_grad(set_to_none=True)
                    scene.optimizer_translation_test.zero_grad(set_to_none=True)
                    scene.scheduler_rotation_test.step()
                    scene.scheduler_translation_test.step()

                if opt_intrinsic:
                    optimizer_u_distortion.step()
                    optimizer_v_distortion.step()
                    optimizer_u_distortion.zero_grad(set_to_none=True)
                    optimizer_v_distortion.zero_grad(set_to_none=True)
                    optimizer_u_radial.step()
                    optimizer_v_radial.step()
                    optimizer_u_radial.zero_grad(set_to_none=True)
                    optimizer_v_radial.zero_grad(set_to_none=True)

                    scene.optimizer_fovx.step()
                    scene.optimizer_fovy.step()
                    scene.optimizer_fovx.zero_grad(set_to_none=True)
                    scene.optimizer_fovy.zero_grad(set_to_none=True)

        torch.save(scene.test_cameras, os.path.join(scene.model_path, 'opt_test_cam.pt'))

    torch.save(scene.test_cameras, os.path.join(scene.model_path, 'opt_test_cam.pt'))

    if not skip_train:
         render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, specular, hybrid, save_distortion_map=save_distortion_map)

    if not skip_test:
         render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, specular, hybrid, distortion_params, u_distortion, v_distortion, u_radial, v_radial, affine_coeff, poly_coeff)

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
    parser.add_argument("--render_fisheyefov_from_colmap_perspective", action="store_true", default=False)
    parser.add_argument("--save_distortion_map", action="store_true", default=False)
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

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.mode, args.hybrid, args.opt_test_cam, args.opt_intrinsic, args.opt_extrinsic, args.render_fisheyefov_from_colmap_perspective, args.save_distortion_map)
