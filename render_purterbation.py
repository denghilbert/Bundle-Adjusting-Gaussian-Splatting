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
import imageio
import numpy as np
import cv2
from utils.camera import Lie

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, specular=None, hybrid=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    lie = Lie()
    pi = 3.1415926
    images = torch.tensor([]).cuda()
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        angle = pi / 80
        name_list = ['rotation_x', 'rotation_y', 'rotation_z', 'translation_x', 'translation_y', 'translation_z', 'focal']
        for i in range(10):
            so3_noise = torch.tensor([0 * angle * i, 0 * angle * i, 0 * angle * i])
            t_noise = torch.tensor([0 * angle * i, 0 * angle * i, angle * i]).numpy()
            Fov_noise = torch.tensor([0 * angle * i,  0 * angle * i]).numpy()
            so3 = lie.so3_to_SO3(so3_noise).cpu().detach().numpy()
            tmp_R = so3 @ view.R
            tmp_T = view.T + t_noise
            tmp_fovx = view.FoVx + Fov_noise[0]
            tmp_fovy = view.FoVy + Fov_noise[1]
            view.reset_extrinsic(tmp_R, tmp_T)
            view.reset_intrinsic(tmp_fovx, tmp_fovy)
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
                results = render(view, gaussians, pipeline, background, mlp_color)
                rendering = results["render"]

            images = torch.cat((images, rendering.unsqueeze(0)))
            torchvision.utils.save_image(rendering, os.path.join(render_path, 'translation_z{}'.format(i) + ".png"))
        torchvision.utils.save_image(torch.mean(images, dim=0), os.path.join(render_path, 'translation_avg_z{}'.format(i) + ".png"))
        import pdb;pdb.set_trace()
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        #torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, mode: str, hybrid: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.asg_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
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
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.mode, args.hybrid)
