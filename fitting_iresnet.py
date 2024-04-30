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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, use_wandb=False, random_init=False, hybrid=False, opt_cam=False, opt_distortion=False, opt_intrinsic=False, r_t_noise=[0., 0.], r_t_lr=[0.001, 0.001], global_alignment_lr=0.001):
    gaussians = GaussianModel(dataset.sh_degree, dataset.asg_degree)
    scene = Scene(dataset, gaussians, random_init=random_init, r_t_noise=r_t_noise, r_t_lr=r_t_lr, global_alignment_lr=global_alignment_lr)
    lens_net = iResNet().cuda()
    l_lens_net = [{'params': lens_net.parameters(), 'lr': 1e-5}]
    optimizer_lens_net = torch.optim.Adam(l_lens_net, eps=1e-15)
    scheduler_lens_net = torch.optim.lr_scheduler.MultiStepLR(optimizer_lens_net, milestones=[2000, 50000], gamma=0.1)
    #scheduler_lens_net = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_lens_net, 'min')
    def zero_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 0.)
            nn.init.constant_(m.bias, 0.01)
    #lens_net.apply(zero_weights)
    #for param in lens_net.parameters():
    #    print(param)
    # Control points
    viewpoint_cam = scene.getTrainCameras().copy()[0]
    width = viewpoint_cam.image_width
    height = viewpoint_cam.image_height
    sample_width = int(width / 8)
    sample_height = int(height / 8)
    K = viewpoint_cam.get_K
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
    P_view_outsidelens_direction = P_view_insidelens_direction
    camera_directions_w_lens = homogenize(P_view_outsidelens_direction)
    gt_points = camera_directions_w_lens.reshape((P_sensor.shape[0], P_sensor.shape[1], 3))[:, :, :2]
    r = torch.sqrt(torch.sum(gt_points**2, dim=-1, keepdim=True))
    inv_r = 1 / r
    theta = torch.atan(r)
    weights = torch.exp(r)
    gt_points = gt_points * inv_r * theta
    gt_points = gt_points.cuda()


    P_view_outsidelens_direction = lens_net.forward(P_view_insidelens_direction)
    control_points = homogenize(P_view_outsidelens_direction)
    control_points = control_points.reshape((P_sensor.shape[0], P_sensor.shape[1], 3))[:, :, :2]
    control_points = nn.Parameter(control_points.cuda().requires_grad_(True))
    optimizer_control_points = torch.optim.Adam([{'params': control_points, 'lr': 0.01}])


    boundary_original_points = P_view_insidelens_direction[-1]

    for idx, points in enumerate([control_points, gt_points]):
        p1 = points.reshape(-1, 2)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(int(control_points.shape[1]/4), int(control_points.shape[0]/4)))
        x = p1[:, 0].detach().cpu().numpy()  # Convert tensor to numpy for plotting
        y = p1[:, 1].detach().cpu().numpy()
        plt.scatter(x, y)
        plt.title('2D Points Plot')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.xlim(-boundary_original_points[0].item() - 0.1, boundary_original_points[0].item() + 0.1)
        plt.ylim(-boundary_original_points[1].item() - 0.1, boundary_original_points[1].item() + 0.1)
        plt.grid(True)
        plt.savefig(os.path.join(scene.model_path, f"init{idx}.png"))


    progress_bar = tqdm(range(0, 10000), desc="Fitting Iresnet")
    for i in range(5000):
        P_view_outsidelens_direction = lens_net.forward(P_view_insidelens_direction)
        control_points = homogenize(P_view_outsidelens_direction)
        control_points = control_points.reshape((P_sensor.shape[0], P_sensor.shape[1], 3))[:, :, :2]
        control_points.retain_grad()

        #loss = torch.linalg.norm(control_points - gt_points, dim=-1).mean()
        #loss = (weights * (control_points - gt_points)**2).mean()
        loss = ((control_points - gt_points)**2).mean()
        progress_bar.set_postfix(loss=loss.item())
        progress_bar.update(1)

        #if i % 10 == 0:

        loss.backward()
        optimizer_lens_net.step()
        optimizer_lens_net.zero_grad(set_to_none = True)
        #scheduler_lens_net.step(loss)
        #optimizer_control_points.step()
        #optimizer_control_points.zero_grad(set_to_none = True)

        if i % 300 == 0:
            #plt.clf()
            #plt.imshow(torch.linalg.norm(control_points - gt_points, dim=-1).detach().cpu().numpy(), cmap='hot', interpolation='nearest')
            #plt.colorbar()
            #plt.savefig(os.path.join(scene.model_path, f"loss_{i}.png"))
            #x, y = control_points.grad[:, :, 0].detach().cpu().numpy(), control_points.grad[:, :, 1].detach().cpu().numpy()
            #grad_x, grad_y = np.sin(x) * np.cos(y), np.cos(x) * np.sin(y)
            #gradient_tensor = np.stack((grad_x, grad_y), axis=-1)
            #magnitude = np.sqrt(gradient_tensor[:,:,0]**2 + gradient_tensor[:,:,1]**2)

            #plt.clf()
            #plt.imshow(magnitude, cmap='viridis')
            #plt.colorbar()
            #plt.title('Gradient Magnitude')
            #plt.savefig(os.path.join(scene.model_path, f"gradient_{i}.png"))
            p1 = control_points.reshape(-1, 2)
            import matplotlib.pyplot as plt
            plt.figure(figsize=(int(control_points.shape[1]/4), int(control_points.shape[0]/4)))
            x = p1[:, 0].detach().cpu().numpy()  # Convert tensor to numpy for plotting
            y = p1[:, 1].detach().cpu().numpy()
            plt.scatter(x, y)
            plt.title('2D Points Plot')
            plt.xlabel('X axis')
            plt.ylabel('Y axis')
            plt.xlim(-boundary_original_points[0].item() - 0.1, boundary_original_points[0].item() + 0.1)
            plt.ylim(-boundary_original_points[1].item() - 0.1, boundary_original_points[1].item() + 0.1)
            plt.grid(True)
            plt.savefig(os.path.join(scene.model_path, f"control_{i}.png"))
            plt.clf()



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
