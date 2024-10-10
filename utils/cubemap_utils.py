import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import os
from scene.dataset_readers import read_intrinsics_binary
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

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

def generate_pts_up_down_left_right(viewpoint_cam, shift_width=0, shift_height=0):
    width = viewpoint_cam.image_width
    height = viewpoint_cam.image_height
    K = viewpoint_cam.get_K
    i, j = np.meshgrid(
        np.linspace(0 + shift_width * width, width + shift_width * width, width),
        np.linspace(0 + shift_height * height, height + shift_height * height, height),
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
    # for debugging
    #print(K @ P_view_insidelens_direction_hom[:1].T)
    #print(K @ P_view_insidelens_direction_hom[-1:].T)
    return P_view_insidelens_direction


def apply_flow_up_down_left_right(viewpoint_cam, rays, img, types="forward", is_fisheye=False):
    width = viewpoint_cam.image_width
    height = viewpoint_cam.image_height
    K = viewpoint_cam.get_K
    r = torch.sqrt(torch.sum(rays**2, dim=-1, keepdim=True))
    if is_fisheye:
        inv_r = 1 / r
        theta = torch.tan(r)
        rays_dis = theta * inv_r * rays
    else:
        rays_dis = rays
    rays_dis_hom = homogenize(rays_dis)

    if types == 'left':
        x = rays_dis_hom[:, 0]  # First column (x)
        y = rays_dis_hom[:, 1]  # Second column (y)
        z = rays_dis_hom[:, 2]  # Third column (z)
        P_left = torch.stack((-z / x, -y / x), dim=1)  # Shape: [N, 2]
        rays_dis_hom = homogenize(P_left)
    elif types == 'right':
        x = rays_dis_hom[:, 0]  # First column (x)
        y = rays_dis_hom[:, 1]  # Second column (y)
        z = rays_dis_hom[:, 2]  # Third column (z)
        P_right = torch.stack((-z / x, y / x), dim=1)  # Shape: [N, 2]
        rays_dis_hom = homogenize(P_right)
    elif types == 'up':
        x = rays_dis_hom[:, 0]  # First column (x)
        y = rays_dis_hom[:, 1]  # Second column (y)
        z = rays_dis_hom[:, 2]  # Third column (z)
        P_up = torch.stack((-x / y, -z / y), dim=1)  # Shape: [N, 2]
        rays_dis_hom = homogenize(P_up)
    elif types == 'down':
        x = rays_dis_hom[:, 0]  # First column (x)
        y = rays_dis_hom[:, 1]  # Second column (y)
        z = rays_dis_hom[:, 2]  # Third column (z)
        P_down = torch.stack((x / y, -z / y), dim=1)  # Shape: [N, 2]
        rays_dis_hom = homogenize(P_down)

    rays_dis_inside = dehomogenize((K @ rays_dis_hom.T).T).reshape(height, width, 2)

    # apply flow field
    x_coords = rays_dis_inside[..., 0]  # Shape: [800, 800]
    y_coords = rays_dis_inside[..., 1]  # Shape: [800, 800]
    x_coords_norm = (x_coords / (img.shape[2] - 1)) * 2 - 1
    y_coords_norm = (y_coords / (img.shape[1] - 1)) * 2 - 1
    grid = torch.stack((x_coords_norm, y_coords_norm), dim=-1)  # Shape: [800, 800, 2]
    grid = grid.unsqueeze(0)
    img_forward_batch = torch.randn((1, 3, 1024, 1024))

    img_forward_batch = img.unsqueeze(0)  # Shape: [1, 3, 800, 800]
    distorted_img = F.grid_sample(
        img_forward_batch,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True)
    distorted_img = distorted_img.squeeze(0)  # Shape: [3, 800, 800]

    return distorted_img, img


def mask_half(image: torch.Tensor, direction: str = "left") -> torch.Tensor:
    _, h, w = image.shape

    # Create a mask with ones for the unmasked area and zeros for the masked area
    mask = torch.ones_like(image)

    if direction == "right":
        # Mask the left half
        mask[:, :, :w // 2] = 0
    elif direction == "left":
        # Mask the right half
        mask[:, :, w // 2:] = 0
    elif direction == "down":
        # Mask the upper half
        mask[:, :h // 2, :] = 0
    elif direction == "up":
        # Mask the lower half
        mask[:, h // 2:, :] = 0
    else:
        raise ValueError("Invalid direction. Choose from 'left', 'right', 'up', 'down'.")

    # Apply the mask to the image (differentiable)
    masked_image = image * mask

    return masked_image, mask
