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

def generate_pts_up_down_left_right(viewpoint_cam, shift_width=0, shift_height=0, sample_rate=1):
    width = viewpoint_cam.image_width
    height = viewpoint_cam.image_height
    K = viewpoint_cam.get_K
    i, j = np.meshgrid(
        np.linspace(0 + shift_width * width, width + shift_width * width, width//sample_rate),
        np.linspace(0 + shift_height * height, height + shift_height * height, height//sample_rate),
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

def interpolate_with_control(control_r, control_theta, r):
    """
    Interpolates control_theta values at specified r locations using linear interpolation.

    Parameters:
    - control_r (torch.Tensor): 1D tensor of reference points for interpolation.
    - control_theta (torch.Tensor): Function values at control_r points.
    - r (torch.Tensor): Target points at which to interpolate control_theta.

    Returns:
    - torch.Tensor: Interpolated values at each point in r, with the same shape as r.
    """
    # Flatten control_r for easy indexing
    control_r_flat = control_r.squeeze()

    # Find indices of the two nearest control_r points for each value in r
    indices = torch.searchsorted(control_r_flat, r.squeeze(), right=True)
    indices = torch.clamp(indices, 1, len(control_r_flat) - 1)

    # Get the lower and upper neighbors for each element in r
    low_indices = indices - 1
    high_indices = indices

    # Fetch the corresponding values from control_r and control_theta
    r_low = control_r_flat[low_indices]
    r_high = control_r_flat[high_indices]
    theta_low = control_theta[low_indices]
    theta_high = control_theta[high_indices]

    # Calculate weights and perform linear interpolation
    weights = (r.squeeze() - r_low) / (r_high - r_low)
    interpolated_theta = (1 - weights).unsqueeze(1) * theta_low + weights.unsqueeze(1) * theta_high

    # Ensure output matches the shape of r
    return interpolated_theta.view_as(r)

def apply_flow_up_down_left_right(viewpoint_cam, rays, rays_residual, img, types="forward", is_fisheye=False, iteration=None, control_r=None, control_theta=None):
    width = viewpoint_cam.image_width
    height = viewpoint_cam.image_height
    K = viewpoint_cam.get_K
    r = torch.sqrt(torch.sum(rays**2, dim=-1, keepdim=True))
    if is_fisheye:
        r[r > 1.392] = 1.392
        theta = torch.tan(r) + interpolate_with_control(control_r, control_theta, r)
        #theta = torch.tan(r)
        inv_r = 1 / (r + 1e-5)

        scale = theta * inv_r
        #scale[scale < 0] = 0 # including scale < 0 can extend to cameras more than 180 degree
        #scale[scale > 10] = 0
        rays_dis = scale * rays
    else:
        rays_dis = rays

    residual = torch.zeros(1, 2, 128, 85)
    #residual = (lens_net.forward(rays_residual, sensor_to_frustum=True) - rays_residual).reshape(height//8, width//8, 2).permute(2, 0, 1).unsqueeze(0)

    #if torch.isnan(residual).any():
    #    import pdb;pdb.set_trace()
    #upsampled_residual = F.interpolate(residual, size=(height, width), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0).reshape(-1, 2)
    #rays_dis_hom = homogenize(rays_dis + upsampled_residual)
    rays_dis_hom = homogenize(rays_dis)

    #xy_points = rays[:, :2].cpu().numpy()
    #plt.figure(figsize=(10, 10))
    #plt.scatter(xy_points[:, 0], xy_points[:, 1], s=10, alpha=0.8)
    #plt.title('2D Scatter Plot of Rays')
    #plt.xlabel('X')
    #plt.ylabel('Y')
    #plt.axis('equal')
    #output_image_path = "output/test/forward_pts.png"
    #plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    #plt.close()

    #xy_points = rays_dis_hom[:, :2].cpu().numpy()
    #plt.figure(figsize=(10, 10))
    #plt.scatter(xy_points[:, 0], xy_points[:, 1], s=10, alpha=0.8)
    #plt.title('2D Scatter Plot of Rays')
    #plt.xlabel('X')
    #plt.ylabel('Y')
    #plt.axis('equal')
    #output_image_path = "output/test/forward_pts_.png"
    #plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    #plt.close()

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

        #xy_points = rays_dis_hom[:, :2].cpu().numpy()
        #xy_points[xy_points>10] = 0
        #xy_points[xy_points<-10] = 0
        #plt.figure(figsize=(10, 10))
        #plt.scatter(xy_points[:, 0], xy_points[:, 1], s=10, alpha=0.8)
        #plt.title('2D Scatter Plot of Rays')
        #plt.xlabel('X')
        #plt.ylabel('Y')
        #plt.axis('equal')
        #output_image_path = "output/test/forward_pts__.png"
        #plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        #plt.close()

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
    #import torchvision
    #torchvision.utils.save_image(img, "output/test/forward_pts___.png")
    #torchvision.utils.save_image(distorted_img, "output/test/forward_pts____.png")
    #import pdb;pdb.set_trace()

    if types == 'forward':
        return distorted_img, img, residual, control_theta
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

