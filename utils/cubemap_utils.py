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


def generate_circular_mask(image_shape: torch.Size, radius: int) -> torch.Tensor:
    """
    Generates a circular mask based on the provided radius.

    Args:
    - image_shape (torch.Size): The shape of the image (C, H, W).
    - radius (int): The radius of the circular mask.

    Returns:
    - torch.Tensor: A circular mask with shape (C, H, W) where the area inside the
      radius from the center is 1 and outside is 0.
    """
    _, h, w = image_shape

    # Create a coordinate grid centered at the image center
    y_center, x_center = h // 2, w // 2
    y_grid, x_grid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")

    # Calculate the distance of each point from the center
    dist_from_center = torch.sqrt((x_grid - x_center) ** 2 + (y_grid - y_center) ** 2)

    # Create the circular mask
    mask = (dist_from_center <= radius).float()

    # Expand the mask to match the shape of the image (C, H, W)
    mask = mask.unsqueeze(0).repeat(3, 1, 1)

    return mask

def interpolate_with_control(control_r, control_theta, r):
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

def differentiable_interpolation(r, control_r, control_theta):
    # Expand dimensions for broadcasting
    r_expanded = r.unsqueeze(1)  # [N, 1, 1]
    control_r_expanded = control_r.unsqueeze(0)  # [1, M, 1]
    
    # Calculate differences between r and control points
    diffs = r_expanded - control_r_expanded  # [N, M, 1]
    
    # Create masks for finding points that bracket each r value
    lower_mask = (diffs >= 0).float()
    upper_mask = (diffs < 0).float()
    
    # Find indices of closest lower and upper control points
    # We use a trick here to get differentiable "argmax-like" behavior
    lower_indices = (lower_mask * torch.arange(control_r.shape[0], device=r.device).float().unsqueeze(0).unsqueeze(-1)).sum(dim=1)
    upper_indices = control_r.shape[0] - 1 - (upper_mask * torch.arange(control_r.shape[0]-1, -1, -1, device=r.device).float().unsqueeze(0).unsqueeze(-1)).sum(dim=1)
    
    # Handle edge cases
    lower_indices = torch.clamp(lower_indices, 0, control_r.shape[0]-2)
    upper_indices = torch.clamp(upper_indices, 1, control_r.shape[0]-1)
    
    # Convert indices to integer for gathering
    lower_indices = lower_indices.long()
    upper_indices = upper_indices.long()
    
    # Gather the bracketing control points and their corresponding theta values
    r1 = control_r[lower_indices]
    r2 = control_r[upper_indices]
    theta1 = control_theta[lower_indices]
    theta2 = control_theta[upper_indices]
    
    # Perform linear interpolation
    # (r - r1) * (theta2 - theta1) / (r2 - r1) + theta1
    result = (r - r1) * (theta2 - theta1) / (r2 - r1) + theta1
    
    return result

def apply_flow_up_down_left_right(viewpoint_cam, rays_dis_hom, img, types="forward", is_fisheye=False, iteration=None):
    width = viewpoint_cam.image_width
    height = viewpoint_cam.image_height
    K = viewpoint_cam.get_K
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


def render_cubemap(render, viewpoint_cam, control_point_sample_scale, cubemap_net, mask_fov90, shift_width, shift_height, gaussians, pipe, background, mlp_color, shift_factors, iteration, hybrid, scene, validation=False, control_theta=None):
    img_list, viewspace_point_tensor_list, visibility_filter_list, radii_list = [], [], [], []
    if validation:
        img_perspective_list = []

    width = viewpoint_cam.image_width
    height = viewpoint_cam.image_height

    rays_base = generate_pts_up_down_left_right(viewpoint_cam, shift_width=0, shift_height=0)
    rays = generate_pts_up_down_left_right(viewpoint_cam, shift_width=0, shift_height=0, sample_rate=control_point_sample_scale)
    render_pkg = render(viewpoint_cam, gaussians, pipe, background, mlp_color, shift_factors, iteration=iteration, hybrid=hybrid, global_alignment=scene.getGlobalAlignment())
    img_forward, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"] * mask_fov90, render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

    # high resolution base distortion
    r_d = torch.sqrt(torch.sum(rays_base**2, dim=-1, keepdim=True))
    inv_r_d = 1 / (r_d + 1e-7)
    #theta = torch.tan(r) + interpolate_with_control(control_r, control_theta, r)
    #theta = torch.tan(r) + differentiable_interpolation(r, control_r, control_theta)
    r_n = torch.tan(r_d)
    scale = r_n * inv_r_d
    rays_dis_base = scale * rays_base

    r_d = torch.sqrt(torch.sum(rays**2, dim=-1, keepdim=True))
    inv_r_d = 1 / (r_d + 1e-7)
    r_d[r_d > 1.5] = 1.5
    r_n = torch.tan(r_d)
    scale = r_n * inv_r_d
    rays_dis = scale * rays

    residual = (cubemap_net.forward(rays_dis, sensor_to_frustum=True) - rays_dis).reshape(height//control_point_sample_scale, width//control_point_sample_scale, 2).permute(2, 0, 1).unsqueeze(0)
    upsampled_residual = F.interpolate(residual, size=(height, width), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0).reshape(-1, 2)
    if torch.isnan(residual).any():
        import pdb;pdb.set_trace()
    rays_dis_hom = homogenize(rays_dis_base + upsampled_residual)

    img_distorted, img_perspective = apply_flow_up_down_left_right(viewpoint_cam, rays_dis_hom, img_forward, types="forward", is_fisheye=True, iteration=iteration)
    img_list.append(img_distorted)
    if validation:
        img_perspective_list.append(img_perspective)
    viewspace_point_tensor_list.append(viewspace_point_tensor)
    visibility_filter_list.append(visibility_filter)
    radii_list.append(radii)

    name = ['up', 'down', 'left', 'right']
    for i, sub_camera in enumerate(viewpoint_cam.sub_cameras):
        if i == 4: break
        render_pkg = render(sub_camera, gaussians, pipe, background, mlp_color, shift_factors, iteration=iteration, hybrid=hybrid, global_alignment=scene.getGlobalAlignment())
        img_up, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"] * mask_fov90, render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        #if name[i] == 'up':
        #    rays_up = generate_pts_up_down_left_right(sub_camera, shift_width=0, shift_height=-shift_height, sample_rate=8)
        #elif name[i] == 'down':
        #    rays_up = generate_pts_up_down_left_right(sub_camera, shift_width=0, shift_height=shift_height, sample_rate=8)
        #elif name[i] == 'left':
        #    rays_up = generate_pts_up_down_left_right(sub_camera, shift_width=shift_width, shift_height=0, sample_rate=8)
        #elif name[i] == 'right':
        #    rays_up = generate_pts_up_down_left_right(sub_camera, shift_width=-shift_width, shift_height=0, sample_rate=8)
        img_distorted, img_perspective = apply_flow_up_down_left_right(sub_camera, rays_dis_hom, img_up, types=name[i], is_fisheye=True)
        img_distorted_masked, half_mask = mask_half(img_distorted, name[i])

        img_list.append(img_distorted_masked)
        viewspace_point_tensor_list.append(viewspace_point_tensor)
        visibility_filter_list.append(visibility_filter)
        radii_list.append(radii)
        if validation:
            img_perspective_list.append(img_perspective)

    if not validation:
        return img_list, viewspace_point_tensor_list, visibility_filter_list, radii_list
    else:
        return img_list, img_perspective_list

