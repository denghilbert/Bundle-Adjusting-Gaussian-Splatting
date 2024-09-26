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
import math
import numpy as np
from typing import NamedTuple
import torch
import torch.nn.functional as F


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t().cuda()
    j = j.t().cuda()
    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    dirs[:, :, 1:3] = -dirs[:, :, 1:3]
    #dirs_d = torch.stack([(i- W*.5)/K[0][0], -(j-H*.5)/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getWorld2View2_torch_tensor(R, t, translate=torch.tensor([.0, .0, .0]).cuda(), scale=torch.tensor(1.0).cuda()):
    Rt = torch.zeros((4, 4)).cuda()
    Rt[:3, :3] =torch.t(R)
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.linalg.inv(C2W)
    return Rt

def getProjectionMatrix(znear, zfar, fovX, fovY):
    if torch.is_tensor(fovX) and torch.is_tensor(fovY):
        tanHalfFovY = torch.tan((fovY / 2))
        tanHalfFovX = torch.tan((fovX / 2))
    else:
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def cubemap_to_perspective(
    img_forward, img_left, img_right, img_up, img_down,
    fov_h_deg, fov_v_deg, output_width, output_height
):
    """
    Converts cubemap images to a perspective projection.

    Args:
        img_forward: PyTorch tensor of shape (C, H, W)
        img_left: PyTorch tensor of shape (C, H, W)
        img_right: PyTorch tensor of shape (C, H, W)
        img_up: PyTorch tensor of shape (C, H, W)
        img_down: PyTorch tensor of shape (C, H, W)
        fov_h_deg: Horizontal field of view in degrees
        fov_v_deg: Vertical field of view in degrees
        output_width: Width of the output image
        output_height: Height of the output image

    Returns:
        output_image: PyTorch tensor of shape (C, output_height, output_width)
    """

    device = img_forward.device
    dtype = img_forward.dtype

    # Convert FOVs to radians
    fov_h_rad = torch.deg2rad(torch.tensor(fov_h_deg, device=device, dtype=dtype))
    fov_v_rad = torch.deg2rad(torch.tensor(fov_v_deg, device=device, dtype=dtype))

    # Compute focal lengths
    focal_length_x = (output_width / 2.0) / torch.tan(fov_h_rad / 2.0)
    focal_length_y = (output_height / 2.0) / torch.tan(fov_v_rad / 2.0)

    # Generate grid of pixel coordinates
    i = torch.linspace(0, output_width - 1, output_width, device=device, dtype=dtype)
    j = torch.linspace(0, output_height - 1, output_height, device=device, dtype=dtype)
    i, j = torch.meshgrid(i, j, indexing='ij')  # Shape: (output_width, output_height)

    # Convert pixel coordinates to camera space coordinates
    x_camera = (i - (output_width / 2.0)) / focal_length_x
    y_camera = ((output_height / 2.0) - j) / focal_length_y  # Invert y-axis
    z_camera = torch.ones_like(x_camera)

    # Normalize the direction vectors
    dirs = torch.stack((x_camera, y_camera, z_camera), dim=-1)  # Shape: (W, H, 3)
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)

    # Determine which face to sample from
    abs_dirs = torch.abs(dirs)
    max_abs_dir, max_dim = abs_dirs.max(dim=-1)  # Shape: (W, H)

    # Initialize the output image
    C = img_forward.shape[0]
    output_image = torch.zeros((C, output_height, output_width), device=device, dtype=dtype)

    # Prepare the face images for grid sampling
    face_imgs = {
        'forward': img_forward.unsqueeze(0),  # Add batch dimension
        'right': img_right.unsqueeze(0),
        'left': img_left.unsqueeze(0),
        'up': img_up.unsqueeze(0),
        'down': img_down.unsqueeze(0)
    }

    # Create a single mask tensor for all faces
    faces_tensor = torch.full((output_width, output_height), -1, device=device, dtype=torch.long)
    faces_tensor[(max_dim == 2) & (dirs[..., 2] > 0)] = 0  # Forward
    faces_tensor[(max_dim == 0) & (dirs[..., 0] > 0)] = 1  # Right
    faces_tensor[(max_dim == 0) & (dirs[..., 0] < 0)] = 2  # Left
    faces_tensor[(max_dim == 1) & (dirs[..., 1] > 0)] = 3  # Up
    faces_tensor[(max_dim == 1) & (dirs[..., 1] < 0)] = 4  # Down

    # Precompute denominators for each face to avoid division by zero
    epsilon = 1e-6

    denominators = torch.where(
        faces_tensor.unsqueeze(-1) == torch.tensor([0, 1, 2, 3, 4], device=device).view(1, 1, 5),
        torch.stack([
            dirs[..., 2],         # Forward: dir_z
            dirs[..., 0],         # Right: dir_x
            -dirs[..., 0],        # Left: -dir_x
            dirs[..., 1],         # Up: dir_y
            -dirs[..., 1],        # Down: -dir_y
        ], dim=-1),
        torch.ones_like(dirs[..., :1]) * epsilon
    )

    # Compute numerators for u and v
    numerators_u = torch.where(
        faces_tensor.unsqueeze(-1) == torch.tensor([0, 1, 2, 3, 4], device=device).view(1, 1, 5),
        torch.stack([
            dirs[..., 0],         # Forward: dir_x
            -dirs[..., 2],        # Right: -dir_z
            dirs[..., 2],         # Left: dir_z
            dirs[..., 0],         # Up: dir_x
            dirs[..., 0],         # Down: dir_x
        ], dim=-1),
        torch.zeros_like(dirs[..., :1])
    )

    numerators_v = torch.where(
        faces_tensor.unsqueeze(-1) == torch.tensor([0, 1, 2, 3, 4], device=device).view(1, 1, 5),
        torch.stack([
            dirs[..., 1],         # Forward: dir_y
            dirs[..., 1],         # Right: dir_y
            dirs[..., 1],         # Left: dir_y
            -dirs[..., 2],        # Up: -dir_z
            dirs[..., 2],         # Down: dir_z
        ], dim=-1),
        torch.zeros_like(dirs[..., :1])
    )

    # Select the appropriate numerators and denominators
    face_indices = faces_tensor.view(-1)
    valid_mask = face_indices != -1
    face_indices_valid = face_indices[valid_mask]
    dirs_valid = dirs.view(-1, 3)[valid_mask]

    denominator = denominators.view(-1, 5)[valid_mask, face_indices_valid]
    numerator_u = numerators_u.view(-1, 5)[valid_mask, face_indices_valid]
    numerator_v = numerators_v.view(-1, 5)[valid_mask, face_indices_valid]

    # Avoid division by zero
    denominator = torch.where(denominator == 0, torch.full_like(denominator, epsilon), denominator)

    # Compute u and v coordinates
    u = numerator_u / denominator
    v = numerator_v / denominator

    # Stack u and v into a grid for grid_sample
    grid = torch.stack((u, v), dim=-1)  # Shape: (N, 2)
    grid = grid.unsqueeze(0).unsqueeze(2)  # Shape: (1, 1, N, 2)

    # Since u and v are in [-1, 1], they can be used directly in grid_sample

    # Prepare the face images list
    face_imgs_list = [face_imgs['forward'], face_imgs['right'], face_imgs['left'], face_imgs['up'], face_imgs['down']]

    # Sample from the faces
    sampled = []
    for idx in range(5):
        mask = face_indices_valid == idx
        if mask.any():
            grid_face = grid[..., mask, :]
            img_face = face_imgs_list[idx]

            # Sample the face image using grid_sample
            sampled_face = F.grid_sample(
                img_face, grid_face, mode='bilinear', padding_mode='border', align_corners=True
            )  # Shape: (1, C, 1, N_face)

            sampled.append((mask, sampled_face.squeeze(0).squeeze(1)))  # (mask, tensor of shape (C, N_face))

    # Assemble the output image
    output_image_flat = output_image.view(C, -1)
    idx = 0
    for mask, sampled_face in sampled:
        output_image_flat[:, valid_mask][..., mask] = sampled_face

    return output_image
