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
import os
from torch import nn
from PIL import Image
import torch.nn.functional as F
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getWorld2View2_torch_tensor, get_rays, fov2focal, focal2fov
from utils.camera import Lie
from utils.general_utils import PILtoTorch


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, intrinsic_matrix, FoVx, FoVy, focal_length_x, focal_length_y, image, gt_alpha_mask, image_name, uid, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", depth=None, ori_path=None, outside_rasterizer=False, test_outside_rasterizer=False, orig_fov_w=0, orig_fov_h=0):
        super(Camera, self).__init__()
        assert orig_fov_w !=0 and orig_fov_h !=0
        self.orig_fov_w = orig_fov_w
        self.orig_fov_h = orig_fov_h

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.intrinsic_matrix = torch.from_numpy(intrinsic_matrix).cuda()
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.focal_x = focal_length_x
        self.focal_y = focal_length_y
        self.image_name = image_name
        self.lie = Lie()

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.depth = torch.Tensor(depth).to(self.data_device) if depth is not None else None

        if gt_alpha_mask is not None:
            #self.original_image *= gt_alpha_mask.to(self.data_device)
            self.original_image *= gt_alpha_mask
        else:
            #self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.original_image *= torch.ones((1, self.image_height, self.image_width))


        self.zfar = 100.0
        self.znear = 0.01

        self.trans = torch.tensor(trans).cuda()
        self.scale = torch.tensor(scale).cuda()

        # represent translation and rotation with so3
        self.init_translation = torch.tensor(T).float().view(-1, 1).cuda()
        self.delta_translation = nn.Parameter(torch.zeros(3, 1).cuda().requires_grad_(True))
        self.translation = self.init_translation + self.delta_translation
        self.init_quaternion = rotation_matrix_to_quaternion(torch.tensor(R).float().t().cuda())
        self.delta_quaternion = nn.Parameter(torch.zeros(4).cuda().requires_grad_(True))
        self.quaternion = self.init_quaternion + self.delta_quaternion
        self.rotation = quaternion_to_rotation_matrix(self.quaternion)
        # get Rt and last row, finally get extrinsic (w2c)
        self.last_row = torch.tensor([[0., 0., 0., 1.]]).cuda()
        self.Rt = torch.cat((self.rotation, self.translation), dim=1)
        self.world_view_transform = torch.cat((self.Rt, self.last_row), dim=0).t()
        self.learnable_fovx = nn.Parameter(torch.tensor(self.FoVx).cuda().requires_grad_(True))
        self.learnable_fovy = nn.Parameter(torch.tensor(self.FoVy).cuda().requires_grad_(True))
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.learnable_fovx, fovY=self.learnable_fovy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        if test_outside_rasterizer:
            self.reset_intrinsic(
                focal2fov(self.focal_x, int(2. * self.orig_fov_w)),
                focal2fov(self.focal_y, int(2. * self.orig_fov_h)),
                self.focal_x,
                self.focal_y,
                int(2. * self.original_image.shape[2]),
                int(2. * self.original_image.shape[1])
            )

        if not test_outside_rasterizer:
            if 'image' in ori_path:
                if os.path.exists(ori_path.split('images')[0] + 'fish/images' + ori_path.split('images')[1]):
                    image = Image.open(ori_path.split('images')[0] + 'fish/images' + ori_path.split('images')[1])
                    orig_w, orig_h = image.size
                    resized_image_rgb = PILtoTorch(image, (orig_w, orig_h))
                    gt_image = resized_image_rgb[:3, ...]
                    self.fish_gt_image = gt_image.clamp(0.0, 1.0)
            else:
                self.fish_gt_image = self.original_image

        if outside_rasterizer:
            self.reset_intrinsic(
                focal2fov(self.focal_x, int(2. * self.orig_fov_w)),
                focal2fov(self.focal_y, int(2. * self.orig_fov_h)),
                self.focal_x,
                self.focal_y,
                int(1. * self.original_image.shape[2]),
                int(1. * self.original_image.shape[1])
            )


    def reset_intrinsic(self, FoVx, FoVy, focal_x, focal_y, width, height):
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.learnable_fovx = nn.Parameter(torch.tensor(FoVx).cuda().requires_grad_(True))
        self.learnable_fovy = nn.Parameter(torch.tensor(FoVy).cuda().requires_grad_(True))
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.learnable_fovx, fovY=self.learnable_fovy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

        self.image_width = width
        self.image_height = height
        self.intrinsic_matrix[0][0] = focal_x
        self.intrinsic_matrix[1][1] = focal_y
        self.intrinsic_matrix[0][2] = width / 2
        self.intrinsic_matrix[1][2] = height / 2

    def perturb_fov(self, scale):
        self.learnable_fovx *= scale
        self.learnable_fovy *= scale

    def reset_extrinsic(self, R, T):
        self.R = R
        self.T = T

        self.init_translation = torch.tensor(T).float().view(-1, 1).cuda()
        self.delta_translation = nn.Parameter(torch.zeros(3, 1).cuda().requires_grad_(True))
        self.translation = self.init_translation + self.delta_translation
        self.init_quaternion = rotation_matrix_to_quaternion(torch.tensor(R).float().t().cuda())
        self.delta_quaternion = nn.Parameter(torch.zeros(4).cuda().requires_grad_(True))
        self.quaternion = self.init_quaternion + self.delta_quaternion
        self.rotation = quaternion_to_rotation_matrix(self.quaternion)
        self.last_row = torch.tensor([[0., 0., 0., 1.]]).cuda()
        self.Rt = torch.cat((self.rotation, self.translation), dim=1)
        self.world_view_transform = torch.cat((self.Rt, self.last_row), dim=0).t()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.learnable_fovx, fovY=self.learnable_fovy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


    def rotate_at_current_location(self, R, cam_center):
        self.R = R
        self.init_quaternion = rotation_matrix_to_quaternion(torch.tensor(R).float().t().cuda())
        self.delta_quaternion = nn.Parameter(torch.zeros(4).cuda().requires_grad_(True))
        self.quaternion = self.init_quaternion + self.delta_quaternion
        self.rotation = quaternion_to_rotation_matrix(self.quaternion)
        self.last_row = torch.tensor([[0., 0., 0., 1.]]).cuda()
        self.Rt = torch.cat((self.rotation, self.translation), dim=1)
        self.world_view_transform = torch.cat((self.Rt, self.last_row), dim=0).t()

        # keep the location of camera center is the same
        c2w = self.world_view_transform.inverse()
        c2w[3, :3] = cam_center
        self.world_view_transform = c2w.inverse()
        self.T = self.world_view_transform[3, :3].cpu().detach().numpy()
        self.init_translation = torch.tensor(self.T).float().view(-1, 1).cuda()
        self.delta_translation = nn.Parameter(torch.zeros(3, 1).cuda().requires_grad_(True))
        self.translation = self.init_translation + self.delta_translation


        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.learnable_fovx, fovY=self.learnable_fovy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def reset_cam_center(self, cam_center):
        c2w = self.world_view_transform.inverse()
        c2w[3, :3] = cam_center
        self.world_view_transform = c2w.inverse()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


    def load2device(self, data_device='cuda'):
        self.original_image = self.original_image.to(data_device)
        self.Rt = self.Rt.to(data_device)
        self.translation = self.translation.to(data_device)
        self.quaternion = self.quaternion.to(data_device)
        self.world_view_transform = self.world_view_transform.to(data_device)
        self.projection_matrix = self.projection_matrix.to(data_device)
        self.full_proj_transform = self.full_proj_transform.to(data_device)
        self.camera_center = self.camera_center.to(data_device)
        self.fid = self.fid.to(data_device)

    @property
    def get_rays(self, noise_d=None, noise_o=None):
        rays_o, rays_d = get_rays(self.image_height, self.image_width, self.intrinsic_matrix, self.world_view_transform.transpose(0, 1).inverse())
        return rays_o, rays_d

    @property
    def get_K(self):
        return self.intrinsic_matrix

    @property
    def get_w2c(self):
        #self.rotation = self.lie.so3_to_SO3(self.so3) # we have error right here, but it doesn't matter
        self.quaternion = self.init_quaternion + self.delta_quaternion
        self.rotation = quaternion_to_rotation_matrix(self.quaternion)
        #self.delta_translation = torch.cat((self.delta_translation_xy, self.delta_translation_z))
        self.translation = self.init_translation + self.delta_translation
        self.Rt = torch.cat((self.rotation, self.translation), dim=1)
        self.world_view_transform = torch.cat((self.Rt, self.last_row), dim=0).t()

        return self.world_view_transform.t()


    def get_c2w(self):
        return self.get_w2c.inverse()

    def get_intrinsic(self):
        return self.projection_matrix

    def get_world_view_transform(self, global_rotation=torch.tensor([[1., 0, 0], [0, 1., 0], [0, 0, 1.]], device='cuda'), global_translation_scale=torch.tensor([1.], device='cuda')):
        self.quaternion = self.init_quaternion + self.delta_quaternion
        self.rotation = global_rotation @ quaternion_to_rotation_matrix(self.quaternion)
        self.translation = self.init_translation + self.delta_translation
        self.Rt = torch.cat((self.rotation, self.translation), dim=1)
        self.world_view_transform = torch.cat((self.Rt, self.last_row), dim=0).t()

        # scaling far/close to the center
        c2w = self.world_view_transform.inverse()
        mask = torch.ones_like(c2w)
        mask[3, :3] = global_translation_scale
        self.world_view_transform = (c2w * mask).inverse()
        return self.world_view_transform

    def get_full_proj_transform(self, global_rotation=torch.tensor([[1., 0, 0], [0, 1., 0], [0, 0, 1.]], device='cuda'), global_translation_scale=torch.tensor([1.], device='cuda')):
        self.world_view_transform = self.get_world_view_transform(global_rotation, global_translation_scale)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.learnable_fovx, fovY=self.learnable_fovy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        return self.full_proj_transform

    def get_camera_center(self, global_rotation=torch.tensor([[1., 0, 0], [0, 1., 0], [0, 0, 1.]], device='cuda'), global_translation_scale=torch.tensor([1.], device='cuda')):
        self.camera_center = self.get_world_view_transform(global_rotation, global_translation_scale).inverse()[3, :3]
        return self.camera_center


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]



def quaternion_to_rotation_matrix(quaternion):
    # Ensure quaternion is normalized
    quaternion = quaternion / torch.norm(quaternion)

    w, x, y, z = quaternion.unbind(-1)

    # Pre-compute repeated values
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz, wx, wy, wz = x * y, x * z, y * z, w * x, w * y, w * z

    # Construct rotation matrix
    R = torch.stack([
        torch.stack([1 - 2 * y2 - 2 * z2, 2 * xy - 2 * wz, 2 * xz + 2 * wy]),
        torch.stack([2 * xy + 2 * wz, 1 - 2 * x2 - 2 * z2, 2 * yz - 2 * wx]),
        torch.stack([2 * xz - 2 * wy, 2 * yz + 2 * wx, 1 - 2 * x2 - 2 * y2])
    ])

    return R

def rotation_matrix_to_quaternion(R):
    t = R.trace()
    if t > 0:
        r = torch.sqrt(1 + t)
        s = 0.5 / r
        w = 0.5 * r
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        r = torch.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        s = 0.5 / r
        w = (R[2, 1] - R[1, 2]) * s
        x = 0.5 * r
        y = (R[0, 1] + R[1, 0]) * s
        z = (R[0, 2] + R[2, 0]) * s
    elif R[1, 1] > R[2, 2]:
        r = torch.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        s = 0.5 / r
        w = (R[0, 2] - R[2, 0]) * s
        x = (R[0, 1] + R[1, 0]) * s
        y = 0.5 * r
        z = (R[1, 2] + R[2, 1]) * s
    else:
        r = torch.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        s = 0.5 / r
        w = (R[1, 0] - R[0, 1]) * s
        x = (R[0, 2] + R[2, 0]) * s
        y = (R[1, 2] + R[2, 1]) * s
        z = 0.5 * r
    return torch.tensor([w, x, y, z]).cuda()
