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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch, ArrayToTorch
from utils.graphics_utils import fov2focal
import json
import os
from PIL import Image
from scipy.spatial.transform import Rotation

WARNED = False


def loadCam(args, id, cam_info, resolution_scale, outside_rasterizer, flow_scale, apply2gt, render_resolution, cubemap, table1):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(orig_h / (resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
        cam_info.intrinsic_matrix[0, 2] = resolution[0] / 2
        cam_info.intrinsic_matrix[1, 2] = resolution[1] / 2

    original_image = cam_info.image
    original_image_resolution = (3, resolution[1], resolution[0])
    loaded_mask = None

    ori_path = cam_info.image_path
    if 'image' in ori_path and 'indoor' not in ori_path:
        if os.path.exists(ori_path.split('images')[0] + 'fish/images' + ori_path.split('images')[1]):
            fish_gt_image = Image.open(ori_path.split('images')[0] + 'fish/images' + ori_path.split('images')[1])
            fish_gt_image_resolution = (3, fish_gt_image.size[1], fish_gt_image.size[0])
        else:
            fish_gt_image = original_image
            fish_gt_image_resolution = (3, fish_gt_image.size[1], fish_gt_image.size[0])
    elif 'indoor' in ori_path:
        fish_gt_image = Image.open(ori_path.split('images')[0] + 'fish/images/' + ori_path.split('images')[1].split('indoor_')[1])
        fish_gt_image_resolution = (3, fish_gt_image.size[1], fish_gt_image.size[0])
    else:
        fish_gt_image = original_image
        fish_gt_image_resolution = (3, fish_gt_image.size[1], fish_gt_image.size[0])

    return Camera(
        colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
        intrinsic_matrix=cam_info.intrinsic_matrix,
        FoVx=cam_info.FovX, FoVy=cam_info.FovY,
        focal_length_x=cam_info.focal_length_x, focal_length_y=cam_info.focal_length_y,
        image=original_image, gt_alpha_mask=loaded_mask,
        fish_gt_image = fish_gt_image,
        image_name=cam_info.image_name, uid=id,
        data_device=args.data_device, depth=cam_info.depth,
        ori_path=ori_path,
        outside_rasterizer=outside_rasterizer,
        orig_fov_w=orig_w,
        orig_fov_h=orig_h,
        original_image_resolution=original_image_resolution,
        fish_gt_image_resolution=fish_gt_image_resolution,
        flow_scale=flow_scale,
        apply2gt=apply2gt,
        render_resolution=render_resolution,
        cubemap=cubemap
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args, outside_rasterizer, flow_scale, apply2gt, render_resolution, cubemap, table1=False):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, outside_rasterizer, flow_scale, apply2gt, render_resolution, cubemap, table1))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id': id,
        'img_name': camera.image_name,
        'width': camera.width,
        'height': camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy': fov2focal(camera.FovY, camera.height),
        'fx': fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def camera_nerfies_from_JSON(path, scale):
    """Loads a JSON camera into memory."""
    with open(path, 'r') as fp:
        camera_json = json.load(fp)

    # Fix old camera JSON.
    if 'tangential' in camera_json:
        camera_json['tangential_distortion'] = camera_json['tangential']

    return dict(
        orientation=np.array(camera_json['orientation']),
        position=np.array(camera_json['position']),
        focal_length=camera_json['focal_length'] * scale,
        principal_point=np.array(camera_json['principal_point']) * scale,
        skew=camera_json['skew'],
        pixel_aspect_ratio=camera_json['pixel_aspect_ratio'],
        radial_distortion=np.array(camera_json['radial_distortion']),
        tangential_distortion=np.array(camera_json['tangential_distortion']),
        image_size=np.array((int(round(camera_json['image_size'][0] * scale)),
                             int(round(camera_json['image_size'][1] * scale)))),
    )

def rotate_camera(viewpoint_cam, deg_x, deg_y, deg_z):
    R = viewpoint_cam.R.T  # World-to-camera rotation matrix
    T = viewpoint_cam.T  # World-to-camera translation matrix
    camera_center_world = -np.dot(R.T, T)
    R_camera_to_world = R.T  # Inverse of the rotation matrix in world-to-camera space

    theta_x = np.deg2rad(deg_x)  # Convert degrees to radians
    theta_y = np.deg2rad(deg_y)  # Convert degrees to radians
    theta_z = np.deg2rad(deg_z)  # Convert degrees to radians
    right_camera = R_camera_to_world[:, 0]   # Right (x-axis)
    up_camera = R_camera_to_world[:, 1]      # Up (y-axis)
    forward_camera = R_camera_to_world[:, 2] # Forward (z-axis)
    Ry = Rotation.from_rotvec(theta_y * up_camera).as_matrix()
    Rx = Rotation.from_rotvec(theta_x * right_camera).as_matrix()
    Rz = Rotation.from_rotvec(theta_z * forward_camera).as_matrix()

    R_camera_to_world_new = np.dot(Rz, np.dot(Rx, np.dot(Ry, R_camera_to_world)))
    R_new = R_camera_to_world_new.T
    T_new = -np.dot(R_new, camera_center_world)

    return R_new, T_new
