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
import sys
from PIL import Image
import random
import imageio
from typing import NamedTuple, Optional
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import trimesh

def load_mesh(path2obj, path2mtl, path2texture):
    Image.MAX_IMAGE_PIXELS = None
    # Load the mesh using trimesh
    mesh = trimesh.load(path2obj, force='mesh')

    # Extract vertices
    vertices = mesh.vertices

    # Extract texture coordinates
    texture_coords = mesh.visual.uv

    # Load the texture image
    texture_image = Image.open(path2texture)
    texture_pixels = np.array(texture_image)

    def get_rgb_from_uv(uv):
        u, v = uv
        h, w, _ = texture_pixels.shape
        x = int(u * (w - 1))
        y = int((1 - v) * (h - 1))
        return texture_pixels[y, x]

    # Map RGB values to vertices using texture coordinates
    vertex_colors = np.array([get_rgb_from_uv(uv) for uv in texture_coords])

    # Separate XYZ coordinates and RGB values
    xyz = vertices
    rgb = vertex_colors

    return xyz, rgb

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    focal_length_x: np.array
    focal_length_y: np.array
    intrinsic_matrix: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    depth: Optional[np.array] = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        #def mat_from_quat(quat):
        #    # quat is [w, x, y, z]
        #    from scipy.spatial.transform import Rotation as R
        #    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]]) # expects [x, y, z, w]
        #    mat = rot.as_matrix()
        #    return mat
        #R_test = mat_from_quat(extr.qvec)

        if intr.model=="SIMPLE_PINHOLE" or intr.model=="RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            intrinsic_matrix = np.array([
                [focal_length_x, 0., width * 0.5],
                [0., focal_length_x, height * 0.5],
                [0., 0., 1.]
            ], dtype=np.float32)
        elif intr.model=="PINHOLE" or intr.model=="OPENCV_FISHEYE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            intrinsic_matrix = np.array([
                [focal_length_x, 0., width * 0.5],
                [0., focal_length_y, height * 0.5],
                [0., 0., 1.]
            ], dtype=np.float32)
            #if 'netflix' in images_folder:
            #    print(intrinsic_matrix)
            #    intrinsic_matrix = np.array([
            #        [fov2focal(FovX, 1000), 0., 1000 * 0.5],
            #        [0., fov2focal(FovY, 666), 666 * 0.5],
            #        [0., 0., 1.]
            #    ], dtype=np.float32)
            #    focal_length_x = intrinsic_matrix[0][0].astype(np.float64)
            #    focal_length_y = intrinsic_matrix[1][1].astype(np.float64)
            #    width = 1000
            #    height = 666
            #    print(intrinsic_matrix)
            #    import pdb;pdb.set_trace()
            if 'colmap' in images_folder and 'eyeful' in images_folder:
                intrinsic_matrix = np.array([
                    [fov2focal(FovX, 684), 0., 684 * 0.5],
                    [0., fov2focal(FovY, 1024), 1024 * 0.5],
                    [0., 0., 1.]
                ], dtype=np.float32)
                focal_length_x = intrinsic_matrix[0][0].astype(np.float64)
                focal_length_y = intrinsic_matrix[1][1].astype(np.float64)
                width = 684
                height = 1024
        elif intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            intrinsic_matrix = np.array([
                [focal_length_x, 0., width * 0.5],
                [0., focal_length_x, height * 0.5],
                [0., 0., 1.]
            ], dtype=np.float32)
        elif intr.model == "FULL_OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            cx = intr.params[2]
            cy = intr.params[3]
            intrinsic_matrix = np.array([
                [focal_length_x, 0., width * 0.5],
                [0., focal_length_x, height * 0.5],
                [0., 0., 1.]
            ], dtype=np.float32)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        # depth = imageio.imread(depth_name)

        if focal_length_y != None:
            cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, focal_length_x=focal_length_x, focal_length_y=focal_length_y, image=image, intrinsic_matrix=intrinsic_matrix,                              image_path=image_path, image_name=image_name, width=width, height=height)
            cam_infos.append(cam_info)
        else:
            cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, focal_length_x=focal_length_x, focal_length_y=focal_length_y, image=image, intrinsic_matrix=intrinsic_matrix,                              image_path=image_path, image_name=image_name, width=width, height=height)
            cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    #normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    #return BasicPointCloud(points=positions, colors=colors, normals=normals)
    return BasicPointCloud(points=positions, colors=colors, normals=None)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8, init_type="sfm", num_pts=100000):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        #cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file.split('sparse')[0] + 'fish/sparse' + cameras_intrinsic_file.split('sparse')[1])
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    #items = list(cam_extrinsics.items())
    #items = items[:2]
    #cam_extrinsics = dict(items)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if 'colmap' in cameras_extrinsic_file and 'eyeful' in cameras_extrinsic_file:
        llffhold = 24
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = cam_infos[:5]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if init_type == "sfm":
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        ply_dense_path = os.path.join(path, "sparse/0/fused.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
    elif init_type == "random":
        ply_path = os.path.join(path, "random.ply")
        print(f"Generating random point cloud ({num_pts})...")

        xyz = np.random.random((num_pts, 3)) * nerf_normalization["radius"]* 3*2 -(nerf_normalization["radius"]*3)

        num_pts = xyz.shape[0]
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        print("Please specify a correct init_type: random or sfm")
        exit(0)

    try:
        if os.path.exists(ply_dense_path):
            pcd = fetchPly(ply_dense_path)
        else:
            pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromVRNeRF(path, transformsfile, white_background, extension=".jpg"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        frames = contents['KRT']

        #random.shuffle(frames)
        #frames = frames[:33]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["cameraId"] + '.jpg')
            #if '0_REN' in cam_name: continue

            intrinsic_matrix = np.array(frame['K']).transpose()
            #if '2_REN' in cam_name:
            #    intrinsic_matrix[0][0] = 2877.69784173
            #    intrinsic_matrix[1][1] = 2877.69784173
            FovX = focal2fov(intrinsic_matrix[0][0], intrinsic_matrix[0][2] * 2)
            FovY = focal2fov(intrinsic_matrix[1][1], intrinsic_matrix[1][2] * 2)
            w2c = np.array(frame['T']).transpose()
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")


            intrinsic_matrix = np.array([
                [fov2focal(FovX, image.size[0]), 0., image.size[0] * 0.5],
                [0., fov2focal(FovY, image.size[1]), image.size[1] * 0.5],
                [0., 0., 1.]
            ], dtype=np.float32)
            focal_length_x = intrinsic_matrix[0][0].astype(np.float64)
            focal_length_y = intrinsic_matrix[1][1].astype(np.float64)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, intrinsic_matrix=intrinsic_matrix, focal_length_x=focal_length_x, focal_length_y=focal_length_y, image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))

    return cam_infos

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        #if len(frames) > 500:
        #    random.shuffle(frames)
        #    frames = frames[:300]

        for idx, frame in enumerate(frames):
            if 'jpg' in frame["file_path"] or 'png' in frame["file_path"]:
                cam_name = os.path.join(path, frame["file_path"])
            else:
                cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            if contents["type"] == "mitsuba":
                c2w[:3, 0:2] *= -1
            else:
                c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            intrinsic_matrix = np.array([
                [fov2focal(FovX, image.size[0]), 0., image.size[0] * 0.5],
                [0., fov2focal(FovY, image.size[1]), image.size[1] * 0.5],
                [0., 0., 1.]
            ], dtype=np.float32)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, focal_length_x=fov2focal(FovX, image.size[0]), focal_length_y=fov2focal(FovY, image.size[1]), image=image, intrinsic_matrix=intrinsic_matrix, image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))

    return cam_infos


def readMetashapeInfo(path, white_background, eval, extension=".png", init_type="sfm", num_pts=100000):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromVRNeRF(path, "cameras.json", white_background, extension)
    print(f"Numer of training cameras: {len(train_cam_infos)}")
    print("Reading Test Transforms")
    test_cam_infos = train_cam_infos.copy()
    random.shuffle(test_cam_infos)
    test_cam_infos = test_cam_infos[:5]

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if init_type == "sfm":
        ply_path = os.path.join(path, "points3d.ply")
        if not os.path.exists(ply_path) or True:
            xyz, rgb = load_mesh(os.path.join(path, 'mesh.obj'), os.path.join(path, 'mesh.mtl'), os.path.join(path, 'mesh.jpg'))
            pcd = BasicPointCloud(points=xyz, colors=rgb, normals=np.zeros((len(xyz), 3)))
            storePly(ply_path, xyz, rgb)
    elif init_type == "random":
        ply_path = os.path.join(path, "random.ply")
        print(f"Generating random point cloud ({num_pts})...")

        xyz = np.random.random((num_pts, 3)) * nerf_normalization["radius"]* 3*2 -(nerf_normalization["radius"]*3)

        num_pts = xyz.shape[0]
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        print("Please specify a correct init_type: random or sfm")
        exit(0)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readNerfSyntheticInfo(path, white_background, eval, extension=".png", init_type="sfm", num_pts=100000, table1=False):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    if not table1:
        test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    elif table1:
        test_cam_infos = readCamerasFromTransforms(path, "transforms_table1.json", white_background, extension)
    else:
        print("you should have table1 json for evaluation")
        exit()

    if not eval:
        train_cam_infos.extend(test_cam_infos)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if init_type == "sfm":
        ply_path = os.path.join(path, "points3d.ply")
        if not os.path.exists(ply_path):
            # Since this data set has no colmap data, we start with random points
            num_pts = 100_000

            if os.path.exists(os.path.join(path, "one_mesh.ply")):
                mesh = trimesh.load(os.path.join(path, "one_mesh.ply"), force='mesh')
                vertices = mesh.vertices
                if len(vertices) > 1000000:
                    subset_indices = np.random.choice(vertices.shape[0], size=1000000, replace=False)
                    vertices = vertices[subset_indices]
                num_pts = len(vertices)
                xyz = vertices
            elif os.path.exists(os.path.join(path, "models")):
                obj_dir = os.path.join(path, "models")
                vertices_list = []
                colors_list = []
                default_color = [255, 255, 255]
                for filename in os.listdir(obj_dir):
                    #if 'Mesh024' in filename or 'Mesh021' in filename and 'lamp' in filename:
                    if 'Mesh020' in filename or 'Mesh021' in filename and 'lamp' in filename:
                        continue
                    if filename.endswith('.obj'):
                        obj_path = os.path.join(obj_dir, filename)
                        mesh = trimesh.load(obj_path)
                        vertices = mesh.vertices
                        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                            colors = mesh.visual.vertex_colors[:, :3]  # Take only RGB (ignore alpha if present)
                        elif hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
                            colors = np.tile(mesh.visual.face_colors[:3], (vertices.shape[0], 1))  # Apply face color to all vertices
                        else:
                            colors = np.tile(default_color, (vertices.shape[0], 1))
                        vertices_list.append(vertices)
                        colors_list.append(colors)

                xyz = np.concatenate(vertices_list, axis=0)
                num_pts = len(xyz)
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
            print(f"Loading point cloud ({num_pts})...")
            storePly(ply_path, xyz, SH2RGB(shs) * 255)
    elif init_type == "random":
        ply_path = os.path.join(path, "random.ply")
        print(f"Generating random point cloud ({num_pts})...")

        xyz = np.random.random((num_pts, 3)) * nerf_normalization["radius"]* 3*2 -(nerf_normalization["radius"]*3)

        num_pts = xyz.shape[0]
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        print("Please specify a correct init_type: random or sfm")
        exit(0)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Metashape" : readMetashapeInfo,
}
