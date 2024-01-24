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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.specular_model import SpecularModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.camera import Lie
import torch

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], random_init=False, r_t_noise=[0., 0.]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.lie = Lie()

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.unnoisy_train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        self.cameras_torch = []
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")

            self.unnoisy_train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            # simply add noise
            so3_noise = torch.randn(len(scene_info.train_cameras), 3) * r_t_noise[0]
            t_noise = (torch.randn(len(scene_info.train_cameras), 3) * r_t_noise[1]).numpy()
            # apply global transformation and rotation
            # so3_noise = torch.tensor([3.1415926 / 4, 0., 0.])[None, :].repeat(len(scene_info.train_cameras), 1)
            #t_noise = torch.tensor([0., 0., 1.])[None, :].repeat(len(scene_info.train_cameras), 1).numpy()

            so3 = self.lie.so3_to_SO3(so3_noise).cpu().detach().numpy()
            for index in range(len(scene_info.train_cameras)):
                tmp_R = so3[index] @ scene_info.train_cameras[index].R
                #tmp_T = so3[index] @ scene_info.train_cameras[index].T + t_noise[index]
                #tmp_T = so3[index] @ (scene_info.train_cameras[index].T + t_noise[index])
                tmp_T = (scene_info.train_cameras[index].T + t_noise[index])
                scene_info.train_cameras[index] = scene_info.train_cameras[index]._replace(T=tmp_T, R=tmp_R)
                #import pdb; pdb.set_trace()
                #print(scene_info.train_cameras[index])
            #import pdb; pdb.set_trace()
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, random_init=random_init)

        # naive implementation to deal with pose noise
        # set camera parameters as learnbale parameters
        l = []
        for camera in self.train_cameras[resolution_scale]:
            #l.append({'params': camera.quaternion, 'lr': 0.001})
            l.append({'params': camera.so3, 'lr': 0.01})
            l.append({'params': camera.translation, 'lr': 0.01})
        #l = [{'params': camera.parameters(), 'lr': 0.01} for camera in self.train_cameras[resolution_scale]]
        self.optimizer = torch.optim.Adam(l, eps=1e-15)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30000, 50000], gamma=1.)


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def get_unnoisy_TrainCameras(self, scale=1.0):
        return self.unnoisy_train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
