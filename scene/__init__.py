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
from scene.cameras import quaternion_to_rotation_matrix, rotation_matrix_to_quaternion
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.camera import Lie
import torch
from torch import nn
from easydict import EasyDict as edict

def procrustes_analysis(X0,X1): # [N,3]
    # translation
    x0_x1 = X0 - X1
    X0 = X0[~(x0_x1 > 1).any(dim=1)]
    X1 = X1[~(x0_x1 > 1).any(dim=1)]
    t0 = X0.mean(dim=0,keepdim=True)
    t1 = X1.mean(dim=0,keepdim=True)
    X0c = X0-t0
    X1c = X1-t1
    # scale
    s0 = (X0c**2).sum(dim=-1).mean().sqrt()
    s1 = (X1c**2).sum(dim=-1).mean().sqrt()
    X0cs = X0c/s0
    X1cs = X1c/s1
    # rotation (use double for SVD, float loses precision)
    U,S,V = (X0cs.t()@X1cs).double().svd(some=True)
    R = (U@V.t()).float()
    if R.det()<0: R[2] *= -1
    # align X1 to X0: X1to0 = (X1-t1)/s1@R.t()*s0+t0
    sim3 = edict(t0=t0[0],t1=t1[0],s0=s0,s1=s1,R=R)
    return sim3

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], random_init=False, r_t_noise=[0., 0.], r_t_lr=[0.001, 0.001], global_alignment_lr=0.001):
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
        self.unnoisy_test_cameras = {}
        self.debug_cameras = {}
        self.test_cameras = {}
        self.global_translation = nn.Parameter(torch.tensor([1.]).cuda().requires_grad_(True))
        self.global_quaternion = nn.Parameter(torch.tensor([1., 0, 0, 0]).cuda().requires_grad_(True))
        self.global_rotation = quaternion_to_rotation_matrix(self.global_quaternion)
        self.sim3 = None

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train_diffusion.json")):
            print("Found transforms_train_diffusion.json file!")
            scene_info = sceneLoadTypeCallbacks["Diffusion"](args.source_path, args.white_background, args.eval, extension=".jpg")
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

            generator = torch.Generator().manual_seed(55)
            self.unnoisy_train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            self.debug_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            # simply add noise
            so3_noise = torch.randn((len(scene_info.train_cameras), 3), generator=generator) * r_t_noise[0]
            t_noise = (torch.randn((len(scene_info.train_cameras), 3), generator=generator) * r_t_noise[1]).numpy()
            #so3_noise = torch.randn((len(scene_info.train_cameras), 3)) * r_t_noise[0]
            #t_noise = (torch.randn((len(scene_info.train_cameras), 3)) * r_t_noise[1]).numpy()

            # add systematic rotation and translation noise to verify preAlignment function
            #so3_noise = torch.ones(len(scene_info.train_cameras), 3) * r_t_noise[0]
            #t_noise = (torch.ones(len(scene_info.train_cameras), 3) * r_t_noise[1]).numpy()

            # apply global transformation and rotation
            # so3_noise = torch.tensor([3.1415926 / 4, 0., 0.])[None, :].repeat(len(scene_info.train_cameras), 1)
            #t_noise = torch.tensor([0., 0., 1.])[None, :].repeat(len(scene_info.train_cameras), 1).numpy()

            so3 = self.lie.so3_to_SO3(so3_noise).cpu().detach().numpy()
            for index in range(len(scene_info.train_cameras)):
                tmp_R = so3[index] @ scene_info.train_cameras[index].R
                #tmp_T = so3[index] @ scene_info.train_cameras[index].T + t_noise[index]
                #tmp_T = so3[index] @ (scene_info.train_cameras[index].T + t_noise[index])
                tmp_T = scene_info.train_cameras[index].T + t_noise[index]
                scene_info.train_cameras[index] = scene_info.train_cameras[index]._replace(T=tmp_T, R=tmp_R)
                #import pdb; pdb.set_trace()
                #print(scene_info.train_cameras[index])
            #import pdb; pdb.set_trace()
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            self.unnoisy_test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, random_init=random_init)

        # naive implementation to deal with pose noise
        # set camera parameters as learnbale parameters
        l_rotation = [{'params': camera.delta_quaternion, 'lr': r_t_lr[0]} for camera in self.train_cameras[resolution_scale]]
        l_rotation_test = [{'params': camera.delta_quaternion, 'lr': 0.0005} for camera in self.test_cameras[resolution_scale]]
        #l_rotation = [{'params': camera.quaternion, 'lr': r_t_lr[0]} for camera in self.train_cameras[resolution_scale]]
        #l_rotation = [{'params': camera.so3, 'lr': 0.01} for camera in self.train_cameras[resolution_scale]]

        l_translation = [{'params': camera.delta_translation, 'lr': r_t_lr[1]} for camera in self.train_cameras[resolution_scale]]
        l_translation_test = [{'params': camera.delta_translation, 'lr': 0.0025} for camera in self.test_cameras[resolution_scale]]
        #l_translation = [{'params': camera.delta_translation_xy, 'lr': r_t_lr[1]} for camera in self.train_cameras[resolution_scale]]
        #l_translation = [{'params': camera.delta_translation_z, 'lr': r_t_lr[1]} for camera in self.train_cameras[resolution_scale]]
        #l_translation = [{'params': camera.translation, 'lr': r_t_lr[1]} for camera in self.train_cameras[resolution_scale]]
        #l = [{'params': camera.parameters(), 'lr': 0.01} for camera in self.train_cameras[resolution_scale]]


        l_fovx = [{'params': camera.learnable_fovx, 'lr': 0.01} for camera in self.train_cameras[resolution_scale]]
        l_fovy = [{'params': camera.learnable_fovy, 'lr': 0.01} for camera in self.train_cameras[resolution_scale]]
        self.optimizer_fovx = torch.optim.Adam(l_fovx, eps=1e-15)
        self.optimizer_fovy = torch.optim.Adam(l_fovy, eps=1e-15)
        self.scheduler_fovx = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_fovx, milestones=[10000], gamma=1.)
        self.scheduler_fovy = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_fovy, milestones=[10000], gamma=1.)
        l_fovx_test = [{'params': camera.learnable_fovx, 'lr': 0.001} for camera in self.test_cameras[resolution_scale]]
        l_fovy_test = [{'params': camera.learnable_fovy, 'lr': 0.001} for camera in self.test_cameras[resolution_scale]]
        self.optimizer_fovx_test = torch.optim.Adam(l_fovx_test, eps=1e-15)
        self.optimizer_fovy_test = torch.optim.Adam(l_fovy_test, eps=1e-15)
        self.scheduler_fovx_test = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_fovx_test, milestones=[10000], gamma=1.)
        self.scheduler_fovy_test = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_fovy_test, milestones=[10000], gamma=1.)

        self.optimizer_rotation = torch.optim.Adam(l_rotation, eps=1e-15)
        self.optimizer_translation = torch.optim.Adam(l_translation, eps=1e-15)
        self.scheduler_rotation = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_rotation, milestones=[10000], gamma=1)
        self.scheduler_translation = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_translation, milestones=[10000], gamma=1)

        self.optimizer_rotation_test = torch.optim.Adam(l_rotation_test, eps=1e-15)
        self.optimizer_translation_test = torch.optim.Adam(l_translation_test, eps=1e-15)
        self.scheduler_rotation_test = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_rotation_test, milestones=[30000], gamma=0.5)
        self.scheduler_translation_test = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_translation_test, milestones=[30000], gamma=0.5)

        l_global_alignment = [{'params': self.global_translation, 'lr': global_alignment_lr}, {'params': self.global_quaternion, 'lr': global_alignment_lr}]
        self.optimizer_global_alignment = torch.optim.Adam(l_global_alignment, eps=1e-15)
        self.scheduler_global_aligment = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_rotation, milestones=[7000, 50000], gamma=1)

        #self.test_cameras[resolution_scale] = self.train_cameras[resolution_scale][90:]
        #self.train_cameras[resolution_scale] = self.train_cameras[resolution_scale][:90]
        #self.unnoisy_train_cameras[resolution_scale] = self.unnoisy_train_cameras[resolution_scale][:90]

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def get_unnoisy_TrainCameras(self, scale=1.0):
        return self.unnoisy_train_cameras[scale]

    def get_unnoisy_TestCameras(self, scale=1.0):
        return self.unnoisy_test_cameras[scale]

    def getTestCameras(self, scale=1.0):
        #if self.sim3 != None:
        #    sim3 = self.sim3
        #    center_pred = torch.stack([camera.get_camera_center() for camera in self.getTrainCameras()], dim=0) # [N,3]
        #    center_GT = torch.stack([camera.get_camera_center() for camera in self.get_unnoisy_TrainCameras()], dim=0) # [N,3]
        #    try:
        #        sim3 = procrustes_analysis(center_GT,center_pred)
        #        self.sim3 = sim3
        #    except:
        #        print("warning: SVD did not converge...")
        #        sim3 = edict(t0=0,t1=0,s0=1,s1=1,R=torch.eye(3,device='cuda'))
        #        self.sim3 = sim3
        #    center_pred2gt = (center_pred - sim3.t1) / sim3.s1@sim3.R.t()*sim3.s0 + sim3.t0
        #    center_gt2pred = (center_GT - sim3.t0) / sim3.s0@sim3.R*sim3.s1 + sim3.t1

        #    R_pred = torch.stack([camera.get_w2c[:3, :3].t() for camera in self.getTrainCameras()], dim=0)
        #    R_gt = torch.stack([camera.get_w2c[:3, :3].t() for camera in self.get_unnoisy_TrainCameras()], dim=0)
        #    t_pred = torch.stack([camera.get_w2c[:3, 3:] for camera in self.getTrainCameras()], dim=0)
        #    t_gt = torch.stack([camera.get_w2c[:3, 3:] for camera in self.get_unnoisy_TrainCameras()], dim=0)

        #    R_pred2gt = R_pred @ sim3.R.t()
        #    R_gt2pred = R_gt @ sim3.R
        #    # we can use this to find t
        #    t_pred2gt = (-R_pred2gt.transpose(1, 2) @ center_pred2gt[..., None])
        #    t_gt2pred = (-R_gt2pred.transpose(1, 2) @ center_gt2pred[..., None])
        #    self.unnoisy_train_cameras[scale][0].get_w2c[:3, :3]
        #    self.train_cameras[scale][0].get_w2c[:3, :3]
        #    copy_gt2pred = self.debug_cameras[scale]
        #    #for camera, R, t in zip(copy_gt2pred, R_gt2pred, t_gt2pred):
        #    for camera, R, t in zip(copy_gt2pred, R_pred, t_pred):
        #        camera.reset_extrinsic(R.cpu().numpy(), t.cpu().numpy())


        #return copy_gt2pred
        return self.test_cameras[scale]

    def getGlobalAlignment(self):
        self.global_rotation = quaternion_to_rotation_matrix(self.global_quaternion)
        return self.global_rotation, self.global_translation

    @torch.no_grad()
    def visTestCameras(self):
        R_pred = torch.stack([camera.get_w2c[:3, :3].t() for camera in self.getTestCameras()], dim=0)
        R_gt = torch.stack([camera.get_w2c[:3, :3].t() for camera in self.get_unnoisy_TestCameras()], dim=0)
        t_pred = torch.stack([camera.get_w2c[:3, 3:] for camera in self.getTestCameras()], dim=0)
        t_gt = torch.stack([camera.get_w2c[:3, 3:] for camera in self.get_unnoisy_TestCameras()], dim=0)

        return torch.cat((R_gt.transpose(1, 2), t_gt), dim=-1), torch.cat((R_pred.transpose(1, 2), t_pred), dim=-1)

    @torch.no_grad()
    def loadAlignCameras(self, if_vis_train=False, if_vis_test=False, camera_uid_list=[], iteration=0, path=''):
        ##############################################################################################################
        if if_vis_test:
            self.train_cameras = torch.load(os.path.join(path, 'opt_cams.pt'))
            self.unnoisy_train_cameras = torch.load(os.path.join(path, 'gt_cams.pt'))

        center_pred = torch.stack([camera.get_camera_center() for camera in self.getTrainCameras()], dim=0) # [N,3]
        center_GT = torch.stack([camera.get_camera_center() for camera in self.get_unnoisy_TrainCameras()], dim=0) # [N,3]
        try:
            sim3 = procrustes_analysis(center_GT,center_pred)
            self.sim3 = sim3
        except:
            print("warning: SVD did not converge...")
            sim3 = edict(t0=0,t1=0,s0=1,s1=1,R=torch.eye(3,device='cuda'))
            self.sim3 = sim3
        if if_vis_train:
            center_pred2gt = (center_pred - sim3.t1) / sim3.s1@sim3.R.t()*sim3.s0 + sim3.t0
            center_gt2pred = (center_GT - sim3.t0) / sim3.s0@sim3.R*sim3.s1 + sim3.t1

            R_pred = torch.stack([camera.get_w2c[:3, :3].t() for camera in self.getTrainCameras()], dim=0)
            R_gt = torch.stack([camera.get_w2c[:3, :3].t() for camera in self.get_unnoisy_TrainCameras()], dim=0)
            t_pred = torch.stack([camera.get_w2c[:3, 3:] for camera in self.getTrainCameras()], dim=0)
            t_gt = torch.stack([camera.get_w2c[:3, 3:] for camera in self.get_unnoisy_TrainCameras()], dim=0)

            R_pred2gt = R_pred @ sim3.R.t()
            R_gt2pred = R_gt @ sim3.R
            # we can use this to find t
            t_pred2gt = (-R_pred2gt.transpose(1, 2) @ center_pred2gt[..., None])
            t_gt2pred = (-R_gt2pred.transpose(1, 2) @ center_gt2pred[..., None])
            if camera_uid_list != []:
                mask = [True if camera.uid in camera_uid_list else False for camera in self.getTrainCameras()]
                return torch.cat((R_gt.transpose(1, 2), t_gt), dim=-1)[mask], torch.cat((R_pred2gt.transpose(1, 2), t_pred2gt), dim=-1)[mask]
            return torch.cat((R_gt.transpose(1, 2), t_gt), dim=-1), torch.cat((R_pred2gt.transpose(1, 2), t_pred2gt), dim=-1)
        ##############################################################################################################

        if if_vis_test and False:
            center_GT = torch.stack([camera.get_camera_center() for camera in self.getTestCameras()], dim=0) # [N,3]
            center_gt2pred = (center_GT - sim3.t0) / sim3.s0@sim3.R*sim3.s1 + sim3.t1
            R_gt = torch.stack([camera.get_w2c[:3, :3].t() for camera in self.getTestCameras()], dim=0)
            t_gt = torch.stack([camera.get_w2c[:3, 3:] for camera in self.getTestCameras()], dim=0)

            R_gt2pred = R_gt @ sim3.R
            t_gt2pred = (-R_gt2pred.transpose(1, 2) @ center_gt2pred[..., None])
            for R, t, camera in zip(R_gt2pred, t_gt2pred, self.test_cameras[1.]):
                camera.reset_extrinsic(R.cpu().numpy(), t.cpu().numpy())

