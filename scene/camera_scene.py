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
import torch
import random
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_load

class CamScene:
    def __init__(self, args, gaussians, load_iteration=None, shuffle=True, h=512, w=512, aspect=-1, eval=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))
        
        if aspect != -1:
            h = 512
            w = 512 * aspect

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            if h == -1 or w == -1:
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, None, eval)
                h = scene_info.train_cameras[0].height
                w = scene_info.train_cameras[0].width
                if w > 1920:
                    scale = w / 1920
                    h /= scale
                    w /= scale
            else:
                scene_info = sceneLoadTypeCallbacks["Colmap_hw"](args.source_path, h, w, None, eval)

        else:
            assert False, "Could not recognize scene type!"
            
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.train_cameras = cameraList_load(scene_info.train_cameras, h, w)
        self.test_cameras = cameraList_load(scene_info.test_cameras, h, w)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            # torch.nn.ModuleList([self.gaussians.recolor, self.gaussians.recolor_upsample, self.gaussians.mlp_head]).load_state_dict(torch.load(os.path.join(self.model_path,
                                                        #    "point_cloud",
                                                        #    "iteration_" + str(self.loaded_iter),
                                                        #    "point_cloud.pth")))
            torch.nn.ModuleList([self.gaussians.recolor, self.gaussians.color_head, self.gaussians.color_upsample]).load_state_dict(torch.load(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.pth")))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
        
    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        
        torch.save(torch.nn.ModuleList([self.gaussians.recolor, self.gaussians.color_head, self.gaussians.color_upsample]).state_dict(), os.path.join(point_cloud_path, "point_cloud.pth"))
        # torch.save(torch.nn.ModuleList([self.gaussians.recolor, self.gaussians.recolor_upsample, self.gaussians.mlp_head]).state_dict(), os.path.join(point_cloud_path, "point_cloud.pth"))
    # def save(self, iteration):
    #     point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
    #     self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras


# class CamScene:
#     def __init__(self, source_path, h=512, w=512, aspect=-1, eval=False):
#         """b
#         :param path: Path to colmap scene main folder.
#         """
#         if aspect != -1:
#             h = 512
#             w = 512 * aspect

#         if os.path.exists(os.path.join(source_path, "sparse")):
#             if h == -1 or w == -1:
#                 scene_info = sceneLoadTypeCallbacks["Colmap"](source_path, None, eval)
#                 h = scene_info.train_cameras[0].height
#                 w = scene_info.train_cameras[0].width
#                 if w > 1920:
#                     scale = w / 1920
#                     h /= scale
#                     w /= scale
#             else:
#                 scene_info = sceneLoadTypeCallbacks["Colmap_hw"](source_path, h, w, None, eval)

#         else:
#             assert False, "Could not recognize scene type!"

#         self.cameras_extent = scene_info.nerf_normalization["radius"]
#         self.train_cameras = cameraList_load(scene_info.train_cameras, h, w)
#         self.test_cameras = cameraList_load(scene_info.test_cameras, h, w)

#     def save(self, iteration):
#         point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
#         self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

#     def getTrainCameras(self):
#         return self.train_cameras

#     def getTestCameras(self):
#         return self.test_cameras
