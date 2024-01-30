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
import sys
sys.path.append("./gaussian-splatting/")
from utils.system_utils import searchForMaxIteration
from scene_2.dataset_readers import sceneLoadTypeCallbacks
from scene_2.gaussian_model import GaussianModel
from arguments import ModelParams
from dataset_readers import SceneInfo, getNerfppNorm, storePly, CameraInfo, focal2fov, fov2focal
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.sh_utils import SH2RGB
import numpy as np
from PIL.Image import Image
from gaussian_model import BasicPointCloud
from typing import List

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        Edit this class to use zero123 images instead of Colmap images
        """
        src_path = args.source_path
        ply_path = os.path.join(src_path, "point_cloud.ply")

        self.gaussians = gaussians

        self.train_cameras = {}
        self.test_cameras = {}

        # if os.path.exists(os.path.join(args.source_path, "sparse")):
        #     scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        # elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        #     print("Found transforms_train.json file, assuming Blender data set!")
        #     scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        # else:
        #     assert False, "Could not recognize scene type!"

        #### Create scene info
        
        # we start with random points
        num_pts = 10_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
        
        train_cam_info = generateCameras(args.frontal_image, args.zero_images)

        scene_info = SceneInfo(point_cloud=pcd,
                        train_cameras=train_cam_info,
                        test_cameras=train_cam_info,
                        nerf_normalization=getNerfppNorm(train_cam_info),
                        ply_path=ply_path)

        if not self.loaded_iter:
            # Save the scene info
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(src_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(src_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # for resolution_scale in resolution_scales:
        #     print("Loading Training Cameras")
        #     self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
        #     print("Loading Test Cameras")
        #     self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        # if self.loaded_iter:
        #     self.gaussians.load_ply(os.path.join(self.model_path,
        #                                                    "point_cloud",
        #                                                    "iteration_" + str(self.loaded_iter),
        #                                                    "point_cloud.ply"))
        # else:
        self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    



def generateCameras(frontal_image: Image, zero_image: Image, fov = 0.6911112070083618):
    '''
    Generate cameras from the output of Zero-123
    '''
    images = [frontal_image]
    sub_images_width = 320
    # split the zero image into 6 images. The images are in a 2x3 grid
    for i in range(6):
        left  = (i % 2)  * sub_images_width
        right = left     + sub_images_width - 1 
        upper = (i // 2) * sub_images_width
        lower = upper + sub_images_width - 1
        images.append(zero_image.crop(box = (left, upper, right, lower)))

    cam_infos = []

    fovx = fov

    azimuths = [0] + np.linspace(30, 330, 6, endpoint=False).tolist()
    elevations = [0, -30, +20, -30, +20, -30, +20, -30]

    for idx in range(len(images)):

        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = np.linalg.inv(np.dot(
            np.dot(
                np.array([
                    [0, 0, 1],
                    [0, 1, 0],
                    [-1, 0, 0],
                ]),
                np.array([
                    [1, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0],
                ])
            ),
            np.dot(
                np.array([
                    [np.cos(np.radians(azimuths[idx])), 0, np.sin(np.radians(azimuths[idx]))],
                    [0, 1, 0],
                    [-np.sin(np.radians(azimuths[idx])), 0, np.cos(np.radians(azimuths[idx]))],
                ]),
                np.array([
                    [1, 0, 0],
                    [0, np.cos(np.radians(elevations[idx])), -np.sin(np.radians(elevations[idx]))],
                    [0, np.sin(np.radians(elevations[idx])), np.cos(np.radians(elevations[idx]))],
                ])
            )
        ))

        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image: Image = images[idx]

        # im_data = np.array(image.convert("RGBA"))

        # bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

        # norm_data = im_data / 255.0
        # arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] # + bg * (1 - norm_data[:, :, 3:4])
        # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

        fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
        FovY = fovy 
        FovX = fovx

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                        image_path=None, image_name=f"image-{idx}", width=image.size[0], height=image.size[1]))
            
    return cam_infos