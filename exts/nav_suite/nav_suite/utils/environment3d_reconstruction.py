# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cv2
import numpy as np
import open3d as o3d
import os
import scipy.spatial.transform as tf
from tqdm import tqdm

import omni.log

from .environment3d_reconstruction_cfg import ReconstructionCfg


class EnvironmentReconstruction:
    """
    Reconstruct 3D Map with depth and optionally semantic images, assumes the ground truth camera odom is known
    Expects following datastructure:

    - cfg.data_dir
        - camera_poses.txt  (format: x y z qw qx qy qz)
        - cfg.depth_cam_name
            - intrinsics.txt
            - distance_to_image_plane
                - xxxx.png  (images should be named with 4 digits, e.g. 0000.png, 0001.png, etc.)
                - xxxx.npy  (arrays should be named with 4 digits, e.g. 0000.npy, 0001.npy, etc.)
        - cfg.sem_cam_name  (optional)
            - intrinsics.txt
            - semantic_segmentation
                - xxxx.png  (images should be named with 4 digits, e.g. 0000.png, 0001.png, etc., RGB images)

    """

    debug = False

    def __init__(self, cfg: ReconstructionCfg):
        # get config
        self._cfg: ReconstructionCfg = cfg
        # read camera params and odom
        self._read_intrinsic()
        self._read_extrinsic()
        # control flag if point-cloud has been loaded
        self._is_constructed = False

        # variables
        self._pcd: o3d.geometry.PointCloud = None

        omni.log.info("Ready to read depth data.")

    ###
    # Operations
    ###

    def depth_reconstruction(self):
        # identify start and end image idx for the reconstruction
        N = len(self.extrinsics)
        self._end_idx = min(self._cfg.max_images, N) if self._cfg.max_images is not None else N

        if self._cfg.point_cloud_batch_size > self._end_idx:
            omni.log.warn(
                "batch size must be smaller or equal than number of images to reconstruct, now set to max value"
                f" {self._end_idx}"
            )
            self._cfg.point_cloud_batch_size = self._end_idx

        omni.log.info(f"total number of images for reconstruction: {int(self._end_idx)}")

        # get pixel tensor for reprojection
        pixels = self._computePixelTensor()

        # init point-cloud
        self._pcd = o3d.geometry.PointCloud()  # point size (n, 3)
        first_batch = True

        # init lists
        points_all = []
        if self._cfg.semantics:
            sem_map_all = []

        for img_counter, img_idx in enumerate(
            tqdm(
                range(self._end_idx),
                desc="Reconstructing 3D Points",
            )
        ):
            im = self._load_depth_image(img_idx)

            # project points in world frame
            rot = tf.Rotation.from_quat(self.extrinsics[img_idx][3:]).as_matrix()
            points = im.reshape(-1, 1) * (rot @ pixels.T).T
            # filter points with 0 depth --> otherwise obstacles at camera position
            non_zero_idx = np.where(points.any(axis=1))[0]

            points_final = points[non_zero_idx] + self.extrinsics[img_idx][:3]

            if self._cfg.semantics:
                sem_annotation, filter_idx = self._get_semantic_image(points_final, img_idx)
                points_all.append(points_final[filter_idx])
                sem_map_all.append(sem_annotation)
            else:
                points_all.append(points_final)

            # update point cloud
            if img_counter % self._cfg.point_cloud_batch_size == 0:
                omni.log.info(f"Updating open3d point cloud with {self._cfg.point_cloud_batch_size} images ...")

                if first_batch:
                    self._pcd.points = o3d.utility.Vector3dVector(np.vstack(points_all))
                    if self._cfg.semantics:
                        self._pcd.colors = o3d.utility.Vector3dVector(np.vstack(sem_map_all) / 255.0)
                    first_batch = False

                else:
                    self._pcd.points.extend(np.vstack(points_all))
                    if self._cfg.semantics:
                        self._pcd.colors.extend(np.vstack(sem_map_all) / 255.0)

                # reset buffer lists
                del points_all
                points_all = []
                if self._cfg.semantics:
                    del sem_map_all
                    sem_map_all = []

                # apply downsampling
                omni.log.info(f"Downsampling point cloud with voxel size {self._cfg.voxel_size} ...")
                self._pcd = self._pcd.voxel_down_sample(self._cfg.voxel_size)

        # add last batch
        if len(points_all) > 0:
            omni.log.info("Updating open3d geometry point cloud with last images ...")
            self._pcd.points.extend(np.vstack(points_all))
            points_all = None
            if self._cfg.semantics:
                self._pcd.colors.extend(np.vstack(sem_map_all) / 255.0)
                sem_map_all = None

            # apply downsampling
            omni.log.info(f"Downsampling point cloud with voxel size {self._cfg.voxel_size} ...")
            self._pcd = self._pcd.voxel_down_sample(self._cfg.voxel_size)

        # update flag
        self._is_constructed = True
        omni.log.info("Construction completed.")

        return

    def show_pcd(self):
        if not self._is_constructed:
            omni.log.warn("No reconstructed cloud")
            return
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=np.min(np.asarray(self._pcd.points), axis=0)
        )
        o3d.visualization.draw_geometries([self._pcd, origin], mesh_show_wireframe=True)  # visualize point cloud
        return

    def save_pcd(self, save_path: str | None = None):
        if not self._is_constructed:
            omni.log.warn("Save points failed, no reconstructed cloud!")

        save_path = save_path if save_path is not None else os.path.join(self._cfg.data_dir)
        omni.log.info("save output files to: " + save_path)

        # save clouds
        o3d.io.write_point_cloud(os.path.join(save_path, "cloud.ply"), self._pcd)
        omni.log.info("Saved point cloud to ply file.")

    @property
    def pcd(self):
        return self._pcd

    ###
    # Helper functions
    ###

    def _read_extrinsic(self):
        """Read the camera extrinsic parameters from file.

        The extrinsic parameters are stored in a text file with the following format: x y z qw qx qy qz and are
        converted here to x y z qx qy qz qw format."""
        self.extrinsics = np.loadtxt(self._cfg.data_dir + "/camera_poses.txt", delimiter=",")

        # modify quaternion to be in the order of x y z w using by scipy
        self.extrinsics[:, 3:] = self.extrinsics[:, [4, 5, 6, 3]]

    def _read_intrinsic(self):
        """Read the camera intrinsic parameters from file."""
        self.K_depth = np.loadtxt(
            os.path.join(self._cfg.data_dir, self._cfg.depth_cam_name, "intrinsics.txt"), delimiter=","
        )
        if self._cfg.semantics:
            self.K_sem = np.loadtxt(
                os.path.join(self._cfg.data_dir, self._cfg.semantic_cam_name, "intrinsics.txt"), delimiter=","
            )

    def _load_depth_image(self, idx: int) -> np.ndarray:
        """Load depth image from file."""

        # get path to images
        img_path = os.path.join(
            self._cfg.data_dir, self._cfg.depth_cam_name, "distance_to_image_plane", str(idx).zfill(4)
        )

        if os.path.isfile(img_path + ".npy"):
            img_array = np.load(img_path + ".npy") / self._cfg.depth_scale
        elif os.path.isfile(img_path + ".png"):
            img_array = cv2.imread(img_path + ".png", cv2.IMREAD_ANYDEPTH) / self._cfg.depth_scale
        else:
            raise FileNotFoundError(f"Depth image {img_path} not found.")

        # set invalid depth values to 0
        img_array[~np.isfinite(img_array)] = 0
        return img_array

    def _computePixelTensor(self):
        depth_img = self._load_depth_image(0)

        # get image plane mesh grid
        pix_u = np.arange(0, depth_img.shape[1])
        pix_v = np.arange(0, depth_img.shape[0])
        grid = np.meshgrid(pix_u, pix_v)
        pixels = np.vstack(list(map(np.ravel, grid))).T
        pixels = np.hstack([pixels, np.ones((len(pixels), 1))])  # add ones for 3D coordinates

        # transform to camera frame
        k_inv = np.linalg.inv(self.K_depth)
        pix_cam_frame = np.matmul(k_inv, pixels.T)
        # reorder to be in "robotics" axis order (x forward, y left, z up)
        return pix_cam_frame[[2, 0, 1], :].T * np.array([1, -1, -1])

    def _get_semantic_image(self, points, idx):
        # load semantic image and pose
        img_path = os.path.join(
            self._cfg.data_dir, self._cfg.semantic_cam_name, "semantic_segmentation", str(idx).zfill(4) + ".png"
        )

        assert os.path.isfile(img_path), f"Semantic image {img_path} not found."
        sem_image = cv2.imread(img_path)  # loads in bgr order
        sem_image = cv2.cvtColor(sem_image, cv2.COLOR_BGR2RGB)
        pose_sem = self.extrinsics[idx]
        # transform points to semantic camera frame
        points_sem_cam_frame = (tf.Rotation.from_quat(pose_sem[3:]).as_matrix().T @ (points - pose_sem[:3]).T).T
        # normalize points
        points_sem_cam_frame_norm = points_sem_cam_frame / points_sem_cam_frame[:, 0][:, np.newaxis]
        # reorder points be camera convention (z-forward)
        points_sem_cam_frame_norm = points_sem_cam_frame_norm[:, [1, 2, 0]] * np.array([-1, -1, 1])
        # transform points to pixel coordinates
        pixels = (self.K_sem @ points_sem_cam_frame_norm.T).T
        # filter points outside of image
        filter_idx = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < sem_image.shape[1])
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < sem_image.shape[0])
        )
        # get semantic annotation
        sem_annotation = sem_image[
            pixels[filter_idx, 1].astype(int),
            pixels[filter_idx, 0].astype(int),
        ]
        # remove all pixels that have no semantic annotation
        non_classified_idx = np.all(sem_annotation == [0, 0, 0], axis=1)
        sem_annotation = sem_annotation[~non_classified_idx]
        filter_idx[np.where(filter_idx)[0][non_classified_idx]] = False

        return sem_annotation, filter_idx
