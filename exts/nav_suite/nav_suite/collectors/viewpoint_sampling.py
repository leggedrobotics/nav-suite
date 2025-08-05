# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import cv2
import numpy as np
import open3d as o3d
import os
import pickle
import random
import time
import torch

import isaaclab.utils.math as math_utils
import omni.log
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import Camera, RayCasterCamera
from isaaclab.sim import SimulationContext

from .viewpoint_sampling_cfg import ViewpointSamplingCfg


class ViewpointSampling:
    def __init__(self, cfg: ViewpointSamplingCfg, scene: InteractiveScene):
        # save cfg and env
        self.cfg = cfg
        self.scene = scene

        # get sim context
        self.sim = SimulationContext.instance()

        # analyse terrains -- check if singleton is used and available
        if (
            hasattr(self.cfg.terrain_analysis.class_type, "instance")
            and self.cfg.terrain_analysis.class_type.instance() is not None
        ):
            self.terrain_analyser = self.cfg.terrain_analysis.class_type.instance()
        else:
            self.terrain_analyser = self.cfg.terrain_analysis.class_type(self.cfg.terrain_analysis, scene=self.scene)

    ###
    # Properties
    ###

    @property
    def samples(self) -> torch.Tensor | list[torch.Tensor]:
        """Get the sampled viewpoints.

        Samples are stored in a torch tensor with the structure
        [x, y, z, qw, qx, qv, qz]
        """
        return self._samples

    @property
    def sliced_bounding_boxes(self) -> list[tuple[float, float, float, float]] | None:
        """Get the sliced bounding boxes.

        The sliced bounding boxes follow the terrain analysis format of the mesh dimensions, i.e. [x_max, y_max, x_min, y_min]
        """
        if self.cfg.sliced_sampling is not None and hasattr(self, "_sliced_bounding_boxes"):
            return self._sliced_bounding_boxes
        else:
            return None

    ###
    # Operations
    ###

    def sample_viewpoints(self, nbr_viewpoints: int, seed: int = 1) -> torch.Tensor | list[torch.Tensor]:
        """Sample viewpoints for the given number of viewpoints and seed.

        Samples are stored in a torch tensor with the structure
        [x, y, z, qw, qx, qv, qz]

        Args:
            nbr_viewpoints (int): The number of viewpoints to sample.
            seed (int, optional): The seed for the random number generator. Defaults to 1.

        Returns:
            torch.Tensor | list[torch.Tensor]: The sampled viewpoints.
        """

        # get the number of slices and their locations
        # NOTE: the number of viewpoints is kept constant per slice
        if self.cfg.sliced_sampling is not None:
            # get mesh size from terrain analysis  -- [x_max, y_max, x_min, y_min]
            mesh_size = self.terrain_analyser.mesh_dimensions
            # get the number of slices in both x and y direction
            nbr_slices_x = int(np.ceil((mesh_size[0] - mesh_size[2]) / self.cfg.sliced_sampling[0]))
            nbr_slices_y = int(np.ceil((mesh_size[1] - mesh_size[3]) / self.cfg.sliced_sampling[1]))
            slice_locations_x = np.linspace(mesh_size[2], mesh_size[0], nbr_slices_x)
            slice_locations_y = np.linspace(mesh_size[3], mesh_size[1], nbr_slices_y)
            # get the slice bounding boxes
            # slide bounding boxes follow the terrain analysis format of the mesh dimensions, i.e. [x_max, y_max, x_min, y_min]
            self._sliced_bounding_boxes = []
            for i in range(nbr_slices_x - 1):
                for j in range(nbr_slices_y - 1):
                    self._sliced_bounding_boxes.append(
                        (slice_locations_x[i + 1], slice_locations_y[j + 1], slice_locations_x[i], slice_locations_y[j])
                    )

            # execute viewpoint sampling for each slice
            self._samples = []
            failed_slices = []
            for i, slice_bounding_box in enumerate(self._sliced_bounding_boxes):
                try:
                    self._samples.append(self._sample_viewpoint_per_area(nbr_viewpoints, seed, slice_bounding_box))
                except Exception as e:
                    failed_slices.append(slice_bounding_box)
                    omni.log.warn(f"Error sampling viewpoints for slice {slice_bounding_box}: {e}")
                    continue

            # remove failed slices
            self._sliced_bounding_boxes = [box for box in self._sliced_bounding_boxes if box not in failed_slices]
            # dump the sliced bounding boxes and samples
            filedir = self.cfg.save_path if self.cfg.save_path else self._get_save_filedir()
            with open(os.path.join(filedir, "sliced_bounding_boxes.pkl"), "wb") as f:
                pickle.dump(self._sliced_bounding_boxes, f)

        else:
            # execute viewpoint sampling for the whole mesh
            self._samples = self._sample_viewpoint_per_area(nbr_viewpoints, seed)

        return self._samples

    def render_viewpoints(
        self, samples: torch.Tensor | list[torch.Tensor] | None = None
    ) -> torch.Tensor | list[torch.Tensor]:
        """Render the images at the given viewpoints and save them to the drive."""
        if samples is None:
            samples = self.samples

        if isinstance(samples, list):
            assert (
                self.sliced_bounding_boxes is not None
            ), "Sliced bounding boxes must be set to render a list of viewpoints samples."
            assert len(samples) == len(
                self.sliced_bounding_boxes
            ), "The number of samples must match the number of sliced bounding boxes."
            render_sample_points = []
            for sample, slice_bounding_box in zip(samples, self.sliced_bounding_boxes):
                print(f"[INFO] Rendering viewpoints for slice {slice_bounding_box}")
                render_sample_points.append(self._render_viewpoints_per_area(sample, slice_bounding_box))
        else:
            render_sample_points = self._render_viewpoints_per_area(samples)

        return render_sample_points

    def check_image_validity(self, image_data: dict) -> torch.Tensor:
        """Return False for valid images."""
        return torch.zeros(self.scene.num_envs, dtype=torch.bool, device=self.scene.device)

    def modify_images(self, image_data: dict) -> dict:
        return image_data

    ###
    # Helper functions
    ###

    def _sample_viewpoint_per_area(
        self, nbr_viewpoints: int, seed: int = 1, slice_bounding_box: tuple[float, float, float, float] | None = None
    ) -> torch.Tensor:

        filedir = self.cfg.save_path if self.cfg.save_path else self._get_save_filedir()
        if slice_bounding_box is not None:
            slice_bounding_box_str = "slice_" + "_".join(
                f"{'n' if x < 0 else ''}{abs(x):.1f}" for x in slice_bounding_box
            )
            filedir = os.path.join(filedir, slice_bounding_box_str)
        elif self.cfg.terrain_analysis.terrain_bounding_box is not None:
            slice_bounding_box_str = "slice_" + "_".join(
                f"{'n' if x < 0 else ''}{abs(x):.1f}" for x in self.cfg.terrain_analysis.terrain_bounding_box
            )
            filedir = os.path.join(filedir, slice_bounding_box_str)
        else:
            slice_bounding_box_str = "full"
        filename = os.path.join(filedir, f"viewpoints_seed{seed}_samples{nbr_viewpoints}.pkl")
        if os.path.isfile(filename):
            with open(filename, "rb") as f:
                data = pickle.load(f)
            # add loaded path dict to data dict
            omni.log.info(f"Loaded {nbr_viewpoints} with seed {seed} for map part {slice_bounding_box_str}.")
            return data
        else:
            omni.log.info(
                f"No viewpoint samples found for seed {seed} and {nbr_viewpoints} samples for map part"
                f" {slice_bounding_box_str}."
            )

        print(f"[INFO] Sampling viewpoints for {slice_bounding_box_str}")

        # analyse terrain if not done yet
        if slice_bounding_box is not None:
            # overwrite the mesh dimensions to influence where the sampling is done
            self.terrain_analyser.reset_graph()
            self.terrain_analyser._mesh_dimensions = slice_bounding_box
            self.terrain_analyser.analyse()
        elif not self.terrain_analyser.complete:
            self.terrain_analyser.analyse()

        # set seed
        random.seed(seed)
        omni.log.info(f"Start sampling {nbr_viewpoints} viewpoints for map part {slice_bounding_box_str}.")

        # samples are organized in [point_idx, neighbor_idx, distance]
        # sample from each point the neighbor with the largest distance
        nbr_samples_per_point = int(np.ceil(nbr_viewpoints / self.terrain_analyser.points.shape[0]).item())
        sample_locations = torch.zeros((nbr_samples_per_point * self.terrain_analyser.points.shape[0], 2))
        sample_locations_count = 0
        curr_point_idx = 0
        while sample_locations_count < nbr_viewpoints:
            # get samples
            sample_idx = self.terrain_analyser.samples[:, 0] == curr_point_idx
            sample_idx_select = torch.randperm(sample_idx.sum())[:nbr_samples_per_point]
            sample_locations[sample_locations_count : sample_locations_count + sample_idx_select.shape[0]] = (
                self.terrain_analyser.samples[sample_idx][sample_idx_select, :2]
            )
            sample_locations_count += sample_idx_select.shape[0]
            curr_point_idx += 1
            # reset point index if all points are sampled
            if curr_point_idx >= self.terrain_analyser.points.shape[0]:
                curr_point_idx = 0

        sample_locations = sample_locations[:sample_locations_count].type(torch.int64)

        # get the z angle of the neighbor that is closest to the origin point
        neighbor_direction = (
            self.terrain_analyser.points[sample_locations[:, 0]] - self.terrain_analyser.points[sample_locations[:, 1]]
        )
        z_angles = torch.atan2(neighbor_direction[:, 1], neighbor_direction[:, 0]).to("cpu")

        # vary the rotation of the forward and horizontal axis (in camera frame) as a uniform distribution within the limits
        x_angles = math_utils.sample_uniform(
            self.cfg.x_angle_range[0], self.cfg.x_angle_range[1], sample_locations_count, device="cpu"
        )
        y_angles = math_utils.sample_uniform(
            self.cfg.y_angle_range[0], self.cfg.y_angle_range[1], sample_locations_count, device="cpu"
        )
        x_angles = torch.deg2rad(x_angles)
        y_angles = torch.deg2rad(y_angles)

        samples = torch.zeros((sample_locations_count, 7))
        samples[:, :3] = self.terrain_analyser.points[sample_locations[:, 0]]
        samples[:, 3:] = math_utils.quat_from_euler_xyz(x_angles, y_angles, z_angles)

        omni.log.info(f"Sampled {sample_locations_count} viewpoints for map part {slice_bounding_box_str}.")

        # save samples
        os.makedirs(filedir, exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(samples, f)

        omni.log.info(f"Saved {sample_locations_count} viewpoints with seed {seed} to {filename}.")

        # debug points and orientation
        if self.cfg.debug_viz:
            env_render_steps = 1000
            marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
            marker_cfg.prim_path = "/Visuals/viewpoints"
            marker_cfg.markers["arrow"].scale = (0.1, 0.1, 0.1)
            self.visualizer = VisualizationMarkers(marker_cfg)
            self.visualizer.visualize(samples[:, :3], samples[:, 3:])

            omni.log.info(f"Visualizing {sample_locations_count} samples for {env_render_steps} render steps...")
            for _ in range(env_render_steps):
                self.sim.render()

            self.visualizer.set_visibility(False)
            omni.log.info("Done visualizing.")

        return samples

    def _render_viewpoints_per_area(
        self, samples: torch.Tensor, slice_bounding_box: tuple[float, float, float, float] | None = None
    ) -> torch.Tensor:
        omni.log.info(f"Start rendering {samples.shape[0]} images.")
        # define how many rounds are necessary to render all viewpoints
        num_rounds = int(np.ceil(samples.shape[0] / self.scene.num_envs))
        # image_idx
        image_idx = [0] * len(self.cfg.cameras)

        filedir = self.cfg.save_path if self.cfg.save_path else self._get_save_filedir()
        if slice_bounding_box is not None:
            slice_bounding_box_str = "slice_" + "_".join(
                f"{'n' if x < 0 else ''}{abs(x):.1f}" for x in slice_bounding_box
            )
            filedir = os.path.join(filedir, slice_bounding_box_str)
        elif self.cfg.terrain_analysis.terrain_bounding_box is not None:
            slice_bounding_box_str = "slice_" + "_".join(
                f"{'n' if x < 0 else ''}{abs(x):.1f}" for x in self.cfg.terrain_analysis.terrain_bounding_box
            )
            filedir = os.path.join(filedir, slice_bounding_box_str)
        filedir = os.path.join(filedir, "images")
        for cam, annotator in self.cfg.cameras.items():
            [os.makedirs(os.path.join(filedir, cam, curr_annotator), exist_ok=True) for curr_annotator in annotator]

        # save camera configurations
        omni.log.info(f"Saving camera configurations to {filedir}.")
        for cam in self.cfg.cameras.keys():
            # perform render steps to fill buffers if usd cameras are used
            if isinstance(self.scene.sensors[cam], Camera):
                for _ in range(5):
                    self.sim.render()
            np.savetxt(
                os.path.join(filedir, cam, "intrinsics.txt"),
                self.scene.sensors[cam].data.intrinsic_matrices[0].cpu().numpy(),
                delimiter=",",
            )

        # init points array for point cloud
        if self.cfg.generate_point_cloud:
            pcd_points = []

        # save images
        samples = samples.to(self.scene.device)
        sample_filter = torch.zeros(samples.shape[0], dtype=torch.bool, device=self.scene.device)
        start_time = time.time()
        for i in range(num_rounds):
            # get samples idx
            samples_idx = torch.arange(i * self.scene.num_envs, min((i + 1) * self.scene.num_envs, samples.shape[0]))
            env_ids = torch.arange(samples_idx.shape[0])
            # set camera positions
            for cam in self.cfg.cameras.keys():
                self.scene.sensors[cam].set_world_poses(
                    positions=samples[samples_idx, :3],
                    orientations=samples[samples_idx, 3:],
                    env_ids=env_ids,
                    convention="world",
                )
            # update simulation
            self.scene.write_data_to_sim()
            # perform render steps to fill buffers if usd cameras are used
            if any([isinstance(self.scene.sensors[cam], Camera) for cam in self.cfg.cameras.keys()]):
                for _ in range(10):
                    self.sim.render()
            # update scene buffers
            self.scene.update(self.sim.get_physics_dt())

            # collect the output
            image_data = {}
            for cam_idx, curr_cam_annotator in enumerate(self.cfg.cameras.items()):
                cam, annotator = curr_cam_annotator

                for curr_annotator in annotator:
                    image_data[(cam, curr_annotator)] = self.scene.sensors[cam].data.output[curr_annotator]

            # check validaty of the image data
            image_filter = self.check_image_validity(image_data)
            sample_filter[samples_idx] = image_filter[env_ids]

            # image modifiers
            image_data = self.modify_images(image_data)

            # render
            for cam_idx, curr_cam_annotator in enumerate(self.cfg.cameras.items()):
                cam, annotator = curr_cam_annotator

                # save images
                for idx in range(samples_idx.shape[0]):
                    if image_filter[idx]:
                        continue

                    for curr_annotator in annotator:
                        # semantic segmentation or RGB
                        if (
                            image_data[(cam, curr_annotator)].shape[-1] == 3
                            or image_data[(cam, curr_annotator)].shape[-1] == 4
                        ):
                            assert cv2.imwrite(
                                os.path.join(filedir, cam, curr_annotator, f"{image_idx[cam_idx]}".zfill(4) + ".png"),
                                cv2.cvtColor(
                                    image_data[(cam, curr_annotator)][idx].cpu().numpy().astype(np.uint8),
                                    cv2.COLOR_RGB2BGR,
                                ),
                            )
                        # depth
                        else:
                            depth_image = image_data[(cam, curr_annotator)][idx]
                            assert cv2.imwrite(
                                os.path.join(filedir, cam, curr_annotator, f"{image_idx[cam_idx]}".zfill(4) + ".png"),
                                np.uint16(depth_image.cpu().numpy() * self.cfg.depth_scale),
                            )

                            # Check if point cloud generation is enabled
                            if self.cfg.generate_point_cloud:
                                # Convert depth image to point cloud
                                intrinsics = self.scene.sensors[cam].data.intrinsic_matrices[0]
                                points = math_utils.unproject_depth(
                                    depth_image, intrinsics, is_ortho=annotator == "distance_to_image_plane"
                                )
                                points = points.reshape(-1, 3)

                                # transform points to world frame
                                points = math_utils.transform_points(
                                    points,
                                    self.scene.sensors[cam].data.pos_w[idx],
                                    self.scene.sensors[cam].data.quat_w_ros[idx],
                                )

                                # filter points that are clipped
                                if self.scene.sensors[cam].cfg.depth_clipping_behavior == "zero":
                                    point_filter = depth_image > 0
                                elif self.scene.sensors[cam].cfg.depth_clipping_behavior == "max" and isinstance(
                                    self.scene.sensors[cam], Camera
                                ):
                                    point_filter = depth_image < self.scene.sensors[cam].cfg.spawn.clipping_range[1]
                                elif self.scene.sensors[cam].cfg.depth_clipping_behavior == "max" and isinstance(
                                    self.scene.sensors[cam], RayCasterCamera
                                ):
                                    point_filter = depth_image < self.scene.sensors[cam].cfg.max_distance
                                else:
                                    point_filter = torch.ones_like(depth_image, dtype=torch.bool)

                                points = points[point_filter.transpose(0, 1).reshape(-1)]

                                # filter points that are outside the slice bounding box
                                # bounding box is in the format [x_max, y_max, x_min, y_min]
                                if slice_bounding_box is not None and self.cfg.slice_pc:
                                    point_filter = (
                                        (points[:, 0] < slice_bounding_box[0])
                                        & (points[:, 0] > slice_bounding_box[2])
                                        & (points[:, 1] < slice_bounding_box[1])
                                        & (points[:, 1] > slice_bounding_box[3])
                                    )
                                    points = points[point_filter]

                                if self.cfg.downsample_point_cloud_factor is not None:
                                    points = points[:: self.cfg.downsample_point_cloud_factor]
                                if self.cfg.downsample_point_cloud_voxel_size is not None:
                                    points = (
                                        torch.unique(
                                            (points // self.cfg.downsample_point_cloud_voxel_size).type(torch.int32),
                                            dim=0,
                                        )
                                        * self.cfg.downsample_point_cloud_voxel_size
                                    )

                                # Append points to the complete point cloud
                                pcd_points.append(points.cpu().numpy())

                    image_idx[cam_idx] += 1

                    if sum(image_idx) % 100 == 0:
                        print(f"Rendered {sum(image_idx)} images in {(time.time() - start_time):.4f}s.")

        # Save the complete point cloud as PLY
        if self.cfg.generate_point_cloud:
            pcd_points = np.concatenate(pcd_points, axis=0)
            print(f"Generating point cloud from {len(pcd_points)} points.")
            if self.cfg.downsample_point_cloud_voxel_size is not None:
                pcd_points = (
                    np.unique((pcd_points // self.cfg.downsample_point_cloud_voxel_size).astype(np.int32), axis=0)
                    * self.cfg.downsample_point_cloud_voxel_size
                )

            ply_filename = os.path.join(filedir, "point_cloud.ply")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_points)
            assert o3d.io.write_point_cloud(ply_filename, pcd)
            omni.log.info(f"Saved complete point cloud to {ply_filename}.")

        # save camera poses
        samples = samples[~sample_filter]
        np.savetxt(os.path.join(filedir, "camera_poses.txt"), samples.cpu().numpy(), delimiter=",")

        return samples

    def _get_save_filedir(self) -> str:
        # get env name
        if isinstance(self.scene.terrain.cfg.usd_path, str):
            terrain_file_path = self.scene.terrain.cfg.usd_path
        else:
            raise KeyError("Only implemented for terrains loaded from usd and matterport")
        env_name = os.path.splitext(terrain_file_path)[0]
        # create directory if necessary
        filedir = os.path.join(terrain_file_path, env_name)
        os.makedirs(filedir, exist_ok=True)
        return filedir
