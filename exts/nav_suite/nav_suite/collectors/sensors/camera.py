

from __future__ import annotations

import cv2
import numpy as np
import open3d as o3d
import os
import time
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import omni.log
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import Camera, RayCasterCamera
from isaaclab.sim import SimulationContext

from .base import SensorBase

if TYPE_CHECKING:
    from .camera_cfg import CameraSensorCfg


class CameraSensor(SensorBase):
    """Camera sensor.

    The resulting folder structure is as follows:

    ``` graphql
    cfg.data_dir
    ├── camera_poses.txt                    # format: x y z qw qx qy qz
    ├── cfg.cameras.keys[0]                 # required
    |   ├── intrinsics.txt                  # K-matrix (3x3)
    |   ├── annotator_0                     # first annotator in the dict of the first camera
    |   |   ├── xxxx.png                    # images saved with 4 digits, e.g. 0000.png
    |   ├── annotator_1                     # second annotator in the dict of the first camera (optional)
    |   |   ├── xxxx.png                    # images saved with 4 digits, e.g. 0000.png
    ├── cfg.cameras.keys[1]                 # optional
    |   ├── intrinsics.txt                  # K-matrix (3x3)
    |   ├── annotator_0                     # first annotator in the dict of the second camera
    |   |   ├── xxxx.png                    # images saved with 4 digits, e.g. 0000.png
    |   ├── annotator_1                     # second annotator in the dict of the second camera (optional)
    |   |   ├── xxxx.png                    # images saved with 4 digits, e.g. 0000.png
    ```

    """

    cfg: CameraSensorCfg

    def __init__(self, cfg: CameraSensorCfg, scene: InteractiveScene):
        super().__init__(cfg, scene)

        # get sim context
        self.sim = SimulationContext.instance()

        # init image index buffer
        self._image_idx = [0] * len(self.cfg.cameras)

        # init time buffer
        self._used_time = 0.0

        # init buffer for point cloud
        if self.cfg.generate_point_cloud:
            self._pcd_points = []

    def pre_collection(self, samples: torch.Tensor, filedir: str):
        # create directory for images
        filedir = os.path.join(filedir, "images")
        for cam, annotator in self.cfg.cameras.items():
            [os.makedirs(os.path.join(filedir, cam, curr_annotator), exist_ok=True) for curr_annotator in annotator]

        # init a filter samples object
        self._sample_filter = torch.zeros(samples.shape[0], dtype=torch.bool, device=self.scene.device)

        # save camera configurations
        omni.log.info(f"Saving camera configurations to {filedir}.")
        for cam in self.cfg.cameras.keys():
            # perform render updates to fill buffers if usd cameras are used
            if isinstance(self.scene.sensors[cam], Camera):
                for _ in range(5):
                    self.sim.render()
            np.savetxt(
                os.path.join(filedir, cam, "intrinsics.txt"),
                self.scene.sensors[cam].data.intrinsic_matrices[0].cpu().numpy(),
                delimiter=",",
            )

    def post_collection(self, samples: torch.Tensor, filedir: str):
        # Save the complete point cloud as PLY
        if self.cfg.generate_point_cloud:
            pcd_points = np.concatenate(self._pcd_points, axis=0)
            print(f"Generating point cloud from {len(pcd_points)} points.")
            if self.cfg.downsample_point_cloud_voxel_size is not None:
                pcd_points = (
                    np.unique((pcd_points // self.cfg.downsample_point_cloud_voxel_size).astype(np.int32), axis=0)
                    * self.cfg.downsample_point_cloud_voxel_size
                )

            ply_filename = os.path.join(filedir, "images", "point_cloud.ply")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_points)
            assert o3d.io.write_point_cloud(ply_filename, pcd)
            omni.log.info(f"Saved complete point cloud to {ply_filename}.")

        # save camera poses
        # TODO: check that this updates the samples in the SensorDataSampling object
        samples = samples[~self._sample_filter]
        np.savetxt(os.path.join(filedir, "images", "camera_poses.txt"), samples.cpu().numpy(), delimiter=",")

    ###
    # Executed logic around the simulation update
    #
    # pre_sim_update
    # scene.write_data_to_sim()
    # post_sim_update
    #
    ###

    def pre_sim_update(self, positions: torch.Tensor, orientations: torch.Tensor, env_ids: torch.Tensor):
        # set camera positions
        for cam in self.cfg.cameras.keys():
            self.scene.sensors[cam].set_world_poses(
                positions=positions,
                orientations=orientations,
                env_ids=env_ids,
                convention="world",
            )

    def post_sim_update(
        self,
        samples_idx: torch.Tensor,
        env_ids: torch.Tensor,
        filedir: str,
        slice_bounding_box: tuple[float, float, float, float] | None = None,
    ):
        # start time
        start_time = time.time()

        # collect the output
        image_data = {}
        for cam_idx, curr_cam_annotator in enumerate(self.cfg.cameras.items()):
            cam, annotator = curr_cam_annotator

            for curr_annotator in annotator:
                image_data[(cam, curr_annotator)] = self.scene.sensors[cam].data.output[curr_annotator]

        # check validaty of the image data
        image_filter = self.check_image_validity(image_data)
        self._sample_filter[samples_idx] = image_filter[env_ids]

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
                            os.path.join(
                                filedir, "images", cam, curr_annotator, f"{self._image_idx[cam_idx]}".zfill(4) + ".png"
                            ),
                            cv2.cvtColor(
                                image_data[(cam, curr_annotator)][idx].cpu().numpy().astype(np.uint8),
                                cv2.COLOR_RGB2BGR,
                            ),
                        )
                    # depth
                    else:
                        depth_image = image_data[(cam, curr_annotator)][idx]
                        assert cv2.imwrite(
                            os.path.join(
                                filedir, "images", cam, curr_annotator, f"{self._image_idx[cam_idx]}".zfill(4) + ".png"
                            ),
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
                            self._pcd_points.append(points.cpu().numpy())

                self._image_idx[cam_idx] += 1

                if sum(self._image_idx) % 100 == 0:
                    print(
                        f"Rendered {sum(self._image_idx)} images in"
                        f" {(time.time() - start_time + self._used_time):.4f}s."
                    )

        # update time
        self._used_time = time.time() - start_time

    ###
    # Helper functions
    ###

    def check_image_validity(self, image_data: dict) -> torch.Tensor:
        """Return False for valid images."""
        return torch.zeros(self.scene.num_envs, dtype=torch.bool, device=self.scene.device)

    def modify_images(self, image_data: dict) -> dict:
        return image_data
