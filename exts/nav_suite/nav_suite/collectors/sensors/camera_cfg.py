# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils.configclass import configclass

from .base_cfg import SensorBaseCfg
from .camera import CameraSensor


@configclass
class CameraSensorCfg(SensorBaseCfg):
    """Camera sensor configuration."""

    class_type: type[CameraSensor] = CameraSensor
    """Class type of the sensor."""

    ###
    # Camera Configuration
    ###
    cameras: dict[str, list[str]] = {
        "camera_0": ["semantic_segmentation"],
        "camera_1": ["distance_to_image_plane"],
    }
    """Dict of cameras and corresponding annotators to use for the sensor data sampling."""

    depth_scale: float = 1000.0
    """Scaling factor for the depth values."""

    ###
    # Point Cloud Generation
    ###
    generate_point_cloud: bool = True
    """Whether to generate a point cloud from the depth images."""

    downsample_point_cloud_factor: int | None = 16
    """Downsample the point cloud generated from each image by a factor.

    Useful to reduce memory usage when generating a large number of data samples.
    Moreover, for high resolution cameras, the point cloud can be very dense without points providing additional information.
    """

    downsample_point_cloud_voxel_size: float | None = 0.05
    """Voxel size to use for the downsampling of the point cloud. If None, no downsampling is applied."""

    slice_pc: bool = False
    """Whether to slice the point cloud into the same slices as the image viewpoints."""
