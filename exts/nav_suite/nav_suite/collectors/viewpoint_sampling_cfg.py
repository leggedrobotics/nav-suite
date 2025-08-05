# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from ..terrain_analysis import TerrainAnalysisCfg, TerrainAnalysisSingletonCfg


@configclass
class ViewpointSamplingCfg:
    """Configuration for the viewpoint sampling."""

    terrain_analysis: TerrainAnalysisCfg | TerrainAnalysisSingletonCfg = TerrainAnalysisCfg(raycaster_sensor="camera_0")
    """Name of the camera object in the scene definition used for the terrain analysis."""

    ###
    # Camera Configuration
    ###
    cameras: dict[str, list[str]] = {
        "camera_0": ["semantic_segmentation"],
        "camera_1": ["distance_to_image_plane"],
    }
    """Dict of cameras and corresponding annotators to use for the viewpoint sampling."""
    depth_scale: float = 1000.0
    """Scaling factor for the depth values."""

    ###
    # Sampling Configuration
    ###
    sample_points: int = 10000
    """Number of random points to sample."""
    x_angle_range: tuple[float, float] = (-2.5, 2.5)
    y_angle_range: tuple[float, float] = (-2, 5)  # negative angle means in isaac convention: look down
    """Range of the x and y angle of the camera (in degrees), will be randomly selected according to a uniform distribution"""
    height: float = 0.5
    """Height to use for the random points."""
    sliced_sampling: tuple[float, float] | None = None
    """Sliced sampling of the viewpoints. If None, no slicing is applied.

    If a tuple is provided, the points will be sampled in slices of the given width and length. That means, the terrain
    will be split into slices and for each slice, the number of viewpoints as given to the ``sample_viewpoints`` method
    is generated.
    """

    ###
    # Point Cloud Generation
    ###
    generate_point_cloud: bool = True
    """Whether to generate a point cloud from the depth images."""
    downsample_point_cloud_factor: int | None = 16
    """Downsample the point cloud generated from each image by a factor.

    Useful to reduce memory usage when generating a large number of viewpoints.
    Moreover, for high resolution cameras, the point cloud can be very dense without points providing additional information.
    """
    downsample_point_cloud_voxel_size: float | None = 0.05
    """Voxel size to use for the downsampling of the point cloud. If None, no downsampling is applied."""
    slice_pc: bool = False
    """Whether to slice the point cloud into the same slices as the viewpoints."""

    ###
    # Saving Configuration
    ###
    save_path: str | None = None
    """Directory to save the viewpoint samples, camera intrinsics and rendered images to.

    If None, the directory is the same as the one of the obj file. Default is None."""

    # debug
    debug_viz: bool = True
    """Whether to visualize the sampled points and orientations."""
