# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class ReconstructionCfg:
    """
    Arguments for 3D reconstruction using depth maps
    """

    # input data parameters
    data_dir: str = MISSING
    """Data directory where the rendered images with the extrinsic and intrinsic camera parameters are stored"""
    depth_cam_name: str = "camera_1"
    """Name of the depth camera in the scene used to render the depth images. Default is 'camera_1'.

    .. note::
        Currently the script only supports rendered depth images from the `distance_to_image_plane` annotator.
    """
    semantic_cam_name: str | None = "camera_0"
    """Name of the semantic camera in the scene used to render the semantic images. Default is 'camera_0'."""

    # reconstruction parameters
    voxel_size: float = 0.05
    """Voxel size for the environment reconstruction in meters.

    The voxel size determines the resolution of the reconstructed 3D environment. For Matterport scenes,
    a voxel size of 0.05 is recommended, while for larger multi-mesh scenes (such as carla), a voxel size should
    be increased (e.g. 0.1 or higher). Default is 0.05."""
    max_images: int | None = 1000
    """Maximum number of images to use for the reconstruction. If None, all images are used. Default is 1000."""
    depth_scale: float = 1000.0
    """Depth scale factor to convert depth values to meters. Default is 1000.0."""
    semantics: bool = True
    """Whether to perform semantic reconstruction.

    Requires semantic images to be present in the data_dir. Default is True."""

    # speed vs. memory trade-off parameters
    point_cloud_batch_size: int = 200
    """Batch size for point cloud generation.

    Defines how many images are added to the pouint-cloud at once. Higher values use more memory but are faster. Default is 200."""
