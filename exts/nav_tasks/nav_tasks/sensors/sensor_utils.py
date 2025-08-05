# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.sensors import RayCasterCameraCfg


def adjust_ray_caster_camera_image_size(
    camera_cfg: RayCasterCameraCfg, downsample_height_factor: int, downsample_width_factor: int
) -> RayCasterCameraCfg:
    """
    Adjust the image size of the camera configuration.

    Args:
        camera_cfg: The camera configuration to adjust.
        downsample_height_factor: The factor to downsample the height of the image by.
        downsample_width_factor: The factor to downsample the width of the image by.

    Returns:
        RayCasterCameraCfg: The adjusted camera configuration.
    """
    # Assert that the downsample factors are factors of the original image size
    assert camera_cfg.pattern_cfg.height % downsample_height_factor == 0, (
        "Height downsample factor should be a factor of the original height, or the pixels will be interpolated and "
        "processing will be slow."
    )
    assert camera_cfg.pattern_cfg.width % downsample_width_factor == 0, (
        "Width downsample factor should be a factor of the original width, or the pixels will be interpolated and "
        "processing will be slow."
    )

    camera_cfg.pattern_cfg.height = camera_cfg.pattern_cfg.height // downsample_height_factor
    camera_cfg.pattern_cfg.width = camera_cfg.pattern_cfg.width // downsample_width_factor
    return camera_cfg
