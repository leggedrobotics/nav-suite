# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Sterolabs Depth Cameras.

The following configuration parameters are available:

* :obj:`ZED_X_NARROW_RAYCASTER_CFG`: The ZED-X Camera with a narrow field of view (4 mm focal length) as an instance
  of :class:`RayCasterCameraCfg`.
* :obj:`ZED_X_MINI_WIDE_RAYCASTER_CFG`: The ZED-X Mini Camera with a wide field of view (2 mm focal length) as an
  instance of :class:`RayCasterCameraCfg`.
* :obj:`ZED_X_NARROW_USD_CFG`: The ZED-X Camera with a narrow field of view (4 mm focal length)  as an
  instance of :class:`CameraCfg`.
* :obj:`ZED_X_MINI_WIDE_USD_CFG`: The ZED-X Mini Camera with a wide field of view (2 mm focal length) as an
  instance of :class:`CameraCfg`.

Reference:

* https://www.stereolabs.com/store/products/zed-x-stereo-camera -> Resources -> DataSheet
* https://www.stereolabs.com/store/products/zed-x-mini-stereo-camera -> Resources -> DataSheet
"""

from isaaclab.sensors import CameraCfg, RayCasterCameraCfg, patterns
from isaaclab.sim.spawners import PinholeCameraCfg

##
# Configuration as RayCasterCameraCfg
##

ZED_X_NARROW_RAYCASTER_CFG = RayCasterCameraCfg(
    pattern_cfg=patterns.PinholeCameraPatternCfg().from_intrinsic_matrix(
        focal_length=38.0,
        intrinsic_matrix=[380.0831, 0.0, 467.7916, 0.0, 380.0831, 262.0532, 0.0, 0.0, 1.0],
        height=540,
        width=960,
    ),
    debug_vis=True,
    max_distance=20.0,
    data_types=["distance_to_image_plane"],
)
"""ZED X Camera with narrow field of view as RayCaster Sensor."""

ZED_X_MINI_WIDE_RAYCASTER_CFG = RayCasterCameraCfg(
    pattern_cfg=patterns.PinholeCameraPatternCfg().from_intrinsic_matrix(
        focal_length=22.0,
        intrinsic_matrix=[369.7771, 0.0, 489.9926, 0.0, 369.7771, 275.9385, 0.0, 0.0, 1.0],
        height=540,
        width=960,
    ),
    debug_vis=True,
    max_distance=20,
    data_types=["distance_to_image_plane"],
)
"""ZED X Mini Camera with wide field of view as RayCaster Sensor."""


##
# Configuration as USD Camera
##

ZED_X_NARROW_USD_CFG = CameraCfg(
    spawn=PinholeCameraCfg().from_intrinsic_matrix(
        focal_length=38.0,
        intrinsic_matrix=[380.0831, 0.0, 467.7916, 0.0, 380.0831, 262.0532, 0.0, 0.0, 1.0],
        height=540,
        width=960,
        clipping_range=(0.01, 20),
    ),
    height=540,
    width=960,
    data_types=["distance_to_image_plane", "rgb"],
)
"""ZED X Camera with narrow field of view as USD Sensor."""


ZED_X_MINI_WIDE_USD_CFG = CameraCfg(
    spawn=PinholeCameraCfg().from_intrinsic_matrix(
        focal_length=22.0,
        intrinsic_matrix=[369.7771, 0.0, 489.9926, 0.0, 369.7771, 275.9385, 0.0, 0.0, 1.0],
        height=540,
        width=960,
        clipping_range=(0.01, 20),
    ),
    height=540,
    width=960,
    data_types=["distance_to_image_plane", "rgb"],
)
"""ZED X Mini Camera with wide field of view as USD Sensor."""
