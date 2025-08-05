# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.sensors.ray_caster import RayCasterCameraCfg
from isaaclab.utils import configclass

from .matterport_raycaster_camera import MatterportRayCasterCamera


@configclass
class MatterportRayCasterCameraCfg(RayCasterCameraCfg):
    """Configuration for the ray-cast camera for Matterport Environments.

    .. note::
        This class depends on the public MatterportRayCasterCamera class, the RSL multi-mesh implementation is
        currently not supported.
    """

    class_type = MatterportRayCasterCamera
    """Name of the specific matterport ray caster camera class."""
