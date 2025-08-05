# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.sensors.ray_caster import RayCasterCfg
from isaaclab.utils import configclass

from .matterport_raycaster import MatterportRayCaster


@configclass
class MatterportRayCasterCfg(RayCasterCfg):
    """Configuration for the ray-cast sensor for Matterport Environments.

    .. note::
        This class depends on the public RayCasterCfg class, the RSL multi-mesh implementation is
        currently not supported.
    """

    class_type = MatterportRayCaster
    """Name of the specific matterport ray caster class."""
