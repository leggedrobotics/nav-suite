# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass

from .corridor import corridor_terrain


@configclass
class CorridorTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a corrdior mesh terrain."""

    function = corridor_terrain

    width_range: tuple[float, float] = (1.5, 5.0)
    """The minimum and maximum width of the corridor (in m)."""

    wall_height: float = 2.5
    """The height of the wall (in m)."""

    wall_thickness: float = 0.2
    """The thickness of the wall (in m)."""

    door_width_range: tuple[float, float] = (1.0, 1.5)
    """The minimum and maximum width of the door (in m)."""

    door_height: float = 2.0
    """The height of the door (in m)."""

    border_width: float = 0.0
    """The width of the border around the terrain (in m). Defaults to 0.0.

    The border is a flat terrain with the same height as the terrain.
    """

    separation_wall_offset_range: tuple[float, float] = (-2.0, 2.0)
    """The minimum and maximum offset of the separation wall from the center (in m)."""
