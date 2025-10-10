# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass

from .corridor_turn import corridor_turn_terrain


@configclass
class CorridorTurnTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with corridors and 90/180 degree turns."""

    function = corridor_turn_terrain
    """Function to generate the terrain."""

    size: tuple[float, float] = (10.0, 10.0)
    """Size of the terrain in meters."""

    wall_height: float = 2.5
    """Height of the corridor walls."""

    wall_thickness: float = 0.2
    """Thickness of the corridor walls."""

    width_range: tuple[float, float] = (2.5, 5.0)
    """The minimum and maximum width of the corridor (in m)."""

    u_top_width_range: tuple[float, float] = (1.5, 3.0)
    """The minimum and maximum width of the top part of the U-shape corridor (in m)."""

    turn_type: str = "90"
    """Type of turn: '90' for L-shape, '180' for U-shape."""

    border_width: float = 0.0
    """The width of the border around the terrain (in m). Defaults to 0.0.

    The border is a flat terrain with the same height as the terrain.
    """

    # Optionally, add more parameters as needed
