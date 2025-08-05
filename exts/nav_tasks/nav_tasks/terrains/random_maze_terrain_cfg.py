# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass

from .random_maze_terrain import random_maze_terrain


@configclass
class RandomMazeTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a random maze terrain"""

    function = random_maze_terrain
    """Function to generate the terrain"""

    resolution: float = 1.0
    """Resolution of the terrain, in this case the maze grid resolution in meters. Default to 1.0"""

    maze_height: float = 1.0
    """Height of the maze walls in meters. Default to 1.0"""

    wall_width: float = 0.2
    """Width of the maze walls in meters. Default to 0.2"""

    # Parameters for the randomization of the obstacles
    max_increase: float = 0.0
    """Maximum increase of the size of the obstacles in meters. Default to 0.0"""

    max_decrease: float = 0.0
    """Maximum decrease of the size of the obstacles in meters. Default to 0.0"""

    width_range: tuple[float, float] = (1.0, 1.0)  # range --> e.g [width*0.0, width*1.3]
    """Range of the width of the obstacles. Default to [1.0, 1.0]"""

    length_range: tuple[float, float] = (1.0, 1.0)
    """Range of the length of the obstacles. Default to [1.0, 1.0]"""

    height_range: tuple[float, float] = (1.0, 1.0)
    """Range of the height of the obstacles. Default to [1.0, 1.0]"""

    num_stairs: int = 0
    """The number of stairs in the maze. Defaults to 0."""

    step_height_range: tuple[float, float] | None = None
    """The minimum and maximum height of the steps (in m).

    .. note::
        Must be provided when num_stairs > 0.
    """

    step_width_range: tuple[float, float] | None = None
    """The minimum and maximum width of the steps (in m).

    .. note::
        Must be provided when num_stairs > 0.
    """

    stairs_platform_width: float = 1.0
    """The width of the platform between the up and down part of the stairs. Defaults to 1.0."""

    # TODO @tasdep add shape type randomisation eg. circles...
