# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass

from .maze_terrain import maze_terrain


@configclass
class MazeTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a Maze terrain"""

    function = maze_terrain
    """Function to generate the terrain"""

    size: tuple[float, float] = (30.0, 30.0)
    """Size of the terrain in meters"""

    resolution: float = 0.05
    """Resolution of the terrain"""

    path_obstacles: str = MISSING
    """Path to the json file containing the obstacles"""

    randomization: dict = {
        "max_increase": 0.0,  # maximum increase of the size of the obstacles in meters
        "max_decrease": 0.0,  # maximum decrease of the size of the obstacles in meters
        "range": {  # range --> e.g [width*0.0, width*1.3] - uniform distribution
            "width": [1.0, 1.0],
            "height": [1.0, 1.0],
            "length": [1.0, 1.0],
            "radius": [1.0, 1.0],
        },
    }
    """Randomization configuration of the obstacles. Default no randomization."""

    difficulty_configuration: dict = {
        "1.0": 1.0,
    }
    """Difficulty configuration of the obstacles. Default to 1.0.

    Setting the probability to 1.0 means that all obstacles are present."""
