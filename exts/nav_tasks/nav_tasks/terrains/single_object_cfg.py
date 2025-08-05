# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass

from .single_object import center_object_pattern, single_object_terrain


@configclass
class SingleObjectTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a stairs and ramp (both next to each other) mesh terrain."""

    function = single_object_terrain

    border_width: float = 0.0
    """The width of the border around the terrain (in m). Defaults to 0.0.

    The border is a flat terrain with the same height as the terrain.
    """

    border_height: float | None = None
    """The height of the border around the terrain (in m). Defaults to None."""

    object_type: Literal["box", "cylinder", "wall"] = "box"
    """The type of the object."""

    dim_range: list[float] = [0.5, 1.5]
    """The dimension of the object.

    If the object is a box, this is the length and width of the box.
    If the object is a cylinder, this is the radius of the cylinder.
    If the object is a wall, this is the length of the wall. The width is fixed to 0.1m.
    """

    height_range: list[float] = [0.5, 1.5]
    """The height of the object. Will be randomly sampled between the two values. Defaults to [0.5, 1.5]."""

    wall_width: float = 0.1
    """The width of the wall (in m). Defaults to 0.1."""

    position_pattern: callable = center_object_pattern
    """The pattern of the object position."""
