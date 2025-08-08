# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import MISSING

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass

from .stairs_ramp_terrain import stairs_ramp_eval_terrain, stairs_ramp_terrain, stairs_ramp_up_down_terrain


@configclass
class StairsRampTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a stairs and ramp (both next to each other) mesh terrain."""

    function = stairs_ramp_terrain

    border_width: float = 0.0
    """The width of the border around the terrain (in m). Defaults to 0.0.

    The border is a flat terrain with the same height as the terrain.
    """

    modify_step_height: bool = False
    """If True, the step height will be modified based on the difficulty. Defaults to False."""

    step_height_range: tuple[float, float] | None = None
    """The minimum and maximum height of the steps (in m).

    .. note::
        Required when :attr:`modify_step_height` is True. Or both, :attr:`modify_step_height` and :attr:`max_height` are False.
    """

    modify_ramp_slope: bool = False
    """If True, the ramp slope will be modified based on the difficulty. Defaults to False."""

    ramp_slope_range: tuple[float, float] | None = None
    """The minimum and maximum slope of the ramp (in degrees).

    .. note::
        Required when :attr:`modify_ramp_slope` is True. Or both, :attr:`modify_ramp_slope` and :attr:`max_height` are False.
    """

    width_randomization: float = 0.0
    """The maximum randomization of the width of the stairs and the ramp (in m). Defaults to 0.0."""

    random_stairs_ramp_position_flipping: bool = False
    """If True, the stairs and the ramp will be randomly flipped. Defaults to False."""

    random_wall_probability: float = 0.1
    """The probability of adding a wall instead of a ramp or stairs obstacle. Defaults to 0.1."""

    all_wall: bool = False
    """If True, all obstacles will be walls. Defaults to False."""

    step_width: float = MISSING
    """The width of the steps (in m)."""

    platform_width: float = 2.0
    """The minimum width of the platform in front and behind the stairs and the ramp. Defaults to 2.0.

    ..note ::
        The platform behind the stairs and the ramp can be extended in the case the maximum height of the stairs
        and the ramp exceeds the :attr:`max_height` attribute (if it is defined).
    """

    max_height: float | None = 2.0
    """The maximum height of the stairs and the ramp (in m). Defaults to 2.0."""

    free_space_front: bool = False
    """Decide if the additional free space due to applied difficulty is in front or behind the obstacle. Defaults to False.

    When increasing the difficulty the stairs and the ramp have a higher incline and therefore require less space.
    The additional free space can be either in front or behind the obstacle."""

    no_free_space_front: bool = True
    """Stairs and Ramp start at the same position. Defaults to False."""

    random_state_file: str | None = None
    """The file to load the numpy random state from to make the terrain generation deterministic. Defaults to None."""


@configclass
class StairsRampEvalTerrainCfg(StairsRampTerrainCfg):
    """Configuration for a stairs and ramp (both next to each other) mesh terrain.

    Eval configuration, both on two sides of a platform.
    """

    function = stairs_ramp_eval_terrain

    center_platform_width: float = 2.0
    """The width of the platform in the center of the terrain"""


@configclass
class StairsRampUpDownTerrainCfg(StairsRampTerrainCfg):
    """Configuration for a stairs and ramp (both next to each other) mesh terrain.

    Eval configuration, both on two sides of a platform.
    """

    function = stairs_ramp_up_down_terrain

    center_platform_width: float = 2.0
    """The width of the platform in the center of the terrain"""
