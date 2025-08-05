# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .goal_command_base import GoalCommandBaseTerm


@configclass
class GoalCommandBaseCfg(CommandTermCfg):
    """Configuration for the terrain-based position command generator."""

    class_type: type = GoalCommandBaseTerm

    vis_line: bool = True
    """Whether to visualize the line from the robot to the goal."""

    asset_name: str = "robot"
    """Name of the asset in the environment for which the commands are generated."""
