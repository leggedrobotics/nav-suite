# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from isaaclab.utils import configclass

from nav_suite.terrain_analysis import TerrainAnalysisCfg, TerrainAnalysisSingletonCfg

from .consecutive_goal_command import ConsecutiveGoalCommand
from .goal_command_base_cfg import GoalCommandBaseCfg


@configclass
class ConsecutiveGoalCommandCfg(GoalCommandBaseCfg):
    """Configuration for the terrain-based position command generator."""

    class_type: type = ConsecutiveGoalCommand

    resample_distance_threshold: float = 0.2
    """Distance threshold for resampling the goals."""

    terrain_analysis: TerrainAnalysisCfg | TerrainAnalysisSingletonCfg = TerrainAnalysisCfg()
    """Configuration for the trajectory sampling."""
