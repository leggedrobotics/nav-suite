# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

from isaaclab.utils import configclass

from nav_suite.collectors import TrajectorySamplingCfg

from .goal_command import GoalCommand
from .goal_command_base_cfg import GoalCommandBaseCfg


@configclass
class GoalCommandCfg(GoalCommandBaseCfg):
    """Configuration for the terrain-based position command generator."""

    class_type: type = GoalCommand

    ###
    # Start-Goal Trajectory Sampling Configuration
    ###

    num_pairs: int = 100
    """Number of start-goal pairs to sample."""

    path_length_range: list[float] = [2.0, 10.0]
    """Range of the sampled trajectories between start and goal."""

    z_offset_spawn: float = 0.1
    """Offset in z direction for the spawn height."""

    traj_sampling: TrajectorySamplingCfg = TrajectorySamplingCfg()
    """Configuration for the actual trajectory sampling class. Defines amount of points in the terrain from which
    the start-goal pairs are sampled."""

    ###
    # Resampling and correct setting of goal commands
    ###

    sampling_mode: Literal["infinite", "autonomous", "bounded"] = "infinite"
    """Mode of the sampling.

    - "infinite": Infinite sampling of the same start-goal pairs until they are resampled by calling
        :meth:`nav_tasks.mdp.GoalCommand.sample_trajectories`.
    - "autonomous": Autonomous resampling of trajectories once all trajectories have been sampled once.
    - "bounded": Bounded sampling of trajectories for a given number of environments. Once all sampled, the last
        command is repeated and no changes are done until :meth:`nav_tasks.mdp.GoalCommand.sample_trajectories` is
        called again.
    """

    reset_pos_term_name: str | None = "reset_base"
    """Name of the termination term that resets the base position.

    This term is normally called before the goal resample and therefore with the old commands. To fix this, we
    call it again after the goal resample."""

    ###
    # Sample distribution over the terrain
    ###

    terrain_level_sampling: bool = False
    """If True, for each robot the terrain level is checked and the path sampled in the same level.

    .. note::
        This is necessary if the terrain levels are controlled by the curriculum. Only works if the terrain type is
        ``generator``.
    """

    subterrain_sampling: bool = False
    """Sampling of start-goal pairs for each subterrain or the whole terrain."""

    subterrain_analysis_cfgs: dict[str, dict] | None = None
    """Dictionary of terrain analysis parameters for each subterrain.

    The keys are the names of the terrains and the values are the corresponding `TerrainAnalysisCfg` configurations.
    If None, the `TerrainAnalysisCfg` of the `traj_sampling` will be used for all terrains. This can be used to
    customize the terrain analysis for each subterrain when each subterrain has been loaded from a USD file (not
    available when a generated terrain with just a single mesh is used).

    .. note::
        The `terrain_bounding_box` parameter will be translated into the local frame of the subterrain. If the prim has
        been rotated, the bounding box will change, as only a axis aligned bounding box is supported.

    .. caution::
        The `TerrainAnalysisCfg` of `TrajectorySamplingCfg` has be the non-singleton version, otherwise the same
        analysis object will be used for all subterrains.
    """
