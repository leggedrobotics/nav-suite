# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from collections.abc import Callable

from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass

from .stateful_rewards import AverageEpisodeVelocityTerm, SteppedProgressTerm


@configclass
class SteppedProgressCfg(RewardTermCfg):

    func: Callable[..., torch.Tensor] = SteppedProgressTerm

    step: float = 0.1
    """Step size for the progress reward percentage."""

    goal_command_generator_name: str = "goal_command"
    """Name of the goal command generator to use.

    .. note::
        The GoalCommand generator should include spawning positions for the robots and the goal,
        i.e., it should be of type :class:`nav_tasks.mdp.GoalCommand` or similar.

    """


@configclass
class AverageEpisodeVelocityCfg(RewardTermCfg):

    func: Callable[..., torch.Tensor] = AverageEpisodeVelocityTerm

    goal_reached_termination_name: str = "goal_reached"
    """Name of the termination condition that indicates the goal has been reached."""
