# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from nav_tasks.mdp import GoalCommand

if TYPE_CHECKING:
    from .stateful_rewards_cfg import AverageEpisodeVelocityCfg, SteppedProgressCfg


class SteppedProgressTerm(ManagerTermBase):
    cfg: SteppedProgressCfg
    risk_grid: torch.Tensor

    def __init__(self, cfg: SteppedProgressCfg, env: ManagerBasedRLEnv):
        # super init just saves the cfg and env
        super().__init__(cfg, env)
        # create a buffer to store the best position of the robot
        self.best_distance_to_goal = torch.zeros(env.num_envs, device=self.device)

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        # get the reward from the function
        # get the position of the robot
        asset: Articulation = env.scene[asset_cfg.name]
        goal_cmd_generator: GoalCommand = env.command_manager.get_term(self.cfg.goal_command_generator_name)  # type: ignore

        orig_distance_to_goal = torch.norm(
            goal_cmd_generator.pos_command_w[:, :2] - goal_cmd_generator.pos_spawn_w[:, :2], dim=1, p=2
        )

        # After reset update the best distance to goal to original for the envs which have been reset
        self.best_distance_to_goal = torch.where(
            torch.isinf(self.best_distance_to_goal), orig_distance_to_goal, self.best_distance_to_goal
        )
        # Note: Using body_pos_w instead of root_pos_w to get the position of the robot.
        cur_distance_goal = torch.norm(
            asset.data.body_pos_w[:, 0, :2] - goal_cmd_generator.pos_command_w[:, :2], dim=1, p=2
        )

        # check if the distance to the goal has increased into the next step size
        updates = self.best_distance_to_goal - cur_distance_goal > (self.cfg.step * orig_distance_to_goal)

        # update best distance to goal
        self.best_distance_to_goal = torch.where(updates, cur_distance_goal, self.best_distance_to_goal)

        # return score for relevant envs
        score = torch.zeros(env.num_envs, device=self.device)
        score[updates] = 1.0
        return score

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resets the manager term.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
        """
        # reset the buffer of last position for finished envs
        self.best_distance_to_goal[env_ids] = torch.inf


class AverageEpisodeVelocityTerm(ManagerTermBase):
    """A reward term that calculates the average velocity of the robot, and returns it as a reward if the robot
    has reached the goal."""

    cfg: AverageEpisodeVelocityCfg

    def __init__(self, cfg: AverageEpisodeVelocityCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # create a buffer to store the best position of the robot
        self.average_velocity = torch.zeros(env.num_envs, device=self.device)
        self.datapoints_per_env = torch.zeros(env.num_envs, device=self.device)

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        # get the reward from the function
        # get the veolcity of the robot
        asset: Articulation = env.scene[asset_cfg.name]
        # Terminations are calculated before rewards so it is valid to fetch the goal_reached signal here.
        at_goal = env.termination_manager.get_term(self.cfg.goal_reached_termination_name)

        current_velocity = torch.norm(asset.data.root_vel_w[:, :2], dim=1, p=2)
        # Calculate new average velocity
        self.average_velocity = (self.average_velocity * self.datapoints_per_env + current_velocity) / (
            self.datapoints_per_env + 1
        )
        self.datapoints_per_env += 1
        return torch.where(at_goal, self.average_velocity, torch.zeros_like(self.average_velocity))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resets the manager term.
        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
        """
        # reset the buffer of last position for finished envs
        self.average_velocity[env_ids] = 0.0
        self.datapoints_per_env[env_ids] = 0.0
