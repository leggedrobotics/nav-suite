# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from nav_tasks.mdp import GoalCommand


class is_successful_terminated_term(ManagerTermBase):
    """Reward the agent for successful termination (e.g. goal reached) that correspond to timeouts.
    The parameters are as follows:
    * attr:`term_keys`: The termination terms to reward. This can be a string, a list of strings
      or regular expressions. Default is ".*" which rewards all timeouts.
    The reward is computed as the sum of the termination terms that are not episodic terminations.
    This means that the reward is 0 if the episode is terminated due to an episodic timeout. Otherwise,
    if two termination terms are active, the reward is 2.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        # find and store the termination terms
        term_keys = cfg.params.get("term_keys", ".*")
        self._term_names = env.termination_manager.find_terms(term_keys)

    def __call__(self, env: ManagerBasedRLEnv, term_keys: str | list[str] = ".*") -> torch.Tensor:
        # Return the unweighted reward for the termination terms
        reset_buf = torch.zeros(env.num_envs, device=env.device)
        for term in self._term_names:
            # Sums over terminations term values to account for multiple terminations in the same step
            reset_buf += env.termination_manager.get_term(term)

        return (reset_buf * (~env.termination_manager.terminated)).float()


def near_goal_stability(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_command_generator_name: str = "goal_command",
) -> torch.Tensor:
    """Reward the agent for being stable near the goal.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.
        goal_command_generator_name: The name of the goal command generator.

    Returns:
        Dense reward [0, +1] based on the distance to the goal and the velocity.

    .. note::
        The selected goal command generator should have ``pos_command_w`` and ``heading_command_w`` attributes. The
        default would be of the class :class:`nav_tasks.mdp.GoalCommand`.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    goal_cmd_generator: GoalCommand = env.command_manager.get_term(goal_command_generator_name)  # type: ignore

    # get data
    distance_goal = torch.norm(asset.data.body_pos_w[:, 0, :2] - goal_cmd_generator.pos_command_w[:, :2], dim=1, p=2)

    angle_goal = torch.abs(wrap_to_pi(asset.data.heading_w - goal_cmd_generator.heading_command_w))

    velocity_norm = torch.norm(asset.data.root_vel_w[:, 0:6], dim=1, p=2)

    # scaling based on distance to goal. Intuition: closer to goal more weight on slowing down, empirical
    # 1.2m away ~= 0
    # 0.5m away ~= 0.75
    # 0m away = 1
    # TODO: add angle distance to scaling
    distance_scaling = torch.exp(-0.6 * (distance_goal + angle_goal) ** 3)

    # reward = 1 for zero velocity, ~=0 for 2 velocity_norm
    reward = torch.exp(-1.5 * velocity_norm) * distance_scaling
    return reward


def near_goal_angle(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_command_generator_name: str = "goal_command",
) -> torch.Tensor:
    """Reward the agent for matching the angle when near the goal.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.
        goal_command_generator_name: The name of the goal command generator.

    Returns:
        Dense reward [0, +1] based on the angle to the goal heading and the distance to the goal.

    .. note::
        The selected goal command generator should have ``pos_command_w`` and ``heading_command_w`` attributes. The
        default would be of the class :class:`nav_tasks.mdp.GoalCommand`.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    goal_cmd_generator: GoalCommand = env.command_manager.get_term(goal_command_generator_name)  # type: ignore

    # angle error of [0,pi]
    angle_goal = torch.abs(wrap_to_pi(asset.data.heading_w - goal_cmd_generator.heading_command_w))

    distance_goal = torch.norm(asset.data.body_pos_w[:, 0, :2] - goal_cmd_generator.pos_command_w[:, :2], dim=1, p=2)
    # scaling based on distance to goal. Intuition: closer to goal more weight on matching angle, empirical
    # 3m away ~= 0
    # 1m away ~= 0.5
    # 0m away = 1
    distance_scaling = torch.exp(-0.5 * distance_goal**2)
    # [0, 1] reward
    reward = (torch.pi - angle_goal) / torch.pi
    reward = reward * distance_scaling
    return reward


def backwards_movement(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward the agent for moving backwards using L2-Kernel

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.

    Returns:
        Dense reward [0, +1] based on the backward velocity.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    forward_velocity = asset.data.root_lin_vel_b[:, 0]
    backward_movement_idx = torch.where(
        forward_velocity < 0.0, torch.ones_like(forward_velocity), torch.zeros_like(forward_velocity)
    )
    reward = torch.square(backward_movement_idx * forward_velocity)
    reward = torch.clip(reward, min=0, max=1.0)
    return reward


def lateral_movement(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Reward the agent for moving lateral using L2-Kernel

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.

    Returns:
        Dense reward [0, +1] based on the lateral velocity.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    lateral_velocity = asset.data.root_lin_vel_b[:, 1]
    reward = torch.square(lateral_velocity)
    reward = torch.clip(reward, min=0, max=1.0)
    return reward
