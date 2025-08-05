# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.types
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers

from nav_tasks.mdp.actions import NavigationSE2Action

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .observations_cfg import PosActionHistoryCfg


class PosActionHistoryTerm(ManagerTermBase):
    """An observation term that embeds the RGB image using the pre-trained DINO."""

    cfg: PosActionHistoryCfg

    def __init__(self, cfg: PosActionHistoryCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # access robot and command term
        self._robot: Articulation = env.scene[self.cfg.robot.name]
        self._command_term: NavigationSE2Action = env.action_manager._terms[self.cfg.command_name]

        # initialize buffers
        self._pose_history = torch.zeros(
            self.num_envs, self.cfg.history_length, self._robot.data.root_pos_w.shape[1], device=self.device
        )
        self._action_history = torch.zeros(
            self.num_envs, self.cfg.history_length, self._command_term.action_dim, device=self.device
        )

        # initialize counter
        self._counter_history = torch.zeros(self.num_envs, device=self.device)
        self._env_reset_flag = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)

        # initialize debug visualizer
        if self.cfg.debug_vis:
            self._debug_vis_marker = VisualizationMarkers(self.cfg.debug_vis_cfg)
            self._debug_vis_marker.set_visibility(True)

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # reset buffers
        self._pose_history[env_ids] = torch.zeros(
            len(env_ids), self.cfg.history_length, self._robot.data.root_pos_w.shape[1], device=self.device
        )
        self._action_history[env_ids] = torch.zeros(
            len(env_ids), self.cfg.history_length, self._command_term.action_dim, device=self.device
        )

        # reset counter
        self._counter_history[env_ids] = 0
        self._env_reset_flag[env_ids] = True

    def _store_history_information(self):
        env_update = self._counter_history % self.cfg.decimation == 0

        # store action and pose history
        self._action_history[env_update] = torch.roll(self._action_history[env_update], shifts=1, dims=1)
        self._action_history[env_update, 0, :] = self._command_term.processed_actions[env_update]
        self._pose_history[env_update] = torch.roll(self._pose_history[env_update], shifts=1, dims=1)
        self._pose_history[env_update, 0, :] = self._robot.data.root_pos_w[env_update]
        self._counter_history[env_update] = 0

        # when the reset flag, set the entire pose history to the current pose
        self._pose_history[self._env_reset_flag] = self._robot.data.root_pos_w[self._env_reset_flag, None, :]
        self._counter_history[self._env_reset_flag] = 0
        self._env_reset_flag[:] = False

        # reset the counter
        self._counter_history += 1

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        # update env information
        self._store_history_information()

        # compute pose history in base frame
        pose_history_base = math_utils.quat_apply_inverse(
            self._robot.data.root_quat_w[:, None, :].repeat(1, self.cfg.history_length, 1),
            (self._pose_history - self._robot.data.root_pos_w[:, None, :]),
        )[..., :2]
        # concatenate action history and pose history
        output = torch.cat([self._action_history, pose_history_base], dim=-1)

        if self.cfg.debug_vis:
            self._debug_vis_marker.visualize(self._pose_history.view(-1, 3))

        # view output as [num_envs, num_obs * ob_dim_single]
        return output.view(self.num_envs, -1)


"""
Actions.
"""


def last_low_level_action(
    env: ManagerBasedRLEnv, action_term: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The last low-level action."""
    action_term: NavigationSE2Action = env.action_manager._terms[action_term]
    return action_term.low_level_actions[:, asset_cfg.joint_ids]


def second_last_low_level_action(
    env: ManagerBasedRLEnv, action_term: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The second to last low level action."""
    action_term: NavigationSE2Action = env.action_manager._terms[action_term]
    return action_term.prev_low_level_actions[:, asset_cfg.joint_ids]


"""
Commands.
"""


def vel_commands(env: ManagerBasedRLEnv, action_term: str) -> torch.Tensor:
    """The velocity command generated by the planner and given as input to the step function"""
    action_term: NavigationSE2Action = env.action_manager._terms[action_term]
    return action_term.processed_actions
