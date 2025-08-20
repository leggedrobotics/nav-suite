# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.assets import check_file_path, read_file

if TYPE_CHECKING:
    from .navigation_actions_cfg import NavigationSE2ActionCfg


class NavigationSE2Action(ActionTerm):
    """Actions to navigate a robot by following some path."""

    cfg: NavigationSE2ActionCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: NavigationSE2ActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # check if policy file exists
        if not check_file_path(cfg.low_level_policy_file):
            raise FileNotFoundError(f"Policy file '{cfg.low_level_policy_file}' does not exist.")
        # load policies
        file_bytes = read_file(self.cfg.low_level_policy_file)
        self.low_level_policy = torch.jit.load(file_bytes, map_location=self.device)
        self.low_level_policy.eval()
        if self.cfg.freeze_low_level_policy:
            self.low_level_policy = torch.jit.freeze(self.low_level_policy)

        # prepare joint position actions
        if not isinstance(self.cfg.low_level_action, list):
            self.cfg.low_level_action = [self.cfg.low_level_action]
        self.low_level_action_terms: list[ActionTerm] = [
            term_cfg.class_type(term_cfg, env) for term_cfg in self.cfg.low_level_action
        ]

        if self.cfg.reorder_joint_list:
            # for policies trained with Isaac Gym or any other engine, reorder the joint based on a provided list of joint names
            self.joint_mapping_gym_to_sim = env.scene["robot"].find_joints(
                env.scene["robot"].joint_names, self.cfg.reorder_joint_list, preserve_order=True
            )[0]

        # parse scale
        if self.cfg.scale is not None:
            if isinstance(self.cfg.scale, (float, int)):
                self._scale = float(self.cfg.scale)
            elif isinstance(self.cfg.scale, list):
                self._scale = torch.tensor([self.cfg.scale], device=self.device).repeat(self.num_envs, 1)
            else:
                raise ValueError(
                    f"Unsupported scale type: {type(self.cfg.scale)}. Supported types are float, int, and list."
                )
        else:
            self._scale = 1

        # parse offset
        if self.cfg.offset is not None:
            if isinstance(self.cfg.offset, (float, int)):
                self._offset = float(self.cfg.offset)
            elif isinstance(self.cfg.offset, list):
                self._offset = torch.tensor([self.cfg.offset], device=self.device).repeat(self.num_envs, 1)
            else:
                raise ValueError(
                    f"Unsupported offset type: {type(self.cfg.offset)}. Supported types are float, int, and list."
                )
        else:
            self._offset = 0

        # parse clip
        if self.cfg.clip_mode != "none":
            if self.cfg.clip_mode == "minmax":
                assert isinstance(
                    self.cfg.clip, list
                ), "Clip must be a list of tuples of (min, max) values if clip mode is 'minmax'"
                assert (
                    len(self.cfg.clip) == self.cfg.action_dim
                ), "Clip must have the same length as the action dimension"
                self._clip = torch.tensor(self.cfg.clip, device=self.device).repeat(self.num_envs, 1, 1)
            elif self.cfg.clip_mode == "tanh":
                pass
            else:
                raise ValueError(
                    f"Unsupported clip mode: {self.cfg.clip_mode}. Supported modes are 'minmax', 'tanh' and 'none'."
                )

        # parse momentum
        if self.cfg.momentum is not None:
            if isinstance(self.cfg.momentum, (float, int)):
                self._momentum = float(self.cfg.momentum)
            elif isinstance(self.cfg.momentum, list):
                self._momentum = torch.tensor([self.cfg.momentum], device=self.device).repeat(self.num_envs, 1)
            else:
                raise ValueError(
                    f"Unsupported momentum type: {type(self.cfg.momentum)}. Supported types are float, int, and list."
                )
        else:
            self._momentum = 0

        # set up buffers
        self._init_buffers()

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self.cfg.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_navigation_velocity_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_navigation_velocity_actions

    @property
    def low_level_actions(self) -> torch.Tensor:
        return self._low_level_actions

    @property
    def prev_low_level_actions(self) -> torch.Tensor:
        return self._prev_low_level_actions

    """
    Operations.
    """

    def process_actions(self, actions):
        """Process low-level navigation actions. This function is called with a frequency of 10Hz"""

        # Store the navigation actions
        self._raw_navigation_velocity_actions[:] = actions
        self._processed_navigation_velocity_actions[:] = actions.clone().view(self.num_envs, self.cfg.action_dim)
        # clip actions
        if self.cfg.clip_mode == "minmax":
            self._processed_navigation_velocity_actions = torch.clamp(
                self._processed_navigation_velocity_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        elif self.cfg.clip_mode == "tanh":
            self._processed_navigation_velocity_actions = torch.tanh(self._processed_navigation_velocity_actions)
        # apply the affine transformations
        self._processed_navigation_velocity_actions *= self._scale
        self._processed_navigation_velocity_actions += self._offset

        # apply a momentum offset
        self._processed_navigation_velocity_actions += (
            torch.norm(self._env.scene.articulations["robot"].data.root_lin_vel_b, dim=1, keepdim=True) * self._momentum
        )

    def apply_actions(self):
        """Apply low-level actions for the simulator to the physics engine. This functions is called with the
        simulation frequency of 200Hz. Since low-level locomotion runs at 50Hz, we need to decimate the actions."""

        if self._counter % self.cfg.low_level_decimation == 0:
            self._counter = 0
            self._prev_low_level_actions[:] = self._low_level_actions.clone()
            # Get low level actions from low level policy
            self._low_level_actions[:] = self.low_level_policy(
                self._env.observation_manager.compute_group(group_name=self.cfg.low_level_obs_group)
            )
            # reorder joints
            if self.cfg.reorder_joint_list is not None:
                self._low_level_actions = self._low_level_actions[:, self.joint_mapping_gym_to_sim]

            # split the actions and apply to each tensor
            idx = 0
            for term in self.low_level_action_terms:
                term_actions = self._low_level_actions[:, idx : idx + term.action_dim]
                term.process_actions(term_actions)
                idx += term.action_dim

        # Apply low level actions
        for term in self.low_level_action_terms:
            term.apply_actions()
        self._counter += 1

    """
    Helper functions
    """

    def _init_buffers(self):
        # Prepare buffers
        self._raw_navigation_velocity_actions = torch.zeros(self.num_envs, self.cfg.action_dim, device=self.device)
        self._processed_navigation_velocity_actions = torch.zeros(
            (self.num_envs, self.cfg.action_dim), device=self.device
        )
        self._low_level_actions = torch.zeros(
            self.num_envs, sum([term.action_dim for term in self.low_level_action_terms]), device=self.device
        )
        self._prev_low_level_actions = torch.zeros_like(self._low_level_actions)
        self._low_level_step_dt = self.cfg.low_level_decimation * self._env.physics_dt
        self._counter = 0
