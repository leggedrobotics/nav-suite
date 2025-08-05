# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""This sub-module contains the common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from isaaclab.terrains import TerrainImporter

from nav_tasks.utils.maths import lin_interp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from nav_tasks.mdp import GoalCommand


def modify_terrain_level(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    promote_ratio_range: tuple[float, float],
    demote_ratio_range: tuple[float, float],
    step_range: tuple[int, int],
    success_termination_term_name: str = "goal_reached",
    failure_termination_term_name: str = "base_contact",
    timeout_termination_term_name: str = "time_out",
) -> torch.Tensor:
    """Curriculum based on whether the robot reached the goal or not

    This term is used to increase the difficulty of the terrain when the robot reaches the goal and decrease the
    difficulty when the robot collides with an obstacle.

    Args:
        env: The learning environment.
        env_ids: The ids of the environments that are affected by the curriculum.
        promote_ratio_range: The range of the promote ratio, [0,1]. 0.5 -> 50% of successful robots will be promoted.
        demote_ratio_range: The range of the demote ratio, [0,1]. 0.5 -> 50% of unsuccessful robots will be demoted.
        step_range: The range of the steps where the curriculum term is active.
        success_termination_term_name: The name of the termination term for successful robots.
        failure_termination_term_name: The name of the termination term for failed robots.
        timeout_termination_term_name: The name of the termination term for timeout robots.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.


    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    terrain: TerrainImporter = env.scene.terrain

    to_promote = env.termination_manager.get_term(success_termination_term_name).clone()[env_ids]
    to_demote = env.termination_manager.get_term(failure_termination_term_name).clone()[env_ids]
    to_demote |= env.termination_manager.get_term(timeout_termination_term_name).clone()[env_ids]

    promote_ratio = lin_interp(*step_range, *promote_ratio_range, env.common_step_counter)

    # Generate a random matrix of the same shape
    random_matrix = torch.rand(to_promote.shape).to(to_promote.device)
    # Create a mask to only promote the values that are greater than the promote ratio
    mask = random_matrix > promote_ratio
    # Apply the mask
    to_promote[mask] = False

    demote_ratio = lin_interp(*step_range, *demote_ratio_range, env.common_step_counter)

    random_matrix = torch.rand(to_demote.shape).to(to_demote.device)
    mask = random_matrix > demote_ratio
    to_demote[mask] = False

    # When reset() is called before the sim starts, don't do the update.
    if env.common_step_counter > 0:
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=to_promote.device)
        terrain.update_env_origins(env_ids, to_promote, to_demote)

    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def modify_goal_distance_in_steps(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    update_rate_steps: float,
    min_path_length_range: tuple[float, float],
    max_path_length_range: tuple[float, float],
    step_range: tuple[int, int],
):
    """Curriculum that increases the goal distance in small steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        update_rate_steps: The number of steps after which the goal distance is increased.
        min_path_length_range: The range of the minimum path length.
        max_path_length_range: The range of the maximum path length.
        step_range: The range of the steps where the curriculum term is active.

    Returns:
        The average goal distance.
    """
    # extract the used quantities (to enable type-hinting)
    goal_cmd_generator: GoalCommand = env.command_manager.get_term("goal_command")

    # compute the new configuration
    # NOTE: env.common_step_counter = learning_iterations * num_steps_per_env
    # FIXME: not sure if this is correct
    min_path_length = lin_interp(*step_range, *min_path_length_range, env.common_step_counter)
    max_path_length = lin_interp(*step_range, *max_path_length_range, env.common_step_counter)

    # Resample trajectories if the number of steps is exceeded
    if (
        env.common_step_counter > goal_cmd_generator.last_update_config_env_step + update_rate_steps
        or env.common_step_counter == 0
    ):
        goal_cmd_generator.update_trajectory_config(
            max_path_length=max_path_length,
            min_path_length=min_path_length,
        )
    # Return average trajectory length
    return goal_cmd_generator.paths[..., -1].mean()


def modify_heading_randomization_linearly(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    event_term_name: str,
    perturbation_range: tuple[float, float],
    step_range: tuple[int, int],
):
    """Curriculum that modifies the "yaw_range" parameter linearly. Can be used for reset_robot_position or
    TerrainAnalysisRootReset, as both use a "yaw_range" addition to spawn conditions.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        event_term_name: The name of the event term.
        perturbation_range: The range of the yaw perturbation over the curriculum.
        step_range: The range of the steps where the curriculum term is active.

    Returns:
        The new perturbation of the randomization term.
    """
    # obtain term settings
    term_cfg = env.event_manager.get_term_cfg(event_term_name)
    # compute the new weight
    # NOTE: env.common_step_counter = learning_iterations * num_steps_per_env
    perturbation = lin_interp(*step_range, *perturbation_range, env.common_step_counter)

    # update term settings
    term_cfg.params["yaw_range"] = (-perturbation, perturbation)  # type: ignore
    env.event_manager.set_term_cfg(event_term_name, term_cfg)
    return perturbation


def modify_goal_conditions(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    termination_term_name: str,
    time_range: tuple[float, float],
    distance_range: tuple[float, float],
    angle_range: tuple[float, float],
    speed_range: tuple[float, float],
    step_range: tuple[int, int],
) -> dict:
    """Curriculum that modifies the termination conditions linearly. Used for the StayedAtGoal termination term.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        termination_term_name: The name of the termination term.
        time_range: The range of the termination time threshold.
        distance_range: The range of the termination distance threshold.
        angle_range: The range of the termination angle threshold.
        speed_range: The range of the termination speed threshold.
        step_range: The range of the steps where the curriculum term is active.

    Returns:
        A dict containing the new termination conditions.
    """
    # obtain term settings
    term_cfg = env.termination_manager.get_term_cfg(termination_term_name)
    # compute the new weights
    # NOTE: env.common_step_counter = learning_iterations * num_steps_per_env)
    params = {
        "time_threshold": lin_interp(*step_range, *time_range, env.common_step_counter),
        "distance_threshold": lin_interp(*step_range, *distance_range, env.common_step_counter),
        "angle_threshold": lin_interp(*step_range, *angle_range, env.common_step_counter),
        "speed_threshold": lin_interp(*step_range, *speed_range, env.common_step_counter),
    }

    # update term settings
    term_cfg.params.update(params)
    env.termination_manager.set_term_cfg(termination_term_name, term_cfg)
    return params


def change_reward_weight(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_name: str,
    weight_range: tuple[float, float],
    step_range: tuple[int, int],
    mode: Literal["linear", "exponential"] = "linear",
):
    """Curriculum that changes the reward weight.

    For the exponential mode, the scaling throughout the defined range is as demonstrated in the following plot:
    .. image:: {NAVSUITE_TASKS_EXT_DIR}/docs/figures/exp_decay.png
        :alt: Exponential Decay

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected. But default in the function call.
        reward_name: The name of the reward term.
        weight_range: The range of the reward weight.
        step_range: The range of the steps where the curriculum term is active.
        mode: The mode of the reward weight change. Options are "linear" or "exponential".

    Returns:
        The new reward weight.
    """
    reward = env.reward_manager.get_term_cfg(reward_name)

    # If before the curriculum starts, keep the original weight
    if env.common_step_counter < step_range[0]:
        return reward.weight

    # If after the curriculum ends, set to final weight
    if env.common_step_counter >= step_range[1]:
        reward.weight = weight_range[1]
        return reward.weight

    # Normalize the progress between 0 and 1
    progress = (env.common_step_counter - step_range[0]) / (step_range[1] - step_range[0])

    if mode == "linear":
        reward.weight = weight_range[0] + (weight_range[1] - weight_range[0]) * progress
    elif mode == "exponential":
        reward.weight = weight_range[0] + (weight_range[1] - weight_range[0]) * math.exp(5 * (progress - 1))

    return reward.weight
