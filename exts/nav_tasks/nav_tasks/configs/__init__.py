# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents, env_cfg_base

gym.register(
    id="NavTasks-DepthImgNavigation-PPO-Anymal-D-DEV",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": env_cfg_base.NavTasksDepthNavEnvCfg_DEV,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOCfgDEV",
    },
)
gym.register(
    id="NavTasks-DepthImgNavigation-PPO-Anymal-D-TRAIN",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": env_cfg_base.NavTasksDepthNavEnvCfg_TRAIN,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOCfg",
    },
)
gym.register(
    id="NavTasks-DepthImgNavigation-PPO-Anymal-D-PLAY",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": env_cfg_base.NavTasksDepthNavEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOCfgDEV",
    },
)
