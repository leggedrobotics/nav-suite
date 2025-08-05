# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from collections.abc import Callable
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

from .observations import PosActionHistoryTerm


@configclass
class PosActionHistoryCfg(ObservationTermCfg):

    func: Callable[..., torch.Tensor] = PosActionHistoryTerm

    robot: SceneEntityCfg = SceneEntityCfg(name="robot")
    """Name of the robot entity which past base states are stored."""

    command_name: str = MISSING
    """Name of the command term which generates the actions for the robot."""

    decimation: int = MISSING
    """Decimation factor for the history."""

    history_length: int = 20
    """Number of past entries to store."""

    debug_vis: bool = True
    """Whether to visualize the pose history."""

    debug_vis_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/PosActionHistory",
        markers={
            "pose_history": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
        },
    )
    """Configuration for the debug visualization."""
