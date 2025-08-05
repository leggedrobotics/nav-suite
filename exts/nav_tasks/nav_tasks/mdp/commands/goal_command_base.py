# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import CUBOID_MARKER_CFG
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.utils.math import quat_from_angle_axis

CYLINDER_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "cylinder": sim_utils.CylinderCfg(
            radius=1,
            height=1,
            axis="X",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    }
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .goal_command_base_cfg import GoalCommandBaseCfg


class GoalCommandBaseTerm(CommandTerm):
    r"""Base class for goal commands.

    This class is used to define the common visualization features for goal commands.
    """

    cfg: GoalCommandBaseCfg
    """Configuration for the command."""

    def __init__(self, cfg: GoalCommandBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # -- goal commands: (x, y, z)
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set the debug visualization for the command.

        Args:
            debug_vis (bool): Whether to enable debug visualization.
        """
        # create markers if necessary for the first time
        # for each marker type check that the correct command properties exist eg. need spawn position for spawn marker
        if debug_vis:
            if not hasattr(self, "box_goal_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/position_goal"
                marker_cfg.markers["cuboid"].size = (0.3, 0.3, 0.3)
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (1.0, 0.15, 0.0)
                marker_cfg.markers["cuboid"].visual_material.roughness = 0.7
                marker_cfg.markers["cuboid"].visual_material.metallic = 1.0
                self.box_goal_visualizer = VisualizationMarkers(marker_cfg)
                self.box_goal_visualizer.set_visibility(True)
            if self.cfg.vis_line and not hasattr(self, "line_to_goal_visualiser"):
                marker_cfg = CYLINDER_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/line_to_goal"
                marker_cfg.markers["cylinder"].height = 1
                marker_cfg.markers["cylinder"].radius = 0.05
                self.line_to_goal_visualiser = VisualizationMarkers(marker_cfg)
                self.line_to_goal_visualiser.set_visibility(True)
        else:
            if hasattr(self, "box_goal_visualizer"):
                self.box_goal_visualizer.set_visibility(False)
            if self.cfg.vis_line and hasattr(self, "line_to_goal_visualiser"):
                self.line_to_goal_visualiser.set_visibility(False)

    def _debug_vis_callback(self, event, env_ids: Sequence[int] | None = None):
        """Callback function for the debug visualization."""
        if env_ids is None:
            env_ids = slice(None)

        # update goal marker if it exists
        self.box_goal_visualizer.visualize(self.pos_command_w[env_ids])

        if self.cfg.vis_line:
            # update the line marker
            # calculate the difference vector between the robot root position and the goal position
            # TODO @tasdep this assumes that robot.data.body_pos_w exists
            difference = self.pos_command_w - self.robot.data.body_pos_w[:, 0, :3]
            translations = self.robot.data.body_pos_w[:, 0, :3]
            # calculate the scale of the arrow (Mx3)
            difference_norm = torch.norm(difference, dim=1)
            # translate half of the length along difference axis
            translations += difference / 2
            # scale along x axis
            scales = torch.vstack(
                [difference_norm, torch.ones_like(difference_norm), torch.ones_like(difference_norm)]
            ).T
            # convert the difference vector to a quaternion
            difference = torch.nn.functional.normalize(difference, dim=1)
            x_vec = torch.tensor([1, 0, 0]).float().to(self.pos_command_w.device)
            angle = -torch.acos(difference @ x_vec)
            axis = torch.linalg.cross(difference, x_vec.expand_as(difference))
            quat = quat_from_angle_axis(angle, axis)

            # TODO @tasdep add the line up in the air when at goal, requires some assumptions about existence of time at goal termination term

            # apply transforms
            self.line_to_goal_visualiser.visualize(
                translations=translations[env_ids], scales=scales[env_ids], orientations=quat[env_ids]
            )
