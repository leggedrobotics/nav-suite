# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Functions to generate different terrains using the ``trimesh`` library."""

from __future__ import annotations

import numpy as np
import trimesh
from typing import TYPE_CHECKING

from isaaclab.terrains.trimesh.utils import *  # noqa: F401, F403
from isaaclab.terrains.trimesh.utils import make_border, make_plane

if TYPE_CHECKING:
    from . import corridor_turn_cfg


# Helper for wall creation
def wall(orientation, start, end, height, thickness):

    center = ((np.array(start) + np.array(end)) / 2).tolist()
    if orientation == "vertical":
        length = np.abs(start[0] - end[0])
        extents = [length, thickness, height]
    elif orientation == "horizontal":
        length = np.abs(start[1] - end[1])
        extents = [thickness, length, height]

    center[2] = height / 2
    return trimesh.creation.box(extents=extents, transform=trimesh.transformations.translation_matrix(center))


def corridor_turn_terrain(difficulty: float, cfg: corridor_turn_cfg.CorridorTurnTerrainCfg):
    """Generate a corridor with a 90 or 180 degree turn."""
    # get the width of the corridor
    width = np.random.uniform(cfg.width_range[0], cfg.width_range[1])
    # get the terrain center
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    # initialize list of meshes
    meshes_list = list()
    # generate a ground plane for the terrain
    ground_plane = make_plane(cfg.size, height=0.0, center_zero=False)
    meshes_list.append(ground_plane)

    # generate the border if needed
    if cfg.border_width > 0.0:
        # obtain a list of meshes for the border
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        make_borders = make_border(cfg.size, border_inner_size, cfg.border_height, border_center)
        # add the border meshes to the list of meshes
        meshes_list += make_borders

    t = cfg.wall_thickness
    h = cfg.wall_height

    if cfg.turn_type == "90":
        # Ensure the size of the terrain is sufficient for the corridor and turn
        if cfg.size[0] / 2 < width / 2 or cfg.size[1] / 2 < width / 2:
            print(
                "Terrain size is too small for the specified corridor width and a 90 degree turn. Adjusting parameters"
                " (terrain may not be useful)."
            )
            if cfg.size[0] / 2 < width / 2:
                width = cfg.size[0] - 0.1
            if cfg.size[1] / 2 < width / 2:
                width = cfg.size[1] - 0.1
        # L-shaped corridor: horizontal then vertical
        # Horizontal segments
        meshes_list.append(
            wall(
                "horizontal",
                [terrain_center[0] + width / 2, terrain_center[1] - width / 2, 0],
                [terrain_center[1] + width / 2, cfg.size[1], 0],
                h,
                t,
            )
        )
        meshes_list.append(
            wall(
                "horizontal",
                [terrain_center[0] - width / 2, terrain_center[1] + width / 2, 0],
                [terrain_center[0] - width / 2, cfg.size[1], 0],
                h,
                t,
            )
        )
        # Vertical segments
        meshes_list.append(
            wall(
                "vertical",
                [terrain_center[0] + width / 2, terrain_center[1] - width / 2, 0],
                [0, terrain_center[1] - width / 2, 0],
                h,
                t,
            )
        )
        meshes_list.append(
            wall(
                "vertical",
                [terrain_center[0] - width / 2, terrain_center[1] + width / 2, 0],
                [0, terrain_center[1] + width / 2, 0],
                h,
                t,
            )
        )
    elif cfg.turn_type == "180":
        # U-shaped corridor: two verticals and a horizontal at the top
        u_top_width = np.random.uniform(cfg.u_top_width_range[0], cfg.u_top_width_range[1])
        # Ensure the size of the terrain is sufficient for the corridor and turn
        if cfg.size[0] / 2 < width or cfg.size[1] / 2 < width + u_top_width / 2:
            print(
                "Terrain size is too small for the specified corridor width and a 180 degree turn. Adjusting parameters"
                " (terrain may not be useful)."
            )
            if cfg.size[0] / 2 < width:
                width = cfg.size[0] / 2 - 0.1
            if cfg.size[1] / 2 < width + u_top_width / 2:
                u_top_width = 2 * (cfg.size[1] / 2 - width) - 0.1
        # U top segments
        meshes_list.append(
            wall(
                "horizontal",
                [terrain_center[0], terrain_center[1] - u_top_width / 2, 0],
                [terrain_center[0], terrain_center[1] + u_top_width / 2, 0],
                h,
                t,
            )
        )
        meshes_list.append(
            wall(
                "horizontal",
                [terrain_center[0] + width, terrain_center[1] - u_top_width / 2 - width, 0],
                [terrain_center[0] + width, terrain_center[1] + u_top_width / 2 + width, 0],
                h,
                t,
            )
        )
        # Right segments
        meshes_list.append(
            wall(
                "vertical",
                [terrain_center[0], terrain_center[1] - u_top_width / 2, 0],
                [0, terrain_center[1] - u_top_width / 2, 0],
                h,
                t,
            )
        )
        meshes_list.append(
            wall(
                "vertical",
                [terrain_center[0] + width, terrain_center[1] - u_top_width / 2 - width, 0],
                [0, terrain_center[1] - u_top_width / 2 - width, 0],
                h,
                t,
            )
        )
        # Vertical segments
        meshes_list.append(
            wall(
                "vertical",
                [terrain_center[0], terrain_center[1] + u_top_width / 2, 0],
                [0, terrain_center[1] + u_top_width / 2, 0],
                h,
                t,
            )
        )
        meshes_list.append(
            wall(
                "vertical",
                [terrain_center[0] + width, terrain_center[1] + u_top_width / 2 + width, 0],
                [0, terrain_center[1] + u_top_width / 2 + width, 0],
                h,
                t,
            )
        )

        # Center wall between the two inner vertical segments
        meshes_list.append(
            wall("vertical", [terrain_center[0] - width, terrain_center[1], 0], [0, terrain_center[1], 0], h, t)
        )
    else:
        raise ValueError("turn_type must be '90' or '180'")

    # compute the origin of the terrain
    origin = np.array([terrain_center[0], terrain_center[1], terrain_center[2]])
    return meshes_list, origin
