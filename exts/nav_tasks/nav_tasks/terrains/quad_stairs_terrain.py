# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import trimesh
from typing import TYPE_CHECKING

from isaaclab.terrains.trimesh.utils import *  # noqa: F401, F403
from isaaclab.terrains.trimesh.utils import make_border, make_plane

if TYPE_CHECKING:
    from . import quad_stairs_terrain_cfg


def quad_pyramid_stairs_terrain(
    difficulty: float, cfg: quad_stairs_terrain_cfg.MeshQuadPyramidStairsCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a four pyramid stair pattern around a center space.

    .. image:: ../../_static/terrains/quad_stairs_terrain.jpg
       :width: 45%
       :align: center

    The terrain is a combination of four pyramid stairs that are placed on each side of
    a center platform with a certain width.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])

    # compute number of steps in x and y direction
    num_steps_x = (cfg.size[0] - 2 * cfg.border_width - cfg.platform_width) / 2 // (2 * cfg.step_width) + 1
    num_steps_y = (cfg.size[1] - 2 * cfg.border_width - cfg.platform_width) / 2 // (2 * cfg.step_width) + 1
    # we take the minimum number of steps in x and y direction
    num_steps = int(min(num_steps_x, num_steps_y))

    # initialize list of meshes
    meshes_list = list()
    # generate a ground plane for the terrain
    ground_plane = make_plane(cfg.size, height=0.0, center_zero=False)
    meshes_list.append(ground_plane)
    # generate the border if needed
    if cfg.border_width > 0.0 and not cfg.holes:
        # obtain a list of meshes for the border
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -step_height / 2]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        make_borders = make_border(cfg.size, border_inner_size, step_height, border_center)
        # add the border meshes to the list of meshes
        meshes_list += make_borders

    # generate the terrain
    # -- compute the position of the center of the terrain and the centers of the stairs ordered above, left, below, right of the center
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    move_x = (terrain_size[0] - cfg.platform_width) / 4 + cfg.platform_width / 2
    move_y = (terrain_size[1] - cfg.platform_width) / 4 + cfg.platform_width / 2
    stairs_centers = [
        [terrain_center[0], terrain_center[1] + move_y, terrain_center[2]],
        [terrain_center[0], terrain_center[1] - move_y, terrain_center[2]],
        [terrain_center[0] + move_x, terrain_center[1], terrain_center[2]],
        [terrain_center[0] - move_x, terrain_center[1], terrain_center[2]],
    ]
    # -- generate the stair pattern
    for k in range(num_steps):
        # check if we need to add holes around the steps
        box_size = (2 * (num_steps - k) * cfg.step_width, 2 * (num_steps - k) * cfg.step_width)
        # compute the quantities of the box
        # -- location
        box_z = terrain_center[2] + k * step_height / 2.0
        box_offset = ((num_steps - k) - 0.5) * cfg.step_width
        # -- dimensions
        box_height = (k + 2) * step_height
        # generate the boxes
        for i in range(4):
            current_center = stairs_centers[i]
            # top/bottom
            box_dims = (box_size[0], cfg.step_width, box_height)
            # -- top
            box_pos = (current_center[0], current_center[1] - box_offset, box_z)
            box_top = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
            # -- bottom
            box_pos = (current_center[0], current_center[1] + box_offset, box_z)
            box_bottom = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
            # right/left
            box_dims = (cfg.step_width, box_size[1], box_height)
            # -- right
            box_pos = (current_center[0] - box_offset, current_center[1], box_z)
            box_right = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
            # -- left
            box_pos = (current_center[0] + box_offset, current_center[1], box_z)
            box_left = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
            # add the boxes to the list of meshes
            meshes_list += [box_top, box_bottom, box_right, box_left]

    origin = np.array([terrain_center[0], terrain_center[1], (num_steps + 1) * step_height])

    return meshes_list, origin
