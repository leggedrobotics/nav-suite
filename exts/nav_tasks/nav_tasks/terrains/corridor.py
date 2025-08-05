# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Functions to generate different terrains using the ``trimesh`` library."""

from __future__ import annotations

import numpy as np
import trimesh
from typing import TYPE_CHECKING, Literal

from isaaclab.terrains.trimesh.utils import *  # noqa: F401, F403
from isaaclab.terrains.trimesh.utils import make_border, make_plane

if TYPE_CHECKING:
    from . import corridor_cfg


def gen_wall_door_mesh(
    cfg: corridor_cfg.CorridorTerrainCfg,
    wall_len: float,
    door_width: float,
    wall_pos: list[float],
    orientation: Literal["horizontal", "vertical"] = "horizontal",
) -> list[trimesh.Trimesh]:
    """Generate a wall with a door in it.

    Args:
        cfg: The configuration for the corridor.
        wall_len: The length of the wall.
        wall_pos: The position of the wall.
        door_width: The width of the door.
        orientation: The orientation of the wall. Can be either "horizontal" or "vertical".

    Returns:
        The tri-mesh of the wall with the door.
    """
    # get the wall and door extents and positions
    door_pos = np.random.uniform(-wall_len / 2 + door_width / 2, wall_len / 2 - door_width / 2)

    # NOTE: as the difference function has kept leftovbers of the mesh, will construct the wall with door
    #       by creating three boxes: box above the door and boxes on each side of the door

    if orientation == "vertical":
        wall_till_door_extents = [wall_len / 2 + door_pos - door_width / 2, cfg.wall_thickness, cfg.wall_height]
        wall_after_door_extents = [wall_len / 2 - door_pos - door_width / 2, cfg.wall_thickness, cfg.wall_height]
        above_door_extents = [door_width, cfg.wall_thickness, cfg.wall_height - cfg.door_height]

        above_door_transform = trimesh.transformations.translation_matrix(
            [wall_pos[0] + door_pos, wall_pos[1], (cfg.wall_height - cfg.door_height) / 2 + cfg.door_height]
        )
        wall_till_door_transform = trimesh.transformations.translation_matrix(
            [wall_pos[0] - wall_len / 2 + wall_till_door_extents[0] / 2, wall_pos[1], cfg.wall_height / 2]
        )
        wall_after_door_transform = trimesh.transformations.translation_matrix(
            [wall_pos[0] + wall_len / 2 - wall_after_door_extents[0] / 2, wall_pos[1], cfg.wall_height / 2]
        )

    elif orientation == "horizontal":
        wall_till_door_extents = [cfg.wall_thickness, wall_len / 2 + door_pos - door_width / 2, cfg.wall_height]
        wall_after_door_extents = [cfg.wall_thickness, wall_len / 2 - door_pos - door_width / 2, cfg.wall_height]
        above_door_extents = [cfg.wall_thickness, door_width, cfg.wall_height - cfg.door_height]

        above_door_transform = trimesh.transformations.translation_matrix(
            [wall_pos[0], wall_pos[1] + door_pos, (cfg.wall_height - cfg.door_height) / 2 + cfg.door_height]
        )
        wall_till_door_transform = trimesh.transformations.translation_matrix(
            [wall_pos[0], wall_pos[1] - wall_len / 2 + wall_till_door_extents[1] / 2, cfg.wall_height / 2]
        )
        wall_after_door_transform = trimesh.transformations.translation_matrix(
            [wall_pos[0], wall_pos[1] + wall_len / 2 - wall_after_door_extents[1] / 2, cfg.wall_height / 2]
        )

    # get the wall and door meshes
    above_door = trimesh.creation.box(extents=above_door_extents, transform=above_door_transform)
    wall_till_door = trimesh.creation.box(extents=wall_till_door_extents, transform=wall_till_door_transform)
    wall_after_door = trimesh.creation.box(extents=wall_after_door_extents, transform=wall_after_door_transform)

    # return the difference between the wall and the door
    return [above_door, wall_till_door, wall_after_door]


def corridor_terrain(
    difficulty: float, cfg: corridor_cfg.CorridorTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a corrdior including doors.

    .. image:: ../../_static/terrains/corridor_terrain.jpg
       :width: 45%
       :align: center

    The terrain is designed to contain a corrdior with a varying width. In addition, it
    contains a wall with a door that separates it in the middle as well as door on each side
    of the corrdior.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # get door width based on difficulty
    door_width = cfg.door_width_range[1] - difficulty * (cfg.door_width_range[1] - cfg.door_width_range[0])

    # get the width of the corridor
    width = np.random.uniform(cfg.width_range[0], cfg.width_range[1])
    # get the terrain center
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    # separation wall ooffset
    separation_wall_offset = np.random.uniform(cfg.separation_wall_offset_range[0], cfg.separation_wall_offset_range[0])

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
        make_borders = make_border(cfg.size, border_inner_size, 0.1, border_center)
        # add the border meshes to the list of meshes
        meshes_list += make_borders

    # create the corridor walls and the separation wall
    # -- create the bottom wall
    meshes_list += gen_wall_door_mesh(
        cfg, width, door_width, [cfg.border_width + cfg.wall_thickness / 2, terrain_center[1]]
    )
    # -- create the top wall
    meshes_list += gen_wall_door_mesh(
        cfg, width, door_width, [cfg.size[0] - cfg.border_width - cfg.wall_thickness / 2, terrain_center[1]]
    )
    # -- create the separation wall
    meshes_list += gen_wall_door_mesh(
        cfg, width, door_width, [terrain_center[0] + separation_wall_offset, terrain_center[1]]
    )
    # -- create the left wall
    meshes_list += gen_wall_door_mesh(
        cfg,
        cfg.size[1] - cfg.border_width * 2,
        door_width,
        [terrain_center[0], terrain_center[0] - width / 2 - cfg.wall_thickness / 2],
        orientation="vertical",
    )
    # -- create the right wall
    meshes_list += gen_wall_door_mesh(
        cfg,
        cfg.size[1] - cfg.border_width * 2,
        door_width,
        [terrain_center[0], terrain_center[0] + width / 2 + cfg.wall_thickness / 2],
        orientation="vertical",
    )

    # compute the origin of the terrain
    origin = np.array([terrain_center[0], terrain_center[1], terrain_center[2]])

    return meshes_list, origin
