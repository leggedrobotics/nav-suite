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
    from . import single_object_cfg


def center_object_pattern(
    cfg: single_object_cfg.SingleObjectTerrainCfg,
) -> tuple[list[tuple[float, float]], np.ndarray]:
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    center = [(terrain_center[0], terrain_center[1])]
    origin = np.array([terrain_center[0] - 0.25 * cfg.size[0], terrain_center[1], 0])
    return center, origin


def cross_object_pattern(cfg: single_object_cfg.SingleObjectTerrainCfg) -> tuple[list[tuple[float, float]], np.ndarray]:
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    centers = [
        (terrain_center[0], terrain_center[1]),
        (terrain_center[0] + 0.2 * cfg.size[0], terrain_center[1] + 0.2 * cfg.size[1]),
        (terrain_center[0] - 0.2 * cfg.size[0], terrain_center[1] + 0.2 * cfg.size[1]),
        (terrain_center[0] + 0.2 * cfg.size[0], terrain_center[1] - 0.2 * cfg.size[1]),
        (terrain_center[0] - 0.2 * cfg.size[0], terrain_center[1] - 0.2 * cfg.size[1]),
    ]
    origin = np.array([terrain_center[0] - 0.25 * cfg.size[0], terrain_center[1], 0])
    return centers, origin


def extended_cross_object_pattern(
    cfg: single_object_cfg.SingleObjectTerrainCfg,
) -> tuple[list[tuple[float, float]], np.ndarray]:
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    centers = [
        (terrain_center[0], terrain_center[1]),
        (terrain_center[0] + 0.2 * cfg.size[0], terrain_center[1] + 0.2 * cfg.size[1]),
        (terrain_center[0] - 0.2 * cfg.size[0], terrain_center[1] + 0.2 * cfg.size[1]),
        (terrain_center[0] + 0.2 * cfg.size[0], terrain_center[1] - 0.2 * cfg.size[1]),
        (terrain_center[0] - 0.2 * cfg.size[0], terrain_center[1] - 0.2 * cfg.size[1]),
        (terrain_center[0] + 0.4 * cfg.size[0], terrain_center[1] + 0.4 * cfg.size[1]),
        (terrain_center[0] - 0.4 * cfg.size[0], terrain_center[1] + 0.4 * cfg.size[1]),
        (terrain_center[0] + 0.4 * cfg.size[0], terrain_center[1] - 0.4 * cfg.size[1]),
        (terrain_center[0] - 0.4 * cfg.size[0], terrain_center[1] - 0.4 * cfg.size[1]),
        (terrain_center[0] + 0.4 * cfg.size[0], terrain_center[1]),
        (terrain_center[0] - 0.4 * cfg.size[0], terrain_center[1]),
        (terrain_center[0], terrain_center[1] + 0.4 * cfg.size[1]),
        (terrain_center[0], terrain_center[1] - 0.4 * cfg.size[1]),
    ]
    origin = np.array([terrain_center[0] - 0.25 * cfg.size[0], terrain_center[1], 0])
    return centers, origin


def single_object_terrain(
    difficulty: float, cfg: single_object_cfg.SingleObjectTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # sample the dimensions of the object
    dim_value = cfg.dim_range[0] + difficulty * (cfg.dim_range[1] - cfg.dim_range[0])

    # initialize list of meshes
    meshes_list = list()
    # generate a ground plane for the terrain
    ground_plane = make_plane(cfg.size, height=0.0, center_zero=False)
    meshes_list.append(ground_plane)
    # generate the border if needed
    if cfg.border_width > 0.0:
        # obtain a list of meshes for the border
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], cfg.border_height / 2]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        make_borders = make_border(cfg.size, border_inner_size, abs(cfg.border_height), border_center)
        # add the border meshes to the list of meshes
        meshes_list += make_borders

    # get the position pattern
    center_positions, origin = cfg.position_pattern(cfg)

    # -- generate the object
    for center_position in center_positions:
        height = (np.random.rand(1) * (cfg.height_range[1] - cfg.height_range[0]) + cfg.height_range[0]).item()
        curr_pos = (center_position[0], center_position[1], height / 2)
        if cfg.object_type == "box":
            curr_dim = (dim_value, dim_value, height)
            meshes_list.append(trimesh.creation.box(curr_dim, trimesh.transformations.translation_matrix(curr_pos)))
        elif cfg.object_type == "cylinder":
            meshes_list.append(
                trimesh.creation.cylinder(
                    radius=dim_value, height=height, transform=trimesh.transformations.translation_matrix(curr_pos)
                )
            )
        elif cfg.object_type == "wall":
            curr_dim = (cfg.wall_width, dim_value, height)
            meshes_list.append(trimesh.creation.box(curr_dim, trimesh.transformations.translation_matrix(curr_pos)))
        else:
            raise ValueError(f"Object type {cfg.object_type} is not supported.")

    return meshes_list, origin
