# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Functions to generate different terrains using the ``trimesh`` library."""

from __future__ import annotations

import math
import numpy as np
import pickle as pkl
import trimesh
from typing import TYPE_CHECKING

import omni.log
from isaaclab.terrains.trimesh.utils import *  # noqa: F401, F403
from isaaclab.terrains.trimesh.utils import make_border, make_plane

if TYPE_CHECKING:
    from . import stairs_ramp_terrain_cfg


def stairs_ramp_terrain(
    difficulty: float, cfg: stairs_ramp_terrain_cfg.StairsRampTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a stairs and ramp next to each other

    .. image:: ../../_static/terrains/stairs_ramp_terrain.jpg
       :width: 45%
       :align: center

    Terrain is designed to have a stairs and ramp next to each other. Can be used for eval purposes with increasing
    step height for the stairs and increasing slope for the ramp.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    if not (cfg.modify_step_height or cfg.modify_ramp_slope):
        omni.log.warn(
            "No change based on difficulty performed because neither step height nor ramp slope are modified."
        )

    # compute number of steps in x and y direction
    ramp_len = cfg.size[0] - 2 * cfg.border_width - 2 * cfg.platform_width
    num_steps = int(ramp_len / cfg.step_width)

    # compute step height and ramp slope based on difficulty
    if cfg.modify_step_height:
        assert cfg.step_height_range is not None, "Step height range must be defined when modifying step height."
        step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
        ramp_height = step_height * num_steps
    elif cfg.modify_ramp_slope:
        assert cfg.ramp_slope_range is not None, "Ramp slope range must be defined when modifying ramp slope."
        ramp_slope = cfg.ramp_slope_range[0] + difficulty * (cfg.ramp_slope_range[1] - cfg.ramp_slope_range[0])
        ramp_height = math.tan(np.deg2rad(ramp_slope)) * ramp_len
        step_height = ramp_height / num_steps
    else:
        assert (
            cfg.step_height_range is not None and cfg.ramp_slope_range is not None
        ), "Step height range  and ramp slope must be defined when neither modifying step height nor ramp slope."
        step_height = cfg.step_height_range[0] if isinstance(cfg.step_height_range, tuple) else cfg.step_height_range
        ramp_slope = cfg.ramp_slope_range[0] if isinstance(cfg.ramp_slope_range, tuple) else cfg.ramp_slope_range
        ramp_height = math.tan(np.deg2rad(ramp_slope)) * ramp_len
        assert np.isclose(step_height * num_steps, ramp_height, atol=0.1), (
            "No difficulty modification is defined and the define step height and ramp slope do not match."
            f"Current step height: {step_height}, current ramp slope: {ramp_slope} with final stairs height of "
            f"{step_height * num_steps} and final ramp length of {ramp_height}."
        )

    if cfg.random_state_file is not None:
        with open(cfg.random_state_file, "rb") as f:
            np_random_state = pkl.load(f)
        np.random.set_state(np_random_state)

    # get the width of the stairs and ramp
    random_width = np.random.uniform(-cfg.width_randomization, cfg.width_randomization)
    stairs_width = (cfg.size[1] - 2 * cfg.border_width) / 2 + random_width
    ramp_width = (cfg.size[1] - 2 * cfg.border_width) / 2 - random_width

    # terrain locations
    # -- compute the position of the center of the terrain and the centers of the stairs as well as ramp
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    if cfg.random_stairs_ramp_position_flipping and np.random.random() > 0.5:
        stairs_center = [terrain_center[0], terrain_center[1] + stairs_width / 2 - random_width]
        ramp_center = [terrain_center[0], terrain_center[1] - ramp_width / 2 - random_width]
    else:
        stairs_center = [terrain_center[0], terrain_center[1] - stairs_width / 2 + random_width]
        ramp_center = [terrain_center[0], terrain_center[1] + ramp_width / 2 + random_width]

    # check if the maximum height is exceeded
    if cfg.max_height is not None and step_height * num_steps > cfg.max_height:
        # adjust the len of the stairs and/ or ramp depending on which is modified by the difficulty, make the other
        # one easier
        if cfg.modify_step_height:
            num_steps = int(cfg.max_height / step_height)
            ramp_height = step_height * num_steps
        elif cfg.modify_ramp_slope:
            ramp_height = cfg.max_height
            ramp_len = ramp_height / math.tan(np.deg2rad(ramp_slope))
            step_height = ramp_height / num_steps
        else:
            ramp_height = cfg.max_height
            step_height = cfg.max_height / num_steps

    # adjust positions of the stairs/ramp depending if additional free space should be in the front or behind
    if cfg.free_space_front:
        stairs_len = num_steps * cfg.step_width
        if stairs_len < ramp_len:
            stairs_center[0] = terrain_center[0] - (ramp_len - stairs_len)
        else:
            ramp_center[0] = terrain_center[0] - (stairs_len - ramp_len)
    elif cfg.no_free_space_front:
        stairs_len = num_steps * cfg.step_width
        if stairs_len > ramp_len:
            ramp_center[0] = ramp_center[0] - (stairs_len - ramp_len) / 2
        else:
            stairs_center[0] = stairs_center[0] - (ramp_len - stairs_len) / 2

    # restrict the space
    # initialize list of meshes
    meshes_list = list()
    # generate a ground plane for the terrain
    ground_plane = make_plane(cfg.size, height=0.0, center_zero=False)
    meshes_list.append(ground_plane)
    # generate the border if needed
    if cfg.border_width > 0.0:
        # obtain a list of meshes for the border
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -step_height / 2]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        make_borders = make_border(cfg.size, border_inner_size, step_height, border_center)
        # add the border meshes to the list of meshes
        meshes_list += make_borders

    # remove obstacle to just have a wall
    replace_stairs = False
    replace_ramp = False
    if cfg.random_wall_probability > 0.0 and np.random.random() < cfg.random_wall_probability:
        if cfg.modify_step_height:
            # replace the stairs with a wall
            replace_stairs = True
            num_steps = 0
        elif cfg.modify_ramp_slope:
            # replace the ramp with a wall
            replace_ramp = True
            ramp_len = 0.0
        else:
            if np.random.random() > 0.5:
                # replace the stairs with a wall
                replace_stairs = True
            else:
                # replace the ramp with a wall
                replace_ramp = True
    if cfg.all_wall:
        replace_stairs = True
        replace_ramp = True
        num_steps = 0
        ramp_len = 0.0

    # generate the terrain
    # -- generate the stair pattern
    if not replace_stairs:
        for k in range(num_steps):
            # compute the quantities of the box
            # -- location
            box_z = terrain_center[2] + k * step_height / 2.0
            box_offset = (k / 2.0 + 0.5) * cfg.step_width
            # -- dimensions
            box_height = (k + 1) * step_height
            # generate the stair
            box_dims = ((num_steps - k) * cfg.step_width, stairs_width, box_height)
            box_pos = (stairs_center[0] + box_offset, stairs_center[1], box_z)
            meshes_list.append(trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos)))

    # -- generate the ramp
    # define the vertices
    if not replace_ramp:
        vertices = np.array([
            [ramp_center[0] - ramp_len / 2, ramp_center[1] - ramp_width / 2, terrain_center[2]],
            [ramp_center[0] - ramp_len / 2, ramp_center[1] + ramp_width / 2, terrain_center[2]],
            [ramp_center[0] + ramp_len / 2, ramp_center[1] + ramp_width / 2, terrain_center[2]],
            [ramp_center[0] + ramp_len / 2, ramp_center[1] - ramp_width / 2, terrain_center[2]],
            [ramp_center[0] + ramp_len / 2, ramp_center[1] - ramp_width / 2, terrain_center[2] + ramp_height],
            [ramp_center[0] + ramp_len / 2, ramp_center[1] + ramp_width / 2, terrain_center[2] + ramp_height],
        ])
        faces = np.array([[0, 1, 2], [0, 2, 3], [3, 2, 4], [2, 5, 4], [0, 3, 4], [0, 4, 5], [0, 5, 1], [1, 5, 2]])
        meshes_list.append(trimesh.Trimesh(vertices=vertices, faces=faces, process=False))

    # -- generate the platform that account for possible varying length of the stairs and ramp
    # define the platform behind the stairs
    platform_start = stairs_center[0] + (num_steps * cfg.step_width) / 2
    platform_end = cfg.size[0] - cfg.border_width
    platform_center = (
        platform_start + (platform_end - platform_start) / 2,
        stairs_center[1],
        terrain_center[2] + ramp_height / 2,
    )
    platform_dims = (platform_end - platform_start, stairs_width, ramp_height)
    meshes_list.append(trimesh.creation.box(platform_dims, trimesh.transformations.translation_matrix(platform_center)))
    # define the platform behind the ramp
    platform_start = ramp_center[0] + ramp_len / 2
    platform_center = (
        platform_start + (platform_end - platform_start) / 2,
        ramp_center[1],
        terrain_center[2] + ramp_height / 2,
    )
    platform_dims = (platform_end - platform_start, ramp_width, ramp_height)
    meshes_list.append(trimesh.creation.box(platform_dims, trimesh.transformations.translation_matrix(platform_center)))
    # compute the origin of the terrain
    origin = np.array([
        min(
            (stairs_center[0] - (num_steps * cfg.step_width) / 2 - cfg.platform_width) / 2,
            (ramp_center[0] - ramp_len / 2 - cfg.platform_width) / 2,
        )
        + cfg.border_width,
        terrain_center[1],
        terrain_center[2],
    ])

    return meshes_list, origin


def stairs_ramp_eval_terrain(
    difficulty: float, cfg: stairs_ramp_terrain_cfg.StairsRampEvalTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a stairs and ramp on two sides of a platform

    Terrain is designed to have a stairs and ramp next to each other. Can be used to eval purposes with increasing
    step height for the stairs and increasing slope for the ramp.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    if not (cfg.modify_step_height or cfg.modify_ramp_slope):
        omni.log.warn(
            "[No change based on difficulty performed because neither step height nor ramp slope are modified."
        )

    # compute number of steps in x and y direction
    ramp_len = (cfg.size[0] - 2 * cfg.border_width - 2 * cfg.platform_width - cfg.center_platform_width) / 2
    num_steps = int(ramp_len / cfg.step_width)

    # compute step height and ramp slope based on difficulty
    if cfg.modify_step_height:
        step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
        ramp_height = step_height * num_steps
    elif cfg.modify_ramp_slope:
        ramp_slope = cfg.ramp_slope_range[0] + difficulty * (cfg.ramp_slope_range[1] - cfg.ramp_slope_range[0])
        ramp_height = math.tan(np.deg2rad(ramp_slope)) * ramp_len
        step_height = ramp_height / num_steps
    else:
        step_height = cfg.step_height_range[0] if isinstance(cfg.step_height_range, tuple) else cfg.step_height_range
        ramp_slope = cfg.ramp_slope_range[0] if isinstance(cfg.ramp_slope_range, tuple) else cfg.ramp_slope_range
        ramp_height = math.tan(np.deg2rad(ramp_slope)) * ramp_len
        assert np.isclose(step_height * num_steps, ramp_height, atol=0.1), (
            "No difficulty modification is defined and the define step height and ramp slope do not match."
            f"Current step height: {step_height}, current ramp slope: {ramp_slope} with final stairs height of "
            f"{step_height * num_steps} and final ramp length of {ramp_height}."
        )

    # get the width of the stairs and ramp
    random_width = np.random.uniform(-cfg.width_randomization, cfg.width_randomization)
    stairs_width = (cfg.size[1] - 2 * cfg.border_width) / 2 + random_width
    ramp_width = (cfg.size[1] - 2 * cfg.border_width) / 2 - random_width

    # terrain locations
    # -- compute the position of the center of the terrain and the centers of the stairs as well as ramp
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]

    if cfg.random_stairs_ramp_position_flipping and np.random.random() > 0.5:
        stairs_center_up = [
            terrain_center[0] + cfg.center_platform_width / 2 + ramp_len / 2,
            terrain_center[1] + stairs_width / 2 - random_width,
        ]
        ramp_center_up = [
            terrain_center[0] + cfg.center_platform_width / 2 + ramp_len / 2,
            terrain_center[1] - ramp_width / 2 - random_width,
        ]
        stairs_center_down = [
            terrain_center[0] - cfg.center_platform_width / 2 - ramp_len / 2,
            terrain_center[1] - stairs_width / 2 + random_width,
        ]
        ramp_center_down = [
            terrain_center[0] - cfg.center_platform_width / 2 - ramp_len / 2,
            terrain_center[1] + ramp_width / 2 + random_width,
        ]
    else:
        stairs_center_up = [
            terrain_center[0] + cfg.center_platform_width / 2 + ramp_len / 2,
            terrain_center[1] - stairs_width / 2 + random_width,
        ]
        ramp_center_up = [
            terrain_center[0] + cfg.center_platform_width / 2 + ramp_len / 2,
            terrain_center[1] + ramp_width / 2 + random_width,
        ]
        stairs_center_down = [
            terrain_center[0] - cfg.center_platform_width / 2 - ramp_len / 2,
            terrain_center[1] + stairs_width / 2 - random_width,
        ]
        ramp_center_down = [
            terrain_center[0] - cfg.center_platform_width / 2 - ramp_len / 2,
            terrain_center[1] - ramp_width / 2 - random_width,
        ]

    # check if the maximum height is exceeded
    if cfg.max_height is not None and step_height * num_steps > cfg.max_height:
        # adjust the len of the stairs and/ or ramp depending on which is modified by the difficulty, make the other
        # one easier
        if cfg.modify_step_height:
            num_steps = int(cfg.max_height / step_height)
            ramp_height = step_height * num_steps
        elif cfg.modify_ramp_slope:
            ramp_height = cfg.max_height
            ramp_len = ramp_height / math.tan(np.deg2rad(ramp_slope))
            step_height = ramp_height / num_steps
        else:
            ramp_height = cfg.max_height
            step_height = cfg.max_height / num_steps

    # adjust positions of the stairs/ramp depending if additional free space should be in the front or behind
    if cfg.free_space_front:
        stairs_len = num_steps * cfg.step_width
        if stairs_len < ramp_len:
            stairs_center_up[0] -= ramp_len - stairs_len
            stairs_center_down[0] += ramp_len - stairs_len
        else:
            ramp_center_up[0] -= stairs_len - ramp_len
            ramp_center_down[0] += stairs_len - ramp_len

    # restrict the space
    # initialize list of meshes
    meshes_list = list()
    # generate a ground plane for the terrain
    ground_plane = make_plane(cfg.size, height=0.0, center_zero=False)
    meshes_list.append(ground_plane)
    # generate the border if needed
    if cfg.border_width > 0.0:
        # obtain a list of meshes for the border
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -step_height / 2]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        make_borders = make_border(cfg.size, border_inner_size, step_height, border_center)
        # add the border meshes to the list of meshes
        meshes_list += make_borders

    # remove obstacle to just have a wall
    replace_stairs = False
    replace_ramp = False
    if cfg.random_wall_probability > 0.0 and np.random.random() < cfg.random_wall_probability:
        if np.random.random() > 0.5:
            # replace the stairs with a wall
            replace_stairs = True
            num_steps = 0
        else:
            # replace the ramp with a wall
            replace_ramp = True
            ramp_len = 0.0
    if cfg.all_wall:
        replace_stairs = True
        replace_ramp = True
        num_steps = 0
        ramp_len = 0.0

    # generate the terrain
    # -- generate the stair pattern
    if not replace_stairs:
        for k in range(num_steps):
            # compute the quantities of the box
            # -- location
            box_z = terrain_center[2] + k * step_height / 2.0
            box_offset = (k / 2.0 + 0.5) * cfg.step_width
            # -- dimensions
            box_height = (k + 1) * step_height
            # generate the stair
            box_dims = ((num_steps - k) * cfg.step_width, stairs_width, box_height)
            box_pos_up = (stairs_center_up[0] + box_offset, stairs_center_up[1], box_z)
            meshes_list.append(trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos_up)))
            box_pos_down = (stairs_center_down[0] - box_offset, stairs_center_down[1], box_z)
            meshes_list.append(trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos_down)))

    # -- generate the ramp
    # define the vertices
    if not replace_ramp:
        vertices_up = np.array([
            [ramp_center_up[0] - ramp_len / 2, ramp_center_up[1] - ramp_width / 2, terrain_center[2]],
            [ramp_center_up[0] - ramp_len / 2, ramp_center_up[1] + ramp_width / 2, terrain_center[2]],
            [ramp_center_up[0] + ramp_len / 2, ramp_center_up[1] + ramp_width / 2, terrain_center[2]],
            [ramp_center_up[0] + ramp_len / 2, ramp_center_up[1] - ramp_width / 2, terrain_center[2]],
            [ramp_center_up[0] + ramp_len / 2, ramp_center_up[1] - ramp_width / 2, terrain_center[2] + ramp_height],
            [ramp_center_up[0] + ramp_len / 2, ramp_center_up[1] + ramp_width / 2, terrain_center[2] + ramp_height],
        ])
        vertices_down = np.array([
            [ramp_center_down[0] + ramp_len / 2, ramp_center_down[1] + ramp_width / 2, terrain_center[2]],
            [ramp_center_down[0] + ramp_len / 2, ramp_center_down[1] - ramp_width / 2, terrain_center[2]],
            [ramp_center_down[0] - ramp_len / 2, ramp_center_down[1] - ramp_width / 2, terrain_center[2]],
            [ramp_center_down[0] - ramp_len / 2, ramp_center_down[1] + ramp_width / 2, terrain_center[2]],
            [ramp_center_down[0] - ramp_len / 2, ramp_center_down[1] + ramp_width / 2, terrain_center[2] + ramp_height],
            [ramp_center_down[0] - ramp_len / 2, ramp_center_down[1] - ramp_width / 2, terrain_center[2] + ramp_height],
        ])
        faces = np.array([[0, 1, 2], [0, 2, 3], [3, 2, 4], [2, 5, 4], [0, 3, 4], [0, 4, 5], [0, 5, 1], [1, 5, 2]])
        meshes_list.append(trimesh.Trimesh(vertices=vertices_up, faces=faces, process=False))
        meshes_list.append(trimesh.Trimesh(vertices=vertices_down, faces=faces, process=False))

    # -- generate the platform that account for possible varying length of the stairs and ramp
    # define the platform behind the stairs
    platform_start = stairs_center_up[0] + (num_steps * cfg.step_width) / 2
    platform_end = cfg.size[0] - cfg.border_width
    platform_center = (
        platform_start + (platform_end - platform_start) / 2,
        stairs_center_up[1],
        terrain_center[2] + ramp_height / 2,
    )
    platform_dims = (platform_end - platform_start, stairs_width, ramp_height)
    meshes_list.append(trimesh.creation.box(platform_dims, trimesh.transformations.translation_matrix(platform_center)))

    platform_start = stairs_center_down[0] - (num_steps * cfg.step_width) / 2
    platform_end = cfg.border_width
    platform_center = (
        platform_start + (platform_end - platform_start) / 2,
        stairs_center_down[1],
        terrain_center[2] + ramp_height / 2,
    )
    platform_dims = (platform_start - platform_end, stairs_width, ramp_height)
    meshes_list.append(trimesh.creation.box(platform_dims, trimesh.transformations.translation_matrix(platform_center)))

    # define the platform behind the ramp
    platform_start = ramp_center_up[0] + ramp_len / 2
    platform_end = cfg.size[0] - cfg.border_width
    platform_center = (
        platform_start + (platform_end - platform_start) / 2,
        ramp_center_up[1],
        terrain_center[2] + ramp_height / 2,
    )
    platform_dims = (platform_end - platform_start, ramp_width, ramp_height)
    meshes_list.append(trimesh.creation.box(platform_dims, trimesh.transformations.translation_matrix(platform_center)))

    platform_start = ramp_center_down[0] - ramp_len / 2
    platform_end = cfg.border_width
    platform_center = (
        platform_start + (platform_end - platform_start) / 2,
        ramp_center_down[1],
        terrain_center[2] + ramp_height / 2,
    )
    platform_dims = (platform_start - platform_end, ramp_width, ramp_height)
    meshes_list.append(trimesh.creation.box(platform_dims, trimesh.transformations.translation_matrix(platform_center)))

    # compute the origin of the terrain
    origin = np.array([terrain_center[0], terrain_center[1], terrain_center[2]])

    return meshes_list, origin


def stairs_ramp_up_down_terrain(
    difficulty: float, cfg: stairs_ramp_terrain_cfg.StairsRampUpDownTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a stairs and ramp on two sides of a platform

    Terrain is designed to have a stairs and ramp next to each other. Can be used to eval purposes with increasing
    step height for the stairs and increasing slope for the ramp.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    if not (cfg.modify_step_height or cfg.modify_ramp_slope):
        omni.log.warn(
            "No change based on difficulty performed because neither step height nor ramp slope are modified."
        )

    # compute number of steps in x and y direction
    ramp_len = (cfg.size[0] - 2 * cfg.border_width - 2 * cfg.platform_width - cfg.center_platform_width) / 2
    num_steps = int(ramp_len / cfg.step_width)

    # compute step height and ramp slope based on difficulty
    if cfg.modify_step_height:
        step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
        ramp_height = step_height * num_steps
    elif cfg.modify_ramp_slope:
        ramp_slope = cfg.ramp_slope_range[0] + difficulty * (cfg.ramp_slope_range[1] - cfg.ramp_slope_range[0])
        ramp_height = math.tan(np.deg2rad(ramp_slope)) * ramp_len
        step_height = ramp_height / num_steps
    else:
        step_height = cfg.step_height_range[0] if isinstance(cfg.step_height_range, tuple) else cfg.step_height_range
        ramp_slope = cfg.ramp_slope_range[0] if isinstance(cfg.ramp_slope_range, tuple) else cfg.ramp_slope_range
        ramp_height = math.tan(np.deg2rad(ramp_slope)) * ramp_len
        assert np.isclose(step_height * num_steps, ramp_height, atol=0.1), (
            "No difficulty modification is defined and the define step height and ramp slope do not match."
            f"Current step height: {step_height}, current ramp slope: {ramp_slope} with final stairs height of "
            f"{step_height * num_steps} and final ramp length of {ramp_height}."
        )

    # get the width of the stairs and ramp
    # NOTE: for the demo case here we flip the randomization
    random_width = -np.random.uniform(-cfg.width_randomization, cfg.width_randomization)
    stairs_width = (cfg.size[1] - 2 * cfg.border_width) / 2 + random_width
    ramp_width = (cfg.size[1] - 2 * cfg.border_width) / 2 - random_width

    # terrain locations
    # -- compute the position of the center of the terrain and the centers of the stairs as well as ramp
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], ramp_height]

    if cfg.random_stairs_ramp_position_flipping and np.random.random() > 0.5:
        stairs_center_up = [
            terrain_center[0] + cfg.center_platform_width / 2 + ramp_len / 2,
            terrain_center[1] + stairs_width / 2 - random_width,
        ]
        ramp_center_up = [
            terrain_center[0] + cfg.center_platform_width / 2 + ramp_len / 2,
            terrain_center[1] - ramp_width / 2 - random_width,
        ]
        stairs_center_down = [
            cfg.center_platform_width / 2 + ramp_len / 2 + cfg.platform_width,
            terrain_center[1] - stairs_width / 2 + random_width,
        ]
        ramp_center_down = [
            cfg.center_platform_width / 2 + ramp_len / 2 + cfg.platform_width,
            terrain_center[1] + ramp_width / 2 + random_width,
        ]
    else:
        stairs_center_up = [
            terrain_center[0] + cfg.center_platform_width / 2 + ramp_len / 2,
            terrain_center[1] - stairs_width / 2 + random_width,
        ]
        ramp_center_up = [
            terrain_center[0] + cfg.center_platform_width / 2 + ramp_len / 2,
            terrain_center[1] + ramp_width / 2 + random_width,
        ]
        stairs_center_down = [
            cfg.center_platform_width / 2 + ramp_len / 2 + cfg.platform_width,
            terrain_center[1] + stairs_width / 2 - random_width,
        ]
        ramp_center_down = [
            cfg.center_platform_width / 2 + ramp_len / 2 + cfg.platform_width,
            terrain_center[1] - ramp_width / 2 - random_width,
        ]

    # check if the maximum height is exceeded
    if cfg.max_height is not None and step_height * num_steps > cfg.max_height:
        # adjust the len of the stairs and/ or ramp depending on which is modified by the difficulty, make the other
        # one easier
        if cfg.modify_step_height:
            num_steps = int(cfg.max_height / step_height)
            ramp_height = step_height * num_steps
        elif cfg.modify_ramp_slope:
            ramp_height = cfg.max_height
            ramp_len = ramp_height / math.tan(np.deg2rad(ramp_slope))
            step_height = ramp_height / num_steps
        else:
            ramp_height = cfg.max_height
            step_height = cfg.max_height / num_steps
        terrain_center[2] = ramp_height

    # adjust positions of the stairs/ramp depending if additional free space should be in the front or behind
    if cfg.free_space_front:
        stairs_len = num_steps * cfg.step_width
        if stairs_len < ramp_len:
            stairs_center_up[0] -= ramp_len - stairs_len
            stairs_center_down[0] += ramp_len - stairs_len
        else:
            ramp_center_up[0] -= stairs_len - ramp_len
            ramp_center_down[0] += stairs_len - ramp_len
    elif cfg.no_free_space_front:
        stairs_len = num_steps * cfg.step_width
        if stairs_len > ramp_len:
            ramp_center_up[0] = ramp_center_up[0] - (stairs_len - ramp_len) / 2
        else:
            stairs_center_up[0] = stairs_center_up[0] - (ramp_len - stairs_len) / 2

    # restrict the space
    # initialize list of meshes
    meshes_list = list()
    # generate a ground plane for the terrain
    ground_plane = make_plane(cfg.size, height=0.0, center_zero=False)
    meshes_list.append(ground_plane)
    # generate the border if needed
    if cfg.border_width > 0.0:
        # obtain a list of meshes for the border
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -step_height / 2]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        make_borders = make_border(cfg.size, border_inner_size, step_height, border_center)
        # add the border meshes to the list of meshes
        meshes_list += make_borders

    # remove obstacle to just have a wall
    replace_stairs = False
    replace_ramp = False
    if cfg.random_wall_probability > 0.0 and np.random.random() < cfg.random_wall_probability:
        if np.random.random() > 0.5:
            # replace the stairs with a wall
            replace_stairs = True
            num_steps = 0
        else:
            # replace the ramp with a wall
            replace_ramp = True
            ramp_len = 0.0
    if cfg.all_wall:
        replace_stairs = True
        replace_ramp = True
        num_steps = 0
        ramp_len = 0.0

    # generate the terrain
    # -- generate the stair pattern
    if not replace_stairs:
        for k in range(num_steps):
            # compute the quantities of the box
            # -- location
            box_z_up = terrain_center[2] + k * step_height / 2.0
            box_z_down = k * step_height / 2.0
            box_offset = (k / 2.0 + 0.5) * cfg.step_width
            # -- dimensions
            box_height = (k + 1) * step_height
            # generate the stair
            box_dims = ((num_steps - k) * cfg.step_width, stairs_width, box_height)
            box_pos_up = (stairs_center_up[0] + box_offset, stairs_center_up[1], box_z_up)
            meshes_list.append(trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos_up)))
            box_pos_down = (stairs_center_down[0] + box_offset, stairs_center_down[1], box_z_down)
            meshes_list.append(trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos_down)))

    # -- generate the ramp
    # define the vertices
    if not replace_ramp:
        vertices_up = np.array([
            [ramp_center_up[0] - ramp_len / 2, ramp_center_up[1] - ramp_width / 2, terrain_center[2]],
            [ramp_center_up[0] - ramp_len / 2, ramp_center_up[1] + ramp_width / 2, terrain_center[2]],
            [ramp_center_up[0] + ramp_len / 2, ramp_center_up[1] + ramp_width / 2, terrain_center[2]],
            [ramp_center_up[0] + ramp_len / 2, ramp_center_up[1] - ramp_width / 2, terrain_center[2]],
            [ramp_center_up[0] + ramp_len / 2, ramp_center_up[1] - ramp_width / 2, terrain_center[2] + ramp_height],
            [ramp_center_up[0] + ramp_len / 2, ramp_center_up[1] + ramp_width / 2, terrain_center[2] + ramp_height],
        ])
        vertices_down = np.array([
            [ramp_center_down[0] - ramp_len / 2, ramp_center_down[1] - ramp_width / 2, 0.0],
            [ramp_center_down[0] - ramp_len / 2, ramp_center_down[1] + ramp_width / 2, 0.0],
            [ramp_center_down[0] + ramp_len / 2, ramp_center_down[1] + ramp_width / 2, 0.0],
            [ramp_center_down[0] + ramp_len / 2, ramp_center_down[1] - ramp_width / 2, 0.0],
            [ramp_center_down[0] + ramp_len / 2, ramp_center_down[1] - ramp_width / 2, terrain_center[2]],
            [ramp_center_down[0] + ramp_len / 2, ramp_center_down[1] + ramp_width / 2, terrain_center[2]],
        ])
        faces = np.array([[0, 1, 2], [0, 2, 3], [3, 2, 4], [2, 5, 4], [0, 3, 4], [0, 4, 5], [0, 5, 1], [1, 5, 2]])
        meshes_list.append(trimesh.Trimesh(vertices=vertices_up, faces=faces, process=False))
        meshes_list.append(trimesh.Trimesh(vertices=vertices_down, faces=faces, process=False))

    # -- generate the platform that account for possible varying length of the stairs and ramp
    # define the platform behind the stairs
    platform_start = stairs_center_up[0] + (num_steps * cfg.step_width) / 2
    platform_end = cfg.size[0] - cfg.border_width
    platform_center = (
        platform_start + (platform_end - platform_start) / 2,
        stairs_center_up[1],
        terrain_center[2] + ramp_height / 2,
    )
    platform_dims = (platform_end - platform_start, stairs_width, ramp_height)
    meshes_list.append(trimesh.creation.box(platform_dims, trimesh.transformations.translation_matrix(platform_center)))

    platform_start = stairs_center_down[0] + (num_steps * cfg.step_width) / 2
    # platform_end = cfg.border_width
    platform_center = (
        platform_start + (platform_end - platform_start) / 2,
        stairs_center_down[1],
        ramp_height / 2,
    )
    platform_dims = (platform_end - platform_start, stairs_width, ramp_height)
    meshes_list.append(trimesh.creation.box(platform_dims, trimesh.transformations.translation_matrix(platform_center)))

    # define the platform behind the ramp
    platform_start = ramp_center_up[0] + ramp_len / 2
    platform_end = cfg.size[0] - cfg.border_width
    platform_center = (
        platform_start + (platform_end - platform_start) / 2,
        ramp_center_up[1],
        terrain_center[2] + ramp_height / 2,
    )
    platform_dims = (platform_end - platform_start, ramp_width, ramp_height)
    meshes_list.append(trimesh.creation.box(platform_dims, trimesh.transformations.translation_matrix(platform_center)))

    platform_start = ramp_center_down[0] + ramp_len / 2
    # platform_end = cfg.border_width
    platform_center = (
        platform_start + (platform_end - platform_start) / 2,
        ramp_center_down[1],
        ramp_height / 2,
    )
    platform_dims = (platform_end - platform_start, ramp_width, ramp_height)
    meshes_list.append(trimesh.creation.box(platform_dims, trimesh.transformations.translation_matrix(platform_center)))

    # compute the origin of the terrain
    origin = np.array([terrain_center[0], terrain_center[1], terrain_center[2]])

    return meshes_list, origin
