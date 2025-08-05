# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Functions to generate different terrains using the ``trimesh`` library."""

from __future__ import annotations

import numpy as np
import torch
import trimesh
from typing import TYPE_CHECKING

from isaaclab.terrains.height_field.hf_terrains import random_uniform_terrain
from isaaclab.terrains.trimesh.utils import *  # noqa: F401, F403
from isaaclab.terrains.trimesh.utils import make_plane

if TYPE_CHECKING:
    from . import pillar_terrain_cfg


def pillar_terrain(
    difficulty: float, cfg: pillar_terrain_cfg.MeshPillarTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a set of repeated boxes and cylinders.

    .. image:: ../../_static/terrains/mesh_pillar_terrain.jpg
       :width: 45%
       :align: center

    The terrain has a ground with a platform in the middle. The objects are randomly placed on the
    terrain s.t. they do not overlap with the platform.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).

    Raises:
        ValueError: If the object type is not supported. It must be either a string or a callable.
    """
    from .pillar_terrain_cfg import MeshPillarTerrainCfg

    # initialize list of meshes
    meshes_list = list()
    # initialize list of object meshes
    object_center_list = list()
    # constants for the terrain
    platform_clearance = 0.1
    # compute quantities
    origin = np.asarray((0.5 * cfg.size[0], 0.5 * cfg.size[1], 0))
    platform_corners = np.asarray([
        [origin[0] - cfg.platform_width / 2, origin[1] - cfg.platform_width / 2],
        [origin[0] + cfg.platform_width / 2, origin[1] + cfg.platform_width / 2],
    ])
    platform_corners[0, :] *= 1 - platform_clearance
    platform_corners[1, :] *= 1 + platform_clearance
    # generate a ground plane for the terrain
    ground_plane = make_plane(cfg.size, height=0.0, center_zero=False)
    meshes_list.append(ground_plane)
    # create rough terrain
    if cfg.rough_terrain:
        cfg.rough_terrain.size = cfg.size
        rough_mesh, _ = random_uniform_terrain(difficulty, cfg.rough_terrain)
        meshes_list += rough_mesh

    for object_cfg in [cfg.box_objects, cfg.cylinder_cfg]:
        # if object type is a string, get the function: make_{object_type}
        if isinstance(object_cfg.object_type, str):
            object_func = globals().get(f"make_{object_cfg.object_type}")
        else:
            object_func = object_cfg.object_type
        if not callable(object_func):
            raise ValueError(f"The attribute 'object_type' must be a string or a callable. Received: {object_func}")

        # Resolve the terrain configuration
        # -- common parameters
        num_objects = object_cfg.num_objects[0] + int(
            difficulty * (object_cfg.num_objects[1] - object_cfg.num_objects[0])
        )
        height = object_cfg.height[0] + difficulty * (object_cfg.height[1] - object_cfg.height[0])
        # -- object specific parameters
        # note: SIM114 requires duplicated logical blocks under a single body.
        if isinstance(object_cfg, MeshPillarTerrainCfg.BoxCfg):
            object_kwargs = {
                "length": object_cfg.length[0] + difficulty * (object_cfg.length[1] - object_cfg.length[0]),
                "width": object_cfg.width[0] + difficulty * (object_cfg.width[1] - object_cfg.width[0]),
                "max_yx_angle": object_cfg.max_yx_angle[0] + difficulty * (
                    object_cfg.max_yx_angle[1] - object_cfg.max_yx_angle[0]
                ),
                "degrees": object_cfg.degrees,
            }
        elif isinstance(object_cfg, MeshPillarTerrainCfg.CylinderCfg):  # noqa: SIM114
            object_kwargs = {
                "radius": object_cfg.radius[0] + difficulty * (object_cfg.radius[1] - object_cfg.radius[0]),
                "max_yx_angle": object_cfg.max_yx_angle[0] + difficulty * (
                    object_cfg.max_yx_angle[1] - object_cfg.max_yx_angle[0]
                ),
                "degrees": object_cfg.degrees,
            }
        else:
            raise ValueError(f"Unknown terrain configuration: {cfg}")

        # sample center for objects
        while True:
            object_centers = np.zeros((num_objects, 3))
            object_centers[:, 0] = np.random.uniform(0, cfg.size[0], num_objects)
            object_centers[:, 1] = np.random.uniform(0, cfg.size[1], num_objects)
            # filter out the centers that are on the platform
            is_within_platform_x = np.logical_and(
                object_centers[:, 0] >= platform_corners[0, 0], object_centers[:, 0] <= platform_corners[1, 0]
            )
            is_within_platform_y = np.logical_and(
                object_centers[:, 1] >= platform_corners[0, 1], object_centers[:, 1] <= platform_corners[1, 1]
            )
            masks = np.logical_and(is_within_platform_x, is_within_platform_y)
            # if there are no objects on the platform, break
            if not np.any(masks):
                break

        # generate obstacles (but keep platform clean)
        for index in range(len(object_centers)):
            # randomize the height of the object
            ob_height = height + np.random.uniform(-cfg.max_height_noise, cfg.max_height_noise)
            object_centers[index, 2] = ob_height / 2
            if ob_height > 0.0:
                object_mesh = object_func(center=object_centers[index], height=ob_height, **object_kwargs)
                meshes_list.append(object_mesh)

        object_center_list.append(object_centers)

    # generate a platform in the middle
    dim = (cfg.platform_width, cfg.platform_width, 0)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], 0)
    platform = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(platform)

    # elevate origin
    # origin[2] = mean_height

    return meshes_list, origin


def pillar_terrain_deterministic(
    difficulty: float, cfg: pillar_terrain_cfg.MeshPillarTerrainDeterministicCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a set of repeated objects that is ordered on a circle around the center.

    .. image:: ../../_static/terrains/mesh_pillar_terrain_deterministic.jpg
       :width: 45%
       :align: center

    The terrain has a ground with a platform in the middle. The objects are placed with a circular pattern
    around the center with the distance from the center to the object increasing for each object. Next
    to the distance, also the height increases for each object when placed further away from the center.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).

    Raises:
        ValueError: If the object type is not supported. It must be either a string or a callable.
    """
    from .pillar_terrain_cfg import MeshPillarTerrainDeterministicCfg

    # initialize list of meshes
    meshes_list = list()
    # initialize list of object meshes
    object_center_list = list()
    # constants for the terrain
    platform_clearance = 0.1
    # compute quantities
    origin = np.asarray((0.5 * cfg.size[0], 0.5 * cfg.size[1], 0))
    platform_corners = np.asarray([
        [origin[0] - cfg.platform_width / 2, origin[1] - cfg.platform_width / 2],
        [origin[0] + cfg.platform_width / 2, origin[1] + cfg.platform_width / 2],
    ])
    platform_corners[0, :] *= 1 - platform_clearance
    platform_corners[1, :] *= 1 + platform_clearance
    # generate a ground plane for the terrain
    ground_plane = make_plane(cfg.size, height=0.0, center_zero=False)
    meshes_list.append(ground_plane)
    # create rough terrain
    if cfg.rough_terrain:
        cfg.rough_terrain.size = cfg.size
        rough_mesh, _ = random_uniform_terrain(difficulty, cfg.rough_terrain)
        meshes_list += rough_mesh

    # get overall number of objects and sample their centers
    num_objects = 0
    for object_cfg in [cfg.box_objects, cfg.cylinder_cfg]:
        num_objects += object_cfg.num_objects[0] + int(
            difficulty * (object_cfg.num_objects[1] - object_cfg.num_objects[0])
        )
    # order the objects in a regular, star shaped pattern
    object_centers = np.zeros((num_objects, 3))
    yaw_sample = torch.linspace(0, 2 * np.pi, num_objects + 1, dtype=torch.float32)[:-1]
    distance = torch.linspace(
        cfg.platform_width, min(0.5 * cfg.size[0], cfg.max_obstacle_distance), num_objects, dtype=torch.float32
    )
    object_centers[:, 0] = distance * torch.cos(yaw_sample) + 0.5 * cfg.size[0]
    object_centers[:, 1] = distance * torch.sin(yaw_sample) + 0.5 * cfg.size[1]
    sampled_idx = 0

    for object_cfg in [cfg.box_objects, cfg.cylinder_cfg]:
        # if object type is a string, get the function: make_{object_type}
        if isinstance(object_cfg.object_type, str):
            object_func = globals().get(f"make_{object_cfg.object_type}")
        else:
            object_func = object_cfg.object_type
        if not callable(object_func):
            raise ValueError(f"The attribute 'object_type' must be a string or a callable. Received: {object_func}")

        # Resolve the terrain configuration
        # -- common parameters
        num_objects = object_cfg.num_objects[0] + int(
            difficulty * (object_cfg.num_objects[1] - object_cfg.num_objects[0])
        )
        height = object_cfg.height[0] + difficulty * (object_cfg.height[1] - object_cfg.height[0])
        # -- object specific parameters
        # note: SIM114 requires duplicated logical blocks under a single body.
        if isinstance(object_cfg, MeshPillarTerrainDeterministicCfg.BoxCfg):
            object_kwargs = {
                "length": object_cfg.length[0] + difficulty * (object_cfg.length[1] - object_cfg.length[0]),
                "width": object_cfg.width[0] + difficulty * (object_cfg.width[1] - object_cfg.width[0]),
                "max_yx_angle": object_cfg.max_yx_angle[0] + difficulty * (
                    object_cfg.max_yx_angle[1] - object_cfg.max_yx_angle[0]
                ),
                "degrees": object_cfg.degrees,
            }
        elif isinstance(object_cfg, MeshPillarTerrainDeterministicCfg.CylinderCfg):  # noqa: SIM114
            object_kwargs = {
                "radius": object_cfg.radius[0] + difficulty * (object_cfg.radius[1] - object_cfg.radius[0]),
                "max_yx_angle": object_cfg.max_yx_angle[0] + difficulty * (
                    object_cfg.max_yx_angle[1] - object_cfg.max_yx_angle[0]
                ),
                "degrees": object_cfg.degrees,
            }
        else:
            raise ValueError(f"Unknown terrain configuration: {cfg}")

        # generate obstacles (but keep platform clean)
        for index in range(num_objects):
            object_mesh = object_func(center=object_centers[index + sampled_idx], height=height, **object_kwargs)
            meshes_list.append(object_mesh)
        sampled_idx += num_objects

        object_center_list.append(object_centers)

    # generate a platform in the middle
    mean_height = np.vstack(object_center_list)[:, 2].mean().item()
    dim = (cfg.platform_width, cfg.platform_width, 0.5 * mean_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.25 * mean_height)
    platform = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(platform)

    # elevate origin
    origin[2] = mean_height

    return meshes_list, origin


def pillar_terrain_planner_test(
    difficulty: float, cfg: pillar_terrain_cfg.MeshPillarPlannerTestTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a set of repeated boxes and cylinders.

    .. image:: ../../_static/terrains/mesh_pillar_terrain.jpg
       :width: 45%
       :align: center

    The terrain has a ground with a platform in the middle. The objects are randomly placed on the
    terrain s.t. they do not overlap with the platform.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).

    Raises:
        ValueError: If the object type is not supported. It must be either a string or a callable.
    """
    from .pillar_terrain_cfg import MeshPillarTerrainCfg

    # initialize list of meshes
    meshes_list = list()
    # initialize list of object meshes
    object_center_list = list()
    # constants for the terrain
    platform_clearance = 0.1
    # compute quantities
    origin = np.asarray((
        0.5 * cfg.size[0] - 0.5 * cfg.goal_platform_location[0],
        0.5 * cfg.size[1] - 0.5 * cfg.goal_platform_location[1],
        0,
    ))
    platform_corners = np.asarray([
        [origin[0] - cfg.platform_width / 2, origin[1] - cfg.platform_width / 2],
        [origin[0] + cfg.platform_width / 2, origin[1] + cfg.platform_width / 2],
    ])
    platform_corners[0, :] *= 1 - platform_clearance
    platform_corners[1, :] *= 1 + platform_clearance
    # goal platform
    goal_corners = np.asarray([
        [
            origin[0] + cfg.goal_platform_location[0] - cfg.platform_width / 2,
            origin[1] + cfg.goal_platform_location[1] - cfg.platform_width / 2,
        ],
        [
            origin[0] + cfg.goal_platform_location[0] + cfg.platform_width / 2,
            origin[1] + cfg.goal_platform_location[1] + cfg.platform_width / 2,
        ],
    ])
    goal_corners[0, :] *= 1 - platform_clearance
    goal_corners[1, :] *= 1 + platform_clearance
    # generate a ground plane for the terrain
    ground_plane = make_plane(cfg.size, height=0.0, center_zero=False)
    meshes_list.append(ground_plane)
    # create rough terrain
    if cfg.rough_terrain:
        cfg.rough_terrain.size = cfg.size
        rough_mesh, _ = random_uniform_terrain(difficulty, cfg.rough_terrain)
        meshes_list += rough_mesh

    for object_cfg in [cfg.box_objects, cfg.cylinder_cfg]:
        # if object type is a string, get the function: make_{object_type}
        if isinstance(object_cfg.object_type, str):
            object_func = globals().get(f"make_{object_cfg.object_type}")
        else:
            object_func = object_cfg.object_type
        if not callable(object_func):
            raise ValueError(f"The attribute 'object_type' must be a string or a callable. Received: {object_func}")

        # Resolve the terrain configuration
        # -- common parameters
        num_objects = object_cfg.num_objects[0] + int(
            difficulty * (object_cfg.num_objects[1] - object_cfg.num_objects[0])
        )
        height = object_cfg.height[0] + difficulty * (object_cfg.height[1] - object_cfg.height[0])
        # -- object specific parameters
        # note: SIM114 requires duplicated logical blocks under a single body.
        if isinstance(object_cfg, MeshPillarTerrainCfg.BoxCfg):
            object_kwargs = {
                "length": object_cfg.length[0] + difficulty * (object_cfg.length[1] - object_cfg.length[0]),
                "width": object_cfg.width[0] + difficulty * (object_cfg.width[1] - object_cfg.width[0]),
                "max_yx_angle": object_cfg.max_yx_angle[0] + difficulty * (
                    object_cfg.max_yx_angle[1] - object_cfg.max_yx_angle[0]
                ),
                "degrees": object_cfg.degrees,
            }
        elif isinstance(object_cfg, MeshPillarTerrainCfg.CylinderCfg):  # noqa: SIM114
            object_kwargs = {
                "radius": object_cfg.radius[0] + difficulty * (object_cfg.radius[1] - object_cfg.radius[0]),
                "max_yx_angle": object_cfg.max_yx_angle[0] + difficulty * (
                    object_cfg.max_yx_angle[1] - object_cfg.max_yx_angle[0]
                ),
                "degrees": object_cfg.degrees,
            }
        else:
            raise ValueError(f"Unknown terrain configuration: {cfg}")

        # sample center for objects
        while True:
            object_centers = np.zeros((num_objects, 3))
            object_centers[:, 0] = np.random.uniform(cfg.border_width, cfg.size[0] - 2 * cfg.border_width, num_objects)
            object_centers[:, 1] = np.random.uniform(cfg.border_width, cfg.size[1] - 2 * cfg.border_width, num_objects)
            # filter out the centers that are on the platform
            is_within_platform_x = np.logical_and(
                object_centers[:, 0] >= platform_corners[0, 0], object_centers[:, 0] <= platform_corners[1, 0]
            )
            is_within_platform_y = np.logical_and(
                object_centers[:, 1] >= platform_corners[0, 1], object_centers[:, 1] <= platform_corners[1, 1]
            )
            is_within_goal_x = np.logical_and(
                object_centers[:, 0] >= goal_corners[0, 0], object_centers[:, 0] <= goal_corners[1, 0]
            )
            is_within_goal_y = np.logical_and(
                object_centers[:, 1] >= goal_corners[0, 1], object_centers[:, 1] <= goal_corners[1, 1]
            )
            masks = np.logical_and(is_within_platform_x, is_within_platform_y)
            masks_goal = np.logical_and(is_within_goal_x, is_within_goal_y)
            # if there are no objects on the platform, break
            if not np.any(masks) and not np.any(masks_goal):
                break

        # generate obstacles (but keep platform clean)
        for index in range(len(object_centers)):
            # randomize the height of the object
            ob_height = height + np.random.uniform(-cfg.max_height_noise, cfg.max_height_noise)
            object_centers[index, 2] = ob_height / 2
            if ob_height > 0.0:
                object_mesh = object_func(center=object_centers[index], height=ob_height, **object_kwargs)
                meshes_list.append(object_mesh)

        object_center_list.append(object_centers)

    # generate a platform in the middle
    dim = (cfg.platform_width, cfg.platform_width, 0)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], 0)
    platform = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(platform)

    return meshes_list, origin
