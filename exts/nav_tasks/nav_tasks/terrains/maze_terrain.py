# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import random
import trimesh
from typing import TYPE_CHECKING

from isaaclab.terrains.trimesh.utils import make_plane

if TYPE_CHECKING:
    from . import maze_terrain_cfg


def maze_terrain(difficulty: float, cfg: maze_terrain_cfg.MazeTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a maze terrain

    .. image:: ../../_static/terrains/maze_training_terrain.jpg
       :width: 30%

    .. image:: ../../_static/terrains/maze_val_easy_terrain.jpg
       :width: 30%

    .. image:: ../../_static/terrains/maze_val_hard_terrain.jpg
       :width: 30%

    Args:
        difficulty (float): difficulty level of the terrain
        cfg (mesh_terrains_cfg.MazeTerrainCfg): configuration of the terrain

    Returns:
        tuple[list[trimesh.Trimesh], np.ndarray]: list of meshes and origin of the terrain
    """
    # Load the data from the JSON file
    with open(cfg.path_obstacles) as f:
        import json

        data = json.load(f)

    # Create a list to hold all the meshes
    meshes = []

    # Process each object in the data
    for obj in data:
        mesh = None
        shape = obj["shape"]
        dims = obj["dimensions"]

        # Based on difficulty level, we want to skip some objects
        if str(difficulty) in cfg.difficulty_configuration:
            if random.random() > cfg.difficulty_configuration[str(difficulty)]:
                continue

        # Create a mesh for each shape
        if shape == "circle":
            # Create a cylinder for circle (because circles in 3D are cylinders)
            radius = max(
                0.0,
                randomize_value(
                    dims["radius"],
                    cfg.randomization["range"]["radius"],
                    cfg.randomization["max_increase"],
                    cfg.randomization["max_decrease"],
                ),
            )
            height = max(
                0.0,
                randomize_value(
                    dims["height"],
                    cfg.randomization["range"]["height"],
                    cfg.randomization["max_increase"],
                    cfg.randomization["max_decrease"],
                ),
            )
            mesh = trimesh.creation.cylinder(
                radius=radius,
                height=height,
                transform=trimesh.transformations.translation_matrix([obj["x"], obj["y"], height / 2]),
            )
        elif shape == "line" or shape == "ceiling":

            # Convert line to a box (since we have thickness)
            start = np.array([obj["x"], obj["y"], dims["height"] / 2])
            end = np.array([obj["x2"], obj["y2"], dims["height"] / 2])

            # Create the box
            length = max(
                0.0,
                randomize_value(
                    np.linalg.norm(end - start),
                    cfg.randomization["range"]["length"],
                    cfg.randomization["max_increase"],
                    cfg.randomization["max_decrease"],
                ),
            )
            width = max(
                0.0,
                randomize_value(
                    dims["thickness"],
                    cfg.randomization["range"]["width"],
                    cfg.randomization["max_increase"],
                    cfg.randomization["max_decrease"],
                ),
            )
            height = max(
                0.0,
                randomize_value(
                    dims["height"],
                    cfg.randomization["range"]["height"],
                    cfg.randomization["max_increase"],
                    cfg.randomization["max_decrease"],
                ),
            )

            # Set the translation
            if shape == "ceiling":
                mesh = trimesh.creation.box(extents=[length, width, width])
                center = np.array([(end[0] + start[0]) / 2, (end[1] + start[1]) / 2, height])
            else:
                mesh = trimesh.creation.box(extents=[length, width, height])
                center = np.array([(end[0] + start[0]) / 2, (end[1] + start[1]) / 2, height / 2])
            mesh.apply_translation(center)

            # Set the rotation
            direction = np.array([0, 0, 1])
            angle = np.arctan2(end[1] - start[1], end[0] - start[0])
            rot_matrix = trimesh.transformations.rotation_matrix(angle, direction, center)
            mesh.apply_transform(rot_matrix)

        if mesh is not None:
            meshes.append(mesh)

    # Union all the meshes to create a single unified mesh
    unified_mesh = trimesh.util.concatenate(meshes)

    # constants for the terrain
    origin = np.asarray((0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0))

    # initialize list of meshes
    meshes_list = list()
    meshes_list.append(unified_mesh)

    # generate a ground plane for the terrain
    ground_plane = make_plane(cfg.size, height=0.0, center_zero=False)
    meshes_list.append(ground_plane)

    return meshes_list, origin


# Helper functions
def randomize_value(value, range, max_increase=np.inf, max_decrease=np.inf):
    """Randomize a value uniformly within a range

    Args:
        value (float): value to be randomized
        range (list[float]): range of the value
        max_increase (float, optional): maximum increase of the value. Defaults to np.inf.
        max_decrease (float, optional): maximum decrease of the value. Defaults to np.inf.

    Returns:
        float: randomized value
    """
    rand_binary = random.randint(0, 1)
    # Make sure that in average, the trajectory is not too far from the original trajectory
    if rand_binary == 0:
        sampling_range = [value * range[0], value]
    else:
        sampling_range = [value, value * range[1]]
    rand_value = np.random.uniform(sampling_range[0], sampling_range[1])
    rand_value = np.clip(rand_value, value - max_decrease, value + max_increase)
    return rand_value
