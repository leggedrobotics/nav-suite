# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import numpy as np
import random
import trimesh
from typing import TYPE_CHECKING

from isaaclab.terrains.trimesh.utils import make_plane

if TYPE_CHECKING:
    from . import random_maze_terrain_cfg


def random_maze_terrain(
    difficulty: float, cfg: random_maze_terrain_cfg.RandomMazeTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a maze terrain

    Args:
        difficulty: difficulty level of the terrain
        cfg: configuration of the terrain

    Returns:
        List of meshes and origin of the terrain
    """

    assert cfg.size[0] % cfg.resolution == 0, "Size must be divisible by resolution"
    assert cfg.size[1] % cfg.resolution == 0, "Size must be divisible by resolution"

    open_probability = min(1.0 - difficulty, 0.6)  # difficulty is in the range [0, 1.0]
    maze = generate_maze((int(cfg.size[0] / cfg.resolution), int(cfg.size[1] / cfg.resolution)), open_probability)

    # Extrustion process
    meshes = []
    wall_indices = np.where(maze == 1)
    wall_indices = np.stack(wall_indices, axis=1)

    # randomize the dimensions of the obstacles
    dim_no_neighbor = np.array([cfg.wall_width, cfg.wall_width, cfg.maze_height])
    dim_below_above = np.array([cfg.resolution / 2, cfg.wall_width, cfg.maze_height])
    dim_left_right = np.array([cfg.wall_width, cfg.resolution / 2, cfg.maze_height])

    # check if there is a wall at any of the neighbors of the cell
    surrounding_wall = np.stack(
        (
            maze[np.clip(wall_indices[:, 0] - 1, 0, maze.shape[0] - 1), wall_indices[:, 1]] == 1,  # below
            maze[np.clip(wall_indices[:, 0] + 1, 0, maze.shape[0] - 1), wall_indices[:, 1]] == 1,  # above
            maze[wall_indices[:, 0], np.clip(wall_indices[:, 1] - 1, 0, maze.shape[1] - 1)] == 1,  # left
            maze[wall_indices[:, 0], np.clip(wall_indices[:, 1] + 1, 0, maze.shape[1] - 1)] == 1,  # right
        ),
        axis=1,
    )

    for i, coord in enumerate(wall_indices):
        if surrounding_wall[i].sum() == 0:
            mesh = trimesh.creation.box(extents=dim_no_neighbor)
            center = np.array(
                [(coord[0] + 0.5) * cfg.resolution, (coord[1] + 0.5) * cfg.resolution, cfg.maze_height / 2]
            )
            meshes.append(mesh.apply_translation(center))
            continue
        if surrounding_wall[i][0]:
            mesh = trimesh.creation.box(extents=dim_below_above)
            center = np.array(
                [(coord[0] + 0.25) * cfg.resolution, (coord[1] + 0.5) * cfg.resolution, cfg.maze_height / 2]
            )
            meshes.append(mesh.apply_translation(center))
        if surrounding_wall[i][1]:
            mesh = trimesh.creation.box(extents=dim_below_above)
            center = np.array(
                [(coord[0] + 0.75) * cfg.resolution, (coord[1] + 0.5) * cfg.resolution, cfg.maze_height / 2]
            )
            meshes.append(mesh.apply_translation(center))
        if surrounding_wall[i][2]:
            mesh = trimesh.creation.box(extents=dim_left_right)
            center = np.array(
                [(coord[0] + 0.5) * cfg.resolution, (coord[1] + 0.25) * cfg.resolution, cfg.maze_height / 2]
            )
            meshes.append(mesh.apply_translation(center))
        if surrounding_wall[i][3]:
            mesh = trimesh.creation.box(extents=dim_left_right)
            center = np.array(
                [(coord[0] + 0.5) * cfg.resolution, (coord[1] + 0.75) * cfg.resolution, cfg.maze_height / 2]
            )
            meshes.append(mesh.apply_translation(center))

    # constants for the terrain
    origin = np.asarray((0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0))

    # generate a ground plane for the terrain
    ground_plane = make_plane(cfg.size, height=0.0, center_zero=False)
    meshes.append(ground_plane)

    # augment with stairs inside the corridors
    if cfg.num_stairs > 0:
        # -- identify suitable locations, i.e. with continuous walls on both sides and 5 consecutive free cells
        suitable_vertical_left = np.all(
            np.stack(
                (
                    np.all(surrounding_wall[:, :1], axis=1),
                    wall_indices[:, 0] > 1,
                    wall_indices[:, 0] < maze.shape[0] - 2,
                    maze[
                        np.clip(wall_indices[:, 0] - 2, 0, maze.shape[0] - 1),
                        np.clip(wall_indices[:, 1] - 1, 0, maze.shape[1] - 1),
                    ]
                    == 0,
                    maze[
                        np.clip(wall_indices[:, 0] - 1, 0, maze.shape[0] - 1),
                        np.clip(wall_indices[:, 1] - 1, 0, maze.shape[1] - 1),
                    ]
                    == 0,
                    maze[wall_indices[:, 0], np.clip(wall_indices[:, 1] - 1, 0, maze.shape[1] - 1)] == 0,
                    maze[
                        np.clip(wall_indices[:, 0] + 1, 0, maze.shape[0] - 1),
                        np.clip(wall_indices[:, 1] - 1, 0, maze.shape[1] - 1),
                    ]
                    == 0,
                    maze[
                        np.clip(wall_indices[:, 0] + 2, 0, maze.shape[0] - 1),
                        np.clip(wall_indices[:, 1] - 1, 0, maze.shape[1] - 1),
                    ]
                    == 0,
                    maze[wall_indices[:, 0], np.clip(wall_indices[:, 1] - 2, 0, maze.shape[1] - 1)]
                    == 1,  # wall on opposite side
                ),
                axis=1,
            ),
            axis=-1,
        )
        suitable_vertical_left_idx = wall_indices[suitable_vertical_left] - np.array([0, 1])
        suitable_vertical_right = np.all(
            np.stack(
                (
                    np.all(surrounding_wall[:, :1], axis=1),
                    wall_indices[:, 0] > 1,
                    wall_indices[:, 0] < maze.shape[0] - 2,
                    maze[
                        np.clip(wall_indices[:, 0] - 2, 0, maze.shape[0] - 1),
                        np.clip(wall_indices[:, 1] + 1, 0, maze.shape[1] - 1),
                    ]
                    == 0,
                    maze[
                        np.clip(wall_indices[:, 0] - 1, 0, maze.shape[0] - 1),
                        np.clip(wall_indices[:, 1] + 1, 0, maze.shape[1] - 1),
                    ]
                    == 0,
                    maze[wall_indices[:, 0], np.clip(wall_indices[:, 1] + 1, 0, maze.shape[1] - 1)] == 0,
                    maze[
                        np.clip(wall_indices[:, 0] + 1, 0, maze.shape[0] - 1),
                        np.clip(wall_indices[:, 1] + 1, 0, maze.shape[1] - 1),
                    ]
                    == 0,
                    maze[
                        np.clip(wall_indices[:, 0] + 2, 0, maze.shape[0] - 1),
                        np.clip(wall_indices[:, 1] + 1, 0, maze.shape[1] - 1),
                    ]
                    == 0,
                    maze[wall_indices[:, 0], np.clip(wall_indices[:, 1] + 2, 0, maze.shape[1] - 1)]
                    == 1,  # wall on opposite side
                ),
                axis=1,
            ),
            axis=-1,
        )
        suitable_vertical_right_idx = wall_indices[suitable_vertical_right] + np.array([0, 1])
        # --only insert stairs where there are walls on both sides
        suitable_vertical_idx = np.unique(
            np.concatenate((suitable_vertical_left_idx, suitable_vertical_right_idx), axis=0), axis=0
        )

        suitable_horizontal_bottom = np.all(
            np.stack(
                (
                    np.all(surrounding_wall[:, 2:], axis=1),
                    wall_indices[:, 1] > 1,
                    wall_indices[:, 1] < maze.shape[1] - 2,
                    maze[
                        np.clip(wall_indices[:, 0] - 1, 0, maze.shape[0] - 1),
                        np.clip(wall_indices[:, 1] - 2, 0, maze.shape[1] - 1),
                    ]
                    == 0,
                    maze[
                        np.clip(wall_indices[:, 0] - 1, 0, maze.shape[0] - 1),
                        np.clip(wall_indices[:, 1] - 1, 0, maze.shape[1] - 1),
                    ]
                    == 0,
                    maze[np.clip(wall_indices[:, 0] - 1, 0, maze.shape[0] - 1), wall_indices[:, 1]] == 0,
                    maze[
                        np.clip(wall_indices[:, 0] - 1, 0, maze.shape[0] - 1),
                        np.clip(wall_indices[:, 1] + 1, 0, maze.shape[1] - 1),
                    ]
                    == 0,
                    maze[
                        np.clip(wall_indices[:, 0] - 1, 0, maze.shape[0] - 1),
                        np.clip(wall_indices[:, 1] + 2, 0, maze.shape[1] - 1),
                    ]
                    == 0,
                    maze[np.clip(wall_indices[:, 0] - 2, 0, maze.shape[0] - 1), wall_indices[:, 1]]
                    == 1,  # wall on opposite side
                ),
                axis=1,
            ),
            axis=-1,
        )
        suitable_horizontal_bottom_idx = wall_indices[suitable_horizontal_bottom] - np.array([1, 0])
        suitable_horizontal_top = np.all(
            np.stack(
                (
                    np.all(surrounding_wall[:, 2:], axis=1),
                    wall_indices[:, 1] > 1,
                    wall_indices[:, 1] < maze.shape[1] - 2,
                    maze[
                        np.clip(wall_indices[:, 0] + 1, 0, maze.shape[0] - 1),
                        np.clip(wall_indices[:, 1] - 2, 0, maze.shape[1] - 1),
                    ]
                    == 0,
                    maze[
                        np.clip(wall_indices[:, 0] + 1, 0, maze.shape[0] - 1),
                        np.clip(wall_indices[:, 1] - 1, 0, maze.shape[1] - 1),
                    ]
                    == 0,
                    maze[np.clip(wall_indices[:, 0] + 1, 0, maze.shape[0] - 1), wall_indices[:, 1]] == 0,
                    maze[
                        np.clip(wall_indices[:, 0] + 1, 0, maze.shape[0] - 1),
                        np.clip(wall_indices[:, 1] + 1, 0, maze.shape[1] - 1),
                    ]
                    == 0,
                    maze[
                        np.clip(wall_indices[:, 0] + 1, 0, maze.shape[0] - 1),
                        np.clip(wall_indices[:, 1] + 2, 0, maze.shape[1] - 1),
                    ]
                    == 0,
                    maze[np.clip(wall_indices[:, 0] + 2, 0, maze.shape[0] - 1), wall_indices[:, 1]]
                    == 1,  # wall on opposite side
                ),
                axis=1,
            ),
            axis=-1,
        )
        suitable_horizontal_top_idx = wall_indices[suitable_horizontal_top] + np.array([1, 0])
        suitable_horizontal_idx = np.unique(
            np.concatenate((suitable_horizontal_bottom_idx, suitable_horizontal_top_idx), axis=0), axis=0
        )

        def filter_subsequent_locations(suitable_idx):
            # enforce a distance of at least 2 between the samples
            # -- Calculate differences in second column for consecutive rows
            diffs = np.diff(suitable_idx[:, 1], prepend=0)
            # -- Create a mask for rows where:
            #    - It's the first occurrence in a group, or
            #    - The difference in the second column is at least 2
            mask = (np.diff(suitable_idx[:, 0], prepend=suitable_idx[0, 0]) != 0) | (diffs >= 2)
            # -- Apply the mask to get the filtered result
            return suitable_idx[mask]

        if len(suitable_vertical_idx) > 0:
            suitable_vertical_idx = filter_subsequent_locations(suitable_vertical_idx)
        if len(suitable_horizontal_idx) > 0:
            suitable_horizontal_idx = filter_subsequent_locations(suitable_horizontal_idx)

            # filter all horizontal idx that are overlapping with the vertical idx
            suitable_vertical_idx_view = np.concatenate(
                (
                    suitable_vertical_idx,
                    suitable_vertical_idx + np.array([1, 0]),
                    suitable_vertical_idx - np.array([1, 0]),
                ),
                axis=0,
            ).view([("", suitable_vertical_idx.dtype)] * 2)
            suitable_horizontal_idx_view = suitable_horizontal_idx.view([("", suitable_horizontal_idx.dtype)] * 2)
            suitable_horizontal_idx = (
                np.setdiff1d(suitable_horizontal_idx_view, suitable_vertical_idx_view)
                .view(suitable_vertical_idx.dtype)
                .reshape(-1, 2)
            )

        # expansion factor due to reduced wall width
        expansion_factor = cfg.resolution - cfg.wall_width

        # sample if the stairs are vertical or horizontal (balanced)
        if len(suitable_vertical_idx) > 0 and len(suitable_horizontal_idx) > 0:
            stairs_orientation = np.concatenate([np.random.rand(1) > 0.5 for _ in range(cfg.num_stairs)])
        elif len(suitable_vertical_idx) > 0:
            stairs_orientation = np.ones(cfg.num_stairs, dtype=bool)
        elif len(suitable_horizontal_idx) > 0:
            stairs_orientation = np.zeros(cfg.num_stairs, dtype=bool)
        else:
            return meshes, origin

        stairs_center = np.zeros((cfg.num_stairs, 2))
        replace = stairs_orientation.sum() > len(suitable_vertical_idx)
        stairs_center[stairs_orientation] = (
            suitable_vertical_idx[
                np.random.choice(suitable_vertical_idx.shape[0], size=stairs_orientation.sum(), replace=replace)
            ]
            + 0.5
        )
        replace = (~stairs_orientation).sum() > len(suitable_horizontal_idx)
        stairs_center[~stairs_orientation] = (
            suitable_horizontal_idx[
                np.random.choice(suitable_horizontal_idx.shape[0], size=(~stairs_orientation).sum(), replace=replace)
            ]
            + 0.5
        )

        # generate the stairs
        assert cfg.step_width_range is not None, "step_height_range must be provided"
        assert cfg.step_height_range is not None, "step_width_range must be provided"
        for stairs_idx in range(cfg.num_stairs):
            step_height = cfg.step_height_range[0] + np.random.rand(1) * (
                cfg.step_height_range[1] - cfg.step_height_range[0]
            )
            step_width = cfg.step_width_range[0] + np.random.rand(1) * (
                cfg.step_width_range[1] - cfg.step_width_range[0]
            )

            num_steps = int((cfg.resolution * 3 - cfg.stairs_platform_width) / (2 * step_width))

            for k in range(num_steps):
                # compute the quantities of the box
                # -- box z offset
                box_z = (k * step_height / 2.0).item()
                # -- dimensions
                box_height = ((k + 1) * step_height).item()
                # generate the stair
                if stairs_orientation[stairs_idx]:
                    box_dims = (
                        (3 * cfg.resolution - k * step_width * 2).item(),
                        cfg.resolution + expansion_factor,
                        box_height,
                    )
                else:
                    box_dims = (
                        cfg.resolution + expansion_factor,
                        (3 * cfg.resolution - k * step_width * 2).item(),
                        box_height,
                    )
                box_pos = (
                    stairs_center[stairs_idx][0] * cfg.resolution,
                    stairs_center[stairs_idx][1] * cfg.resolution,
                    box_z,
                )
                meshes.append(trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos)))

    return meshes, origin


# Helper functions
def randomize_array(
    value: np.ndarray, range: tuple[float, float], max_increase: float = np.inf, max_decrease: float = np.inf
) -> np.ndarray:
    """Randomize a value uniformly within a range

    Args:
        value: value to be randomized
        range: range of the value
        max_increase: maximum increase of the value. Defaults to np.inf.
        max_decrease: maximum decrease of the value. Defaults to np.inf.

    Returns:
        The randomized value within the given range.
    """
    rand_value = np.random.uniform(value * range[0], value * range[1], value.shape)
    rand_value = np.clip(rand_value, value - max_decrease, value + max_increase)
    return rand_value


def round_up_to_odd(n: float) -> int:
    """
    Rounds up a given number to the nearest odd integer.
    Args:
        n: The number to be rounded up.
    Returns:
        The nearest odd integer greater than or equal to the input number.
    """
    rounded = math.ceil(n)
    return rounded if rounded % 2 != 0 else rounded + 1


def generate_maze(size: tuple[int, int], open_probability: float = 0.1) -> np.ndarray:
    """Generates a random maze using depth-first search algorithm. Guaranteed to be solvable between any free space.

    Args:
        size: The size of the maze. The actual size will be adjusted to be odd if an even number is provided.
        open_probability: The probability of introducing random openings in the maze. Default is 0.1.

    Returns:
        A 2D numpy array representing the generated maze, where 0 indicates a path and 1 indicates a wall.
    """

    # Ensure the size is odd
    size_x, size_y = size
    size_x = size_x if size_x % 2 == 1 else size_x + 1
    size_y = size_y if size_y % 2 == 1 else size_y + 1

    # Generate a maze using depth-first search to ensure solvability
    maze = np.ones((size_x, size_y), dtype=np.uint8)
    stack = [(0, 0)]
    maze[0, 0] = 0
    prev_direction = (2, 0)

    def is_valid(x, y):
        return 0 <= x < size_x and 0 <= y < size_y

    def get_neighbors(x, y):
        directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        if random.random() > 0.5 and is_valid(x + prev_direction[0], y + prev_direction[1]):
            return [(x + prev_direction[0], y + prev_direction[1])]
        else:
            return [(x + dx, y + dy) for dx, dy in directions if is_valid(x + dx, y + dy)]

    while stack:
        x, y = stack[-1]
        neighbors = [(nx, ny) for nx, ny in get_neighbors(x, y) if maze[nx, ny] == 1]
        if neighbors:
            nx, ny = random.choice(neighbors)
            prev_direction = (nx - x, ny - y)
            maze[(x + nx) // 2, (y + ny) // 2] = 0
            maze[nx, ny] = 0
            stack.append((nx, ny))
        else:
            stack.pop()

    # Introduce random openings based on a probability
    open_percentage = np.sum(maze == 0) / maze.size
    if open_percentage < open_probability:
        closed_mask = np.where(maze == 1)
        # get random indices to open
        num_open = int(open_probability * maze.size) - np.sum(maze == 0)
        open_indices = random.sample(range(len(closed_mask[0])), num_open)
        open_mask = (closed_mask[0][open_indices], closed_mask[1][open_indices])
        maze[open_mask] = 0

    return maze
