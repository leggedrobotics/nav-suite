# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

from isaaclab.utils import configclass

from .terrain_analysis import TerrainAnalysis, TerrainAnalysisSingleton


@configclass
class TerrainAnalysisCfg:

    class_type: type = TerrainAnalysis
    """Type of the terrain analysis to use."""

    robot_height: float = 0.6
    """Height of the robot"""

    wall_height: float = 1.0
    """Height of the walls.

    Wall filtering will start rays from that height and filter all that hit the mesh within 0.3m. This will always
    define the raycasting height."""

    max_sample_height: float | None = None
    """Maximum height of the sample points.

    If None, the max height is defined by the wall height. This flag can be useful in a map with vary different heights
    (e.g. a city map) where all the building should be filtered out."""

    robot_buffer_spawn: float = 0.7
    """Robot buffer for spawn location"""

    sample_points: int = 1000
    """Number of nodes in the tree"""

    max_path_length: float = 10.0
    """Maximum distance from the start location to the goal location"""

    door_filtering: bool = False
    """Account for doors when doing the height difference based edge filtering. Default is False.

    Normally, the height of the terrain is just determined by top-down raycasting. If True, there will be an additional
    raycasting 0.1m above the ground. If a upward pointing ray does not yield the same height as the top-down ray, the
    algorithms assumes that there is a door and a new height is determined."""

    door_height_threshold: float = 1.5
    """Threshold of the door height for the door detection.

    As some objects are composed out of multiple layers of meshes (e.g. stairs as combination of boxes), a door will be
    identified as a height difference of the top-down ray and the upward ray of at least this threshold."""

    num_connections: int = 5
    """Number of connections to make in the graph"""

    raycaster_sensor: str | None = None
    """Name of the raycaster sensor to use for terrain analysis.

    If None, the terrain analysis will be done on the USD stage. For matterport environments,
    the IsaacLab raycaster sensor can be used as the ply mesh is a single mesh. On the contrary,
    for unreal engine meshes (as they consists out of multiple meshes), raycasting should be
    performed over the USD stage. Default is None."""

    grid_resolution: float = 0.1
    """Resolution of the grid to check for not traversable edges"""

    height_diff_threshold: float | None = 0.3
    """Threshold for height difference between two points.

    If None, no height difference threshold is applied."""

    min_height_diff_edge_filter: bool = False
    """Filter investigated edges in the height difference filter is both are on the same height. Default is False.

    If True, the height difference filter will only be applied if the two points are on different heights.
    This can lead to a speed up if the graph is large. If False, the height difference filter will be applied to all."""

    viz_graph: bool = True
    """Visualize the graph after the construction for a short amount of time."""

    viz_height_map: bool = True
    """Visualize the height map after the construction for a short amount of time."""

    viz_duration: int = 1000
    """Number of render steps of the visualization."""

    semantic_point_filter: bool = True
    """Filter points based on semantic classes.

    .. note::
        This will only filter points if a semantic cost mapping is provided."""

    semantic_edge_filter: bool = True
    """Filter edges based on semantic classes.

    .. note::
        This will only filter edges if a semantic cost mapping is provided."""

    semantic_cost_mapping: str | None = None
    """Path to the semantic cost mapping file.

    The file should be a YAML file with the following format:

    .. code-block:: yaml

        semantic_costs:
          - class_name_1: cost_1
          - class_name_2: cost_2
          - ...
    """

    semantic_cost_threshold: float = 0.5
    """Threshold for semantic cost filtering"""

    no_class_cost: Literal["max", "min"] = "max"
    """Which cost to assign to points with no class.

    .. note::
        This will only filter points if a semantic cost mapping is provided."""

    # dimension limiting
    dim_limiter_prim: str | None = None
    """Prim name that should be used to limit the dimensions of the mesh.

    All meshes including this prim string are used to set the range in which the graph is constructed and samples are
    generated. If None, all meshes are considered.

    .. note::
        Only used if not a raycaster sensor is passed to the terrain analysis.
    """

    max_terrain_size: float | None = None
    """Maximum size of the terrain in meters.

    This can be useful when e.g. a ground plan is given and the entire anlaysis would run out of memory. If None, the
    entire terrain is considered.
    """

    terrain_bounding_box: tuple[float, float, float, float] | None = None
    """Bounding box where terrain analysis is performed. If None, no bounding box is used.

    The bounding box is defined using the corner coordinate definition, i.e. [x_min, y_min, x_max, y_max].
    - **x_min**: The x-coordinate of the top-left (or bottom-left) corner.
    - **y_min**: The y-coordinate of the top-left (or bottom-left) corner.
    - **x_max**: The x-coordinate of the bottom-right (or top-right) corner.
    - **y_max**: The y-coordinate of the bottom-right (or top-right) corner.
    It is aligned with the world axes.
    """

    keep_paths_in_subterrain: bool = False
    """Whether all paths should start and end within the same subterrain - for curriculum based terrains."""

    # randomization of generated sample points
    sample_height_random_range: tuple[float, float] | None = None
    """Random range for the height randomization of the sample points which are used to construct the traversability graph.

    If None, the height will be the robot height above the terrain. This should be used whenever the traversability graph is used to sample start and goal points for navigation.
    A randomization range can be useful when e.g. sampling viewpoints for environment reconstruction.
    """


@configclass
class TerrainAnalysisSingletonCfg(TerrainAnalysisCfg):
    """Configuration for the terrain analysis singleton."""

    class_type: type = TerrainAnalysisSingleton
    """Type of the terrain analysis to use."""
