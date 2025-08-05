# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from nav_suite.utils.obj_converter_cfg import ObjConverterCfg

from .nav_terrain_importer import NavTerrainImporter


@configclass
class NavTerrainImporterCfg(TerrainImporterCfg):
    class_type: type = NavTerrainImporter
    """The class name of the terrain importer."""

    usd_uniform_env_spacing: float | None = None
    """Grid-like environment spacing for USD terrains. Defaults to None.

    For a USD terrain with a single origin, split the mesh into different origins with a uniform spacing."""

    regular_spawning: bool = False
    """Whether to spawn the robots on all terrain origins in a regular pattern. Defaults to False."""

    asset_converter: ObjConverterCfg = ObjConverterCfg()
    """The configuration for the obj asset converter.

    If an `.obj` file, e.g. from matterport, is given, it will be automatically converted using the configuration defined here. """

    groundplane: bool = True
    """Whether to import the ground plane."""

    sem_mesh_to_class_map: str | None = None
    """Path to the mesh to semantic class mapping file.

    If set, semantic classes will be added to the scene. Default is None."""

    duplicate_cfg_file: str | list | None = None
    """Configuration file(s) to duplicate prims in the scene.

    Selected prims are clone by the provided factor and moved to the defined location. Default is None."""

    people_config_file: str | None = None
    """Path to the people configuration file.

    If set, people defined in the Nvidia Nucleus can be added to the scene. Default is None."""

    scale: tuple[float, float, float] | dict[str, tuple[float, float, float]] = (1.0, 1.0, 1.0)
    """The scale of the terrain. Default is (1.0, 1.0, 1.0).

    If a dictionary is provided, the keys are the names of the terrains and the values are the scales.
    """

    usd_path: str | dict[str, str] | None = None
    """The path to the USD file containing the terrain. Only used if ``terrain_type`` is set to "usd".

    If a dictionary is provided, the keys are the names of the terrains and the values are the paths to the USD files.
    """

    add_colliders: bool = False
    """Add colliders to meshes"""
