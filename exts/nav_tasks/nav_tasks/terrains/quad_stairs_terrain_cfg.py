# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPyramidStairsTerrainCfg
from isaaclab.utils import configclass

from .quad_stairs_terrain import quad_pyramid_stairs_terrain


@configclass
class MeshQuadPyramidStairsCfg(MeshPyramidStairsTerrainCfg):

    function = quad_pyramid_stairs_terrain
    """The function to call to evaluate the terrain."""
