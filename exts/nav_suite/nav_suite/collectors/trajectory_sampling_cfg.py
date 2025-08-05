# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from ..terrain_analysis import TerrainAnalysisCfg, TerrainAnalysisSingletonCfg


@configclass
class TrajectorySamplingCfg:
    """Configuration for the trajectory sampling."""

    # sampling
    sample_points: int = 10000
    """Number of random points to sample."""

    height: float = 0.5
    """Height to use for the random points."""

    enable_saved_paths_loading: bool = True
    """Load saved paths if they exist. Default to True.

    This will be guarantee reproducibility of the generated paths for the same terrain. Paths are saved in the same
    directory as the mesh file (for usd/obj/... files) or if not available under logs. The saved paths follow the
    following naming convention: paths_seed{seed}_paths{num_path}_min{min_len}_max{max_len}.pkl
    """

    terrain_analysis: TerrainAnalysisCfg | TerrainAnalysisSingletonCfg = TerrainAnalysisCfg(raycaster_sensor="camera_0")
    """Name of the camera object in the scene definition used for the terrain analysis."""
