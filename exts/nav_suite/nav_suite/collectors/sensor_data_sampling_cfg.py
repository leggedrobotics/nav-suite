# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import MISSING

from isaaclab.utils import configclass

from ..terrain_analysis import TerrainAnalysisCfg, TerrainAnalysisSingletonCfg
from .sensors.base_cfg import SensorBaseCfg


@configclass
class SensorDataSamplingCfg:
    """Configuration for the sensor data sampling."""

    ###
    # Sensor Configuration
    ###
    sensor_data_handlers: list[SensorBaseCfg] = MISSING
    """List of sensor data handlers that include the logic of how the sensor data is treated."""

    ###
    # Sampling Configuration
    ###
    terrain_analysis: TerrainAnalysisCfg | TerrainAnalysisSingletonCfg = TerrainAnalysisCfg(raycaster_sensor="camera_0")
    """Name of the camera object in the scene definition used for the terrain analysis."""

    sample_points: int = 10000
    """Number of random points to sample."""

    x_angle_range: tuple[float, float] = (-2.5, 2.5)
    y_angle_range: tuple[float, float] = (-2, 5)  # negative angle means in isaac convention: look down
    """Range of the x and y angle of the camera (in degrees), will be randomly selected according to a uniform distribution"""

    height: float = 0.5
    """Height to use for the random points."""

    sliced_sampling: tuple[float, float] | None = None
    """Sliced sampling of the sample points. If None, no slicing is applied.

    If a tuple is provided, the points will be sampled in slices of the given width and length. That means, the terrain
    will be split into slices and for each slice, the number of sample points as given to the
    :func:`nav_suite.collectors.sensor_data_sampling.SensorDataSampling.sample_sensor_data` method is generated.
    """

    ###
    # Saving Configuration
    ###
    save_path: str | None = None
    """Directory to save the sensor data samples.

    If None, the directory is the same as the one of the obj file. Default is None."""

    # debug
    debug_viz: bool = True
    """Whether to visualize the sampled points and orientations."""
