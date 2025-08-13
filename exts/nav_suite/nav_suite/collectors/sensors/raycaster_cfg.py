

from dataclasses import MISSING

from isaaclab.managers import ObservationTermCfg
from isaaclab.utils.configclass import configclass

from .base_cfg import SensorBaseCfg
from .raycaster import RaycasterSensor


@configclass
class RaycasterSensorCfg(SensorBaseCfg):
    """Raycaster sensor configuration."""

    class_type: type[RaycasterSensor] = RaycasterSensor
    """Class type of the sensor."""

    ###
    # Raycaster Configuration
    ###

    sensor_name: str = MISSING
    """Name of the sensor to use for the raycaster."""

    asset_name: str = MISSING
    """Name of the asset under which the raycaster is spawned.

    Need to access the asset to set the world pose of the raycaster as the sensor itself is virtual and therefore does
    not support to set the world pose directly.
    """

    obs_term_cfg: list[ObservationTermCfg] | ObservationTermCfg | None = None
    """Observation term configuration.

    If provided, the output of the observation term will be saved. If None, the :attr:`ray_hits_w` will be saved.
    """
