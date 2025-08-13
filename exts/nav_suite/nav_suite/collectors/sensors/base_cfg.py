

from isaaclab.utils.configclass import configclass

from .base import SensorBase


@configclass
class SensorBaseCfg:
    """Base configuration for sensors."""

    class_type: type[SensorBase] = SensorBase
    """Class type of the sensor."""

    requires_render: bool = False
    """Whether the sensor requires render steps to fill the buffers.

    For USD/ tiled cameras, this has to be set to True. For raycaster cameras, this should be set to False to speedup
    the sampling process.
    """
