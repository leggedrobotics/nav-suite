# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from isaaclab.scene import InteractiveScene

if TYPE_CHECKING:
    from .base_cfg import SensorBaseCfg


class SensorBase(ABC):
    """Base class for data extraction logic of sensor data.

    The individual function are executed in the following logic:

    pre_collection

    for i in range(num_collection_rounds):
        pre_sim_update
        scene.write_data_to_sim()
        post_sim_update

    post_collection
    """

    def __init__(self, cfg: SensorBaseCfg, scene: InteractiveScene):
        self.cfg = cfg
        self.scene = scene

    @abstractmethod
    def pre_collection(self, samples: torch.Tensor, filedir: str):
        """Pre collection hook."""
        pass

    @abstractmethod
    def post_collection(self, samples: torch.Tensor, filedir: str):
        """Post collection hook."""
        pass

    @abstractmethod
    def pre_sim_update(self, positions: torch.Tensor, orientations: torch.Tensor, env_ids: torch.Tensor):
        """Pre simulation step hook."""
        pass

    @abstractmethod
    def post_sim_update(
        self,
        samples_idx: torch.Tensor,
        env_ids: torch.Tensor,
        filedir: str,
        slice_bounding_box: tuple[float, float, float, float] | None = None,
    ):
        """Post simulation step hook."""
        pass
