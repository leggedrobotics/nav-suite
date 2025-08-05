# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from isaaclab.sensors.ray_caster import RayCasterData


class MatterportRayCasterData(RayCasterData):
    ray_class_ids: torch.Tensor = None
    """The class ids for each ray hit."""
