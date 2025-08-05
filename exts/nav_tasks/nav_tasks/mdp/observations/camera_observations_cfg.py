# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from collections.abc import Callable
from typing import Literal

from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from .camera_observations import DINOEmbeddedRGBImageTerm, EmbeddedDepthImageTerm


@configclass
class EmbeddedDepthImageCfg(ObservationTermCfg):

    func: Callable[..., torch.Tensor] = EmbeddedDepthImageTerm

    sensor_cfg: SceneEntityCfg | None = None
    """Name of the camera sensor configuration to use."""


@configclass
class DINOEmbeddedRGBImageCfg(ObservationTermCfg):

    func: Callable[..., torch.Tensor] = DINOEmbeddedRGBImageTerm

    sensor_cfg: SceneEntityCfg | None = None
    """Name of the camera sensor configuration to use."""

    backbone_size: Literal["small", "base", "large", "giant"] = "small"
    """Size of the backbone to use.

        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    """

    with_registers: bool = False
    """Whether to use the DINO model with registers."""
