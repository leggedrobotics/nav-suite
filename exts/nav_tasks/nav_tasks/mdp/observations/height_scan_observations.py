# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.warp import raycast_dynamic_meshes

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def height_scan_bounded(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - 0.5
    height = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    # assign max distance to inf values
    height[torch.isinf(height)] = sensor.cfg.max_distance
    height[torch.isnan(height)] = sensor.cfg.max_distance
    return height


def height_scan_clipped(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.5,
    clip_height: tuple[float, float] = (-1.0, 0.5),
) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # get the bounded height scan
    height = height_scan_bounded(env, sensor_cfg, offset)
    # clip to max observable height
    height = torch.clip(height, clip_height[0], clip_height[1])

    return height


def height_scan_square(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    shape: list[int] | None = None,
    offset: float = 0.5,
    clip_height: tuple[float, float] = (-1.0, 0.5),
) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame given in the square pattern of the sensor."""
    # call regular height scanner function
    height = height_scan_clipped(env, sensor_cfg, offset=offset, clip_height=clip_height)
    shape = shape if shape is not None else [int(math.sqrt(height.shape[1])), int(math.sqrt(height.shape[1]))]
    # unflatten the height scan to make use of spatial information
    height_square = torch.unflatten(height, 1, (shape[0], shape[1]))
    # NOTE: the height scan is mirrored as the pattern is created from neg to pos whereas in the robotics frame, the left of
    # the robot is positive and the right is negative
    height_square = torch.flip(height_square, dims=[1])
    # unqueeze to make compatible with convolutional layers
    return height_square.unsqueeze(1)


def height_scan_door_recognition(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    shape: list[int] | None = None,
    door_height_thres: float = 1.25,
    offset: float = 0.5,
    clip_height: tuple[float, float] = (-1.0, 0.5),
    return_height: bool = True,
) -> torch.Tensor | None:
    """Height scan that xplicitly accounts for doors in the scene.

    Doors should be recognized by performing two more raycasting operations: shortly above the ground up and down.
    Then it will be checked if the up raycast has a hit and if the hit is lower than the initial raycast.
    Moreover, it is checked if the distance between up and down raycast is above a certain threshold.

    Args:

    """

    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    # get the sensor hit points
    ray_origins = sensor.data.ray_hits_w.clone()

    # we raycast one more time shortly above the ground up and down, if the up raycast hits and is lower than the
    # initial raycast, a potential door is detected
    ray_origins[..., 2] = 0.5
    ray_directions = torch.zeros_like(ray_origins)
    ray_directions[..., 2] = -1.0

    hit_point_down = raycast_dynamic_meshes(
        ray_origins,
        ray_directions,
        mesh_ids_wp=sensor._mesh_ids_wp,  # list with shape num_envs x num_meshes_per_env
        max_dist=sensor.cfg.max_distance,
        mesh_positions_w=sensor._mesh_positions_w if sensor.cfg.track_mesh_transforms else None,
        mesh_orientations_w=sensor._mesh_orientations_w if sensor.cfg.track_mesh_transforms else None,
    )[0]

    ray_directions[..., 2] = 1.0

    hit_point_up = raycast_dynamic_meshes(
        ray_origins,
        ray_directions,
        mesh_ids_wp=sensor._mesh_ids_wp,  # list with shape num_envs x num_meshes_per_env
        max_dist=sensor.cfg.max_distance,
        mesh_positions_w=sensor._mesh_positions_w if sensor.cfg.track_mesh_transforms else None,
        mesh_orientations_w=sensor._mesh_orientations_w if sensor.cfg.track_mesh_transforms else None,
    )[0]

    lower_height = (
        (hit_point_up[..., 2] < (sensor.data.ray_hits_w[..., 2] - 1e-3))
        & torch.isfinite(hit_point_up[..., 2])
        & ((hit_point_up[..., 2] - hit_point_down[..., 2]) > door_height_thres)
        & torch.isfinite(hit_point_down[..., 2])
    )

    # overwrite the data
    sensor.data.ray_hits_w[lower_height] = hit_point_down[lower_height]

    # debug
    if False:
        env_render_steps = 1000

        # provided height scan
        positions = sensor.data.ray_hits_w.clone()
        # flatten positions
        positions = positions.view(-1, 3)

        # in headless mode, we cannot visualize the graph and omni.debug.draw is not available
        try:
            import omni.isaac.debug_draw._debug_draw as omni_debug_draw

            draw_interface = omni_debug_draw.acquire_debug_draw_interface()
            draw_interface.draw_points(
                positions.tolist(),
                [(1.0, 0.5, 0, 1)] * positions.shape[0],
                [5] * positions.shape[0],
            )

            sim = SimulationContext.instance()
            for _ in range(env_render_steps):
                sim.render()

            # clear the drawn points and lines
            draw_interface.clear_points()
            draw_interface.clear_lines()

        except ImportError:
            print("[WARNING] Cannot visualize occluded height scan in headless mode.")

    if return_height:
        # call regular height scanner function
        return height_scan_square(env, sensor_cfg, shape, offset, clip_height)
    else:
        return None


@configclass
class HeightScanOcculusionModifierCfg:
    """Configuration for the HeightScanOcculusionModifier."""

    height_scan_func: callable = MISSING
    """The height scan function to modify."""

    sensor_cfg: SceneEntityCfg = MISSING
    """The sensor configuration."""

    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    """The asset configuration."""

    env_ratio: float | None = None
    """The ratio of environments to apply the occlusion to."""

    sensor_offsets: list[list[float]] | list[float] | None = None
    """The sensor offset to account for the sensor's position."""

    offset_threshold: float = 0.5  # 0.01
    """The distance threshold to consider a point as occluded."""

    def __post_init__(self):
        assert (
            self.env_ratio is None or 0.0 <= self.env_ratio <= 1.0
        ), "The environment ratio must be between 0.0 and 1.0."


class HeightScanOcculusionModifier:
    """Modify height scan to account for occulsions in the terrain that cannot be observed by the sensor."""

    def __init__(self, cfg: HeightScanOcculusionModifierCfg):
        self.cfg = cfg

    def _setup(self, env: ManagerBasedRLEnv):
        # extract the used quantities (to enable type-hinting)
        self._sensor: RayCaster = env.scene.sensors[self.cfg.sensor_cfg.name]
        self._asset: Articulation = env.scene[self.cfg.asset_cfg.name]
        # account for the sensor offset
        if self.cfg.sensor_offsets is not None:
            if isinstance(self.cfg.sensor_offsets[0], list):
                self._sensor_offset_tensor = (
                    torch.tensor(self.cfg.sensor_offsets, device=self._asset.device)
                    .unsqueeze(1)
                    .repeat(1, env.num_envs, 1)
                )
            else:
                self._sensor_offset_tensor = torch.tensor(
                    [[self.cfg.sensor_offsets]], device=self._asset.device
                ).repeat(1, env.num_envs, 1)
        else:
            self._sensor_offset_tensor = None
        # get the sensors where occlusion should be applied
        if self.cfg.env_ratio is not None:
            self._env_ids = torch.randperm(env.num_envs, device=env.device)[: int(self.cfg.env_ratio * env.num_envs)]
        else:
            self._env_ids = slice(None)

    def _get_occuled_points(self, robot_position: torch.Tensor) -> torch.Tensor:
        robot_position = robot_position[:, None, :].repeat(1, self._sensor.data.ray_hits_w.shape[1], 1)
        ray_directions = self._sensor.data.ray_hits_w - robot_position

        # NOTE: ray directions can never be inf or nan, otherwise the raycasting takes forever
        ray_directions[torch.isinf(ray_directions)] = 0.0
        ray_directions[torch.isnan(ray_directions)] = 0.0

        # raycast from the robot to intended hit positions
        ray_hits_w = raycast_dynamic_meshes(
            robot_position,
            ray_directions,
            mesh_ids_wp=self._sensor._mesh_ids_wp,  # list with shape num_envs x num_meshes_per_env
            max_dist=self._sensor.cfg.max_distance,
            mesh_positions_w=self._sensor._mesh_positions_w if self._sensor.cfg.track_mesh_transforms else None,
            mesh_orientations_w=self._sensor._mesh_orientations_w if self._sensor.cfg.track_mesh_transforms else None,
        )[0]

        # get not visible parts of the height-scan
        unseen = torch.norm(ray_hits_w - self._sensor.data.ray_hits_w, dim=2) > self.cfg.offset_threshold

        return unseen

    def __call__(self, env: ManagerBasedRLEnv, *args, **kwargs) -> torch.Tensor:
        """Modify the height scan to account for occulsions in the terrain that cannot be observed by the sensor."""

        # setup the modifier
        if not hasattr(self, "_sensor"):
            self._setup(env)

        # account for the sensor offset
        if self._sensor_offset_tensor is not None:
            unseen = torch.zeros(
                (self._sensor_offset_tensor.shape[0], *self._sensor.data.ray_hits_w.shape[:-1]),
                device=self._asset.device,
                dtype=torch.bool,
            )
            for offset_idx in range(self._sensor_offset_tensor.shape[0]):
                robot_position = self._asset.data.root_pos_w + math_utils.quat_rotate(
                    self._asset.data.root_quat_w, self._sensor_offset_tensor[offset_idx]
                )
                unseen[offset_idx] = self._get_occuled_points(robot_position)
            unseen = torch.all(unseen, dim=0)
        else:
            robot_position = self._asset.data.root_pos_w

        # overwrite the data
        unseen[self._env_ids] = False
        if torch.any(unseen):
            unseen_points = self._sensor.data.ray_hits_w[unseen]
            unseen_points[..., 2] = self._sensor.cfg.max_distance
            self._sensor.data.ray_hits_w[unseen] = unseen_points

        # return the modified height scan
        return self.cfg.height_scan_func(env, *args, **kwargs)

    def __name__(self):
        return "HeightScanOcculusionModifier"


class HeightScanOcculusionDoorRecognitionModifier(HeightScanOcculusionModifier):

    def __init__(self, cfg: HeightScanOcculusionModifierCfg):
        super().__init__(cfg)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        shape: list[int] | None = None,
        door_height_thres: float = 1.25,
        offset: float = 0.5,
        clip_height: tuple[float, float] = (-1.0, 0.5),
    ):
        height_scan_door_recognition(env, sensor_cfg, door_height_thres=door_height_thres, return_height=False)
        return super().__call__(env, sensor_cfg=sensor_cfg, shape=shape, offset=offset, clip_height=clip_height)

    def __name__(self):
        return "HeightScanOcculusionDoorRecognitionModifier"


def height_scan_square_exp_occlu(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    shape: list[int] | None = None,
    offset: float = 0.5,
    clip_height: tuple[float, float] = (-1.0, 0.5),
) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame given in the square pattern of the sensor.

    Explicitly account for occulsions of the terrain.

    FIXME: IMPLEMENT AGAIN AS MODIFIER WITH NEW ISAAC SIM RELEASE"""

    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    # get the sensor hit points
    ray_hits = sensor.data.ray_hits_w.clone()
    # account for the sensor offset
    robot_position = asset.data.root_pos_w + math_utils.quat_rotate(
        asset.data.root_quat_w, torch.tensor([[0.4, 0.0, 0.0]], device=asset.device).repeat(env.num_envs, 1)
    )
    robot_position = robot_position[:, None, :].repeat(1, ray_hits.shape[1], 1)
    ray_directions = ray_hits - robot_position

    # NOTE: ray directions can never be inf or nan, otherwise the raycasting takes forever
    ray_directions[torch.isinf(ray_directions)] = 0.0
    ray_directions[torch.isnan(ray_directions)] = 0.0

    # raycast from the robot to intended hit positions
    ray_hits_w = raycast_dynamic_meshes(
        robot_position,
        ray_directions,
        mesh_ids_wp=sensor._mesh_ids_wp,  # list with shape num_envs x num_meshes_per_env
        max_dist=sensor.cfg.max_distance,
        mesh_positions_w=sensor._mesh_positions_w if sensor.cfg.track_mesh_transforms else None,
        mesh_orientations_w=sensor._mesh_orientations_w if sensor.cfg.track_mesh_transforms else None,
    )[0]

    # get not visible parts of the height-scan
    unseen = torch.norm(ray_hits_w - ray_hits, dim=2) > 0.01

    # overwrite the data
    if torch.any(unseen):
        unseen_points = sensor.data.ray_hits_w[unseen]
        unseen_points[..., 2] = sensor.cfg.max_distance
        sensor.data.ray_hits_w[unseen] = unseen_points

    # debug
    if False:
        env_render_steps = 1000

        # provided height scan
        positions = sensor.data.ray_hits_w.clone()
        # flatten positions
        positions = positions.view(-1, 3)

        # in headless mode, we cannot visualize the graph and omni.debug.draw is not available
        try:
            import omni.isaac.debug_draw._debug_draw as omni_debug_draw

            draw_interface = omni_debug_draw.acquire_debug_draw_interface()
            draw_interface.draw_points(
                positions.tolist(),
                [(1.0, 0.5, 0, 1)] * positions.shape[0],
                [5] * positions.shape[0],
            )

            sim = SimulationContext.instance()
            for _ in range(env_render_steps):
                sim.render()

            # clear the drawn points and lines
            draw_interface.clear_points()
            draw_interface.clear_lines()

        except ImportError:
            print("[WARNING] Cannot visualize occluded height scan in headless mode.")

    # run regular height scan
    return height_scan_square(env, sensor_cfg, shape, offset, clip_height)


def height_scan_square_exp_occlu_with_door_recognition(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    shape: list[int] | None = None,
    door_height_thres: float = 1.25,
    offset: float = 0.5,
    clip_height: tuple[float, float] = (-1.0, 0.5),
) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame given in the square pattern of the sensor.

    Explicitly account for occulsions of the terrain and doors in the scene.
    """

    height_scan_door_recognition(
        env,
        sensor_cfg,
        shape,
        door_height_thres=door_height_thres,
        offset=offset,
        clip_height=clip_height,
        return_height=False,
    )
    return height_scan_square_exp_occlu(env, asset_cfg, sensor_cfg, shape, offset, clip_height)
