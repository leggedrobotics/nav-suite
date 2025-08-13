

from __future__ import annotations

import numpy as np
import os
import torch
from typing import TYPE_CHECKING

from isaaclab.scene import InteractiveScene
from isaaclab.sensors import RayCaster
from isaaclab.utils import noise

from .base import SensorBase

if TYPE_CHECKING:
    from .raycaster_cfg import RaycasterSensorCfg


class RaycasterSensor(SensorBase):
    """Raycaster sensor."""

    cfg: RaycasterSensorCfg

    def __init__(self, cfg: RaycasterSensorCfg, scene: InteractiveScene):
        super().__init__(cfg, scene)

        # init index buffer
        self._idx = 0

        # check that the sensor is a raycaster
        if not isinstance(self.scene.sensors[self.cfg.sensor_name], RayCaster):
            raise ValueError("Sensor is not a raycaster.")

        # check that the observations term cfg is a list
        if self.cfg.obs_term_cfg is not None and not isinstance(self.cfg.obs_term_cfg, list):
            self.cfg.obs_term_cfg = [self.cfg.obs_term_cfg]

    def pre_collection(self, samples: torch.Tensor, filedir: str):
        # create directory for raycaster data
        filedir = os.path.join(filedir, "raycaster")
        os.makedirs(filedir, exist_ok=True)

    def post_collection(self, samples: torch.Tensor, filedir: str):
        # save raycaster poses
        np.savetxt(os.path.join(filedir, "raycaster_poses.txt"), samples.cpu().numpy(), delimiter=",")

    def pre_sim_update(self, positions: torch.Tensor, orientations: torch.Tensor, env_ids: torch.Tensor):
        # as the raycaster itself is a virtual sensor, it does not directly support to set the world poses
        # instead, we need to set the world poses of the xform / articulation / rigid body under which it is spawned

        try:
            asset = self.scene.articulations[self.cfg.asset_name]
        except KeyError:
            asset = None

        if asset is None:
            try:
                asset = self.scene.rigid_objects[self.cfg.asset_name]
            except KeyError:
                asset = None

        assert asset is not None, "Asset name not found in scene. Has to be an articulation or rigid body."

        # set the world pose of the asset
        asset.write_root_pose_to_sim(
            torch.cat([positions[env_ids], orientations[env_ids]], dim=-1),
            env_ids=env_ids.to(self.scene.device),
        )

    def post_sim_update(
        self,
        samples_idx: torch.Tensor,
        env_ids: torch.Tensor,
        filedir: str,
        slice_bounding_box: tuple[float, float, float, float] | None = None,
    ):
        if self.cfg.obs_term_cfg is not None:
            # construct a dummy class with the current scene (no other attribute can be accesses)

            class ManagerBasedEnvDummy:
                def __init__(self, scene: InteractiveScene):
                    self.scene = scene

            env_dummy = ManagerBasedEnvDummy(self.scene)

            output = {}

            for obs_term_cfg in self.cfg.obs_term_cfg:
                # get the mdp output
                curr_output = obs_term_cfg.func(env_dummy, **obs_term_cfg.params)

                # apply post-processing
                if obs_term_cfg.modifiers is not None:
                    for modifier in obs_term_cfg.modifiers:
                        curr_output = modifier.func(curr_output, **modifier.params)
                if isinstance(obs_term_cfg.noise, noise.NoiseCfg):
                    curr_output = obs_term_cfg.noise.func(curr_output, obs_term_cfg.noise)
                elif (
                    isinstance(obs_term_cfg.noise, noise.NoiseModelCfg)
                    and obs_term_cfg.noise.func is not None
                ):
                    curr_output = obs_term_cfg.noise.func(curr_output)
                if obs_term_cfg.clip:
                    curr_output = curr_output.clip_(min=obs_term_cfg.clip[0], max=obs_term_cfg.clip[1])
                if obs_term_cfg.scale is not None:
                    curr_output = curr_output.mul_(obs_term_cfg.scale)

                # save the output for the current function
                output[obs_term_cfg.func.__name__] = curr_output

        else:
            output = self.scene.sensors[self.cfg.sensor_name].data.ray_hits_w

        for env_id in env_ids:
            if isinstance(output, dict):
                for key, value in output.items():
                    np.save(os.path.join(filedir, "raycaster", key, f"{self._idx}".zfill(4) + ".npy"), value[env_id].cpu().numpy())
            else:
                # save the output
                np.save(os.path.join(filedir, "raycaster", f"{self._idx}".zfill(4) + ".npy"), output[env_id].cpu().numpy())

            # increment the index
            self._idx += 1
