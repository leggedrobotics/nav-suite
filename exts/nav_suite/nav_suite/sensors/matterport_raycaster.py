# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import os
import torch
import trimesh
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
import omni.physics.tensors.impl.api as physx
import pandas as pd
import warp as wp
from isaaclab.sensors.ray_caster import RayCaster
from isaaclab.utils.math import convert_quat, quat_apply, quat_apply_yaw
from isaaclab.utils.warp import raycast_mesh
from isaacsim.core.prims import XFormPrim

from nav_suite import NAVSUITE_DATA_DIR

from .matterport_raycaster_data import MatterportRayCasterData

if TYPE_CHECKING:
    from .matterport_raycaster_cfg import MatterportRayCasterCfg


class MatterportRayCaster(RayCaster):
    """A ray-casting sensor for matterport meshes.

    The ray-caster uses a set of rays to detect collisions with meshes in the scene. The rays are
    defined in the sensor's local coordinate frame. The sensor can be configured to ray-cast against
    a set of meshes with a given ray pattern.

    The meshes are parsed from the list of primitive paths provided in the configuration. These are then
    converted to warp meshes and stored in the `warp_meshes` list. The ray-caster then ray-casts against
    these warp meshes using the ray pattern provided in the configuration.

    .. note::
        Currently, only static meshes are supported. Extending the warp mesh to support dynamic meshes
        is a work in progress.

    .. note::
        This class depends on the public RayCaster class, the RSL multi-mesh implementation is
        currently not supported.
    """

    cfg: MatterportRayCasterCfg
    """The configuration parameters."""

    def __init__(self, cfg: MatterportRayCasterCfg):
        """Initializes the ray-caster object.

        Args:
            cfg (MatterportRayCasterCfg): The configuration parameters.
        """
        # initialize base class
        super().__init__(cfg)

        # face id to semantic category id mapping
        self.face_id_category_mapping = {}

        self._data = MatterportRayCasterData()

    def _initialize_impl(self):
        super()._initialize_impl()

        # load categort id to class mapping (name and id of mpcat40 redcued class set)
        # More Information: https://github.com/niessner/Matterport/blob/master/data_organization.md#house_segmentations
        mapping = pd.read_csv(NAVSUITE_DATA_DIR + "/matterport/category_mapping.tsv", sep="\t")
        self.mapping_mpcat40 = torch.tensor(mapping["mpcat40index"].to_numpy(), device=self._device, dtype=torch.long)
        self.classes_mpcat40 = pd.read_csv(NAVSUITE_DATA_DIR + "/matterport/mpcat40.tsv", sep="\t")[
            "mpcat40"
        ].to_numpy()

        # init buffer for semantic class of the rays
        self._data.ray_class_ids = torch.zeros(self._num_envs, self.num_rays, device=self._device, dtype=torch.long)

    def _initialize_warp_meshes(self):
        # check if mesh is already loaded
        assert len(self.cfg.mesh_prim_paths) == 1, "Currently only one Matterport Environment is supported."

        for mesh_prim_path in self.cfg.mesh_prim_paths:
            if mesh_prim_path in self.meshes and mesh_prim_path in self.face_id_category_mapping:
                continue

            # find ply
            if os.path.isabs(mesh_prim_path):
                file_path = mesh_prim_path
                assert os.path.isfile(mesh_prim_path), f"No .ply file found under absolute path: {mesh_prim_path}"
            else:
                file_path = os.path.join(NAVSUITE_DATA_DIR, mesh_prim_path)
                assert os.path.isfile(
                    file_path
                ), f"No .ply file found under relative path to extension data: {file_path}"

            # load ply
            curr_trimesh = trimesh.load(file_path)

            if mesh_prim_path not in self.meshes:
                # Convert trimesh into wp mesh
                mesh_wp = wp.Mesh(
                    points=wp.array(curr_trimesh.vertices.astype(np.float32), dtype=wp.vec3, device=self._device),
                    indices=wp.array(curr_trimesh.faces.astype(np.int32).flatten(), dtype=int, device=self._device),
                )
                # save mesh
                self.meshes[mesh_prim_path] = mesh_wp

            if mesh_prim_path not in self.face_id_category_mapping:
                # create mapping from face id to semantic category id
                # get raw face information
                faces_raw = curr_trimesh.metadata["_ply_raw"]["face"]["data"]
                omni.log.info(f"Raw face information of type {faces_raw.dtype}")
                # get face categories
                face_id_category_mapping = torch.tensor(
                    [single_face[3] for single_face in faces_raw], device=self._device
                )
                # save mapping
                self.face_id_category_mapping[mesh_prim_path] = face_id_category_mapping

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        # obtain the poses of the sensors
        if isinstance(self._view, XFormPrim):
            pos_w, quat_w = self._view.get_world_poses(env_ids)
        elif isinstance(self._view, physx.ArticulationView):
            pos_w, quat_w = self._view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        elif isinstance(self._view, physx.RigidBodyView):
            pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        else:
            raise RuntimeError(f"Unsupported view type: {type(self._view)}")
        # note: we clone here because we are read-only operations
        pos_w = pos_w.clone()
        quat_w = quat_w.clone()
        # apply drift
        pos_w += self.drift[env_ids]
        # store the poses
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w[env_ids] = quat_w

        # ray cast based on the sensor poses
        if self.cfg.attach_yaw_only:
            # only yaw orientation is considered and directions are not rotated
            ray_starts_w = quat_apply_yaw(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = self.ray_directions[env_ids]
        else:
            # full orientation is considered
            ray_starts_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])
        # ray cast and store the hits
        # TODO: Make this work for multiple meshes?
        self._data.ray_hits_w[env_ids], _, _, ray_face_ids = raycast_mesh(
            ray_starts_w,
            ray_directions_w,
            max_dist=self.cfg.max_distance,
            mesh=self.meshes[self.cfg.mesh_prim_paths[0]],
            return_face_id=True,
        )
        # assign each hit the semantic class
        face_id = self.face_id_category_mapping[self.cfg.mesh_prim_paths[0]][ray_face_ids.flatten().type(torch.long)]
        # map category index to reduced set
        self._data.ray_class_ids[env_ids] = self.mapping_mpcat40[face_id.type(torch.long) - 1].reshape(len(env_ids), -1)
