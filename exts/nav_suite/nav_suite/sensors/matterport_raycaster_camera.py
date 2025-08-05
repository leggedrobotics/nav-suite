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
from typing import ClassVar

import isaaclab.utils.math as math_utils
import omni.log
import pandas as pd
import warp as wp
from isaaclab.sensors import RayCasterCamera, RayCasterCameraCfg
from isaaclab.utils.warp import raycast_mesh

from nav_suite import NAVSUITE_DATA_DIR


class MatterportRayCasterCamera(RayCasterCamera):
    """Ray-caster camera for Matterport environments.

    .. note::
        This class depends on the public RayCasterCamera class, the RSL multi-mesh implementation is
        currently not supported.
    """

    UNSUPPORTED_TYPES: ClassVar[dict] = {
        "rgb",
        "instance_id_segmentation",
        "instance_segmentation",
        "skeleton_data",
        "motion_vectors",
        "bounding_box_2d_tight",
        "bounding_box_2d_loose",
        "bounding_box_3d",
    }
    """Data types that are not supported by the ray-caster."""

    def __init__(self, cfg: RayCasterCameraCfg):
        # initialize base class
        super().__init__(cfg)

        # mapping from face_id to semantic category id
        self.face_id_category_mapping = {}

    def _check_supported_data_types(self, cfg: RayCasterCameraCfg):
        # check if there is any intersection in unsupported types
        # reason: we cannot obtain this data from simplified warp-based ray caster
        common_elements = set(cfg.data_types) & MatterportRayCasterCamera.UNSUPPORTED_TYPES
        if common_elements:
            raise ValueError(
                f"RayCasterCamera class does not support the following sensor types: {common_elements}."
                "\n\tThis is because these sensor types cannot be obtained in a fast way using ''warp''."
                "\n\tHint: If you need to work with these sensor types, we recommend using the USD camera"
                " interface from the isaaclab.sensors.camera module."
            )

    def _initialize_impl(self):
        super()._initialize_impl()

        # load categort id to class mapping (name and id of mpcat40 redcued class set)
        # More Information: https://github.com/niessner/Matterport/blob/master/data_organization.md#house_segmentations
        mapping = pd.read_csv(NAVSUITE_DATA_DIR + "/matterport/category_mapping.tsv", sep="\t")
        self.mapping_mpcat40 = torch.tensor(mapping["mpcat40index"].to_numpy(), device=self._device, dtype=torch.long)
        self.classes_mpcat40 = pd.read_csv(NAVSUITE_DATA_DIR + "/matterport/mpcat40.tsv", sep="\t")[
            "mpcat40"
        ].to_numpy()
        self._color_mapping()

    def _color_mapping(self):
        # load defined colors for mpcat40
        mapping_40 = pd.read_csv(NAVSUITE_DATA_DIR + "/matterport/mpcat40.tsv", sep="\t")
        color = mapping_40["hex"].to_numpy()
        self.color = torch.tensor(
            [(int(color[i][1:3], 16), int(color[i][3:5], 16), int(color[i][5:7], 16)) for i in range(len(color))],
            device=self._device,
            dtype=torch.uint8,
        )

    def _initialize_warp_meshes(self):
        # only one mesh is supported
        assert len(self.cfg.mesh_prim_paths) == 1, "Currently only one Matterport Environment is supported."

        # check if mesh is already loaded
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
        # increment frame count
        self._frame[env_ids] += 1

        # compute poses from current view
        pos_w, quat_w = self._compute_camera_world_poses(env_ids)
        # update the data
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w_world[env_ids] = quat_w

        # note: full orientation is considered
        ray_starts_w = math_utils.quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
        ray_starts_w += pos_w.unsqueeze(1)
        ray_directions_w = math_utils.quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])
        # ray cast and store the hits
        # TODO: Make ray-casting work for multiple meshes?
        # necessary for regular dictionaries.
        self.ray_hits_w, ray_depth, ray_normal, ray_face_ids = raycast_mesh(
            ray_starts_w,
            ray_directions_w,
            mesh=self.meshes[self.cfg.mesh_prim_paths[0]],
            max_dist=self.cfg.max_distance,
            return_distance=any(
                [name in self.cfg.data_types for name in ["distance_to_image_plane", "distance_to_camera"]]
            ),
            return_normal="normals" in self.cfg.data_types,
            return_face_id="semantic_segmentation" in self.cfg.data_types,
        )
        # update output buffers
        if "distance_to_image_plane" in self.cfg.data_types:
            # note: data is in camera frame so we only take the first component (z-axis of camera frame)
            distance_to_image_plane = (
                math_utils.quat_apply(
                    math_utils.quat_inv(quat_w).repeat(1, self.num_rays),
                    (ray_depth[:, :, None] * ray_directions_w),
                )
            )[:, :, 0]
            self._data.output["distance_to_image_plane"][env_ids] = distance_to_image_plane.view(-1, *self.image_shape)
        if "distance_to_camera" in self.cfg.data_types:
            self._data.output["distance_to_camera"][env_ids] = ray_depth.view(-1, *self.image_shape)
        if "normals" in self.cfg.data_types:
            self._data.output["normals"][env_ids] = ray_normal.view(-1, *self.image_shape, 3)
        if "semantic_segmentation" in self._data.output.keys():  # noqa: SIM118
            # get the category index of the hit faces (category index from unreduced set = ~1600 classes)
            face_id = self.face_id_category_mapping[self.cfg.mesh_prim_paths[0]][
                ray_face_ids.flatten().type(torch.long)
            ]
            # map category index to reduced set
            face_id_mpcat40 = self.mapping_mpcat40[face_id.type(torch.long) - 1]
            # get the color of the face
            face_color = self.color[face_id_mpcat40]
            # reshape and transpose to get the correct orientation
            self._data.output["semantic_segmentation"][env_ids] = face_color.view(-1, *self.image_shape, 3)

    def _create_buffers(self):
        """Create the buffers to store data."""
        # prepare drift
        self.drift = torch.zeros(self._view.count, 3, device=self.device)
        # create the data object
        # -- pose of the cameras
        self._data.pos_w = torch.zeros((self._view.count, 3), device=self._device)
        self._data.quat_w_world = torch.zeros((self._view.count, 4), device=self._device)
        # -- intrinsic matrix
        self._data.intrinsic_matrices = torch.zeros((self._view.count, 3, 3), device=self._device)
        self._data.intrinsic_matrices[:, 2, 2] = 1.0
        self._data.image_shape = self.image_shape
        # -- output data
        # create the buffers to store the annotator data.
        self._data.output = {}
        self._data.info = [{name: None for name in self.cfg.data_types}] * self._view.count
        for name in self.cfg.data_types:
            if name in ["distance_to_image_plane", "distance_to_camera"]:
                shape = (self.cfg.pattern_cfg.height, self.cfg.pattern_cfg.width)
                dtype = torch.float32
            elif name in ["normals"]:
                shape = (self.cfg.pattern_cfg.height, self.cfg.pattern_cfg.width, 3)
                dtype = torch.float32
            elif name in ["semantic_segmentation"]:
                shape = (self.cfg.pattern_cfg.height, self.cfg.pattern_cfg.width, 3)
                dtype = torch.uint8
            else:
                raise ValueError(f"Unknown data type: {name}")
            # store the data
            self._data.output[name] = torch.zeros((self._view.count, *shape), dtype=dtype, device=self._device)

    """
    Debug Visualization
    """

    # def _debug_vis_callback(self, event):
    #     """Callback for the debug visualization event."""
    #     super()._debug_vis_callback(event)

    #     if not hasattr(self, "image_visualizer") or not SimulationContext.instance().is_playing():
    #         # Visualizers have not been created yet.
    #         return
    #     # update the visualization
    #     self.image_visualizer.update_image(self._data.output["semantic_segmentation"].view(-1, 3).cpu().numpy())

    # def _set_debug_vis_impl(self, debug_vis: bool):
    #     """Set the debug visualization implementation.

    #     Args:
    #         debug_vis: Whether to enable or disable debug visualization.
    #     """
    #     super()._set_debug_vis_impl(debug_vis)

    #     if not hasattr(self, "_viewer_env_idx"):
    #         self._viewer_env_idx = 0

    #     if not hasattr(self, "_vis_frame"):
    #         raise RuntimeError("No frame set for debug visualization.")

    #     # Clear internal visualizers
    #     self._action_term_visualizers = []
    #     self._vis_frame.clear()

    #     if debug_vis:
    #         # create a subscriber for the post update event if it doesn't exist
    #         if not hasattr(self, "_debug_vis_handle") or self._debug_vis_handle is None:
    #             app_interface = omni.kit.app.get_app_interface()
    #             self._debug_vis_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
    #                 lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event)
    #             )
    #     else:
    #         # remove the subscriber if it exists
    #         if self._debug_vis_handle is not None:
    #             self._debug_vis_handle.unsubscribe()
    #             self._debug_vis_handle = None

    #         self._vis_frame.visible = False
    #         return

    #     self._vis_frame.visible = True

    #     with self._vis_frame:
    #         with VStack():
    #             # Add a plot in a collapsible frame for each action term
    #             for name, terms in self.get_active_iterable_terms():
    #                 frame = CollapsableFrame(
    #                     name,
    #                     collapsed=False,
    #                     style={"border_color": 0xFF8A8777, "padding": 4},
    #                 )
    #                 with frame:
    #                     plot = LiveLinePlot(
    #                         y_data=[[term] for term in terms],
    #                         plot_height=150,
    #                         show_legend=True,
    #                     )
    #                     self._action_term_visualizers.append(plot)
    #                 frame.collapsed = True

    #     self._debug_vis = debug_vis
