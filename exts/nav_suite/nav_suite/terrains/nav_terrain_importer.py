# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import numpy as np
import os
import torch
import trimesh
import yaml
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import omni
import omni.log
from isaaclab.terrains import TerrainImporter
from isaaclab.terrains.utils import create_prim_from_mesh
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaacsim.core.utils.semantics import add_update_semantics, check_missing_semantics, remove_all_semantics
from pxr import Gf, UsdGeom

from nav_suite.utils.obj_converter import ObjConverter
from nav_suite.utils.prims import get_all_prims_including_str

if TYPE_CHECKING:
    from .terrain_importer_cfg import NavTerrainImporterCfg


class NavTerrainImporter(TerrainImporter):
    """
    A unified environment importer that can handle both USD environments (e.g. from UnrealEngine), generated
    environments, and OBJ files (e.g., Matterport scans).

    Supports features like semantic labeling, mesh duplication, and people placement.
    Furthermore, multi-usd environments are supported.
    """

    cfg: NavTerrainImporterCfg

    def __init__(self, cfg: NavTerrainImporterCfg) -> None:
        # Actual import happens here
        super().__init__(cfg)

        # Add collision and physics properties to a environment if specified
        if self.cfg.add_colliders:
            self._apply_physics_properties()

        # Add a ground plane if specified
        if self.cfg.groundplane:
            ground_plane_cfg = sim_utils.GroundPlaneCfg(physics_material=self.cfg.physics_material)
            ground_plane = ground_plane_cfg.func("/World/GroundPlane", ground_plane_cfg)
            ground_plane.visible = False

        # add usd meshes (e.g. for people)
        if self.cfg.people_config_file:
            self._insert_people()

        # assign semantic labels
        if self.cfg.sem_mesh_to_class_map:
            self._add_semantics()

    """
    Overwrite Environment Origins Method
    """

    def configure_env_origins(self, origins: np.ndarray | torch.Tensor | None = None):
        """Configure the origins of the environments based on the added terrain.

        Args:
            origins: The origins of the sub-terrains. Shape is (num_rows, num_cols, 3).

        .. note::
            This method adds the options to add grid-like origins to a USD terrain.
        """
        # decide whether to compute origins in a grid or based on curriculum
        terrain_origins_source = self.cfg.custom_origins if self.cfg.custom_origins is not None else origins

        if terrain_origins_source is not None:
            # convert to torch tensor if needed
            if isinstance(terrain_origins_source, np.ndarray):
                terrain_origins_source = torch.from_numpy(terrain_origins_source)
            # store the origins
            self.terrain_origins = terrain_origins_source.to(self.device, dtype=torch.float)
            # compute environment origins
            self.env_origins = self._compute_env_origins_curriculum(self.cfg.num_envs, self.terrain_origins)
        # uniform env_spacing for usd_size
        elif self.cfg.usd_uniform_env_spacing is not None and self.cfg.terrain_type == "usd":
            # get the size of the mesh
            prim = prim_utils.get_prim_at_path(self.cfg.prim_path + "/terrain")
            bbox = UsdGeom.Boundable(prim).ComputeWorldBound(0, UsdGeom.Tokens.default_).ComputeAlignedBox()
            # set the env_spacing
            grid_x, grid_y = torch.meshgrid(
                torch.arange(
                    bbox.GetMin()[0] + self.cfg.usd_uniform_env_spacing / 2,
                    bbox.GetMax()[0] - self.cfg.usd_uniform_env_spacing / 2,
                    self.cfg.usd_uniform_env_spacing,
                ),
                torch.arange(
                    bbox.GetMin()[1] + self.cfg.usd_uniform_env_spacing / 2,
                    bbox.GetMax()[1] - self.cfg.usd_uniform_env_spacing / 2,
                    self.cfg.usd_uniform_env_spacing,
                ),
                indexing="ij",
            )
            self.env_origins = torch.stack(
                (grid_x.flatten(), grid_y.flatten(), torch.zeros_like(grid_x.flatten())), dim=1
            ).to(self.device)
            # make length equal to number of envs
            # Calculate the number of repetitions needed
            repetitions = max(1, (self.cfg.num_envs // self.env_origins.shape[0]) + 1)
            self.env_origins = self.env_origins.repeat(repetitions, 1)[: self.cfg.num_envs]
        else:
            self.terrain_origins = None
            # check if env spacing is valid
            if self.cfg.env_spacing is None:
                raise ValueError("Environment spacing must be specified for configuring grid-like origins.")
            # compute environment origins
            self.env_origins = self._compute_env_origins_grid(self.cfg.num_envs, self.cfg.env_spacing)

    def _compute_env_origins_curriculum(self, num_envs: int, origins: torch.Tensor) -> torch.Tensor:
        """Compute the origins of the environments defined by the sub-terrains origins.

        .. note::
            Adds option to spawn regular on all origins or random on the grid.
        """
        # extract number of rows and cols
        num_rows, num_cols = origins.shape[:2]
        # maximum initial level possible for the terrains
        if self.cfg.max_init_terrain_level is None:
            max_init_level = num_rows - 1
        else:
            max_init_level = min(self.cfg.max_init_terrain_level, num_rows - 1)
        # store maximum terrain level possible
        self.max_terrain_level = num_rows
        # check if spawn on random terrain origins or in a regular pattern on all origins
        if self.cfg.regular_spawning:
            repeated_terrain_origins = origins[: max_init_level + 1, :].reshape(-1, 3)
            return repeated_terrain_origins.repeat(math.ceil(num_envs / repeated_terrain_origins.shape[0]), 1)[
                :num_envs
            ]

        else:
            # define all terrain levels and types available
            if hasattr(self.cfg, "random_seed") and self.cfg.random_seed is not None:
                torch.manual_seed(self.cfg.random_seed)
            self.terrain_levels = torch.randint(0, max_init_level + 1, (num_envs,), device=self.device)
            self.terrain_types = torch.div(
                torch.arange(num_envs, device=self.device),
                (num_envs / num_cols),
                rounding_mode="floor",
            ).to(torch.long)
            # create tensor based on number of environments
            env_origins = torch.zeros(num_envs, 3, device=self.device)
            env_origins[:] = origins[self.terrain_levels, self.terrain_types]
            return env_origins

    """
    Assign Semantic Labels
    """

    def _add_semantics(self):
        # remove all previous semantic labels
        remove_all_semantics(prim_utils.get_prim_at_path(self.cfg.prim_path + "/terrain"), recursive=True)

        # get all meshes where semantic labels are missing
        missing_prims = check_missing_semantics(self.cfg.prim_path + "/terrain")
        omni.log.info(f"Total of {len(missing_prims)} meshes in the scene, start assigning semantic class ...")

        # mapping from prim name to class
        with open(self.cfg.sem_mesh_to_class_map) as stream:
            class_keywords = yaml.safe_load(stream)

        if "default" in class_keywords:
            default_class = class_keywords["default"]
            del class_keywords["default"]
        else:
            default_class = None

        # make all the string lower case
        keywords_class_mapping_lower = {
            key: [value_single.lower() for value_single in value] for key, value in class_keywords.items()
        }

        # loop over them, assign labels
        success_prims = np.zeros(len(missing_prims), dtype=bool)
        for missing_prim_idx, missing_prim in enumerate(missing_prims):
            missing_prim_lower = missing_prim.lower()
            for class_name, keywords in keywords_class_mapping_lower.items():
                if any([keyword in missing_prim_lower for keyword in keywords]):
                    add_update_semantics(prim_utils.get_prim_at_path(missing_prim), class_name)
                    success_prims[missing_prim_idx] = True
                    break

        if default_class is not None and not np.all(success_prims):
            omni.log.warn(
                f"Missing semantic classes on {np.sum(~success_prims)} meshes, assigning default class"
                f" {default_class} to missing meshes."
            )
            for missing_prim_idx, missing_prim in enumerate(missing_prims):
                if not success_prims[missing_prim_idx]:
                    add_update_semantics(prim_utils.get_prim_at_path(missing_prim), default_class)
        elif not np.all(success_prims):
            raise ValueError("Not all meshes are assigned a semantic class!")

        omni.log.info("Semantic mapping done.")

    """
    Duplicate Meshes and add People
    """

    def mesh_duplicator(self, duplicate_cfg_filepath: str):
        """Duplicate prims in the scene."""

        with open(duplicate_cfg_filepath) as stream:
            multipy_cfg: dict = yaml.safe_load(stream)

        # get the stage
        stage = omni.usd.get_context().get_stage()

        # init counter
        add_counter = 0

        for value in multipy_cfg.values():
            # get the prim that should be duplicated
            prims = get_all_prims_including_str(self.cfg.prim_path + "/terrain", value["prim"])

            if len(prims) == 0:
                omni.log.info(f"Could not find prim {value['prim']}, no replication possible!")
                continue

            if value.get("only_first_match", True):
                prims = [prims[0]]

            # make translations a list of lists in the case only a single translation is given
            if not isinstance(value["translation"][0], list):
                value["translation"] = [value["translation"]]

            # iterate over translations and their factor
            for translation_idx, curr_translation in enumerate(value["translation"]):
                for copy_idx in range(value.get("factor", 1)):
                    for curr_prim in prims:
                        # get the path of the current prim
                        curr_prim_path = curr_prim.GetPath().pathString
                        # copy path
                        new_prim_path = os.path.join(
                            curr_prim_path + f"_tr{translation_idx}_cp{copy_idx}" + value.get("suffix", "")
                        )

                        success = omni.usd.duplicate_prim(
                            stage=stage,
                            prim_path=curr_prim_path,
                            path_to=new_prim_path,
                            duplicate_layers=True,
                        )
                        assert success, f"Failed to duplicate prim '{curr_prim_path}'"

                        # get crosswalk prim
                        prim = prim_utils.get_prim_at_path(new_prim_path)
                        xform = UsdGeom.Mesh(prim).AddTranslateOp()
                        xform.Set(
                            Gf.Vec3d(curr_translation[0], curr_translation[1], curr_translation[2]) * (copy_idx + 1)
                        )

                        # update counter
                        add_counter += 1

        omni.log.info(f"Number of added prims: {add_counter} from file {duplicate_cfg_filepath}")

    def _insert_people(self):
        # load people config file
        with open(self.cfg.people_config_file) as stream:
            people_cfg: dict = yaml.safe_load(stream)

        for key, person_cfg in people_cfg.items():
            self.insert_single_person(
                person_cfg["prim_name"],
                person_cfg["translation"],
                scale_people=person_cfg.get("scale", 1.0),
                usd_path=person_cfg.get("usd_path", "People/Characters/F_Business_02/F_Business_02.usd"),
            )
            # TODO: allow for movement of the people

        omni.log.info(f"Number of people added: {len(people_cfg)}")

    @staticmethod
    def insert_single_person(
        prim_name: str,
        translation: list,
        scale_people: float = 1.0,
        usd_path: str = "People/Characters/F_Business_02/F_Business_02.usd",
    ) -> None:
        person_prim = prim_utils.create_prim(
            prim_path=os.path.join("/World/People", prim_name),
            translation=tuple(translation),
            usd_path=os.path.join(ISAAC_NUCLEUS_DIR, usd_path),
            scale=(scale_people, scale_people, scale_people),
        )

        if isinstance(person_prim.GetAttribute("xformOp:orient").Get(), Gf.Quatd):
            person_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
        else:
            person_prim.GetAttribute("xformOp:orient").Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        add_update_semantics(person_prim, "person")

        # add collision body
        UsdGeom.Mesh(person_prim)

    """
    Import USD method with scale option
    """

    def import_usd(self, name: str, usd_path: str | dict[str, str]):
        """Import a mesh from a USD file.
        This function imports a USD file into the simulator as a terrain. It parses the USD file and
        stores the mesh under the prim path ``cfg.prim_path/{key}``. If multiple meshes are present in
        the USD file, only the first mesh is imported.
        The function doe not apply any material properties to the mesh. The material properties should
        be defined in the USD file.
        Args:
            name: The name of the imported terrain. This name is used to create the USD prim
                corresponding to the terrain.
            usd_path: The path to the USD file.
        Raises:
            ValueError: If a terrain with the same name already exists.
        """

        def import_single_usd(name: str, usd_path: str, scale: tuple[float, float, float]):
            # create prim path for the terrain
            prim_path = self.cfg.prim_path + f"/{name}"
            # check if key exists
            if prim_path in self.terrain_prim_paths:
                raise ValueError(
                    f"A terrain with the name '{name}' already exists. Existing terrains:"
                    f" {', '.join(self.terrain_names)}."
                )
            # store the mesh name
            self.terrain_prim_paths.append(prim_path)

            # add the prim path
            cfg = sim_utils.UsdFileCfg(usd_path=usd_path, scale=scale)
            terrain_prim = cfg.func(prim_path, cfg)
            return terrain_prim

        # add the prim path
        if isinstance(usd_path, str):
            usd_path = self._convert_obj_to_usd(usd_path)
            import_single_usd(name, usd_path, self.cfg.scale)
        elif isinstance(usd_path, dict):
            # NOTE: mesh are added along the y-axis
            curr_translation = 0.0
            for idx, (terrain_name, curr_usd_path) in enumerate(usd_path.items()):
                curr_usd_path = self._convert_obj_to_usd(curr_usd_path)
                terrain_prim = import_single_usd(
                    terrain_name,
                    curr_usd_path,
                    self.cfg.scale[terrain_name] if isinstance(self.cfg.scale, dict) else self.cfg.scale,
                )
                # get the extent of the terrain
                bbox = UsdGeom.Boundable(terrain_prim).ComputeWorldBound(0, UsdGeom.Tokens.default_).ComputeAlignedBox()
                if idx != 0:
                    # update translation with the minimum extend of the terrain with a 1m buffer
                    curr_translation -= bbox.GetMin()[1] - 1.0
                # add translation to the terrain
                terrain_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, curr_translation, 0))
                # update translation for the next terrain
                curr_translation += bbox.GetMax()[1]

        # modify mesh if duplication config is provided
        if self.cfg.duplicate_cfg_file and isinstance(self.cfg.duplicate_cfg_file, str):
            self.mesh_duplicator(self.cfg.duplicate_cfg_file)
        elif self.cfg.duplicate_cfg_file and isinstance(self.cfg.duplicate_cfg_file, list):
            [self.mesh_duplicator(duplicate_cfg_file) for duplicate_cfg_file in self.cfg.duplicate_cfg_file]
        else:
            omni.log.info("No mesh duplication executed.")

    def import_mesh(self, name: str, mesh: trimesh.Trimesh | dict[float | str, trimesh.Trimesh]):
        """Import a mesh into the simulator.

        The mesh is imported into the simulator under the prim path ``cfg.prim_path/{key}``. The created path
        contains the mesh as a :class:`pxr.UsdGeom` instance along with visual or physics material prims.

        Args:
            name: The name of the imported terrain. This name is used to create the USD prim
                corresponding to the terrain.
            mesh: The mesh to import.

        Raises:
            ValueError: If a terrain with the same name already exists.
        """
        if isinstance(mesh, trimesh.Trimesh):
            mesh = {None: mesh}
        else:
            # save the mesh descriptor
            self.mesh_descriptors = {
                self.cfg.prim_path + f"/{name}_{curr_mesh_descriptor}".replace(".", "_"): curr_mesh_descriptor
                for curr_mesh_descriptor in mesh.keys()
            }

        for idx, (curr_mesh_descriptor, curr_mesh) in enumerate(mesh.items()):
            # create prim path for the terrain
            if curr_mesh_descriptor is None:
                prim_path = self.cfg.prim_path + f"/{name}"
            else:
                prim_path = self.cfg.prim_path + f"/{name}_{curr_mesh_descriptor}".replace(".", "_")
            # check if key exists
            if prim_path in self.terrain_prim_paths:
                raise ValueError(
                    f"A terrain with the name '{name}' already exists. Existing terrains:"
                    f" {', '.join(self.terrain_names)}."
                )
            # store the mesh name
            self.terrain_prim_paths.append(prim_path)

            # import the mesh
            create_prim_from_mesh(
                prim_path,
                curr_mesh,
                visual_material=self.cfg.visual_material,
                physics_material=self.cfg.physics_material,
            )

    """
    Helper functions
    """

    def _convert_obj_to_usd(self, file_path: str) -> str:
        # check if the file is already a usd file
        base_path, _ = os.path.splitext(file_path)

        if os.path.exists(base_path + ".usd"):
            omni.log.info(f"Loading environment from {base_path}.usd")
            return base_path + ".usd"
        elif os.path.exists(base_path + ".usda"):
            omni.log.info(f"Loading environment from {base_path}.usda")
            return base_path + ".usda"
        elif os.path.exists(base_path + ".usdc"):
            omni.log.info(f"Loading environment from {base_path}.usdc")
            return base_path + ".usdc"

        elif os.path.exists(base_path + ".obj"):
            omni.log.info(f"Loading environment from {base_path}.obj")

            from isaacsim.core.utils import extensions

            extensions.enable_extension("omni.kit.asset_converter")

            self.cfg.asset_converter.asset_path = file_path
            self.cfg.asset_converter.usd_dir = os.path.dirname(file_path)
            self.cfg.asset_converter.usd_file_name = os.path.basename(file_path.replace(".obj", ".usd"))
            self.cfg.asset_converter.force_usd_conversion = True

            # Create converter instance
            converter_instance = ObjConverter(self.cfg.asset_converter)

            # Convert the asset
            converter_instance._convert_asset(self.cfg.asset_converter)

            self.cfg.usd_path = base_path + ".usd"

            # update the usd_path
            return base_path + ".usd"

        else:
            raise ValueError(f"Could not find any environment file in {base_path}. Please check asset.")

    def _apply_physics_properties(self):

        # Look for mesh in the Environment prim
        mesh_prim = sim_utils.get_first_matching_child_prim(
            self.cfg.prim_path,
            lambda prim: prim.GetTypeName() == "Mesh",
        )
        # Check if the mesh is valid
        if mesh_prim is None:
            omni.log.warn(f"Could not find any meshes in {self.cfg.usd_path}. Please check asset.")
            return

        # Apply collider properties
        collider_cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        sim_utils.define_collision_properties(self.cfg.prim_path, collider_cfg)

        # Create and bind physics material
        physics_material_cfg: sim_utils.RigidBodyMaterialCfg = self.cfg.physics_material
        physics_material_cfg.func(f"{self.cfg.prim_path}/physicsMaterial", self.cfg.physics_material)
        sim_utils.bind_physics_material(self.cfg.prim_path, f"{self.cfg.prim_path}/physicsMaterial")

        # Update stage for any remaining process
        stage_utils.update_stage()
        omni.log.info("Environment setup complete...")
