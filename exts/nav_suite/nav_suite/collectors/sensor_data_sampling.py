from __future__ import annotations

import numpy as np
import os
import pickle
import random
import torch

import isaaclab.utils.math as math_utils
import omni.log
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

from .sensor_data_sampling_cfg import SensorDataSamplingCfg


class SensorDataSampling:
    def __init__(self, cfg: SensorDataSamplingCfg, scene: InteractiveScene):
        # check the config for any missing values
        cfg.validate()

        # save cfg and env
        self.cfg = cfg
        self.scene = scene

        # get sim context
        self.sim = SimulationContext.instance()

        # analyse terrains -- check if singleton is used and available
        if (
            hasattr(self.cfg.terrain_analysis.class_type, "instance")
            and self.cfg.terrain_analysis.class_type.instance() is not None
        ):
            self.terrain_analyser = self.cfg.terrain_analysis.class_type.instance()
        else:
            self.terrain_analyser = self.cfg.terrain_analysis.class_type(self.cfg.terrain_analysis, scene=self.scene)

        # init the sensor data handlers
        self.sensor_data_handlers = [
            sensor_cfg.class_type(sensor_cfg, scene=self.scene) for sensor_cfg in self.cfg.sensor_data_handlers
        ]

    ###
    # Properties
    ###

    @property
    def samples(self) -> torch.Tensor | list[torch.Tensor]:
        """Get the sample points.

        Samples are stored in a torch tensor with the structure
        [x, y, z, qw, qx, qv, qz]
        """
        return self._samples

    @property
    def sliced_bounding_boxes(self) -> list[tuple[float, float, float, float]] | None:
        """Get the sliced bounding boxes.

        The sliced bounding boxes follow the terrain analysis format of the mesh dimensions, i.e. [x_max, y_max, x_min, y_min]
        """
        if self.cfg.sliced_sampling is not None and hasattr(self, "_sliced_bounding_boxes"):
            return self._sliced_bounding_boxes
        else:
            return None

    ###
    # Operations
    ###

    def sample_sensor_data(self, nbr_samples: int, seed: int = 1) -> torch.Tensor | list[torch.Tensor]:
        """Sample sensor data for the given number of samples and seed.

        Samples are stored in a torch tensor with the structure
        [x, y, z, qw, qx, qv, qz]

        Args:
            nbr_samples (int): The number of sample points.
            seed (int, optional): The seed for the random number generator. Defaults to 1.

        Returns:
            torch.Tensor | list[torch.Tensor]: The sampled points.
        """

        # get the number of slices and their locations
        # NOTE: the number of sample points is kept constant per slice
        if self.cfg.sliced_sampling is not None:
            # get mesh size from terrain analysis  -- [x_max, y_max, x_min, y_min]
            mesh_size = self.terrain_analyser.mesh_dimensions
            # get the number of slices in both x and y direction
            nbr_slices_x = int(np.ceil((mesh_size[0] - mesh_size[2]) / self.cfg.sliced_sampling[0]))
            nbr_slices_y = int(np.ceil((mesh_size[1] - mesh_size[3]) / self.cfg.sliced_sampling[1]))
            slice_locations_x = np.linspace(mesh_size[2], mesh_size[0], nbr_slices_x)
            slice_locations_y = np.linspace(mesh_size[3], mesh_size[1], nbr_slices_y)
            # get the slice bounding boxes
            # slide bounding boxes follow the terrain analysis format of the mesh dimensions, i.e. [x_max, y_max, x_min, y_min]
            self._sliced_bounding_boxes = []
            for i in range(nbr_slices_x - 1):
                for j in range(nbr_slices_y - 1):
                    self._sliced_bounding_boxes.append(
                        (slice_locations_x[i + 1], slice_locations_y[j + 1], slice_locations_x[i], slice_locations_y[j])
                    )

            # execute sensor data sampling for each slice
            self._samples = []
            failed_slices = []
            for i, slice_bounding_box in enumerate(self._sliced_bounding_boxes):
                try:
                    self._samples.append(self._sample_point_per_area(nbr_samples, seed, slice_bounding_box))
                except Exception as e:
                    failed_slices.append(slice_bounding_box)
                    omni.log.warn(f"Error sampling sensor data for slice {slice_bounding_box}: {e}")
                    continue

            # remove failed slices
            self._sliced_bounding_boxes = [box for box in self._sliced_bounding_boxes if box not in failed_slices]
            # dump the sliced bounding boxes and samples
            filedir = self.cfg.save_path if self.cfg.save_path else self._get_save_filedir()
            with open(os.path.join(filedir, "sliced_bounding_boxes.pkl"), "wb") as f:
                pickle.dump(self._sliced_bounding_boxes, f)

        else:
            # execute sample point sampling for the whole mesh
            self._samples = self._sample_point_per_area(nbr_samples, seed)

        return self._samples

    def render_sensor_data(
        self, samples: torch.Tensor | list[torch.Tensor] | None = None
    ) -> torch.Tensor | list[torch.Tensor]:
        """Render the sensor data at the given sample points and save them to the drive."""
        if samples is None:
            samples = self.samples

        if isinstance(samples, list):
            assert (
                self.sliced_bounding_boxes is not None
            ), "Sliced bounding boxes must be set to render a list of sample points."
            assert len(samples) == len(
                self.sliced_bounding_boxes
            ), "The number of samples must match the number of sliced bounding boxes."
            render_sample_points = []
            for sample, slice_bounding_box in zip(samples, self.sliced_bounding_boxes):
                print(f"[INFO] Rendering sample points for slice {slice_bounding_box}")
                render_sample_points.append(self._render_sensor_data_per_area(sample, slice_bounding_box))
        else:
            render_sample_points = self._render_sensor_data_per_area(samples)

        return render_sample_points

    ###
    # Helper functions
    ###

    def _sample_point_per_area(
        self, nbr_samples: int, seed: int = 1, slice_bounding_box: tuple[float, float, float, float] | None = None
    ) -> torch.Tensor:

        filedir = self.cfg.save_path if self.cfg.save_path else self._get_save_filedir()
        if slice_bounding_box is not None:
            slice_bounding_box_str = "slice_" + "_".join(
                f"{'n' if x < 0 else ''}{abs(x):.1f}" for x in slice_bounding_box
            )
            filedir = os.path.join(filedir, slice_bounding_box_str)
        elif self.cfg.terrain_analysis.terrain_bounding_box is not None:
            slice_bounding_box_str = "slice_" + "_".join(
                f"{'n' if x < 0 else ''}{abs(x):.1f}" for x in self.cfg.terrain_analysis.terrain_bounding_box
            )
            filedir = os.path.join(filedir, slice_bounding_box_str)
        else:
            slice_bounding_box_str = "full"
        filename = os.path.join(filedir, f"sample_points_seed{seed}_samples{nbr_samples}.pkl")
        if os.path.isfile(filename):
            with open(filename, "rb") as f:
                data = pickle.load(f)
            # add loaded path dict to data dict
            omni.log.info(f"Loaded {nbr_samples} with seed {seed} for map part {slice_bounding_box_str}.")
            return data
        else:
            omni.log.info(
                f"No sample points found for seed {seed} and {nbr_samples} samples for map part"
                f" {slice_bounding_box_str}."
            )

        print(f"[INFO] Sampling sample points for {slice_bounding_box_str}")

        # analyse terrain if not done yet
        if slice_bounding_box is not None:
            # overwrite the mesh dimensions to influence where the sampling is done
            self.terrain_analyser.reset_graph()
            self.terrain_analyser._mesh_dimensions = slice_bounding_box
            self.terrain_analyser.analyse()
        elif not self.terrain_analyser.complete:
            self.terrain_analyser.analyse()

        # set seed
        random.seed(seed)
        omni.log.info(f"Start sampling {nbr_samples} sample points for map part {slice_bounding_box_str}.")

        # samples are organized in [point_idx, neighbor_idx, distance]
        # sample from each point the neighbor with the largest distance
        nbr_samples_per_point = int(np.ceil(nbr_samples / self.terrain_analyser.points.shape[0]).item())
        sample_locations = torch.zeros((nbr_samples_per_point * self.terrain_analyser.points.shape[0], 2))
        sample_locations_count = 0
        curr_point_idx = 0
        while sample_locations_count < nbr_samples:
            # get samples
            sample_idx = self.terrain_analyser.samples[:, 0] == curr_point_idx
            sample_idx_select = torch.randperm(sample_idx.sum())[:nbr_samples_per_point]
            sample_locations[sample_locations_count : sample_locations_count + sample_idx_select.shape[0]] = (
                self.terrain_analyser.samples[sample_idx][sample_idx_select, :2]
            )
            sample_locations_count += sample_idx_select.shape[0]
            curr_point_idx += 1
            # reset point index if all points are sampled
            if curr_point_idx >= self.terrain_analyser.points.shape[0]:
                curr_point_idx = 0

        sample_locations = sample_locations[:sample_locations_count].type(torch.int64)

        # get the z angle of the neighbor that is closest to the origin point
        neighbor_direction = (
            self.terrain_analyser.points[sample_locations[:, 0]] - self.terrain_analyser.points[sample_locations[:, 1]]
        )
        z_angles = torch.atan2(neighbor_direction[:, 1], neighbor_direction[:, 0]).to("cpu")

        # vary the rotation of the forward and horizontal axis (in camera frame) as a uniform distribution within the limits
        x_angles = math_utils.sample_uniform(
            self.cfg.x_angle_range[0], self.cfg.x_angle_range[1], sample_locations_count, device="cpu"
        )
        y_angles = math_utils.sample_uniform(
            self.cfg.y_angle_range[0], self.cfg.y_angle_range[1], sample_locations_count, device="cpu"
        )
        x_angles = torch.deg2rad(x_angles)
        y_angles = torch.deg2rad(y_angles)

        samples = torch.zeros((sample_locations_count, 7))
        samples[:, :3] = self.terrain_analyser.points[sample_locations[:, 0]]
        samples[:, 3:] = math_utils.quat_from_euler_xyz(x_angles, y_angles, z_angles)

        omni.log.info(f"Sampled {sample_locations_count} sample points for map part {slice_bounding_box_str}.")

        # save samples
        os.makedirs(filedir, exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(samples, f)

        omni.log.info(f"Saved {sample_locations_count} sample points with seed {seed} to {filename}.")

        # debug points and orientation
        if self.cfg.debug_viz:
            env_render_steps = 1000
            marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
            marker_cfg.prim_path = "/Visuals/sample_points"
            marker_cfg.markers["arrow"].scale = (0.1, 0.1, 0.1)
            self.visualizer = VisualizationMarkers(marker_cfg)
            self.visualizer.visualize(samples[:, :3], samples[:, 3:])

            omni.log.info(f"Visualizing {sample_locations_count} samples for {env_render_steps} render steps...")
            for _ in range(env_render_steps):
                self.sim.render()

            self.visualizer.set_visibility(False)
            omni.log.info("Done visualizing.")

        return samples

    def _render_sensor_data_per_area(
        self, samples: torch.Tensor, slice_bounding_box: tuple[float, float, float, float] | None = None
    ) -> torch.Tensor:
        omni.log.info(f"Start rendering sensor data at {samples.shape[0]} points.")
        # define how many rounds are necessary to render all sample points
        num_rounds = int(np.ceil(samples.shape[0] / self.scene.num_envs))

        filedir = self.cfg.save_path if self.cfg.save_path else self._get_save_filedir()
        if slice_bounding_box is not None:
            slice_bounding_box_str = "slice_" + "_".join(
                f"{'n' if x < 0 else ''}{abs(x):.1f}" for x in slice_bounding_box
            )
            filedir = os.path.join(filedir, slice_bounding_box_str)
        elif self.cfg.terrain_analysis.terrain_bounding_box is not None:
            slice_bounding_box_str = "slice_" + "_".join(
                f"{'n' if x < 0 else ''}{abs(x):.1f}" for x in self.cfg.terrain_analysis.terrain_bounding_box
            )
            filedir = os.path.join(filedir, slice_bounding_box_str)

        # pre_collection callback
        for sensor_data_handler in self.sensor_data_handlers:
            sensor_data_handler.pre_collection(samples, filedir)

        # save images
        samples = samples.to(self.scene.device)
        for i in range(num_rounds):
            # get samples idx
            samples_idx = torch.arange(i * self.scene.num_envs, min((i + 1) * self.scene.num_envs, samples.shape[0]))
            env_ids = torch.arange(samples_idx.shape[0])

            # execute pre_sim_update callback
            for sensor_data_handler in self.sensor_data_handlers:
                sensor_data_handler.pre_sim_update(samples[samples_idx, :3], samples[samples_idx, 3:], env_ids)

            # update simulation
            self.scene.write_data_to_sim()

            # perform render steps to fill buffers if needed by any sensor
            if any([sensor_data_handler.cfg.requires_render for sensor_data_handler in self.sensor_data_handlers]):
                for _ in range(10):
                    self.sim.render()

            # update scene buffers
            self.scene.update(self.sim.get_physics_dt())

            # execute post_sim_update callback
            for sensor_data_handler in self.sensor_data_handlers:
                sensor_data_handler.post_sim_update(samples_idx, env_ids, filedir, slice_bounding_box)

        # execute post_collection callback
        for sensor_data_handler in self.sensor_data_handlers:
            sensor_data_handler.post_collection(samples, filedir)

        return samples

    def _get_save_filedir(self) -> str:
        # get env name
        if isinstance(self.scene.terrain.cfg.usd_path, str):
            terrain_file_path = self.scene.terrain.cfg.usd_path
        else:
            raise KeyError("Only implemented for terrains loaded from usd and matterport")
        env_name = os.path.splitext(terrain_file_path)[0]
        # create directory if necessary
        filedir = os.path.join(terrain_file_path, env_name)
        os.makedirs(filedir, exist_ok=True)
        return filedir
