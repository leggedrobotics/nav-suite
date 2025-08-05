# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import math
import os
import pickle
import random
import torch

import omni.log
from isaaclab.scene import InteractiveScene

from .trajectory_sampling_cfg import TrajectorySamplingCfg


class TrajectorySampling:
    def __init__(self, cfg: TrajectorySamplingCfg, scene: InteractiveScene):
        # save cfg and env
        self.cfg = cfg
        self.scene = scene

    def sample_paths(self, num_paths, min_path_length, max_path_length, seed: int = 1) -> torch.Tensor:
        """
        Sample Trajectories over the entire terrain.

        Args:
            num_paths: Number of paths to sample per terrain.
            min_path_length: Minimum path length.
            max_path_length: Maximum path length.
            seed: Random seed.
            filter_target_within_terrain: If True, the target point will be within the same terrain as the start point.

        Returns:
            A tensor of shape [num_paths, 7] containing the sampled paths.
        """

        # load paths if they exist
        if self.cfg.enable_saved_paths_loading:
            filename = self._get_save_path_trajectories(seed, num_paths, min_path_length, max_path_length)
            if os.path.isfile(filename):
                with open(filename, "rb") as f:
                    saved_paths = pickle.load(f)
                # add loaded path dict to data dict
                omni.log.info(
                    f"Loaded {num_paths} with [{min_path_length},{max_path_length}] length generated with seed {seed}."
                )
                return saved_paths

        # analyse terrain if not done yet
        if not hasattr(self, "terrain_analyser"):
            # check if singleton is used and available
            if (
                hasattr(self.cfg.terrain_analysis.class_type, "instance")
                and self.cfg.terrain_analysis.class_type.instance() is not None
            ):
                self.terrain_analyser = self.cfg.terrain_analysis.class_type.instance()
            else:
                self.terrain_analyser = self.cfg.terrain_analysis.class_type(
                    self.cfg.terrain_analysis, scene=self.scene
                )
        if not self.terrain_analyser.complete:
            self.terrain_analyser.analyse()

        # map distance to idx pairs
        random.seed(seed)

        # get index of samples within length
        within_length = (self.terrain_analyser.samples[:, 2] > min_path_length) & (
            self.terrain_analyser.samples[:, 2] <= max_path_length
        )

        # apply the within_length filter first
        filtered_samples = self.terrain_analyser.samples[within_length]

        # randomly permute the filtered samples
        rand_idx = torch.randperm(filtered_samples.shape[0], device=self.terrain_analyser.device)

        # select the samples
        selected_samples = filtered_samples[rand_idx][:num_paths]

        # filter edge cases
        if selected_samples.shape[0] == 0:
            raise ValueError(f"No paths found with length [{min_path_length},{max_path_length}]")
        if selected_samples.shape[0] < num_paths:
            omni.log.warn(
                f"Only {selected_samples.shape[0]} paths found with length [{min_path_length},{max_path_length}]"
                f" instead of {num_paths}"
            )

        # get start, goal and path length
        data = torch.zeros((selected_samples.shape[0], 7))
        data[:, :3] = self.terrain_analyser.points[selected_samples[:, 0].type(torch.int64)]
        data[:, 3:6] = self.terrain_analyser.points[selected_samples[:, 1].type(torch.int64)]
        data[:, 6] = selected_samples[:, 2]

        # save data as pickle
        if self.cfg.enable_saved_paths_loading:
            filename = self._get_save_path_trajectories(seed, num_paths, min_path_length, max_path_length)
            with open(filename, "wb") as f:
                pickle.dump(data, f)

        # define start points
        return data

    def sample_paths_by_terrain(
        self,
        num_paths,
        min_path_length,
        max_path_length,
        seed: int = 1,
        filter_target_within_terrain: bool = True,
        terrain_level_sampling: bool = False,
    ) -> torch.Tensor:
        """
        Sample Trajectories by subterrains.

        Args:
            num_paths: Number of paths to sample per terrain.
            min_path_length: Minimum path length.
            max_path_length: Maximum path length.
            seed: Random seed.
            filter_target_within_terrain: If True, the target point will be within the same terrain as the start point.
            terrain_level_sampling: If True, num_paths paths will be sampled for each terrain level instead of num_paths paths for the entire terrain.

        Returns:
            A tensor of shape [row, col, num_paths, 7] containing the sampled paths.
        """

        # load paths if they exist
        if self.cfg.enable_saved_paths_loading:
            if self.scene.terrain.cfg.terrain_type == "generator":
                omni.log.warn(
                    "You are loading pre-computed paths for a terrain that is being generated live. "
                    "Make sure the same random seed has been set."
                )
            filename = self._get_save_path_trajectories(seed, num_paths, min_path_length, max_path_length)
            if os.path.isfile(filename):
                with open(filename, "rb") as f:
                    saved_paths = pickle.load(f)
                    omni.log.info(
                        f"Loaded {num_paths} with [{min_path_length},{max_path_length}] length generated with"
                        f" seed {seed}."
                    )
                    return saved_paths

        assert self.scene.terrain.terrain_origins is not None, (
            "Sampling paths by terrains needs terrain origins. If you are using a USD, make sure you have a "
            "version of IsaacLab-Internal that assigns terrain_origins for USDs in terrain_importer."
        )

        # analyse terrain if not done yet
        if not hasattr(self, "terrain_analyser"):
            # check if singleton is used and available
            if (
                hasattr(self.cfg.terrain_analysis.class_type, "instance")
                and self.cfg.terrain_analysis.class_type.instance() is not None
            ):
                self.terrain_analyser = self.cfg.terrain_analysis.class_type.instance()
            else:
                self.terrain_analyser = self.cfg.terrain_analysis.class_type(
                    self.cfg.terrain_analysis, scene=self.scene
                )
        if not self.terrain_analyser.complete:
            self.terrain_analyser.analyse()

        # map distance to idx pairs
        random.seed(seed)

        # get index of samples within length
        within_length = (self.terrain_analyser.samples[:, 2] > min_path_length) & (
            self.terrain_analyser.samples[:, 2] <= max_path_length
        )

        # apply the within_length filter
        filtered_samples = self.terrain_analyser.samples[within_length]
        # returns a tensor [row_idx, col_idx]
        filtered_samples_subterrains_origins = self.terrain_analyser.sample_terrain_origins[within_length]

        # filter if start and end point within the same terrain
        if filter_target_within_terrain:
            filtered_samples_subterrains_targets = self.terrain_analyser.sample_terrain_targets[within_length]

            # filter target points within the same terrain as the start points
            same_terrain = torch.all(
                filtered_samples_subterrains_origins == filtered_samples_subterrains_targets, dim=-1
            )
            filtered_samples = filtered_samples[same_terrain]
            filtered_samples_subterrains_origins = filtered_samples_subterrains_origins[same_terrain]

        # randomly permute the filtered samples
        rand_idx = torch.randperm(filtered_samples.shape[0], device=self.terrain_analyser.device)

        # select the samples
        randomized_samples = filtered_samples[rand_idx]
        randomized_samples_subterrains_origins = filtered_samples_subterrains_origins[rand_idx]

        # filter edge cases
        assert (
            randomized_samples.shape[0] > 0
        ), f"[ERROR] No paths found with length [{min_path_length},{max_path_length}]"
        if randomized_samples.shape[0] < num_paths:
            omni.log.warn(
                f"Only {randomized_samples.shape[0]} paths found with length"
                f" [{min_path_length},{max_path_length}] instead of {num_paths}"
            )

        # Make a samples by terrain tensor for easy indexing in goal_command. We need the equivalent number of paths
        # per terrain, so we take the min number of paths in the terrains and trim each terrain's paths to that number.
        num_rows, num_cols = self.scene.terrain.terrain_origins.shape[:2]
        if terrain_level_sampling:
            terrain_levels, samples_per_terrain_level = torch.unique(
                randomized_samples_subterrains_origins[:, 0], return_counts=True
            )
            assert len(terrain_levels) == num_rows, "Not all terrain levels have paths."
            if samples_per_terrain_level.min().item() < num_paths:
                omni.log.warn(
                    f"Only {samples_per_terrain_level.min().item()} paths found for terrain level "
                    f"{terrain_levels[samples_per_terrain_level.min().item()]} instead of {num_paths}"
                )
                samples_per_terrain_level = samples_per_terrain_level.min().item()
            else:
                samples_per_terrain_level = num_paths

            samples_by_terrain = torch.zeros(num_rows, samples_per_terrain_level, 7)
            for row in range(num_rows):
                mask = randomized_samples_subterrains_origins[:, 0] == row
                clipped = randomized_samples[mask][:samples_per_terrain_level]
                samples_by_terrain[row, :, :3] = self.terrain_analyser.points[clipped[:, 0].type(torch.int64)]
                samples_by_terrain[row, :, 3:6] = self.terrain_analyser.points[clipped[:, 1].type(torch.int64)]
                samples_by_terrain[row, :, 6] = clipped[:, 2].type(torch.int64)
        else:
            subterrain_idx_origins = (
                randomized_samples_subterrains_origins[:, 0] * num_cols + randomized_samples_subterrains_origins[:, 1]
            )
            env_samples, samples_per_terrain = torch.unique(subterrain_idx_origins, return_counts=True)
            assert len(env_samples) == num_rows * num_cols, "Not all terrains have paths."
            if samples_per_terrain.min().item() < num_paths / (num_rows * num_cols):
                omni.log.warn(
                    f"Only {samples_per_terrain.min().item()} paths found per terrain instead of"
                    f" {num_paths / (num_rows * num_cols)}"
                )
                samples_per_terrain = samples_per_terrain.min().item()
            else:
                samples_per_terrain = math.floor(num_paths / (num_rows * num_cols))

            # Make the return tensor, of shape [num_terrain_levels, num_terrain_types, num_paths, 7]
            samples_by_terrain = torch.zeros(num_rows, num_cols, samples_per_terrain, 7)
            for row, col in itertools.product(range(num_rows), range(num_cols)):
                mask = subterrain_idx_origins == int(row * num_cols + col)
                clipped = randomized_samples[mask][:samples_per_terrain]
                samples_by_terrain[row, col, :, :3] = self.terrain_analyser.points[clipped[:, 0].type(torch.int64)]
                samples_by_terrain[row, col, :, 3:6] = self.terrain_analyser.points[clipped[:, 1].type(torch.int64)]
                samples_by_terrain[row, col, :, 6] = clipped[:, 2].type(torch.int64)

        # save curr_data as pickle
        if self.cfg.enable_saved_paths_loading:
            filename = self._get_save_path_trajectories(seed, num_paths, min_path_length, max_path_length)
            with open(filename, "wb") as f:
                pickle.dump(samples_by_terrain, f)

        # define start points
        return samples_by_terrain

    ###
    # Save paths
    ###

    def _get_save_path_trajectories(self, seed, num_path: int, min_len: float, max_len: float) -> str:
        filename = f"paths_seed{seed}_paths{num_path}_min{min_len}_max{max_len}.pkl"
        # get env name
        if isinstance(self.scene.terrain.cfg.usd_path, str):
            terrain_file_path = self.scene.terrain.cfg.usd_path
        else:
            terrain_file_path = None
            omni.log.info("Terrain is generated, trajectories will be saved under 'logs' directory.")

        if terrain_file_path:
            env_name = os.path.splitext(terrain_file_path)[0]
            # create directory if necessary
            filedir = os.path.join(terrain_file_path, env_name)
            os.makedirs(filedir, exist_ok=True)
            return os.path.join(filedir, filename)
        else:
            os.makedirs("logs", exist_ok=True)
            log_path = os.path.join("logs", filename)
            return os.path.abspath(log_path)
