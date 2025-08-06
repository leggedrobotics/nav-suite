Changelog
---------

0.3.9 (2025-08-04)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

- Add height scan observation terms that include following terms:
  - :meth:`nav_tasks.mdp.observations.height_scan_observations.height_scan_bounded`
  - :meth:`nav_tasks.mdp.observations.height_scan_observations.height_scan_clipped`
  - :meth:`nav_tasks.mdp.observations.height_scan_observations.height_scan_square`
  - :meth:`nav_tasks.mdp.observations.height_scan_observations.height_scan_door_recognition`
  - :meth:`nav_tasks.mdp.observations.height_scan_observations.height_scan_square_exp_occlu`
  - :meth:`nav_tasks.mdp.observations.height_scan_observations.height_scan_square_exp_occlu_with_door_recognition`
  - :class:`nav_tasks.mdp.observations.height_scan_observations.HeightScanOcculusionModifier`
  - :class:`nav_tasks.mdp.observations.height_scan_observations.HeightScanOcculusionDoorRecognitionModifier`


0.3.8 (2025-06-11)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

- Changed ``quat_rotate`` to ``quat_apply`` to faster implementation of IsaacLab


0.3.7 (2025-05-20)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

- Added new environment configs :file:`nav_tasks/configs/env_cfg_base.py` to support training and playing with
  RSL-RL.
- Added new agent configs :file:`nav_tasks/configs/agents.py` to support training and playing with RSL-RL.
- Added new :file:`scripts/nav_tasks/test_training.py` to test training of RSL-RL with CLI arguments.


0.3.6 (2025-05-07)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

- Added support for sampling paths by terrain level :attr:`nav_tasks.mdp.commands.GoalCommandCfg.terrain_level_sampling`
  following addition of :meth:`nav_suite.collectors.TrajectorySampling.sample_paths_by_terrain`.
- Added autonomous resampling to :attr:`nav_tasks.mdp.commands.GoalCommand` to avoid curriculum solution for resampling
  paths.
- Added clipping, scaling and offsetting of the commands in the :class:`nav_tasks.mdp.actions.NavigationSE2Action`

Changed
^^^^^^^

- Changed prev. attribute :attr:`nav_tasks.mdp.commands.GoalCommandCfg.infite_sampling` to
  :attr:`nav_tasks.mdp.commands.GoalCommandCfg.sampling_mode` to define  different sampling modes: ``infinite``,
  ``autonomous``, and ``bounded`` in a single argument.
- Updated :meth:`nav_tasks.mdp.curriculums.modify_terrain_level` to use termination term names for promotion and demotion logic.
- Change the :attr:`nav_tasks.mdp.commands.GoalCommandCfg.trajectory_config` into individual attributes:
  - :attr:`nav_tasks.mdp.commands.GoalCommandCfg.num_pairs`
  - :attr:`nav_tasks.mdp.commands.GoalCommandCfg.path_length_range`


0.3.5 (2025-05-06)
~~~~~~~~~~~~~~~~~~

Fixes
^^^^^^

- Fixes passing of the scene to the TerrainAnalysis in :class:`nav_tasks.mdp.commands.ConsecutiveGoalCommand`


0.3.4 (2025-04-28)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

- Added subterrain support to :class:`nav_tasks.mdp.commands.GoalCommandCfg` (i.e., analyse multiple terrains
  independently and then generate goal commands for all of them)..
- Added default base position option to :meth:`nav_tasks.mdp.events.reset_robot_position`.

Changed
^^^^^^^

- Updated mean path length computation in:meth:`nav_tasks.mdp.curriculums.modify_goal_distance_in_steps` to use true value.
- Updated base pos addition in :meth:`nav_tasks.mdp.events.reset_robot_position` to be optional (per default false)
- Updated :class:``nav_tasks.mdp.events.TerrainAnalysisRootReset` to support singleton pattern.
- Changed :meth:`nav_tasks.mdp.commands.GoalCommand.update_trajectory_config` from hardcoded default values to use the
  values from the config if None is passed.

Removed
^^^^^^

- Removed :attr:`nav_tasks.mdp.commands.GoalCommand:num_paths`, which did not reflect the true number of paths but
  just the intended number of paths.


0.3.3 (2025-04-13)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

- Added new observation term :class:`nav_tasks.mdp.observations.PosActionHistoryCfg` for including the history of the
  robot's position and action.
- Introduced a new curriculum term :meth:`nav_tasks.mdp.curriculum.change_reward_weight` to adjust a reward weight
  during training (either linearly or exponentially).

Changed
^^^^^^^

- Changed :class:`nav_tasks.mdp.events.reset_robot_position` to allow reset to default joint states and variable velocities.


0.3.2 (2025-03-31)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

- Added :class:`nav_tasks.mdp.observations.camera_observations.DINOEmbeddedRGBImageCfg` to embed RGB images using a DINO model


0.3.1 (2025-03-05)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

- Fixed logging statements to use ``omni.log`` instead of print statements.

Changed
^^^^^^^

- Updated the :meth:`nav_tasks.mdp.curriculums.modify_goal_distance_in_steps` function with a note about its correctness.


0.3.0 (2025-02-26)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

- Added back the infinite sampling option for :class:`nav_tasks.mdp.commands.GoalCommand`

Fixed
^^^^^

- Updates to new naming conventions and structure of IsaacLab 2.0.1


0.2.7 (2025-02-11)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

- Fixed the :func:`nav_tasks.terrains.random_maze_terrain:random_maze_terrain` for the case that no stairs are added


0.2.6 (2025-02-04)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

- Changed :class:`nav_tasks.mdp.commands.GoalCommandCfg`'s ``trajectory_config`` to pass single numbers instead of
  lists.
- Changed :class:`nav_tasks.mdp.commands.GoalCommand` to call the new ``sample_paths_by_terrain`` function from the
  trajectory sampler, so that it can filter commands to those in the same sub-terrain as the agent's assigned
  curriculum sub-terrain.
- Removed the ability to prevent goals being infinitely sampled in :class:`nav_tasks.mdp.commands.GoalCommand`, in order
  to simplify the implementation logic.
- Changed to ``omni.log`` instead of print statements


0.2.5 (2025-02-04)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

- Added goal and start poinmt perturbation to  :class:`nav_tasks.mdp.commands.FixedGoalCommand`
- Added :class:`nav_tasks.mdp.terrains.MeshPillarPlannerTestTerrainCfg` for a pillar terrain where the start and goal
  positions are obstacle free
- Added :class:`nav_tasks.mdp.terrains.StairsRampUpDownTerrainCfg` for a terrain where a stairs/ramp that goes up on
  the one side of the central platform and down on the other side

Changed
^^^^^^^

- Changed :class:`nav_tasks.mdp.commands.FixedGoalCommand` to fit the intervace of :class:`nav_tasks.mdp.commands.GoalCommand`
  and allow for a specific number of trajectories to be sampled and executed
- Changed color of goal marker and make line between robot position and goal option in :class:`nav_tasks.mdp.commands.BaseGoalCommand`


Fixed
^^^^^

- Fixed reset of :class:`nav_tasks.mdp.commands.GoalCommand`


0.2.4 (2024-10-18)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

- Removes the necessity that the goal generator used for the :meth:`nav_tasks.mdp.terminations.at_goal` has an ``heading_command_w`` attribute


0.2.3 (2024-10-16)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

- Removed the robot height offset from spawn positions in :class:`nav_tasks.mdp.commands.GoalCommand`.
  The robot height offset is now added by the :class:`nav_tasks.nav_collectors.terrain_analysis.TerrainAnalysis`,
  which stops terrain analysis removing paths that are traversible because of mesh intersections.

0.2.2 (2024-10-14)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

- Add an observation :class:`nav_tasks.mdp.observations.EmbeddedDepthImageTerm` that returns an embedding of a depth
  image. The embedding is generated using a pre-trained model. For visibility, the model structure is included as
  :class:`nav_tasks.mdp.observations.depth_embedder.DepthEmbedder`.


0.2.1 (2024-10-09)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

- Add :class:`nav_tasks.mdp.commands.GoalCommandBase` with config class :class:`nav_tasks.mdp.commands.GoalCommandBaseCfg`
  that provides a base class for all goal command generatos with common tools for debug visualization
- Added curriculum terms to
    - modify the terrain level :meth:`nav_tasks.mdp.curriculum.modify_terrain_level`,
    - modify the goal distance :meth:`nav_tasks.mdp.curriculum.modify_goal_distance_in_steps`,
    - modify the heading randomization :meth:`nav_tasks.mdp.curriculum.modify_heading_randomization_linearly`,
    - modify the goal conditions :meth:`nav_tasks.mdp.curriculum.modify_goal_conditions`
- Added event terms to reset robot position to position defined by command generator :meth:`nav_tasks.mdp.events.reset_robot_position`
- Added camera observation terms :meth:`nav_tasks.mdp.observations.camera_image`
- Added reward terms
    - Stability of robot when near the goal :meth:`nav_tasks.mdp.rewards.rewards.near_goal_stability`
    - heading error when near goal :meth:`nav_tasks.mdp.rewards.rewards.near_goal_angle`
    - backwards movement :meth:`nav_tasks.mdp.rewards.rewards.backwards_movement`
    - lateral movement :meth:`nav_tasks.mdp.rewards.rewards.lateral_movement`
- Added stateful rewards terms
    - discrete stepped distance to goal :class:`nav_tasks.mdp.rewards.stateful_rewards.SteppedProgressTerm`
    - average episode velocity :class:`nav_tasks.mdp.rewards.stateful_rewards.AverageEpisodeVelocityTerm`
- Add terminations terms
    - time out proportional to goal distance :meth:`nav_tasks.mdp.terminations.proportional_time_out`
    - stayed at goal for set time :class:`nav_tasks.mdp.terminations.StayedAtGoal`
- Add stereolabs Depth Camera configurations and camera downsampling
- Add random maze terrain with guaranteed solvability


0.2.0 (2024-09-18)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

- Changed to IsaacLab and renamed extension to ``nav_tasks``


0.1.0 (2024-09-01)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

- Add :class:`nav_tasks.mdp.actions.NavigationSE2Action` that receive a velocity as input argument and
  use a pre-trained locomotion policy to translate the command into joint actions.
- Add :func:`nav_tasks.mdp.terminations.at_goal` which terminates the agent once it reaches its goal.
- Add a set of terrains
  - :class:`nav_tasks.mdp.terrains.CorridorTerrainCfg` class
  - :class:`nav_tasks.mdp.terrains.MazeTerrainCfg` class
  - :class:`nav_tasks.mdp.terrains.MeshPillarTerrainCfg` class
  - :class:`nav_tasks.mdp.terrains.StairsRampTerrainCfg` class
  - :class:`nav_tasks.mdp.terrains.MeshQuadPyramidStairsCfg` class


0.0.7 (2024-09-01)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

- Added safety margin to :class:`nav_tasks.mdp.events.TerrainAnalysisRootReset` to prevent spawning inside the ground


0.0.6 (2024-08-09)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

- Introduce a reset event :class:`nav_tasks.mdp.events.TerrainAnalysisRootReset` that places the asset on the
  free points determined by the :class:`nav_collectors.terrain_analysis.TerrainAnalysis`

Changed
^^^^^^^

- The :class:`nav_collectors.terrain_analysis.TerrainAnalysis` available in all GoalCommand generators is now
  exposed as :attr:`nav_tasks.mdp.commands.FixGoalCommand.analysis`,
  :attr:`nav_tasks.mdp.commands.GoalCommand.analysis` and
  :attr:`nav_tasks.mdp.commands.ConsecutiveGoalCommand.analysis`


0.0.5 (2024-08-08)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

- Adds option to :class:`nav_tasks.mdp.commands.FixGoalCommand` to elevate the goal position by the terrain height
  at the goal position


0.0.4 (2024-08-08)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

- Adds option to :class:`nav_tasks.mdp.commands.FixGoalCommand` that either the defined terrain origins or the
  terrain centers can be used to reference the goal in the case the terrain origins are offsetted from the center.

Fixed
^^^^^

- Fixes visualization in :class:`nav_tasks.mdp.commands.GoalCommand` where the arrow size was not correct and
  generated an error if an empty env_ids list was passed


0.0.3 (2024-07-31)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

- Adds the :class:`nav_tasks.mdp.commands.ConsecutiveGoalCommand` with corresponding configclass
  :class:`nav_tasks.mdp.commands.ConsecutiveGoalCommandCfg` generating goal positions close to the spawn
  position and then always a new one once the previous one is reached up to a threshold
- Adds an arrow marker to :class:`nav_tasks.mdp.commands.GoalCommand` to visualize the direction of the goal position


0.0.2 (2024-07-28)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

- Adds the :class:`nav_tasks.mdp.commands.FixGoalCommand` with corresponding config
  class :class:`nav_tasks.mdp.commands.FixGoalCommandCfg` that generates goal positions with a fix distance to
  the terrain origin


0.0.1 (2024-07-06)
~~~~~~~~~~~~~~~~~~

Added
^^^^^
- Adds the :class:`nav_tasks.mdp.commands.GoalCommand` with corresponding config
  class :class:`nav_tasks.mdp.commands.GoalCommandCfg`
