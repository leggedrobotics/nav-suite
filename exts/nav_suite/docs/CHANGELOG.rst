Changelog
---------


0.2.5 (2025-08-13)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

- Added sampling support to other sensor data (such as RayCasters) to :class:`nav_suite.collectors.SensorDataSampling`
  for sampling sensor data from the environment
    - RayCaster implementation in :class:`nav_suite.collectors.sensors.RayCasterSensor`

Changed
^^^^^^^

- Renamed :class:`nav_suite.collectors.ViewpointSampling` to :class:`nav_suite.collectors.SensorSampling` and
  :class:`nav_suite.collectors.ViewpointSamplingCfg` to :class:`nav_suite.collectors.SensorSamplingCfg`
- Sensor data extraction is now done in individual classes for each sensor type. The logic for camera data prev.
  included in :class:`nav_suite.collectors.ViewpointSampling` is now extracted in
  :class:`nav_suite.collectors.sensors.CameraSensor`.


0.2.4 (2025-08-08)
~~~~~~~~~~~~~~~~~~

Fixes
^^^^^

- Fixed dtype mismatch in :class:`nav_suite.collectors.TrajectorySampling` by removing the casting of trajectory lengths to ``torch.int64``.
- Renamed configuration option ``attach_yaw_only`` to  ``ray_alignment`` according to IsaacLab API changes in 2.1.1


0.2.3 (2025-06-11)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

- Changed ``quat_rotate`` to ``quat_apply`` to faster implementation of IsaacLab

Fixes
^^^^^

- Fixed scale when importing single usd meshes in :class:`nav_suite.importer.NavTerrainImporter`


0.2.2 (2025-05-15)
~~~~~~~~~~~~~~~~~~

Fixes
^^^^^

- Fixed a bug in :class:`nav_suite.terrains.terrain_analysis.TerrainAnalysis` where the semantic cost mapping was not
  being applied correctly.


0.2.1 (2025-05-07)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

- Added :meth:`nav_suite.terrains.nav_terrain_importer.NavTerrainImporter.import_mesh` to add multi-mesh support for
  generated terrains to the terrain importer. For this change, the ``TerrainGenerator`` has to generate multiple meshes as
  output.

Changed
^^^^^^^

- Updated :meth:`nav_suite.collectors.TrajectorySampling.sample_paths_by_terrain`  to handle terrain level sampling.

Fixes
^^^^^^

- Fixed terrain naming in :meth:`nav_suite.terrains.nav_terrain_importer.NavTerrainImporter._compute_env_origins_curriculum`.


0.2.0 (2025-05-06)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

- Refactored :class:`nav_suite.terrains.matterport_importer` and :class:`nav_suite.terrains.unreal_importer` into
  new :class:`nav_suite.terrains.nav_terrain_importer`


0.1.1 (2025-05-06)
~~~~~~~~~~~~~~~~~~

Fixes
^^^^^^^

- Fixes raycasting in :meth:`nav_collectors.terrain_analysis.TerrainAnalysis.door_filtering` to use the correct kernel.


0.1.0 (2025-04-14)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

- Merged nav_collectors and nav_importer extensions into nav_suite extension.
- Moved semantic costs values to yaml files in the data folder.


Previous Changelog nav_collectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toggle::

  0.3.3 (2025-04-28)
  ~~~~~~~~~~~~~~~~~~

  Added
  ^^^^^

  - Added :class:`nav_collectors.terrain_analysis.TerrainAnalysisSingleton` for singleton pattern support while the
    :class:`nav_collectors.terrain_analysis.TerrainAnalysis` is changed to be a new instance every time.
  - Added multi-mesh raycasting support to :class:`nav_collectors.terrain_analysis.TerrainAnalysis`.

  Changed
  ^^^^^^^

  - Updated :class:`nav_collectors.collectors.TrajectorySampling` and :class:`nav_collectors.collectors.ViewpointSampling`
    to support singleton terrain analysis.

  0.3.2 (2025-04-13)
  ~~~~~~~~~~~~~~~~~~

  Added
  ^^^^^

  - Visualization of graph points in :class:`nav_collectors.terrain_analysis.TerrainAnalysis` with :attr:`viz_graph` option.

  Changed
  ^^^^^^^

  - Updated door filtering in :class:`nav_collectors.terrain_analysis.TerrainAnalysis` to be applied for both height map and graph point filtering.


  0.3.1 (2025-04-03)
  ~~~~~~~~~~~~~~~~~~

  Changed
  ^^^^^^^

  - Replace :meth:`nav_importer.utils.prims.get_all_meshes` with :meth:`sim_utils.get_all_matching_child_prims`


  0.3.0 (2025-04-01)
  ~~~~~~~~~~~~~~~~~~

  Changed
  ^^^^^^^

  - Remove GUI of the extension


  0.2.6 (2025-03-28)
  ~~~~~~~~~~~~~~~~~~

  Changed
  ^^^^^^^

  - Change :class:`terrain_analysis.TerrainAnalysis` to be an instance to avoid recalculate them multiple times.


  0.2.5 (2025-03-27)
  ~~~~~~~~~~~~~~~~~~

  Added
  ^^^^^

  - Added option to splice the pc in :class:`nav_collectors.collectors.ViewpointSampling`.


  0.2.4 (2025-03-22)
  ~~~~~~~~~~~~~~~~~~

  Added
  ^^^^^

  - Introduced :attr:`nav_collectors.terrain_analysis.TerrainAnalysisCfg.terrain_bounding_box` to define a bounding box for terrain analysis.


  0.2.3 (2025-03-21)
  ~~~~~~~~~~~~~~~~~~

  Fixed
  ^^^^^

  - Fixed raycasting distance in :class:`terrain_analysis.TerrainAnalysis` for planes.


  0.2.2 (2025-03-20)
  ~~~~~~~~~~~~~~~~~~

  Fixed
  ^^^^^

  - Fixed raycasting distance in :class:`terrain_analysis.TerrainAnalysis` to reach the lowest points of the terrain.


  0.2.1 (2025-03-05)
  ~~~~~~~~~~~~~~~~~~

  Added
  ^^^^^

  - Added support for sliced sampling and point cloud generation in :class:`nav_collectors.collectors.ViewpointSampling`
  - Added :class:`nav_collectors.configs.mountain_class_cost:MountainSemanticCostMapping` for Mountain semantic cost mapping.

  Fixed
  ^^^^^

  - Corrected the file paths in the README for standalone scripts.
  - Fixed semantic filtering and make height different filtering optional in :class:`nav_collectors.terrain_analysis.TerrainAnalysis`.


  0.2.0 (2025-02-26)
  ~~~~~~~~~~~~~~~~~~

  Fixed
  ^^^^^

  - Updates to new naming conventions and structure of IsaacLab 2.0.1
  - Fixed examples :meth:`nav_collectors.collectors.TrajectorySampling:sample_paths` to
    account for the changed type (no list anymore) for the ``num_path``, ``min_path_length`` and ``max_path_length`` parameters.
  - Fixed extension :class:`nav_collectors.scripts.NavCollectorExtension` to account
    for the changed type (no list anymore) for the ``num_path``, ``min_path_length`` and ``max_path_length`` parameters.
  - Fixed :class:`nav_collectors.terrain_analysis.TerrainAnalysis` for changes in the raycaster in IsaacLab 2.0.1


  0.1.6 (2025-02-04)
  ~~~~~~~~~~~~~~~~~~

  Added
  ^^^^^

  - Added sample_paths_by_terrain function to :class:`nav_collectors.collectors.TrajectorySampling` that separates paths
    by the subterrain that they belong to.
  - Added indexing samples by subterrain to :class:`terrain_analysis.TerrainAnalysis`, and visualizing graph nodes by
    subterrain.

  Changed
  ^^^^^^^

  - Changed to ``omni.log`` instead of print statements


  0.1.5 (2025-02-04)
  ~~~~~~~~~~~~~~~~~~

  Added
  ^^^^^

  - Added :meth:`nav_collectors.terrain_analysis.TerrainAnalysis.shortest_path_lengths` to get the shortest length between
    points given the build traversability graph


  0.1.4 (2024-10-17)
  ~~~~~~~~~~~~~~~~~~

  Fixed
  ^^^^^

  - Fixed a bug in :class:`nav_collectors.collectors.TrajectorySampling` that was causing incorrect sampling of paths of
    desired length.


  0.1.3 (2024-10-16)
  ~~~~~~~~~~~~~~~~~~

  Fixed
  ^^^^^

  - Set the height of the sampled points in the :class:`nav_collectors.terrain_analysis.TerrainAnalysisCfg` to the robot
    height to avoid removing traversible paths because they intersect with rough terrain at ground height.


  0.1.2 (2024-10-09)
  ~~~~~~~~~~~~~~~~~~

  Fixed
  ^^^^^

  - Set the height of the sampled points in the :class:`nav_collectors.terrain_analysis.TerrainAnalysisCfg` to the height
    of the terrain at the sampled point


  0.1.1 (2024-10-07)
  ~~~~~~~~~~~~~~~~~~

  Changed
  ^^^^^^^

  - Removed ``InteractiveSceneCfg`` from :class:`nav_collectors.collectors.TrajectorySamplingCfg` and
    :class:`nav_collectors.collectors.ViewpointSamplingCfg`. Instead, the scene now has to be passed through
    the collector classes


  0.1.0 (2024-09-18)
  ~~~~~~~~~~~~~~~~~~

  Changed
  ^^^^^^^

  - Changed to IsaacLab and renamed extension to ``nav_collectors``


  0.0.10 (2024-09-18)
  ~~~~~~~~~~~~~~~~~~

  Added
  ^^^^^

  - Added :attr:`nav_collectors.terrain_analysis.TerrainAnalysisCfg.max_terrain_size` to limit the size of the terrain
    in the terrain analysis module and avoid memory issues


  0.0.9 (2024-09-01)
  ~~~~~~~~~~~~~~~~~~

  Fixed
  ^^^^^

  - Fixes wrong threshold value in :attr:`nav_collectors.terrain_analysis.TerrainAnalysis.construct_height_map` to do the
    door filtering correctly


  0.0.8 (2024-08-09)
  ~~~~~~~~~~~~~~~~~~

  Added
  ^^^^^

  - Add functionality :attr:`nav_collectors.terrain_analysis.TerrainAnalysis.get_height` to get the height of
    points in the terrain from the generated height height-map


  0.0.7 (2024-08-01)
  ~~~~~~~~~~~~~~~~~~

  Fixed
  ^^^^^

  - Fixed height-map computation in :class:`nav_collectors.terrain_analysis.TerrainAnalysis` when door filtering is activated
    and objects such as stairs are present, which were identified as doors by requiring a minimum door height.
  - Fixed a device error in the :class:`nav_collectors.collectors.TrajectorySampling` due to samples in
    :class:`nav_collectors.terrain_analysis.TerrainAnalysis` being now on GPU when the whole process is run on GPU.


  0.0.6 (2024-07-31)
  ~~~~~~~~~~~~~~~~~~

  Added
  ^^^^^

  - Expose :attr:`nav_collectors.terrain_analysis.TerrainAnalysis.isolated_points_ids` to get the ids of isolated
    points in the terrain analysis which are not automatically removed from :attr:`nav_collectors.terrain_analysis.TerrainAnalysis.points`

  Fixed
  ^^^^^

  - Fixed unnecessary configs parameters in configclass :class:`nav_collectors.collectors.ExplorationCfg`


  0.0.5 (2024-07-29)
  ~~~~~~~~~~~~~~~~~~

  Changed
  ^^^^^^^

  - Change :class:`nav_collectors.terrain_analysis.TerrainAnalysis` to execute all raycasting operations on the device
    of the scene

  Fixed
  ^^^^^

  - Fixed issue with :class:`nav_collectors.terrain_analysis.TerrainAnalysis` to new version of the multi-mesh raycaster


  0.0.4 (2024-07-28)
  ~~~~~~~~~~~~~~~~~~

  Added
  ^^^^^

  - Expose :attr:`nav_collectors.terrain_analysis.TerrainAnalysis.height_grid` and
    :attr:`nav_collectors.terrain_analysis.TerrainAnalysis.mesh_dimensions` within the TerrainAnalysis Module
  - Add :attr:`nav_collectors.collectors.TrajectorySamplingCfg.enable_saved_paths_loading` to enabled/ disable
    loading of generated path in the trajectory sampling
  - Expose :attr:`nav_collectors.terrain_analysis.TerrainAnalysisCfg.viz_height_map` to enable/ disable the
    visualization of the generated height grid

  Changed
  ^^^^^^^

  - Change the logic of :func:`nav_collectors.terrain_analysis.TerrainAnalysis._edge_filter_height_diff`
    to not assume a concrete mesh but instead also support a hallow one

  Fixed
  ^^^^^

  - Fixed support for multi-mesh raycaster


  0.0.3 (2024-07-08)
  ~~~~~~~~~~~~~~~~~~

  Fixed
  ^^^^^

  - Fixes infinite loop in :class:`nav_collectors.collectors.ViewpointSampling` when not all samples are
    generated in the first iteration through the traversability graph.


  0.0.2 (2024-05-02)
  ~~~~~~~~~~~~~~~~~~

  Added
  ^^^^^
  - Add filtering of empty nodes from the graph in :class:`nav_collectors.terrain_analysis.TerrainAnalysis`

  Fixed
  ^^^^^
  - Remove unnecessary config params from :class:`nav_collectors.collectors.TrajectorySamplingCfg`

  Changed
  ^^^^^^^
  - Restructured :class:`nav_collectors.collectors.TerrainAnalysis` to an own directory
    :class:`nav_collectors.terrain_analysis.TerrainAnalysis` and made corresponding changes to the imports.


  0.0.1 (2024-05-02)
  ~~~~~~~~~~~~~~~~~~

  Added
  ^^^^^
  - Trajectories and Viewpoint sampling from any environment with terrain analysis module.


Previous Changelog nav_importer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toggle::

  0.3.4 (2025-04-28)
  ~~~~~~~~~~~~~~~~~~

  Added
  ^^^^^

  - Added multi-USD support in :class:`nav_importer.importer.UnRealImporter`.


  0.3.3 (2025-04-28)
  ~~~~~~~~~~~~~~~~~~

  Added
  ^^^^^

  - Added option for regular spawning of robots in :class:`nav_importer.importer.UnRealImporter`.
  - Added option to define grid-like environment origins for usd assets in :class:`nav_importer.importer.UnRealImporter`.


  0.3.2 (2025-04-13)
  ~~~~~~~~~~~~~~~~~~

  Added
  ^^^^^

  - Added scale option to :class:`nav_importer.importer.UnRealImporterCfg` for scaling the imported assets.


  0.3.1 (2025-04-03)
  ~~~~~~~~~~~~~~~~~~

  Changed
  ^^^^^^^

  - Removed storage of warp meshes in :class:`nav_importer.importer.importer`
  - Moved :file:`exts/nav_importer/nav_importer/scripts/utils/convert_obj.py` to general scripts folder :file:`scripts/tools/convert_obj.py`
  - Rename general :file:`importer` to :file:`matterport_importer`
  - Removed now obsolete :meth:`nav_importer.utils.prims.get_all_meshes`


  0.3.0 (2025-04-01)
  ~~~~~~~~~~~~~~~~~~

  Changed
  ^^^^^^^

  - Removed GUI of the extension
  - Replace logging from from ``carb.log`` with ``omni.log``


  0.2.2 (2025-03-26)
  ~~~~~~~~~~~~~~~~~~

  Fixed
  ^^^^^

  - Fixed missing cameras enabled in the carla import example.


  0.2.1 (2025-03-05)
  ~~~~~~~~~~~~~~~~~~

  Fixed
  ^^^^^

  - Fixed semantic mapping in :class:`nav_importer.importer.UnRealImporter` to handle missing semantics.


  0.2.0 (2025-02-26)
  ~~~~~~~~~~~~~~~~~~

  Fixed
  ^^^^^

  - Updates to new naming conventions and structure of IsaacLab 2.0.1
  - Fixed :class:`nav_importer.sensors.MatterportRayCaster` and :class:`nav_importer.sensors.MatterportRayCasterCamera`
    for changes in the raycaster in IsaacLab 2.0.1

  Changed
  ^^^^^^^

  - Remove classvar ``face_id_category_mapping`` in :class:`nav_importer.sensors.MatterportRayCaster`
    and changed to class attribute


  0.1.2 (2025-02-05)
  ~~~~~~~~~~~~~~~~~~

  Changed
  ^^^^^^^

  - Changed to ``omni.log`` instead of print statements


  0.1.1 (2024-10-07)
  ~~~~~~~~~~~~~~~~~~

  Added
  ^^^^^

  - Add ``warehouse.usd`` asset to the repo, dadd other environments as download links to README

  Changed
  ^^^^^^^

  - Rename ``DATA_DIR`` to ``NAVSUITE_IMPORTER_DATA_DIR``


  0.1.0 (2024-09-18)
  ~~~~~~~~~~~~~~~~~~

  Changed
  ^^^^^^^

  - Changed to IsaacLab and renamed extension to ``nav_importer``


  0.0.2 (2024-07-06)
  ~~~~~~~~~~~~~~~~~~

  Fixed
  ^^^^^

  - Fixed the obj importer :class:`nav_importer.utils.ObjConverter`


  0.0.1 (2024-05-02)
  ~~~~~~~~~~~~~~~~~~

  Added
  ^^^^^
  - Added first version of the extension
