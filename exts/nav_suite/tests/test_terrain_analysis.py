# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import os
import torch

import pytest
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sim import build_simulation_context
from isaaclab.utils import configclass
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG

from nav_suite import NAVSUITE_TEST_ASSETS_DIR
from nav_suite.terrain_analysis import TerrainAnalysis, TerrainAnalysisCfg
from nav_suite.terrains import NavTerrainImporterCfg


@configclass
class BasicSceneCfg(InteractiveSceneCfg):
    """Configuration for a basic test scene with terrain."""

    terrain = NavTerrainImporterCfg(
        prim_path="/World/Ground",
        terrain_type="usd",
        usd_path=os.path.join(NAVSUITE_TEST_ASSETS_DIR, "terrains", "ground_plane.usda"),
        num_envs=1,
        env_spacing=2.0,
        add_colliders=False,
    )

    robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # raycaster for terrain analysis
    raycaster = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/lidar_cage",
        update_period=0,
        debug_vis=False,
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.1,
            size=(1.0, 1.0),
        ),
        mesh_prim_paths=["/World/Ground"],
        attach_yaw_only=True,
    )


@pytest.fixture(params=["cuda", "cpu"])
def device(request):
    """Fixture providing both cuda and cpu devices for testing."""
    return request.param


@pytest.fixture
def simulation_context(device):
    """Fixture for managing simulation context across tests."""
    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        yield sim


@pytest.fixture
def scene(simulation_context):
    """Fixture providing an InteractiveScene with terrain."""
    scene_cfg = BasicSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    simulation_context.reset()
    return scene


@pytest.fixture
def terrain_analysis_test(scene):
    """Fixture providing terrain analysis with standard 3x3 height grid."""

    # Create terrain analysis configuration (following matterport_viewpoint_sampling.py pattern)
    terrain_analysis_cfg = TerrainAnalysisCfg(
        grid_resolution=0.1,
        sample_points=10,  # Use small number for faster tests
        viz_graph=False,  # Disable visualization for tests
        viz_height_map=False,
        semantic_cost_mapping=None,
    )

    terrain_analysis = TerrainAnalysis(terrain_analysis_cfg, scene=scene)

    # Standard 3x3 height grid created on the scene's device
    terrain_analysis._height_grid = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        device=scene.device,
    )

    # Standard mesh dimensions: [x_max, y_max, x_min, y_min]
    terrain_analysis._mesh_dimensions = [0.3, 0.3, 0.0, 0.0]

    return terrain_analysis


@pytest.fixture
def terrain_analysis_real(scene):
    """Fixture providing terrain analysis that will construct its own height map from the scene."""

    # Create terrain analysis configuration with raycaster sensor for mesh-based raycasting
    terrain_analysis_cfg = TerrainAnalysisCfg(
        grid_resolution=0.1,
        sample_points=10,  # Use small number for faster tests
        viz_graph=False,  # Disable visualization for tests
        viz_height_map=False,
        semantic_cost_mapping=None,
        raycaster_sensor="raycaster",  # Use the raycaster sensor defined in scene
    )

    terrain_analysis = TerrainAnalysis(terrain_analysis_cfg, scene=scene)

    return terrain_analysis


def test_get_height_single_position(terrain_analysis_test):
    """Test get_height with a single position."""
    # Test a position that should map to grid index [0, 0]
    positions = torch.tensor([
        [0.05, 0.05],  # Grid index [0, 0] -> height 1.0
    ])
    heights = terrain_analysis_test.get_height(positions)

    expected_height = torch.tensor([1.0])
    assert torch.equal(heights.cpu(), expected_height), f"Expected {expected_height}, got {heights.cpu()}"


def test_get_height_multiple_positions(terrain_analysis_test):
    """Test get_height with multiple positions."""
    # Test multiple positions
    positions = torch.tensor([
        [0.05, 0.05],  # Grid index [0, 0] -> height 1.0
        [0.15, 0.25],  # Grid index [1, 2] -> height 6.0
        [0.25, 0.15],  # Grid index [2, 1] -> height 8.0
    ])
    heights = terrain_analysis_test.get_height(positions)

    expected_heights = torch.tensor([1.0, 6.0, 8.0])  # Heights at [0,0], [1,2] and [2,1]
    assert torch.equal(heights.cpu(), expected_heights), f"Expected {expected_heights}, got {heights.cpu()}"


def test_get_height_boundary_clamping(terrain_analysis_test):
    """Test that positions outside grid bounds are clamped correctly."""
    # Test positions outside bounds
    positions = torch.tensor([
        [-0.1, -0.1],  # Outside bounds, should clamp to [0, 0]
        [0.5, 0.5],  # Outside bounds, should clamp to [2, 2]
    ])
    heights = terrain_analysis_test.get_height(positions)

    expected_heights = torch.tensor([1.0, 9.0])  # Heights at [0,0] and [2,2]
    assert torch.equal(heights.cpu(), expected_heights), f"Expected {expected_heights}, got {heights.cpu()}"


def test_get_height_exact_grid_boundaries(terrain_analysis_test):
    """Test positions that fall exactly on grid boundaries."""
    # Test positions on exact boundaries
    positions = torch.tensor([
        [0.0, 0.0],  # Exact corner -> [0, 0]
        [0.2, 0.2],  # Grid boundary -> [2, 2]
        [0.1, 0.0],  # Edge position -> [1, 0]
    ])
    heights = terrain_analysis_test.get_height(positions)

    expected_heights = torch.tensor([1.0, 9.0, 4.0])  # Heights at [0,0], [2,2] and [1,0]
    assert torch.equal(heights.cpu(), expected_heights), f"Expected {expected_heights}, got {heights.cpu()}"


def test_get_height_empty_input(terrain_analysis_test):
    """Test get_height with empty input tensor."""
    # Test empty input
    positions = torch.empty((0, 2))
    heights = terrain_analysis_test.get_height(positions)

    assert heights.shape == (0,), f"Expected empty tensor, got shape {heights.shape}"


def test_analyse_basic_functionality(terrain_analysis_real):
    """Test that analyse() completes without errors and sets expected attributes."""

    # Run analysis - this will automatically setup the raycaster and construct height map
    terrain_analysis_real.analyse()

    # Verify analysis completed and required attributes are set
    assert terrain_analysis_real.complete, "TerrainAnalysis should be complete after analyse()"
    assert hasattr(terrain_analysis_real, "graph"), "graph attribute should be set after analyse()"
    assert hasattr(terrain_analysis_real, "samples"), "samples attribute should be set after analyse()"
    assert hasattr(terrain_analysis_real, "points"), "points attribute should be set after analyse()"
    assert terrain_analysis_real.graph is not None, "graph should not be None after analyse()"
    assert terrain_analysis_real.samples is not None, "samples should not be None after analyse()"
    assert terrain_analysis_real.points is not None, "points should not be None after analyse()"
