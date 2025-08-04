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

from isaaclab.sim import build_simulation_context

from nav_suite import NAVSUITE_TEST_ASSETS_DIR
from nav_suite.terrains import NavTerrainImporter, NavTerrainImporterCfg
from nav_suite.terrain_analysis import TerrainAnalysis, TerrainAnalysisCfg


# Test Fixtures
@pytest.fixture(params=["cuda:0", "cpu"])
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
def basic_terrain_analysis_cfg():
    """Fixture for basic terrain analysis configuration."""
    return TerrainAnalysisCfg(
        grid_resolution=0.1,
        sample_points=10,  # Use small number for faster tests
        viz_graph=False,   # Disable visualization for tests
        viz_height_map=False,
        semantic_cost_mapping=None,
    )


@pytest.fixture
def terrain_importer(simulation_context):
    """Fixture that creates a basic terrain importer for testing."""
    terrain_cfg = NavTerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=os.path.join(NAVSUITE_TEST_ASSETS_DIR, "terrains", "ground_plane.usda"),
        num_envs=1,
        env_spacing=2.0,
        add_colliders=False,
    )
    return NavTerrainImporter(terrain_cfg)


@pytest.fixture
def scene(device, terrain_importer):
    """Fixture that creates a scene-like object for testing."""
    # Set terrain_origins to None to avoid warnings
    terrain_importer.terrain_origins = None
    # Add device attribute to terrain_importer to act as scene
    terrain_importer.device = device
    # Use terrain_importer as both scene and terrain
    terrain_importer.terrain = terrain_importer
    return terrain_importer


def test_get_height_single_position(scene, basic_terrain_analysis_cfg):
    """Test get_height with a single position."""
    # Create terrain analysis instance
    terrain_analysis = TerrainAnalysis(basic_terrain_analysis_cfg, scene)

    # Set up a known height grid for testing
    terrain_analysis._height_grid = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])

    # Set mesh dimensions: [x_max, y_max, x_min, y_min]
    terrain_analysis._mesh_dimensions = [0.3, 0.3, 0.0, 0.0]

    # Test a position that should map to grid index [0, 0]
    positions = torch.tensor([[0.05, 0.05]])
    heights = terrain_analysis.get_height(positions)

    expected_height = torch.tensor([1.0])
    assert torch.allclose(heights, expected_height), f"Expected {expected_height}, got {heights}"


def test_get_height_multiple_positions(scene, basic_terrain_analysis_cfg):
    """Test get_height with multiple positions."""
    # Create terrain analysis instance
    terrain_analysis = TerrainAnalysis(basic_terrain_analysis_cfg, scene)

    # Set up a known height grid for testing
    terrain_analysis._height_grid = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])

    # Set mesh dimensions: [x_max, y_max, x_min, y_min]
    terrain_analysis._mesh_dimensions = [0.3, 0.3, 0.0, 0.0]

    # Test multiple positions
    positions = torch.tensor([
        [0.05, 0.05],  # Grid index [0, 0] -> height 1.0
        [0.15, 0.25],  # Grid index [1, 2] -> height 6.0
        [0.25, 0.15]   # Grid index [2, 1] -> height 8.0
    ])
    heights = terrain_analysis.get_height(positions)

    expected_heights = torch.tensor([1.0, 6.0, 8.0])
    assert torch.allclose(heights, expected_heights), f"Expected {expected_heights}, got {heights}"


def test_get_height_boundary_clamping(scene, basic_terrain_analysis_cfg):
    """Test that positions outside grid bounds are clamped correctly."""
    # Create terrain analysis instance
    terrain_analysis = TerrainAnalysis(basic_terrain_analysis_cfg, scene)

    # Set up a known height grid for testing
    terrain_analysis._height_grid = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])

    # Set mesh dimensions: [x_max, y_max, x_min, y_min]
    terrain_analysis._mesh_dimensions = [0.3, 0.3, 0.0, 0.0]

    # Test positions outside bounds
    positions = torch.tensor([
        [-0.1, -0.1],  # Outside bounds, should clamp to [0, 0]
        [0.5, 0.5]     # Outside bounds, should clamp to [2, 2]
    ])
    heights = terrain_analysis.get_height(positions)

    expected_heights = torch.tensor([1.0, 9.0])  # Heights at [0,0] and [2,2]
    assert torch.allclose(heights, expected_heights), f"Expected {expected_heights}, got {heights}"


def test_get_height_exact_grid_boundaries(scene, basic_terrain_analysis_cfg):
    """Test positions that fall exactly on grid boundaries."""
    # Create terrain analysis instance
    terrain_analysis = TerrainAnalysis(basic_terrain_analysis_cfg, scene)

    # Set up a known height grid for testing
    terrain_analysis._height_grid = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])

    # Set mesh dimensions: [x_max, y_max, x_min, y_min]
    terrain_analysis._mesh_dimensions = [0.3, 0.3, 0.0, 0.0]

    # Test positions on exact boundaries
    positions = torch.tensor([
        [0.0, 0.0],    # Exact corner -> [0, 0]
        [0.2, 0.2],    # Grid boundary -> [2, 2]
        [0.1, 0.0]     # Edge position -> [1, 0]
    ])
    heights = terrain_analysis.get_height(positions)

    expected_heights = torch.tensor([1.0, 9.0, 4.0])
    assert torch.allclose(heights, expected_heights), f"Expected {expected_heights}, got {heights}"


def test_get_height_empty_input(scene, basic_terrain_analysis_cfg):
    """Test get_height with empty input tensor."""
    # Create terrain analysis instance
    terrain_analysis = TerrainAnalysis(basic_terrain_analysis_cfg, scene)

    # Set up a known height grid for testing
    terrain_analysis._height_grid = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])

    # Set mesh dimensions: [x_max, y_max, x_min, y_min]
    terrain_analysis._mesh_dimensions = [0.3, 0.3, 0.0, 0.0]

    # Test empty input
    positions = torch.empty((0, 2))
    heights = terrain_analysis.get_height(positions)

    assert heights.shape == (0,), f"Expected empty tensor, got shape {heights.shape}"


def test_get_height_with_different_devices(scene, basic_terrain_analysis_cfg, device):
    """Test get_height works correctly with different devices."""
    # Create terrain analysis instance
    terrain_analysis = TerrainAnalysis(basic_terrain_analysis_cfg, scene)

    # Set up a known height grid for testing
    terrain_analysis._height_grid = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0]
    ]).to(device)

    # Set mesh dimensions: [x_max, y_max, x_min, y_min]
    terrain_analysis._mesh_dimensions = [0.2, 0.2, 0.0, 0.0]

    # Test position on the specified device
    positions = torch.tensor([[0.05, 0.05]]).to(device)
    heights = terrain_analysis.get_height(positions)

    expected_height = torch.tensor([1.0])
    assert torch.allclose(heights.cpu(), expected_height), f"Expected {expected_height}, got {heights.cpu()}"
    assert heights.device.type == device.split(":")[0], f"Expected device {device}, got {heights.device}"
