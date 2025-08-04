# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import contextlib
import math
import os
import tempfile
import torch
import trimesh
import yaml
from typing import Literal

import isaacsim.core.utils.stage as stage_utils
import pytest
from isaaclab.sim import build_simulation_context, get_first_matching_child_prim
from isaacsim.core.utils.semantics import get_semantics

from nav_suite import NAVSUITE_TEST_ASSETS_DIR
from nav_suite.terrains import NavTerrainImporter, NavTerrainImporterCfg


# Helper Functions
def assert_mesh_prim_exists(prim_path):
    """Helper to validate mesh prim existence and structure."""
    stage = stage_utils.get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    assert prim.IsValid(), f"Prim at {prim_path} is not valid"

    mesh_child = get_first_matching_child_prim(prim_path, lambda x: x.GetTypeName() == "Mesh")
    assert mesh_child is not None, f"No mesh child found at {prim_path}"
    return prim, mesh_child


def create_test_mesh(mesh_type: Literal["triangle", "square", "tetrahedron"] = "triangle"):
    """Create test meshes for different test scenarios."""
    if mesh_type == "triangle":
        vertices = [[0, 0, 0], [1, 0, 0], [0.5, 1, 0]]
        faces = [[0, 1, 2]]
    elif mesh_type == "square":
        vertices = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
        faces = [[0, 1, 2], [0, 2, 3]]
    elif mesh_type == "tetrahedron":
        vertices = [[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0.5, 1]]
        faces = [[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]]
    else:
        raise ValueError(f"Unknown mesh type: {mesh_type}")

    return trimesh.Trimesh(vertices=vertices, faces=faces)


# Test Fixtures
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
def basic_terrain_config():
    """Fixture for basic terrain configuration."""
    return NavTerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=os.path.join(NAVSUITE_TEST_ASSETS_DIR, "terrains", "ground_plane.usda"),
        num_envs=1,
        env_spacing=2.0,
        add_colliders=False,
    )


@pytest.fixture
def obj_terrain_config():
    """Fixture for OBJ terrain configuration."""
    return NavTerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=os.path.join(NAVSUITE_TEST_ASSETS_DIR, "terrains", "ground_plane_obj.obj"),
        num_envs=1,
        env_spacing=2.0,
        add_colliders=False,
    )


@pytest.fixture
def terrain_config_factory():
    """Factory for creating terrain configurations with common defaults."""

    def _create_config(overrides=None):
        base_config = {
            "prim_path": "/World/ground",
            "terrain_type": "usd",
            "usd_path": os.path.join(NAVSUITE_TEST_ASSETS_DIR, "terrains", "ground_plane.usda"),
            "num_envs": 1,
            "env_spacing": 2.0,
            "add_colliders": False,
        }
        if overrides:
            base_config.update(overrides)
        return NavTerrainImporterCfg(**base_config)

    return _create_config


@pytest.fixture
def temp_yaml_file():
    """Fixture for creating temporary YAML files with automatic cleanup."""
    temp_files = []

    def _create_temp_file(content_dict):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
            yaml.dump(content_dict, temp_file)
            temp_files.append(temp_file.name)
            return temp_file.name

    yield _create_temp_file

    # Cleanup
    for file_path in temp_files:
        with contextlib.suppress(Exception):
            if os.path.exists(file_path):
                os.unlink(file_path)


# ============================================================================
# NavTerrainImporter Method Tests
# ============================================================================


def test_import_mesh_single(simulation_context, terrain_config_factory):
    """Tests the import_mesh method directly with a simple trimesh object."""

    # Import dummy usd file but not actually needed for actual testing
    terrain_importer_cfg = terrain_config_factory({"groundplane": False})
    terrain_importer = NavTerrainImporter(terrain_importer_cfg)

    # Create a simple tetrahedron mesh using helper function
    test_mesh = create_test_mesh("tetrahedron")

    # Call import_mesh directly
    terrain_importer.import_mesh("test_terrain", test_mesh)

    # Verify the mesh was imported correctly
    assert len(terrain_importer.terrain_prim_paths) >= 1
    expected_prim_path = "/World/ground/test_terrain"
    assert expected_prim_path in terrain_importer.terrain_prim_paths

    # Verify the mesh prim exists using helper function
    assert_mesh_prim_exists(expected_prim_path)


def test_import_mesh_multiple(simulation_context, terrain_config_factory):
    """Tests the import_mesh method with multiple meshes (dictionary input)."""

    # Import dummy usd file but not actually needed for actual testing
    terrain_importer_cfg = terrain_config_factory({"groundplane": False})
    terrain_importer = NavTerrainImporter(terrain_importer_cfg)

    # Create meshes using helper function
    mesh1 = create_test_mesh("triangle")
    mesh2 = create_test_mesh("square")

    # Create dictionary of meshes
    mesh_dict = {"triangle": mesh1, "square": mesh2}

    # Call import_mesh with dictionary
    terrain_importer.import_mesh("multi_terrain", mesh_dict)

    # Verify both meshes were imported
    assert len(terrain_importer.terrain_prim_paths) >= 2

    expected_paths = ["/World/ground/multi_terrain_triangle", "/World/ground/multi_terrain_square"]

    for expected_path in expected_paths:
        assert expected_path in terrain_importer.terrain_prim_paths
        # Verify the mesh prim exists using helper function
        assert_mesh_prim_exists(expected_path)


def test_mesh_duplication(simulation_context, basic_terrain_config, temp_yaml_file):
    """Tests the mesh_duplicator method of NavTerrainImporter.

    The naming convention for duplicated prims follows this pattern:
    {original_path}_tr{translation_idx}_cp{copy_idx}{suffix}
    where:
    - translation_idx: index of the translation in the list of translations
    - copy_idx: index of the copy (0 to factor-1)
    - suffix: optional suffix specified in the config
    """
    # Create duplication configuration
    duplication_config = {
        "duplicate_prim": {
            "prim": "terrain",  # The prim to duplicate
            "translation": [1.0, 0.0, 0.0],  # Translation for each copy
            "factor": 2,  # Number of copies to make
            "suffix": "_copy",  # Suffix for the duplicated prims
        }
    }
    temp_file_path = temp_yaml_file(duplication_config)

    # Create the terrain importer with duplication config
    terrain_importer_cfg = NavTerrainImporterCfg(
        prim_path=basic_terrain_config.prim_path,
        terrain_type=basic_terrain_config.terrain_type,
        usd_path=basic_terrain_config.usd_path,
        num_envs=basic_terrain_config.num_envs,
        env_spacing=basic_terrain_config.env_spacing,
        add_colliders=basic_terrain_config.add_colliders,
        duplicate_cfg_file=temp_file_path,
    )
    terrain_importer = NavTerrainImporter(terrain_importer_cfg)

    # Check that the original terrain exists
    original_prim_path = terrain_importer.cfg.prim_path + "/terrain"
    assert original_prim_path in terrain_importer.terrain_prim_paths

    # Check that the duplicated prims exist
    stage = stage_utils.get_current_stage()

    # Check first copy (tr0 = first translation, cp0 = first copy)
    first_copy_path = original_prim_path + "/mesh_tr0_cp0_copy"
    first_copy_prim = stage.GetPrimAtPath(first_copy_path)
    assert first_copy_prim.IsValid()

    # Check second copy (tr0 = first translation, cp1 = second copy)
    second_copy_path = original_prim_path + "/mesh_tr0_cp1_copy"
    second_copy_prim = stage.GetPrimAtPath(second_copy_path)
    assert second_copy_prim.IsValid()

    # Verify translations
    first_copy_xform = first_copy_prim.GetAttribute("xformOp:translate").Get()
    second_copy_xform = second_copy_prim.GetAttribute("xformOp:translate").Get()

    # First copy should be at [1, 0, 0] (translation * 1)
    assert first_copy_xform[0] == 1.0
    assert first_copy_xform[1] == 0.0
    assert first_copy_xform[2] == 0.0

    # Second copy should be at [2, 0, 0] (translation * 2)
    assert second_copy_xform[0] == 2.0
    assert second_copy_xform[1] == 0.0
    assert second_copy_xform[2] == 0.0


def test_people_insertion(simulation_context, terrain_config_factory, temp_yaml_file):
    """Tests the insert_single_person static method and people insertion functionality."""

    # Create a temporary people config file
    people_config = {
        "person1": {"prim_name": "test_person1", "translation": [1.0, 2.0, 0.0], "scale": 1.2},
        "person2": {"prim_name": "test_person2", "translation": [3.0, 4.0, 0.0], "scale": 1.0},
    }
    temp_file_path = temp_yaml_file(people_config)

    # Create the terrain importer with people config
    terrain_importer_cfg = terrain_config_factory({"people_config_file": temp_file_path})
    NavTerrainImporter(terrain_importer_cfg)

    # Get the stage
    stage = stage_utils.get_current_stage()

    # Verify each person's properties
    for person_id, person_cfg in people_config.items():
        person_path = f"/World/People/{person_cfg['prim_name']}"
        person_prim = stage.GetPrimAtPath(person_path)
        assert person_prim.IsValid(), f"Person prim at {person_path} is not valid"

        # Verify transform
        xform = person_prim.GetAttribute("xformOp:translate").Get()
        assert xform == tuple(person_cfg["translation"]), f"Unexpected translation for {person_id}"

        # Verify scale
        scale = person_prim.GetAttribute("xformOp:scale").Get()
        expected_scale = (person_cfg["scale"],) * 3
        assert all(
            round(s, 2) == round(es, 2) for s, es in zip(scale, expected_scale)
        ), f"Unexpected scale for {person_id}"

        # Verify semantic label
        semantic_info = get_semantics(person_prim)
        assert (
            semantic_info is not None and "Semantics" in semantic_info
        ), f"Missing semantic information for {person_id}"
        semantic_label = semantic_info["Semantics"][1]
        assert semantic_label == "person", f"Expected semantic label 'person' for {person_id}, got '{semantic_label}'"


def test_usd_uniform_env_spacing(simulation_context, terrain_config_factory):
    """Tests that usd_uniform_env_spacing creates a grid of environment origins over the mesh bounding box."""

    # Test values
    spacing = 10
    num_envs = 10

    terrain_importer_cfg = terrain_config_factory({
        "num_envs": num_envs,
        "usd_uniform_env_spacing": spacing,
    })
    terrain_importer = NavTerrainImporter(terrain_importer_cfg)

    expected_coords = [
        (-20.0, -20.0),
        (-20.0, -10.0),
        (-20.0, 0.0),
        (-20.0, 10.0),
        (-10.0, -20.0),
        (-10.0, -10.0),
        (-10.0, 0.0),
        (-10.0, 10.0),
        (0.0, -20.0),
        (0.0, -10.0),
    ]
    # Only take as many as num_envs
    expected_coords = expected_coords[:num_envs]

    env_origins = terrain_importer.env_origins.cpu().numpy()
    assert env_origins.shape == (num_envs, 3)
    for i, (x, y) in enumerate(expected_coords):
        assert abs(env_origins[i][0] - x) < 1e-3, f"x coord mismatch at {i}: {env_origins[i][0]} vs {x}"
        assert abs(env_origins[i][1] - y) < 1e-3, f"y coord mismatch at {i}: {env_origins[i][1]} vs {y}"
        assert abs(env_origins[i][2] - 0) < 1e-3, f"z coord mismatch at {i}: {env_origins[i][2]} vs 0"


def test_regular_spawning(simulation_context, terrain_config_factory):
    """Tests regular_spawning behavior with a custom grid of origins."""

    num_envs = 10

    origins = torch.tensor([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
    ])

    terrain_importer_cfg = terrain_config_factory({
        "num_envs": num_envs,
        "regular_spawning": True,
        "custom_origins": origins,
    })
    terrain_importer = NavTerrainImporter(terrain_importer_cfg)

    # Compute expected origins by repeating the custom origins
    expected_origins = origins.repeat(math.ceil(num_envs / origins.shape[0]), 1, 1).reshape(-1, 3)[:num_envs]

    assert torch.allclose(terrain_importer.env_origins.cpu(), expected_origins)


def test_random_spawning(simulation_context, terrain_config_factory):
    """Tests random spawning behavior with a custom grid of origins."""

    num_envs = 10
    random_seed = 42

    origins = torch.tensor([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
    ])

    terrain_importer_cfg = terrain_config_factory({
        "num_envs": num_envs,
        "regular_spawning": False,
        "custom_origins": origins,
        "groundplane": False,
        "random_seed": random_seed,
    })

    terrain_importer = NavTerrainImporter(terrain_importer_cfg)

    # The env_origins should be a random selection from the grid, reproducible with the same seed
    env_origins_first = terrain_importer.env_origins.cpu().numpy()

    # Re-create with the same seed to check determinism
    terrain_importer_cfg2 = terrain_config_factory({
        "num_envs": num_envs,
        "regular_spawning": False,
        "custom_origins": origins,
        "groundplane": False,
        "random_seed": random_seed,
    })

    terrain_importer2 = NavTerrainImporter(terrain_importer_cfg2)
    env_origins_second = terrain_importer2.env_origins.cpu().numpy()

    # Check that seeding logic is correct
    assert (env_origins_first == env_origins_second).all(), "Random spawning is not deterministic with the same seed"

    origins_flat = origins.reshape(-1, 3)

    # Check that all generated origins are in the original provided origins
    for eo in env_origins_first:
        assert any((eo == origin).all() for origin in origins_flat), f"Origin {eo} not in provided grid"


def test_semantic_import(simulation_context, terrain_config_factory, temp_yaml_file):
    """Tests the _add_semantics method through semantic labels importing."""

    # Create a temporary keyword mapping file
    keyword_mapping = {
        "default": "ground",
        "tree": ["Tree", "Forest", "Pine", "Leaves"],
        "water": ["River", "Water"],
        "ground": ["Landscape", "Grass"],
    }
    temp_file_path = temp_yaml_file(keyword_mapping)

    # Create the terrain importer with semantic mapping
    terrain_importer_cfg = terrain_config_factory({
        "usd_path": os.path.join(NAVSUITE_TEST_ASSETS_DIR, "terrains", "natural_terrain.usda"),
        "sem_mesh_to_class_map": temp_file_path,
    })
    terrain_importer = NavTerrainImporter(terrain_importer_cfg)

    # Get the stage
    stage = stage_utils.get_current_stage()

    # Get the terrain prim
    terrain_prim = stage.GetPrimAtPath(terrain_importer.cfg.prim_path + "/terrain")
    assert terrain_prim.IsValid(), "Terrain prim is not valid"

    # Verify ground/grass semantic
    ground_mesh = get_first_matching_child_prim(terrain_prim.GetPath(), lambda x: x.GetName() == "Ground")
    assert ground_mesh is not None, "Ground mesh not found"
    ground_semantic = get_semantics(ground_mesh)
    assert ground_semantic is not None, "Missing semantic information for ground mesh"
    assert "Semantics" in ground_semantic, "Missing Semantics in semantic info for ground mesh"
    assert (
        ground_semantic["Semantics"][1] == "ground"
    ), f"Expected semantic label 'ground' for ground mesh, got '{ground_semantic['Semantics'][1]}'"

    # Verify tree semantic
    tree_prim = get_first_matching_child_prim(terrain_prim.GetPath(), lambda x: x.GetName() == "Tree")
    assert tree_prim is not None, "Tree prim not found"

    # Verify water semantic
    water_mesh = get_first_matching_child_prim(terrain_prim.GetPath(), lambda x: x.GetName() == "Water")
    assert water_mesh is not None, "Water mesh not found"
    water_semantic = get_semantics(water_mesh)
    assert water_semantic is not None, "Missing semantic information for water mesh"
    assert "Semantics" in water_semantic, "Missing Semantics in semantic info for water mesh"
    assert (
        water_semantic["Semantics"][1] == "water"
    ), f"Expected semantic label 'water' for water mesh, got '{water_semantic['Semantics'][1]}'"


@pytest.mark.parametrize("add_colliders", [True, False])
def test_collision_properties(simulation_context, terrain_config_factory, add_colliders):
    """Tests the _apply_physics_properties method through collision properties configuration."""

    terrain_importer_cfg = terrain_config_factory({"add_colliders": add_colliders})
    terrain_importer = NavTerrainImporter(terrain_importer_cfg)

    # check if mesh prim path exists
    mesh_prim_path = terrain_importer.cfg.prim_path + "/terrain"
    assert mesh_prim_path in terrain_importer.terrain_prim_paths

    # Get schemas applied to the prim
    stage = stage_utils.get_current_stage()
    prim = stage.GetPrimAtPath(terrain_importer.cfg.prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim path '{terrain_importer.cfg.prim_path}' is not valid.")
    applied_schemas = prim.GetAppliedSchemas()

    if add_colliders:
        # check that the collision schema is applied
        assert "PhysicsCollisionAPI" in applied_schemas
    else:
        # check that the collision schema is not applied
        assert "PhysicsCollisionAPI" not in applied_schemas


def test_scale_parameter(simulation_context, terrain_config_factory):
    """Tests the import_usd method with scale parameter."""

    test_scale = (2.0, 2.0, 2.0)
    terrain_importer_cfg = terrain_config_factory({"scale": test_scale})
    terrain_importer = NavTerrainImporter(terrain_importer_cfg)

    # check that the terrain was imported correctly
    assert terrain_importer.terrain_prim_paths is not None
    assert len(terrain_importer.terrain_prim_paths) == 1

    # check if mesh prim path exists
    mesh_prim_path = terrain_importer.cfg.prim_path + "/terrain"
    assert mesh_prim_path in terrain_importer.terrain_prim_paths

    # check that the mesh prim exists and has the correct scale
    prim, mesh_prim = assert_mesh_prim_exists(mesh_prim_path)

    # Scale attribute is only defined on the parent prim
    parent_prim = mesh_prim.GetParent()
    scale_attr = parent_prim.GetAttribute("xformOp:scale")
    scale_value = scale_attr.Get() if scale_attr is not None else None

    assert scale_attr is not None and scale_attr.HasAuthoredValue()
    assert scale_value == test_scale


def test_usd_import(simulation_context, basic_terrain_config):
    """Tests the import_usd method with basic USD import."""
    terrain_importer = NavTerrainImporter(basic_terrain_config)

    # check that the terrain was imported correctly
    assert terrain_importer.terrain_prim_paths is not None
    assert len(terrain_importer.terrain_prim_paths) == 1

    # check if mesh prim path exists
    mesh_prim_path = terrain_importer.cfg.prim_path + "/terrain"
    assert mesh_prim_path in terrain_importer.terrain_prim_paths

    # check that the mesh prim exists
    assert_mesh_prim_exists(mesh_prim_path)


def test_obj_import(simulation_context, obj_terrain_config):
    """Tests the _convert_obj_to_usd method through OBJ file import."""
    # import the terrain
    terrain_importer = NavTerrainImporter(obj_terrain_config)

    # check that the terrain was imported correctly
    assert terrain_importer.terrain_prim_paths is not None
    assert len(terrain_importer.terrain_prim_paths) == 1

    # check if mesh prim path exists
    mesh_prim_path = terrain_importer.cfg.prim_path + "/terrain"
    assert mesh_prim_path in terrain_importer.terrain_prim_paths

    # check that the mesh prim exists
    assert_mesh_prim_exists(mesh_prim_path)

    # check that the usd_path was updated to the usd file
    assert os.path.basename(terrain_importer.cfg.usd_path).endswith(".usd")

    # cleanup: delete the generated files if they exist
    generated_files = [
        os.path.join(NAVSUITE_TEST_ASSETS_DIR, "terrains", "ground_plane_obj.usd"),
        os.path.join(NAVSUITE_TEST_ASSETS_DIR, "terrains", ".asset_hash"),
        os.path.join(NAVSUITE_TEST_ASSETS_DIR, "terrains", "config.yaml"),
    ]

    for file_path in generated_files:
        with contextlib.suppress(Exception):
            if os.path.exists(file_path):
                os.remove(file_path)


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_malformed_duplicate_config_error(simulation_context, basic_terrain_config, temp_yaml_file):
    """Test error handling for malformed duplication configuration files."""
    # Create malformed YAML content
    malformed_config = {
        "duplicate_prim": {
            "prim": "terrain",
            "translation": "invalid_translation_format",  # Should be a list
            "factor": "not_a_number",  # Should be an integer
        }
    }
    temp_file_path = temp_yaml_file(malformed_config)

    terrain_config = NavTerrainImporterCfg(
        prim_path=basic_terrain_config.prim_path,
        terrain_type=basic_terrain_config.terrain_type,
        usd_path=basic_terrain_config.usd_path,
        num_envs=basic_terrain_config.num_envs,
        env_spacing=basic_terrain_config.env_spacing,
        add_colliders=basic_terrain_config.add_colliders,
        duplicate_cfg_file=temp_file_path,
    )

    with pytest.raises((ValueError, TypeError, KeyError)):
        NavTerrainImporter(terrain_config)


def test_incomplete_semantic_mapping_file_error(simulation_context, basic_terrain_config, temp_yaml_file):
    """Test error handling for incomplete semantic mapping files."""
    # Create invalid semantic mapping
    invalid_mapping = {
        "incomplete_mapping": ["Tree", "Forest"],
        # Missing "mesh" mapping defined in the ground_plane.usd test file
    }
    temp_file_path = temp_yaml_file(invalid_mapping)

    terrain_config = NavTerrainImporterCfg(
        prim_path=basic_terrain_config.prim_path,
        terrain_type=basic_terrain_config.terrain_type,
        usd_path=basic_terrain_config.usd_path,
        num_envs=basic_terrain_config.num_envs,
        env_spacing=basic_terrain_config.env_spacing,
        add_colliders=basic_terrain_config.add_colliders,
        sem_mesh_to_class_map=temp_file_path,
    )

    try:
        # Should throw error that not all meshes have semantic mapping
        NavTerrainImporter(terrain_config)
        assert False
    except (ValueError):
        pass
