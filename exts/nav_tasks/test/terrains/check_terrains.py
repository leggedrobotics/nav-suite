# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Launch Isaac Sim Simulator first."""

import argparse

parser = argparse.ArgumentParser(description="Generate terrains using trimesh")
parser.add_argument(
    "--headless", action="store_true", default=False, help="Don't create a window to display each output."
)
args_cli = parser.parse_args()

from isaaclab.app import AppLauncher

# launch omniverse app
# note: we only need to do this because of `TerrainImporter` which uses Omniverse functions
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import argparse
import os
import trimesh

from isaaclab.terrains.utils import color_meshes_by_height

import nav_tasks.terrains as mesh_gen
from nav_tasks import NAVSUITE_TASKS_DATA_DIR


def test_corridor_terrain(difficulty: float, output_dir: str, headless: bool):
    # parameters for the terrain
    cfg = mesh_gen.CorridorTerrainCfg(size=(8.0, 8.0))
    # generate the terrain
    meshes, origin = cfg.function(difficulty=difficulty, cfg=cfg)
    # add colors to the meshes based on the height
    colored_mesh = color_meshes_by_height(meshes)
    # add a marker for the origin
    origin_transform = trimesh.transformations.translation_matrix(origin)
    origin_marker = trimesh.creation.axis(origin_size=0.1, transform=origin_transform)
    # visualize the meshes
    scene = trimesh.Scene([colored_mesh, origin_marker])
    # save the scene to a png file
    data = scene.save_image(resolution=(640, 480))
    # write the image to a file
    with open(os.path.join(output_dir, "corridor_terrain.jpg"), "wb") as f:
        f.write(data)
    # show the scene in a window
    if not headless:
        trimesh.viewer.SceneViewer(scene=scene, caption="Corridor Terrain")


def test_maze_terrain(
    difficulty: float, terrain_json: str, terrain_name: str, size: tuple[float, float], output_dir: str, headless: bool
):
    # parameters for the terrain
    cfg = mesh_gen.MazeTerrainCfg(size=size, path_obstacles=terrain_json)
    # generate the terrain
    meshes, origin = cfg.function(difficulty=difficulty, cfg=cfg)
    # add colors to the meshes based on the height
    colored_mesh = color_meshes_by_height(meshes)
    # add a marker for the origin
    origin_transform = trimesh.transformations.translation_matrix(origin)
    origin_marker = trimesh.creation.axis(origin_size=0.1, transform=origin_transform)
    # visualize the meshes
    scene = trimesh.Scene([colored_mesh, origin_marker])
    # save the scene to a png file
    data = scene.save_image(resolution=(640, 480))
    # write the image to a file
    with open(os.path.join(output_dir, terrain_name + ".jpg"), "wb") as f:
        f.write(data)
    # show the scene in a window
    if not headless:
        trimesh.viewer.SceneViewer(scene=scene, caption=terrain_name)


def test_mesh_pillar_terrain(difficulty: float, deterministic: bool, output_dir: str, headless: bool):
    # parameters for the terrain
    box_objects = mesh_gen.MeshPillarTerrainCfg.BoxCfg(
        width=(0.5, 1.0), length=(0.2, 0.5), max_yx_angle=(0, 10), height=(0.5, 3), num_objects=(5, 5)
    )
    cylinder_cfg = mesh_gen.MeshPillarTerrainCfg.CylinderCfg(
        radius=(0.3, 0.5), max_yx_angle=(0, 5), height=(0.5, 5), num_objects=(7, 7)
    )
    if deterministic:
        cfg = mesh_gen.MeshPillarTerrainDeterministicCfg(
            size=(8.0, 8.0), box_objects=box_objects, cylinder_cfg=cylinder_cfg
        )
    else:
        cfg = mesh_gen.MeshPillarTerrainCfg(size=(8.0, 8.0), box_objects=box_objects, cylinder_cfg=cylinder_cfg)
    # generate the terrain
    meshes, origin = cfg.function(difficulty=difficulty, cfg=cfg)
    # add colors to the meshes based on the height
    colored_mesh = color_meshes_by_height(meshes)
    # add a marker for the origin
    origin_transform = trimesh.transformations.translation_matrix(origin)
    origin_marker = trimesh.creation.axis(origin_size=0.1, transform=origin_transform)
    # visualize the meshes
    scene = trimesh.Scene([colored_mesh, origin_marker])
    # save the scene to a png file
    data = scene.save_image(resolution=(640, 480))
    # write the image to a file
    terrain_name = "mesh_pillar_terrain_deterministic" if deterministic else "mesh_pillar_terrain"
    with open(os.path.join(output_dir, terrain_name + ".jpg"), "wb") as f:
        f.write(data)
    # show the scene in a window
    if not headless:
        trimesh.viewer.SceneViewer(scene=scene, caption="Mesh Pillar Terrain")


def test_quad_stairs_terrain(difficulty: float, output_dir: str, headless: bool):
    # parameters for the terrain
    cfg = mesh_gen.MeshQuadPyramidStairsCfg(size=(8.0, 8.0), step_width=0.3, step_height_range=(0.2, 0.5))
    # generate the terrain
    meshes, origin = cfg.function(difficulty=difficulty, cfg=cfg)
    # add colors to the meshes based on the height
    colored_mesh = color_meshes_by_height(meshes)
    # add a marker for the origin
    origin_transform = trimesh.transformations.translation_matrix(origin)
    origin_marker = trimesh.creation.axis(origin_size=0.1, transform=origin_transform)
    # visualize the meshes
    scene = trimesh.Scene([colored_mesh, origin_marker])
    # save the scene to a png file
    data = scene.save_image(resolution=(640, 480))
    # write the image to a file
    with open(os.path.join(output_dir, "quad_stairs_terrain.jpg"), "wb") as f:
        f.write(data)
    # show the scene in a window
    if not headless:
        trimesh.viewer.SceneViewer(scene=scene, caption="Quad Stairs Terrain")


def test_stairs_ramp_terrain(difficulty: float, output_dir: str, headless: bool):
    # parameters for the terrain
    cfg = mesh_gen.StairsRampTerrainCfg(
        size=(8.0, 8.0),
        modify_ramp_slope=True,
        ramp_slope_range=(5, 45),
        step_width=0.3,
        random_wall_probability=0.0,
    )
    # generate the terrain
    meshes, origin = cfg.function(difficulty=difficulty, cfg=cfg)
    # add colors to the meshes based on the height
    colored_mesh = color_meshes_by_height(meshes)
    # add a marker for the origin
    origin_transform = trimesh.transformations.translation_matrix(origin)
    origin_marker = trimesh.creation.axis(origin_size=0.1, transform=origin_transform)
    # visualize the meshes
    scene = trimesh.Scene([colored_mesh, origin_marker])
    # save the scene to a png file
    data = scene.save_image(resolution=(640, 480))
    # write the image to a file
    with open(os.path.join(output_dir, "stairs_ramp_terrain.jpg"), "wb") as f:
        f.write(data)
    # show the scene in a window
    if not headless:
        trimesh.viewer.SceneViewer(scene=scene, caption="Stairs Ramp Terrain")


def test_stairs_ramp_eval_terrain(difficulty: float, output_dir: str, headless: bool):
    # parameters for the terrain
    cfg = mesh_gen.StairsRampEvalTerrainCfg(
        size=(8.0, 8.0),
        modify_ramp_slope=True,
        ramp_slope_range=(5, 45),
        step_width=0.3,
        random_wall_probability=0.0,
    )
    # generate the terrain
    meshes, origin = cfg.function(difficulty=difficulty, cfg=cfg)
    # add colors to the meshes based on the height
    colored_mesh = color_meshes_by_height(meshes)
    # add a marker for the origin
    origin_transform = trimesh.transformations.translation_matrix(origin)
    origin_marker = trimesh.creation.axis(origin_size=0.1, transform=origin_transform)
    # visualize the meshes
    scene = trimesh.Scene([colored_mesh, origin_marker])
    # save the scene to a png file
    data = scene.save_image(resolution=(640, 480))
    # write the image to a file
    with open(os.path.join(output_dir, "stairs_ramp_eval_terrain.jpg"), "wb") as f:
        f.write(data)
    # show the scene in a window
    if not headless:
        trimesh.viewer.SceneViewer(scene=scene, caption="Stairs Ramp Eval Terrain")


def test_stairs_ramp_up_down_terrain(difficulty: float, output_dir: str, headless: bool):
    # parameters for the terrain
    cfg = mesh_gen.StairsRampUpDownTerrainCfg(
        size=(8.0, 8.0),
        modify_ramp_slope=True,
        ramp_slope_range=(5, 45),
        step_width=0.3,
        random_wall_probability=0.0,
    )
    # generate the terrain
    meshes, origin = cfg.function(difficulty=difficulty, cfg=cfg)
    # add colors to the meshes based on the height
    colored_mesh = color_meshes_by_height(meshes)
    # add a marker for the origin
    origin_transform = trimesh.transformations.translation_matrix(origin)
    origin_marker = trimesh.creation.axis(origin_size=0.1, transform=origin_transform)
    # visualize the meshes
    scene = trimesh.Scene([colored_mesh, origin_marker])
    # save the scene to a png file
    data = scene.save_image(resolution=(640, 480))
    # write the image to a file
    with open(os.path.join(output_dir, "stairs_ramp_up_down_terrain.jpg"), "wb") as f:
        f.write(data)
    # show the scene in a window
    if not headless:
        trimesh.viewer.SceneViewer(scene=scene, caption="Stairs Ramp Up Down Terrain")


def test_random_maze_terrain(difficulty: float, output_dir: str, headless: bool):
    # parameters for the terrain
    cfg = mesh_gen.RandomMazeTerrainCfg(
        size=(20.0, 20.0),
        resolution=1.0,
        maze_height=1.0,
        max_increase=10.0,
        max_decrease=10.0,
        width_range=(0.5, 1.0),
        length_range=(0.5, 1.0),
        height_range=(0.7, 1.3),
        num_stairs=2,
        step_height_range=(0.2, 0.25),
        step_width_range=(0.2, 0.3),
    )
    # generate the terrain
    meshes, origin = cfg.function(difficulty=difficulty, cfg=cfg)
    # add colors to the meshes based on the height
    colored_mesh = color_meshes_by_height(meshes)
    # add a marker for the origin
    origin_transform = trimesh.transformations.translation_matrix(origin)
    origin_marker = trimesh.creation.axis(origin_size=0.1, transform=origin_transform)
    # visualize the meshes
    scene = trimesh.Scene([colored_mesh, origin_marker])
    # save the scene to a png file
    data = scene.save_image(resolution=(640, 480))
    # write the image to a file
    with open(os.path.join(output_dir, "random_maze_terrain.jpg"), "wb") as f:
        f.write(data)
    # show the scene in a window
    if not headless:
        trimesh.viewer.SceneViewer(scene=scene, caption="Random Maze Terrain")


def main():
    # Create directory to dump results
    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "terrains", "trimesh")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # Read headless mode
    headless = args_cli.headless
    # generate terrains
    test_corridor_terrain(difficulty=0.0, output_dir=output_dir, headless=headless)
    test_maze_terrain(
        difficulty=0.0,
        terrain_json=os.path.join(NAVSUITE_TASKS_DATA_DIR, "maze_terrain_json", "Training.json"),
        terrain_name="maze_training_terrain",
        size=(30, 30),
        output_dir=output_dir,
        headless=headless,
    )
    test_maze_terrain(
        difficulty=0.0,
        terrain_json=os.path.join(NAVSUITE_TASKS_DATA_DIR, "maze_terrain_json", "Validation_Easy.json"),
        terrain_name="maze_val_easy_terrain",
        size=(15, 15),
        output_dir=output_dir,
        headless=headless,
    )
    test_maze_terrain(
        difficulty=0.0,
        terrain_json=os.path.join(NAVSUITE_TASKS_DATA_DIR, "maze_terrain_json", "Validation_Hard.json"),
        terrain_name="maze_val_hard_terrain",
        size=(15, 15),
        output_dir=output_dir,
        headless=headless,
    )
    test_mesh_pillar_terrain(difficulty=0.0, deterministic=True, output_dir=output_dir, headless=headless)
    test_mesh_pillar_terrain(difficulty=0.0, deterministic=False, output_dir=output_dir, headless=headless)
    test_quad_stairs_terrain(difficulty=0.0, output_dir=output_dir, headless=headless)
    test_stairs_ramp_terrain(difficulty=0.0, output_dir=output_dir, headless=headless)
    test_stairs_ramp_eval_terrain(difficulty=0.0, output_dir=output_dir, headless=headless)
    test_stairs_ramp_up_down_terrain(difficulty=0.0, output_dir=output_dir, headless=headless)
    test_random_maze_terrain(difficulty=0.9, output_dir=output_dir, headless=headless)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
