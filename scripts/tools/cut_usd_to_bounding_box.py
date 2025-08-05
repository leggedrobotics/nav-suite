# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Script to cut a USD file to a certain bounding box.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Filter USD scene by bounding box.")
parser.add_argument("input_usd", help="Path to input USD file")
parser.add_argument("output_dir", help="Path to output directory")
parser.add_argument("--min_bounds", type=float, nargs=2, help="Min bounding box coordinates (x y z)")
parser.add_argument("--max_bounds", type=float, nargs=2, help="Max bounding box coordinates (x y z)")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import os
import trimesh
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import isaaclab.sim as sim_utils
import omni.usd
from isaaclab.utils.mesh import create_trimesh_from_geom_mesh
from pxr import Gf, Sdf, Usd, UsdGeom


def recursive_non_instance_prim_finder(prim):
    if not prim.IsInstanceProxy():
        return prim
    return recursive_non_instance_prim_finder(prim.GetParent())


def cut_mesh_to_bounding_box(stage, mesh_prim, min_bounds, max_bounds) -> bool:
    # check if mesh complete inside the bounding box
    bbox = UsdGeom.Boundable(mesh_prim).ComputeWorldBound(0, UsdGeom.Tokens.default_).ComputeAlignedBox()
    if (
        bbox.GetMin()[0] >= min_bounds[0]
        and bbox.GetMax()[0] <= max_bounds[0]
        and bbox.GetMin()[1] >= min_bounds[1]
        and bbox.GetMax()[1] <= max_bounds[1]
    ):
        return False

    # create trimesh from mesh
    vertices, faces = create_trimesh_from_geom_mesh(mesh_prim)
    # transform to world coordinates
    transform = np.asarray(omni.usd.get_world_transform_matrix(mesh_prim)).T
    vertices = (transform @ np.hstack([vertices, np.ones((vertices.shape[0], 1))]).T).T[:, :3]
    mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    # create bounding box planes
    bbox_means = (np.array(min_bounds) + np.array(max_bounds)) / 2
    bbox_centers = np.array([
        [bbox_means[0], min_bounds[1], 0],
        [bbox_means[0], max_bounds[1], 0],
        [min_bounds[0], bbox_means[1], 0],
        [max_bounds[0], bbox_means[1], 0],
    ])
    bbox_normals = np.array([[0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]])
    # slice the mesh with the bounding box planes
    for center, normal in zip(bbox_centers, bbox_normals):
        mesh_trimesh = mesh_trimesh.slice_plane(center, normal)
    # original mesh path
    mesh_path = mesh_prim.GetPath().pathString
    mesh_path = mesh_path.removeprefix("/Root/").replace("/", "_")
    # apply root transform to the mesh
    root_prim = stage.GetPrimAtPath("/Root")
    transform_root = np.asarray(omni.usd.get_world_transform_matrix(root_prim)).T
    mesh_trimesh.apply_transform(np.linalg.inv(transform_root))
    # add as usd mesh back to the stage
    clipped_mesh_prim = UsdGeom.Mesh.Define(stage, "/Root/clipped/" + mesh_path)
    clipped_mesh_prim.CreatePointsAttr([Gf.Vec3f(*v) for v in mesh_trimesh.vertices])
    clipped_mesh_prim.CreateFaceVertexCountsAttr([3] * len(mesh_trimesh.faces))
    clipped_mesh_prim.CreateFaceVertexIndicesAttr(mesh_trimesh.faces.flatten().tolist())

    return True


def filter_usd_by_bbox(input_usd, output_dir, min_bounds, max_bounds):
    # Load the USD Stage
    stage = Usd.Stage.Open(input_usd)
    if not stage:
        print(f"Error: Could not open USD file {input_usd}")
        return

    def is_within_bounds(prim):
        """Check if a prim's bounding box is at least partially inside the given bounds"""
        bbox = UsdGeom.Boundable(prim).ComputeWorldBound(0, UsdGeom.Tokens.default_).ComputeAlignedBox()

        # 4 cases when no intersection:
        outside_bbox_mask = (
            bbox.GetMax()[0] < min_bounds[0]
            or bbox.GetMin()[0] > max_bounds[0]
            or bbox.GetMax()[1] < min_bounds[1]
            or bbox.GetMin()[1] > max_bounds[1]
        )

        return not outside_bbox_mask

    def filter_point_instancer(instancer, stage):
        """Filter PointInstancer to keep only instances within the bounding box."""
        positions = np.asarray(instancer.GetPositionsAttr().Get() or [])
        orientations = np.asarray(instancer.GetOrientationsAttr().Get() or [])  # order: x, y, z, w
        scales = np.asarray(instancer.GetScalesAttr().Get() or [])
        if positions is None or len(positions) == 0 or orientations is None or len(orientations) == 0:
            return False

        proto_index_attr = instancer.GetProtoIndicesAttr()
        proto_indices = np.array(proto_index_attr.Get())
        keep_proto_indices = np.zeros(len(proto_indices), dtype=bool)

        instancer_transform = np.asarray(omni.usd.get_world_transform_matrix(instancer.GetPrim())).T

        # Retrieve the prototype extent (bounding boxes of instances)
        proto_paths = instancer.GetPrototypesRel().GetTargets()
        for idx, proto_path in enumerate(proto_paths):
            proto_prim = stage.GetPrimAtPath(proto_path)
            if not proto_prim:
                continue

            # Get the extent of the prototype
            bbox = UsdGeom.Boundable(proto_prim).ComputeWorldBound(0, UsdGeom.Tokens.default_).ComputeAlignedBox()
            # transform the bounding box to coordinates
            corners = np.array([
                [bbox.GetMin()[0], bbox.GetMin()[1], bbox.GetMin()[2]],
                [bbox.GetMin()[0], bbox.GetMax()[1], bbox.GetMin()[2]],
                [bbox.GetMax()[0], bbox.GetMin()[1], bbox.GetMin()[2]],
                [bbox.GetMax()[0], bbox.GetMax()[1], bbox.GetMin()[2]],
            ])

            # get the indices of the instances that are part of this prototype
            proto_indexes = np.nonzero(proto_indices == idx)[0]

            # construct the transformation matrices
            transform_matrix = np.eye(4)[None, :, :].repeat(len(proto_indexes), axis=0)
            transform_matrix[:, :3, 3] = positions[proto_indexes]
            transform_matrix[:, :3, :3] = (
                Rotation.from_quat(orientations[proto_indexes]).as_matrix() * scales[proto_indexes][:, np.newaxis, :]
            )
            transform_matrix = instancer_transform @ transform_matrix

            # apply the transformation matrix to the bounding box
            corners_world = (transform_matrix @ np.hstack([corners, np.ones((corners.shape[0], 1))]).T)[:, :3, :]
            corners_world = corners_world.transpose(0, 2, 1)[..., :2]
            # check which any of the corners are inside the bounding box
            corner_inside_bbox_mask = np.any(
                np.all(np.logical_and(corners_world >= min_bounds, corners_world <= max_bounds), axis=2), axis=1
            )
            # check if the entire bounding box is inside the bounding box
            inside_bbox_mask = np.all(
                np.logical_and(corners_world[:, 0] <= min_bounds, corners_world[:, -1] >= max_bounds), axis=1
            )

            keep_proto_indices[proto_indexes] = np.logical_or(inside_bbox_mask, corner_inside_bbox_mask)

        if np.any(keep_proto_indices):
            # set a mask for the instances that are kept
            instancer.GetPositionsAttr().Set(positions[keep_proto_indices])
            instancer.GetOrientationsAttr().Set(orientations[keep_proto_indices])
            instancer.GetScalesAttr().Set(scales[keep_proto_indices])
            instancer.GetProtoIndicesAttr().Set(proto_indices[keep_proto_indices])
            return True

        return False

    print(
        "Searching for all meshes and point instancers under root prim",
        stage.GetPseudoRoot().GetChildren()[0].GetPath(),
    )
    mesh_prims = sim_utils.get_all_matching_child_prims(
        stage.GetPseudoRoot().GetChildren()[0].GetPath(), lambda prim: prim.GetTypeName() == "Mesh", stage=stage
    )
    point_instancers = sim_utils.get_all_matching_child_prims(
        stage.GetPseudoRoot().GetChildren()[0].GetPath(),
        lambda prim: prim.GetTypeName() == "PointInstancer",
        stage=stage,
    )

    # filter all meshes that are under a point instancer
    filtered_mesh_prims = [
        mesh_prim
        for mesh_prim in mesh_prims
        if not any(
            Sdf.Path.HasPrefix(mesh_prim.GetPath(), instancer_path.GetPath()) for instancer_path in point_instancers
        )
    ]

    # **Group meshes by their parent Xform**
    xform_children_map = {}
    for mesh in filtered_mesh_prims:
        parent_xform = mesh.GetParent()
        if parent_xform not in xform_children_map:
            xform_children_map[parent_xform] = []
        xform_children_map[parent_xform].append(mesh)

    # **Filter meshes by bounding box**
    xforms_to_remove = []
    prims_to_remove = set()

    for xform, meshes in tqdm(xform_children_map.items(), desc="Filtering Xform groups"):
        meshes_outside_bbox = [mesh for mesh in meshes if not is_within_bounds(mesh)]

        # cut the meshes to the bounding box
        mesh_not_removed = [mesh for mesh in meshes if mesh not in meshes_outside_bbox]
        for mesh in mesh_not_removed:
            mesh_cut = cut_mesh_to_bounding_box(stage, mesh, min_bounds, max_bounds)
            if mesh_cut:
                meshes_outside_bbox.append(mesh)

        if len(meshes_outside_bbox) == len(meshes):
            # **All meshes under this Xform are out of bounds → mark Xform for removal**
            if xform.IsInstanceProxy():
                xform_parent = recursive_non_instance_prim_finder(xform)
                xforms_to_remove.append(xform_parent)
            else:
                xforms_to_remove.append(xform)
        else:
            # **Only some meshes should be removed → keep the Xform, remove only those meshes**
            prims_to_remove.update(meshes_outside_bbox)

    # Process PointInstancer prims separately
    point_instancers_to_remove = []
    for prim in tqdm(point_instancers, desc="Processing PointInstancers"):
        if not filter_point_instancer(UsdGeom.PointInstancer(prim), stage):
            point_instancers_to_remove.append(prim)

    # Remove unwanted prims
    for prim in tqdm(point_instancers_to_remove, desc="Removing PointInstancers"):
        prim.SetActive(False)

    # remove all xforms where all underling meshes have been removed
    for xform in tqdm(xforms_to_remove, desc="Removing Xforms"):
        try:
            if not xform.IsValid():
                continue

            remove_success = stage.RemovePrim(xform.GetPath())
            if not remove_success:
                xform.SetActive(False)
        except Exception as e:
            print(f"Error removing prim {xform.GetPath()}: {e}")

    for prim in tqdm(prims_to_remove, desc="Removing unwanted single prims"):
        if not prim.IsValid():
            continue

        if prim.IsInstanceProxy():
            continue

        try:
            remove_success = stage.RemovePrim(prim.GetPath())
            if not remove_success:
                prim.SetActive(False)
        except Exception as e:
            print(f"Error removing prim {prim.GetPath()}: {e}")

    # get the basename of the input usd file and append the bounding box to it
    os.makedirs(output_dir, exist_ok=True)
    output_usd = os.path.join(
        output_dir,
        os.path.basename(input_usd).replace(
            ".usd", f"_bbox_{min_bounds[0]}_{min_bounds[1]}_{max_bounds[0]}_{max_bounds[1]}.usd"
        ),
    )
    # Save the reduced USD scene
    stage.GetRootLayer().Export(output_usd)
    print(f"Reduced USD scene saved to {output_usd}")


if __name__ == "__main__":
    filter_usd_by_bbox(args_cli.input_usd, args_cli.output_dir, args_cli.min_bounds, args_cli.max_bounds)
    # Close the simulator
    simulation_app.close()
