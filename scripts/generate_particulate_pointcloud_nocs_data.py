#!/usr/bin/env python3
"""Generate Particulate point-cloud samples with part/global NOCS targets.

This script reuses ``generate_particulate_pointcloud_data.py`` for the S2U
simulation, TSDF reconstruction, motion labels, multiprocessing, and metadata.
It only augments each sample with ``nocs_p`` and ``nocs_g``:

* ``nocs_p``: per-part normalized object coordinates.
* ``nocs_g``: global/rest-object normalized object coordinates.

The normalization formula matches ``tools/dataset.py``:

    (points - bbox_min) * factor + 0.5 - 0.5 * (bbox_max - bbox_min) * factor

where ``factor = 1 / ||bbox_max - bbox_min||_2``.

Objects that do not ship a ``bounding_box.json`` (which the S2U simulator needs
to size/place the object) are handled exactly as in
``generate_particulate_pointcloud_twopose_data.py``: when an Articraft-10K
``fit`` source is passed (``--category`` / ``--fit-dir`` / ``--all-categories``),
each ``.tar.gz`` archive is extracted into a staging object-set directory and a
``bounding_box.json`` is synthesized from the URDF's visual geometry before the
base single-pose generator runs.

Point-cloud source (``--point-sampling``)
-----------------------------------------
By default the point cloud is produced by multi-view depth scanning + TSDF
fusion (the base generator's ``acquire_part_labeled_point_cloud``). Passing
``--point-sampling`` instead samples the cloud directly from the part surface
meshes at the object's current pose (area-weighted across parts). Everything
downstream -- stratified downsampling, normalization, NOCS/NPCS targets -- is
identical; only the acquisition method changes. Tune density with
``--sampling-points`` (total surface points before downsampling) and
``--sampling-min-per-part``.

Object coverage (``--one-per-object``)
--------------------------------------
By default, ``--num-scenes`` scenes choose staged objects randomly. Passing
``--one-per-object`` instead sorts all staged URDF paths and generates exactly
one sample for each object. ``--num-scenes`` is ignored in this mode. With
multiple workers, sample indices are divided between workers but still map to
the same sorted object list, so objects are neither duplicated nor omitted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

import generate_particulate_pointcloud_data as particulate
import generate_particulate_pointcloud_twopose_data as twopose


_BASE_CREATE_TRAINING_SAMPLE = particulate.create_training_sample
_BASE_SCHEMA_DESCRIPTION = particulate.schema_description
_BASE_GENERATE_WORKER = particulate.generate_worker
_BASE_PARSE_ARGS = particulate.parse_args

_RUNTIME_POINT_SAMPLING = False
_RUNTIME_SAMPLING_POINTS = 50000
_RUNTIME_SAMPLING_MIN_PER_PART = 512


def transform_points(transform: Any, points: np.ndarray) -> np.ndarray:
    return transform.rotation.apply(points) + transform.translation


def inverse_transform_points(transform: Any, points: np.ndarray) -> np.ndarray:
    inverse = transform.inverse()
    return transform_points(inverse, points)


def transform_matrix(transform: Any) -> np.ndarray:
    return transform.as_matrix().astype(np.float32)


def transform_matrix_array(
    transforms: Mapping[int, Any],
    ordered_owners: Sequence[int],
) -> np.ndarray:
    return np.stack(
        [transform_matrix(transforms[owner_id]) for owner_id in ordered_owners],
        axis=0,
    )


def relative_transform_matrix(parent: Any, child: Any) -> np.ndarray:
    return (parent.inverse().as_matrix() @ child.as_matrix()).astype(np.float32)


def relative_transform_matrix_array(
    parent: Any,
    transforms: Mapping[int, Any],
    ordered_owners: Sequence[int],
) -> np.ndarray:
    return np.stack(
        [
            relative_transform_matrix(parent, transforms[owner_id])
            for owner_id in ordered_owners
        ],
        axis=0,
    )


def nocs_factor_and_corners(points: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise particulate.SampleGenerationError(
            "NOCS normalization needs a non-empty (N, 3) point set"
        )
    bbox_min = np.amin(points, axis=0, keepdims=True)
    bbox_max = np.amax(points, axis=0, keepdims=True)
    extent = bbox_max - bbox_min
    norm = float(np.linalg.norm(extent))
    if norm < 1e-10:
        raise particulate.SampleGenerationError("NOCS bounding box is degenerate")
    return 1.0 / norm, bbox_min, bbox_max


def normalize_nocs(
    points: np.ndarray,
    factor: float,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> np.ndarray:
    extent = bbox_max - bbox_min
    return (
        (points - bbox_min) * factor
        + np.array([[0.5, 0.5, 0.5]], dtype=np.float64)
        - 0.5 * extent * factor
    )


def get_owner_world_transforms(
    sim: Any,
    owner_ids: Sequence[int],
    frame_source_link_ids: Sequence[int] | None = None,
) -> Dict[int, Any]:
    from s2u.utils.saver import get_body_pose, get_link_pose

    bullet = sim.world.p
    client_id = getattr(bullet, "_client", 0)
    transforms: Dict[int, Any] = {}
    source_ids = owner_ids if frame_source_link_ids is None else frame_source_link_ids
    if len(source_ids) != len(owner_ids):
        raise ValueError("owner IDs and frame-source IDs must have equal lengths")
    for owner_id, source_id in zip(owner_ids, source_ids):
        if source_id == -1:
            transforms[owner_id] = get_body_pose(sim.object.uid, client_id)
        else:
            transforms[owner_id] = get_link_pose(
                (sim.object.uid, source_id),
                client_id,
            )
    return transforms


def sampled_points_in_owner_frames(
    points_world: np.ndarray,
    part_ids: np.ndarray,
    ordered_owners: Sequence[int],
    owner_transforms: Mapping[int, Any],
) -> np.ndarray:
    local_points = np.zeros_like(points_world, dtype=np.float64)
    for part_id, owner_id in enumerate(ordered_owners):
        mask = part_ids == part_id
        if not np.any(mask):
            continue
        local_points[mask] = inverse_transform_points(
            owner_transforms[owner_id],
            points_world[mask],
        )
    return local_points


def mesh_vertices_from_path(path: Path) -> np.ndarray:
    import trimesh

    mesh = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [
            geometry
            for geometry in mesh.geometry.values()
            if len(geometry.vertices) > 0
        ]
        if not meshes:
            return np.empty((0, 3), dtype=np.float64)
        return np.concatenate(
            [np.asarray(geometry.vertices, dtype=np.float64) for geometry in meshes],
            axis=0,
        )
    return np.asarray(mesh.vertices, dtype=np.float64)


def primitive_bbox_points(geometry_type: int, dimensions: np.ndarray) -> np.ndarray:
    import pybullet

    dimensions = np.asarray(dimensions, dtype=np.float64)
    if geometry_type == pybullet.GEOM_BOX:
        if dimensions.size != 3:
            return np.empty((0, 3), dtype=np.float64)
        half = dimensions.reshape(3) * 0.5
    elif geometry_type == pybullet.GEOM_SPHERE:
        radius = float(dimensions.reshape(-1)[0]) if dimensions.size else 0.0
        half = np.array([radius, radius, radius], dtype=np.float64)
    elif geometry_type == pybullet.GEOM_CYLINDER:
        values = dimensions.reshape(-1)
        if values.size < 2:
            return np.empty((0, 3), dtype=np.float64)
        radius = float(values[1])
        height = float(values[0])
        half = np.array([radius, radius, height * 0.5], dtype=np.float64)
    else:
        return np.empty((0, 3), dtype=np.float64)

    signs = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    return signs * half.reshape(1, 3)


def visual_geometry_points(visual: Tuple[Any, ...], object_path: Path) -> np.ndarray:
    import pybullet

    geometry_type = int(visual[2])
    dimensions = np.asarray(visual[3], dtype=np.float64)
    mesh_name = visual[4].decode("utf-8") if isinstance(visual[4], bytes) else visual[4]

    if geometry_type == pybullet.GEOM_MESH and mesh_name:
        mesh_path = Path(mesh_name)
        if not mesh_path.is_absolute():
            mesh_path = object_path.parent / mesh_path
        vertices = mesh_vertices_from_path(mesh_path)
        if len(vertices) == 0:
            return vertices
        if dimensions.size == 3:
            vertices = vertices * dimensions.reshape(1, 3)
        elif dimensions.size == 1:
            vertices = vertices * float(dimensions[0])
        return vertices

    return primitive_bbox_points(geometry_type, dimensions)


def collect_visual_reference_points(
    sim: Any,
    structure: Mapping[str, Any],
    owner_transforms: Mapping[int, Any],
    object_path: Path,
    mode: str,
) -> List[List[np.ndarray]]:
    from s2u.utils.saver import get_mesh_pose

    bullet = sim.world.p
    client_id = getattr(bullet, "_client", 0)
    num_parts = len(structure["ordered_owners"])
    points_by_part: List[List[np.ndarray]] = [[] for _ in range(num_parts)]
    base_transform = owner_transforms[-1]

    for visual in bullet.getVisualShapeData(sim.object.uid):
        source_link = int(visual[1])
        if source_link not in structure["source_link_to_part"]:
            continue
        part_id = int(structure["source_link_to_part"][source_link])
        geometry_points = visual_geometry_points(visual, object_path)
        if len(geometry_points) == 0:
            continue

        _, _, _, visual_world_transform = get_mesh_pose(visual, client_id)
        points_world = transform_points(visual_world_transform, geometry_points)

        if mode == "part":
            owner_id = int(structure["ordered_owners"][part_id])
            points = inverse_transform_points(owner_transforms[owner_id], points_world)
        elif mode == "global":
            points = inverse_transform_points(base_transform, points_world)
        else:
            raise ValueError(f"Unknown reference point mode: {mode}")
        points_by_part[part_id].append(points)

    return points_by_part


def reset_movable_joints_to_rest(
    sim: Any,
    all_joint_info: Mapping[int, Tuple[Any, ...]],
) -> None:
    bullet = sim.world.p
    for joint_id, joint_data in all_joint_info.items():
        if int(joint_data[2]) in particulate.SUPPORTED_MOVABLE_TYPES:
            bullet.resetJointState(sim.object.uid, joint_id, 0.0)


def concatenate_or_none(chunks: Sequence[np.ndarray]) -> np.ndarray | None:
    valid = [chunk for chunk in chunks if chunk is not None and len(chunk) > 0]
    if not valid:
        return None
    return np.concatenate(valid, axis=0)


def build_nocs_targets(
    sim: Any,
    all_joint_info: Mapping[int, Tuple[Any, ...]],
    structure: Mapping[str, Any],
    points_world: np.ndarray,
    part_ids: np.ndarray,
) -> Dict[str, np.ndarray]:
    ordered_owners = list(structure["ordered_owners"])
    frame_source_link_ids = list(
        structure.get("frame_source_link_ids", ordered_owners)
    )
    object_path = Path(sim.object_urdfs[sim.object_idx])
    current_transforms = get_owner_world_transforms(
        sim,
        ordered_owners,
        frame_source_link_ids,
    )
    part_coords = sampled_points_in_owner_frames(
        points_world,
        part_ids,
        ordered_owners,
        current_transforms,
    )
    part_reference = collect_visual_reference_points(
        sim,
        structure,
        current_transforms,
        object_path,
        mode="part",
    )

    bullet = sim.world.p
    state_id = bullet.saveState()
    try:
        reset_movable_joints_to_rest(sim, all_joint_info)
        rest_transforms = get_owner_world_transforms(
            sim,
            ordered_owners,
            frame_source_link_ids,
        )
        global_reference = collect_visual_reference_points(
            sim,
            structure,
            rest_transforms,
            object_path,
            mode="global",
        )
    finally:
        bullet.restoreState(state_id)
        bullet.removeState(state_id)

    global_coords = np.zeros_like(points_world, dtype=np.float64)
    base_rest_transform = rest_transforms[-1]
    base_current_transform = current_transforms[-1]
    for part_id, owner_id in enumerate(ordered_owners):
        mask = part_ids == part_id
        if not np.any(mask):
            continue
        points_rest_world = transform_points(
            rest_transforms[owner_id],
            part_coords[mask],
        )
        global_coords[mask] = inverse_transform_points(
            base_rest_transform,
            points_rest_world,
        )

    nocs_p = np.zeros_like(points_world, dtype=np.float64)
    part_factors = np.zeros(len(ordered_owners), dtype=np.float32)
    part_bbox_min = np.zeros((len(ordered_owners), 3), dtype=np.float32)
    part_bbox_max = np.zeros((len(ordered_owners), 3), dtype=np.float32)
    for part_id in range(len(ordered_owners)):
        mask = part_ids == part_id
        reference_points = concatenate_or_none(part_reference[part_id])
        if reference_points is None:
            reference_points = part_coords[mask]
        factor, bbox_min, bbox_max = nocs_factor_and_corners(reference_points)
        nocs_p[mask] = normalize_nocs(
            part_coords[mask],
            factor,
            bbox_min,
            bbox_max,
        )
        part_factors[part_id] = factor
        part_bbox_min[part_id] = bbox_min.reshape(3)
        part_bbox_max[part_id] = bbox_max.reshape(3)

    global_reference_points = concatenate_or_none(
        [
            concatenate_or_none(part_chunks)
            for part_chunks in global_reference
        ]
    )
    if global_reference_points is None:
        global_reference_points = global_coords
    global_factor, global_bbox_min, global_bbox_max = nocs_factor_and_corners(
        global_reference_points
    )
    nocs_g = normalize_nocs(
        global_coords,
        global_factor,
        global_bbox_min,
        global_bbox_max,
    )

    return {
        "nocs_p": nocs_p.astype(np.float32),
        "nocs_g": nocs_g.astype(np.float32),
        "nocs_part_coords": part_coords.astype(np.float32),
        "nocs_global_coords": global_coords.astype(np.float32),
        "nocs_part_factor": part_factors,
        "nocs_part_bbox_min": part_bbox_min,
        "nocs_part_bbox_max": part_bbox_max,
        "nocs_global_factor": np.asarray(global_factor, dtype=np.float32),
        "nocs_global_bbox_min": global_bbox_min.reshape(3).astype(np.float32),
        "nocs_global_bbox_max": global_bbox_max.reshape(3).astype(np.float32),
        "urdf_frame_source_link_ids": np.asarray(
            frame_source_link_ids,
            dtype=np.int16,
        ),
        "urdf_object_current_frame": transform_matrix(base_current_transform),
        "urdf_link_current_frames": transform_matrix_array(
            current_transforms,
            ordered_owners,
        ),
        "urdf_link_current_frames_object": relative_transform_matrix_array(
            base_current_transform,
            current_transforms,
            ordered_owners,
        ),
        "urdf_object_rest_frame": transform_matrix(base_rest_transform),
        "urdf_link_rest_frames": transform_matrix_array(
            rest_transforms,
            ordered_owners,
        ),
        "urdf_link_rest_frames_object": relative_transform_matrix_array(
            base_rest_transform,
            rest_transforms,
            ordered_owners,
        ),
    }


def collect_part_meshes(
    sim: Any,
    structure: Mapping[str, Any],
    object_path: Path,
    center: np.ndarray,
    scale: float,
) -> Dict[str, np.ndarray]:
    """Collect per-part mesh vertices (posed, normalized) and triangle faces.

    Vertices are in the same normalized coordinate space as the point cloud:
      verts_normalized = (verts_world - center) * scale
    Use normalization_center and normalization_scale from the sample to recover
    world coordinates.

    Output arrays use a concatenated-plus-offsets layout:
      mesh_vertices[mesh_vertex_offsets[i] : mesh_vertex_offsets[i+1]]  → part i verts
      mesh_faces[mesh_face_offsets[i]   : mesh_face_offsets[i+1]]     → part i faces
    Face indices are local to each part's vertex block (zero-based per part).
    """
    import trimesh
    import pybullet
    from s2u.utils.saver import get_mesh_pose

    bullet = sim.world.p
    client_id = getattr(bullet, "_client", 0)
    num_parts = len(structure["ordered_owners"])
    part_verts: List[List[np.ndarray]] = [[] for _ in range(num_parts)]
    part_faces: List[List[np.ndarray]] = [[] for _ in range(num_parts)]

    for visual in bullet.getVisualShapeData(sim.object.uid):
        source_link = int(visual[1])
        if source_link not in structure["source_link_to_part"]:
            continue
        part_id = int(structure["source_link_to_part"][source_link])
        geometry_type = int(visual[2])
        dimensions = np.asarray(visual[3], dtype=np.float64)
        mesh_name = visual[4].decode("utf-8") if isinstance(visual[4], bytes) else visual[4]

        verts: np.ndarray | None = None
        faces: np.ndarray | None = None
        if geometry_type == pybullet.GEOM_MESH and mesh_name:
            mesh_path = Path(mesh_name)
            if not mesh_path.is_absolute():
                mesh_path = object_path.parent / mesh_name
            loaded = trimesh.load(str(mesh_path), force="mesh", process=False)
            if isinstance(loaded, trimesh.Scene):
                parts = [g for g in loaded.geometry.values() if len(g.vertices) > 0]
                loaded = trimesh.util.concatenate(parts) if parts else None
            if loaded is not None and len(loaded.vertices) > 0:
                verts = np.asarray(loaded.vertices, dtype=np.float64)
                faces = np.asarray(loaded.faces, dtype=np.int32)
                if dimensions.size == 3:
                    verts = verts * dimensions.reshape(1, 3)
                elif dimensions.size == 1:
                    verts = verts * float(dimensions[0])

        if verts is None or len(verts) == 0:
            continue

        _, _, _, visual_world_transform = get_mesh_pose(visual, client_id)
        verts_world = transform_points(visual_world_transform, verts)
        verts_normalized = (verts_world - center.reshape(1, 3)) * scale

        face_offset = sum(len(v) for v in part_verts[part_id])
        part_verts[part_id].append(verts_normalized)
        if faces is not None:
            part_faces[part_id].append(faces + face_offset)

    all_v_chunks: List[np.ndarray] = []
    all_f_chunks: List[np.ndarray] = []
    v_offsets = [0]
    f_offsets = [0]
    for pid in range(num_parts):
        if part_verts[pid]:
            v = np.concatenate(part_verts[pid], axis=0)
            all_v_chunks.append(v)
            v_offsets.append(v_offsets[-1] + len(v))
        else:
            v_offsets.append(v_offsets[-1])
        if part_faces[pid]:
            f = np.concatenate(part_faces[pid], axis=0)
            all_f_chunks.append(f)
            f_offsets.append(f_offsets[-1] + len(f))
        else:
            f_offsets.append(f_offsets[-1])

    all_v = (
        np.concatenate(all_v_chunks, axis=0).astype(np.float32)
        if all_v_chunks
        else np.empty((0, 3), dtype=np.float32)
    )
    all_f = (
        np.concatenate(all_f_chunks, axis=0).astype(np.int32)
        if all_f_chunks
        else np.empty((0, 3), dtype=np.int32)
    )
    return {
        "mesh_vertices": all_v,
        "mesh_faces": all_f,
        "mesh_vertex_offsets": np.asarray(v_offsets, dtype=np.int32),
        "mesh_face_offsets": np.asarray(f_offsets, dtype=np.int32),
    }


def _visual_world_mesh(
    visual: Tuple[Any, ...],
    object_path: Path,
    client_id: int,
) -> Any | None:
    """Build a world-posed ``trimesh`` for one PyBullet visual (mesh or primitive).

    Returns ``None`` if the visual has no usable surface geometry.  Primitive
    dimensions follow PyBullet's ``getVisualShapeData`` convention: box = full
    extents ``[x, y, z]``; cylinder = ``[length, radius, 0]``; sphere =
    ``[radius, ...]``.
    """
    import trimesh
    import pybullet
    from s2u.utils.saver import get_mesh_pose

    geometry_type = int(visual[2])
    dimensions = np.asarray(visual[3], dtype=np.float64)
    mesh_name = visual[4].decode("utf-8") if isinstance(visual[4], bytes) else visual[4]

    mesh = None
    if geometry_type == pybullet.GEOM_MESH and mesh_name:
        mesh_path = Path(mesh_name)
        if not mesh_path.is_absolute():
            mesh_path = object_path.parent / mesh_path
        loaded = trimesh.load(str(mesh_path), force="mesh", process=False)
        if isinstance(loaded, trimesh.Scene):
            parts = [g for g in loaded.geometry.values() if len(g.vertices) > 0]
            loaded = trimesh.util.concatenate(parts) if parts else None
        if loaded is not None and len(loaded.vertices) > 0:
            mesh = loaded
            if dimensions.size == 3:
                mesh.apply_scale(dimensions.reshape(3))
            elif dimensions.size == 1:
                mesh.apply_scale(float(dimensions[0]))
    elif geometry_type == pybullet.GEOM_BOX:
        mesh = trimesh.creation.box(extents=dimensions.reshape(-1)[:3])
    elif geometry_type == pybullet.GEOM_SPHERE:
        mesh = trimesh.creation.icosphere(radius=float(dimensions.reshape(-1)[0]))
    elif geometry_type == pybullet.GEOM_CYLINDER:
        values = dimensions.reshape(-1)
        mesh = trimesh.creation.cylinder(
            radius=float(values[1]), height=float(values[0])
        )

    if mesh is None or len(mesh.vertices) == 0:
        return None
    _, _, _, visual_world_transform = get_mesh_pose(visual, client_id)
    mesh.apply_transform(np.asarray(visual_world_transform.as_matrix(), dtype=np.float64))
    return mesh


def sample_part_labeled_point_cloud(
    sim: Any,
    part_source_links: Mapping[int, Sequence[int]],
    total_points: int,
    min_points_per_part: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop-in replacement for ``acquire_part_labeled_point_cloud`` that samples
    the point cloud directly from the part surface meshes rather than fusing
    multi-view depth scans.

    Points are sampled at the object's *current* joint pose, area-weighted across
    parts (with a per-part minimum), and labelled with the pipeline part index.
    Returns ``(points_world, labels, normals)`` with the same contract as the
    scanning acquisition, so the downstream stratified downsample / normalization
    / NOCS logic is unchanged.
    """
    import trimesh

    bullet = sim.world.p
    client_id = getattr(bullet, "_client", 0)
    object_path = Path(sim.object_urdfs[sim.object_idx])

    part_ids = sorted(part_source_links)  # part indices, in pipeline order
    row_of_link: Dict[int, int] = {}
    for row, part_id in enumerate(part_ids):
        for link in part_source_links[part_id]:
            row_of_link[int(link)] = row

    meshes_by_row: List[List[Any]] = [[] for _ in part_ids]
    for visual in bullet.getVisualShapeData(sim.object.uid):
        row = row_of_link.get(int(visual[1]))
        if row is None:
            continue
        mesh = _visual_world_mesh(visual, object_path, client_id)
        if mesh is not None and len(mesh.faces) > 0:
            meshes_by_row[row].append(mesh)

    combined: List[Any] = []
    areas = np.zeros(len(part_ids), dtype=np.float64)
    for row in range(len(part_ids)):
        if meshes_by_row[row]:
            mesh = trimesh.util.concatenate(meshes_by_row[row])
            combined.append(mesh)
            areas[row] = float(getattr(mesh, "area", 0.0))
        else:
            combined.append(None)

    total_area = float(areas.sum())
    if total_area <= 0.0:
        raise particulate.SampleGenerationError(
            "Point sampling found no surface geometry on the object"
        )

    points_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []
    normal_chunks: List[np.ndarray] = []
    for row in range(len(part_ids)):
        mesh = combined[row]
        if mesh is None or areas[row] <= 0.0:
            raise particulate.SampleGenerationError(
                f"Part {row} has no surface geometry to sample"
            )
        count = max(
            int(min_points_per_part),
            int(round(total_points * areas[row] / total_area)),
        )
        try:
            points_local, face_indices = trimesh.sample.sample_surface(
                mesh, count, seed=row
            )
        except TypeError:  # older trimesh without the seed kwarg
            points_local, face_indices = trimesh.sample.sample_surface(mesh, count)
        points_chunks.append(np.asarray(points_local, dtype=np.float64))
        label_chunks.append(np.full(len(points_local), row, dtype=np.int16))
        normal_chunks.append(
            np.asarray(mesh.face_normals[face_indices], dtype=np.float64)
        )

    points = np.concatenate(points_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)
    normals = np.concatenate(normal_chunks, axis=0)
    return points, labels, normals


def _make_sampling_acquire(total_points: int, min_points_per_part: int):
    """Return an ``acquire_part_labeled_point_cloud``-compatible callable that
    samples surface points instead of scanning (scan-only args are ignored)."""

    def acquire(
        sim: Any,
        part_source_links: Mapping[int, Sequence[int]],
        num_views: int = 0,
        tsdf_resolution: int = 0,
        min_segmentation_score: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return sample_part_labeled_point_cloud(
            sim, part_source_links, total_points, min_points_per_part
        )

    return acquire


def validate_nocs(sample: Mapping[str, np.ndarray]) -> None:
    points = sample["points"]
    for key in ("nocs_p", "nocs_g"):
        value = sample[key]
        if value.shape != points.shape:
            raise particulate.SampleGenerationError(
                f"{key} must have shape {points.shape}, got {value.shape}"
            )
        if not np.isfinite(value).all():
            raise particulate.SampleGenerationError(f"{key} contains non-finite values")

    num_parts = int(sample["num_valid_parts"])
    frame_keys = (
        "urdf_link_current_frames",
        "urdf_link_current_frames_object",
        "urdf_link_rest_frames",
        "urdf_link_rest_frames_object",
    )
    for key in frame_keys:
        value = sample[key]
        if value.shape != (num_parts, 4, 4):
            raise particulate.SampleGenerationError(
                f"{key} must have shape {(num_parts, 4, 4)}, got {value.shape}"
            )
        if not np.isfinite(value).all():
            raise particulate.SampleGenerationError(f"{key} contains non-finite values")

    for key in ("urdf_object_current_frame", "urdf_object_rest_frame"):
        value = sample[key]
        if value.shape != (4, 4):
            raise particulate.SampleGenerationError(
                f"{key} must have shape {(4, 4)}, got {value.shape}"
            )
        if not np.isfinite(value).all():
            raise particulate.SampleGenerationError(f"{key} contains non-finite values")


def create_training_sample(
    sim: Any,
    args: Any,
    rng: np.random.Generator,
    sample_index: int,
) -> Dict[str, np.ndarray]:
    all_joint_info = particulate.get_all_joint_info(sim)
    structure = particulate.build_part_structure(
        all_joint_info,
        max_parts=(
            len(all_joint_info) + 1
            if args.max_moving_joints is not None
            else args.max_parts
        ),
        require_serial=not args.allow_tree,
    )
    if args.max_moving_joints is not None:
        structure = particulate.limit_moving_joints(
            all_joint_info,
            structure,
            max_moving_joints=args.max_moving_joints,
            max_parts=args.max_parts,
        )

    sample = _BASE_CREATE_TRAINING_SAMPLE(sim, args, rng, sample_index)
    points_world = (
        sample["points"].astype(np.float64)
        / float(sample["normalization_scale"])
        + sample["normalization_center"].astype(np.float64).reshape(1, 3)
    )
    nocs = build_nocs_targets(
        sim,
        all_joint_info,
        structure,
        points_world,
        sample["part_ids"].astype(np.int16),
    )
    sample.update(nocs)
    sample["nocs_gt"] = sample["nocs_p"]
    sample["nocs_gt_g"] = sample["nocs_g"]
    validate_nocs(sample)
    if getattr(args, "save_mesh", False):
        sample.update(
            collect_part_meshes(
                sim,
                structure,
                Path(sim.object_urdfs[sim.object_idx]),
                sample["normalization_center"].astype(np.float64),
                float(sample["normalization_scale"]),
            )
        )
    return sample


def schema_description() -> Dict[str, str]:
    schema = _BASE_SCHEMA_DESCRIPTION()
    schema.update(
        {
            "nocs_p": "(N,3) float32 part NOCS coordinates",
            "nocs_g": "(N,3) float32 global/rest-object NOCS coordinates",
            "nocs_gt": "Alias of nocs_p for tools/dataset.py naming compatibility",
            "nocs_gt_g": "Alias of nocs_g for tools/dataset.py naming compatibility",
            "nocs_part_coords": "(N,3) float32 coordinates before part NOCS normalization",
            "nocs_global_coords": "(N,3) float32 coordinates before global NOCS normalization",
            "nocs_part_factor": "(P,) float32 per-part NOCS normalization factors",
            "nocs_global_factor": "scalar float32 global NOCS normalization factor",
            "nocs_part_bbox_min": "(P,3) float32 per-part bbox minima",
            "nocs_part_bbox_max": "(P,3) float32 per-part bbox maxima",
            "nocs_global_bbox_min": "(3,) float32 global/rest bbox minimum",
            "nocs_global_bbox_max": "(3,) float32 global/rest bbox maximum",
            "urdf_frame_source_link_ids": (
                "(P,) int16 PyBullet source link ID for each stored frame; "
                "normally -1 is the base frame, while an articraft_global "
                "wrapper uses its fixed original-base child ID"
            ),
            "urdf_object_current_frame": (
                "(4,4) float32 transform from object/base canonical frame to "
                "current world frame"
            ),
            "urdf_link_current_frames": (
                "(P,4,4) float32 transforms from link canonical frames to "
                "current world frame"
            ),
            "urdf_link_current_frames_object": (
                "(P,4,4) float32 current link canonical frames expressed in "
                "the current object/base frame"
            ),
            "urdf_object_rest_frame": (
                "(4,4) float32 transform from object/base canonical frame to "
                "world after resetting movable URDF joints to zero"
            ),
            "urdf_link_rest_frames": (
                "(P,4,4) float32 transforms from link canonical frames to "
                "world after resetting movable URDF joints to zero"
            ),
            "urdf_link_rest_frames_object": (
                "(P,4,4) float32 rest link canonical frames expressed in the "
                "rest object/base canonical frame"
            ),
        }
    )
    return schema


def _staging_arg_parser() -> argparse.ArgumentParser:
    """Options that select and stage Articraft-10K ``fit`` objects.

    These mirror ``generate_particulate_pointcloud_twopose_data.py`` and are
    consumed here (not by the base single-pose generator), so they are stripped
    from ``sys.argv`` before ``particulate.main()`` parses the rest.
    """

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--articraft-root",
        type=Path,
        default=Path("~/Articraft-10K"),
        help="Root of the Articraft-10K dataset",
    )
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        help="Sub-category name whose 'fit' objects are used (repeatable)",
    )
    parser.add_argument(
        "--fit-dir",
        action="append",
        default=[],
        help="Explicit path to a 'fit' folder (repeatable)",
    )
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="Use every 'fit' folder found under --articraft-root",
    )
    parser.add_argument(
        "--dataset-type",
        choices=("train", "val", "test"),
        default="train",
        help="Dataset type (used to select categories when --all-categories)",
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=None,
        help="Where to extract objects (default: <root>/staging)",
    )
    parser.add_argument(
        "--restage",
        action="store_true",
        help="Re-extract archives and recompute bounding boxes even if staged",
    )
    parser.add_argument(
        "--bbox-pose-samples",
        type=int,
        default=16,
        help=(
            "Random joint configurations (plus rest + both extremes) unioned "
            "when synthesizing each object's bounding box. 0 = rest only"
        ),
    )
    parser.add_argument(
        "--point-sampling",
        action="store_true",
        help=(
            "Sample the point cloud directly from part surface meshes instead of "
            "multi-view depth scanning + TSDF fusion"
        ),
    )
    parser.add_argument(
        "--sampling-points",
        type=int,
        default=50000,
        help=(
            "Total surface points sampled (area-weighted across parts) when "
            "--point-sampling; downsampled to --num-points afterwards"
        ),
    )
    parser.add_argument(
        "--sampling-min-per-part",
        type=int,
        default=512,
        help="Minimum surface points sampled per part when --point-sampling",
    )
    return parser


def _peek_base_options(remaining: Sequence[str]) -> argparse.Namespace:
    """Read the base-generator options staging needs, without consuming them."""

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("root", type=Path)
    parser.add_argument("--object-set", default="articraft_fit")
    parser.add_argument(
        "--sim-src",
        type=Path,
        default=Path("~/Articulated_object_simulation-main/src"),
    )
    parser.add_argument("--pos-rot", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--is-syn",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    known, _ = parser.parse_known_args(list(remaining))
    return known


def _stage_fit_objects(
    staging_args: argparse.Namespace,
    remaining: List[str],
) -> List[str]:
    """Stage the selected Articraft ``fit`` objects and point the base generator
    at the staged object set. Returns the updated base-generator argv."""

    base_opts = _peek_base_options(remaining)
    sim_src = base_opts.sim_src.expanduser().resolve()
    object_set = base_opts.object_set
    root = base_opts.root.expanduser().resolve()
    staging_root = staging_args.staging_root or (root / "staging")
    staging_root = staging_root.expanduser().resolve()
    articraft_root = staging_args.articraft_root.expanduser().resolve()
    articraft_root = staging_args.articraft_root
    print(f"articraft_root = {staging_args.articraft_root}", flush=True)
    is_syn = base_opts.is_syn
    if is_syn is None:
        is_syn = "syn" in object_set.lower()

    fit_dirs = twopose.collect_fit_dirs(
        articraft_root,
        staging_args.category,
        staging_args.fit_dir,
        staging_args.all_categories,
        staging_args.dataset_type,
    )
    print(f"Staging objects from {len(fit_dirs)} 'fit' folder(s)...", flush=True)
    object_set_dir, _ = twopose.stage_objects(
        fit_dirs,
        staging_root,
        object_set,
        sim_src,
        overwrite=staging_args.restage,
        bbox_options={
            "is_syn": is_syn,
            "pos_rot": bool(base_opts.pos_rot),
            "num_pose_samples": staging_args.bbox_pose_samples,
        },
    )
    staged_count = len([p for p in object_set_dir.iterdir() if p.is_dir()])
    print(f"Staged {staged_count} objects into {object_set_dir}", flush=True)

    # Append the staged locations so they override any earlier user values when
    # the base generator re-parses the argv (argparse keeps the last occurrence).
    return remaining + [
        "--object-set",
        object_set,
        "--urdf-root",
        str(staging_root),
    ]


def _parse_args_with_nocs_runtime() -> argparse.Namespace:
    """Attach NOCS-only acquisition settings to the worker namespace."""
    args = _BASE_PARSE_ARGS()
    args.point_sampling = _RUNTIME_POINT_SAMPLING
    args.sampling_points = _RUNTIME_SAMPLING_POINTS
    args.sampling_min_per_part = _RUNTIME_SAMPLING_MIN_PER_PART
    return args


def generate_nocs_worker(
    args: argparse.Namespace,
    worker_id: int,
    sample_indices: Sequence[int],
) -> List[str]:
    """Reinstall NOCS hooks inside spawned workers before base generation."""
    particulate.create_training_sample = create_training_sample
    particulate.schema_description = schema_description
    if args.point_sampling:
        particulate.acquire_part_labeled_point_cloud = _make_sampling_acquire(
            args.sampling_points,
            args.sampling_min_per_part,
        )
    return _BASE_GENERATE_WORKER(args, worker_id, sample_indices)


def main() -> None:
    global _RUNTIME_POINT_SAMPLING
    global _RUNTIME_SAMPLING_POINTS
    global _RUNTIME_SAMPLING_MIN_PER_PART

    particulate.create_training_sample = create_training_sample
    particulate.schema_description = schema_description
    particulate.generate_worker = generate_nocs_worker
    particulate.parse_args = _parse_args_with_nocs_runtime

    staging_args, remaining = _staging_arg_parser().parse_known_args()

    _RUNTIME_POINT_SAMPLING = staging_args.point_sampling
    _RUNTIME_SAMPLING_POINTS = staging_args.sampling_points
    _RUNTIME_SAMPLING_MIN_PER_PART = staging_args.sampling_min_per_part

    if staging_args.point_sampling:
        # Replace multi-view scanning with direct surface sampling.  Assigning the
        # module attribute handles the current process; generate_nocs_worker
        # reinstalls this hook in each spawned worker process.
        particulate.acquire_part_labeled_point_cloud = _make_sampling_acquire(
            staging_args.sampling_points,
            staging_args.sampling_min_per_part,
        )
        print(
            "Point cloud source: surface sampling "
            f"(~{staging_args.sampling_points} pts total, "
            f"min {staging_args.sampling_min_per_part}/part); "
            "multi-view scanning disabled.",
            flush=True,
        )

    use_fit = bool(
        staging_args.category
        or staging_args.fit_dir
        or staging_args.all_categories
    )
    if use_fit:
        remaining = _stage_fit_objects(staging_args, remaining)

    sys.argv = [sys.argv[0], *remaining]
    particulate.main()


if __name__ == "__main__":
    main()
