#!/usr/bin/env python3
"""Cluster Articraft URDFs by articulated structure and rest-state geometry.

The primary descriptor represents each rigid articulated segment in its own
pose-invariant geometric form.  It is compared together with the ordered
kinematic chain (R/P types, consecutive joint-axis relations, and normalized
pivot spacing).  Whole-object HOG at the URDF rest state is retained as a
lower-weight cue so that a cluster is also useful for choosing consistent
NOCS/NPCS rest-state conventions.

The three mixed distances are clustered with k-medoids rather than k-means.
This supports variable-length kinematic chains and makes every cluster centre
a real object that can be inspected in the verification GUI.  No depth
renderer, lighting, texture, or RGB appearance is used.

Example
-------
    conda run -n serial_arti python scripts/cluster_urdf_hog.py \
        --category articulated_task_lamp --clusters 3

Outputs are written under
``data/urdfs/HOGClusters/<category>/<sub-category>/k_<K>``:

* ``assignments.csv`` and ``assignments.json``;
* ``features.npz`` for later analysis;
* one three-view point-projection PNG per object; and
* one labelled contact sheet per cluster, ordered from the cluster medoid out.

Dependencies beyond the Articraft canonicalization stack are scikit-image and
Pillow.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# Allow ``python scripts/cluster_urdf_hog.py`` from any working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from PIL import Image, ImageDraw, ImageFont
    from skimage.feature import hog
except ImportError as exc:  # pragma: no cover - only exercised in incomplete envs
    raise SystemExit(
        "cluster_urdf_hog.py requires Pillow and scikit-image. "
        "Install them or run it in the serial_arti environment.\n"
        f"Missing dependency: {exc}"
    ) from exc

from articraft_canon import canonical_zero, dataset, geometry as geo, parse
from articraft_canon.dataset import ArchiveRef


LOGGER = logging.getLogger("cluster_urdf_hog")
_VIEW_PLANES = ((0, 1), (0, 2), (1, 2))  # XY, XZ, YZ
_VIEW_NAMES = ("XY", "XZ", "YZ")


@dataclass
class ObjectFeature:
    """Articulation-invariant and rest-layout descriptors for one object."""

    ref: ArchiveRef
    points_used: int
    rest_hog: np.ndarray
    segment_features: np.ndarray
    joint_features: np.ndarray
    projections: list[np.ndarray]
    canonical_fallback: bool = False
    mimic_joints: tuple[str, ...] = ()
    joint_signature: tuple[str, ...] = ()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster a category's URDFs using link-local geometry, kinematic "
            "structure, rest-state HOG, and k-medoids."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=_REPO_ROOT / "data/urdfs/Dataset",
        help="Articraft Dataset root (default: %(default)s).",
    )
    parser.add_argument(
        "--category",
        required=True,
        help="Top-level category folder to cluster, e.g. robotic_arm.",
    )
    parser.add_argument(
        "--sub-category",
        action="append",
        default=None,
        help="Optional leaf sub-category filter; repeat to select several.",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "val", "test"),
        default=None,
        help="Optional split filter; repeat to select several.",
    )
    parser.add_argument(
        "--include-not-fit",
        action="store_true",
        help="Include not-fit archives. By default only fit objects are used.",
    )
    parser.add_argument(
        "--clusters",
        "-k",
        type=int,
        required=True,
        help="Number of k-medoids clusters.",
    )
    parser.add_argument(
        "--rest-state",
        choices=("source", "canonical-zero"),
        default="source",
        help=(
            "Use the source URDF q=0 state (default), or first apply the "
            "project's automatic canonical-zero procedure."
        ),
    )
    parser.add_argument(
        "--unreachable",
        choices=("mid-limit", "clamp", "extend-limits"),
        default="mid-limit",
        help="Fallback policy used only with --rest-state canonical-zero.",
    )
    parser.add_argument(
        "--pose-normalization",
        choices=("pca", "source"),
        default="pca",
        help=(
            "PCA-align each rest cloud for global-pose invariance (default), "
            "or retain source URDF axes."
        ),
    )
    parser.add_argument(
        "--points-per-link",
        type=int,
        default=4096,
        help="Surface samples drawn while loading each link (default: %(default)s).",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=16000,
        help="Maximum whole-object samples used for projections (default: %(default)s).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Square occupancy projection size in pixels (default: %(default)s).",
    )
    parser.add_argument(
        "--splat-radius",
        type=int,
        default=1,
        help="Radius of each projected point in pixels (default: %(default)s).",
    )
    parser.add_argument(
        "--orientations",
        type=int,
        default=9,
        help="Number of unsigned HOG orientation bins (default: %(default)s).",
    )
    parser.add_argument(
        "--pixels-per-cell",
        type=int,
        default=8,
        help="HOG cell width/height in pixels (default: %(default)s).",
    )
    parser.add_argument(
        "--cells-per-block",
        type=int,
        default=2,
        help="HOG block width/height in cells (default: %(default)s).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Seed for point subsampling and k-medoids (default: %(default)s).",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=30,
        help="Independent k-medoids initializations (default: %(default)s).",
    )
    parser.add_argument(
        "--joint-signature-weight",
        type=float,
        default=None,
        help=(
            "Deprecated alias for --kinematic-weight, retained for old command "
            "lines. If provided, it overrides --kinematic-weight."
        ),
    )
    parser.add_argument(
        "--part-weight",
        type=float,
        default=0.55,
        help=(
            "Weight for articulation-invariant rigid-segment geometry "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--kinematic-weight",
        type=float,
        default=0.25,
        help="Weight for joint types, axes, and pivot layout (default: %(default)s).",
    )
    parser.add_argument(
        "--rest-hog-weight",
        type=float,
        default=0.20,
        help="Weight for whole-object rest-state HOG (default: %(default)s).",
    )
    parser.add_argument(
        "--shape-hist-bins",
        type=int,
        default=12,
        help=(
            "Bins in each per-segment radial and pair-distance histogram "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--d2-pairs",
        type=int,
        default=4096,
        help=(
            "Random point pairs used by each segment's D2 descriptor "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help=(
            "Maximum k-medoids update iterations per initialization "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--strict-serial",
        action="store_true",
        help="Reject fixed side branches instead of retaining their geometry.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N matching archives (useful for a quick test).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output folder. Default: data/urdfs/HOGClusters/<category>/"
            "<sub-category>/k_<K>."
        ),
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.clusters < 1:
        raise SystemExit("--clusters must be at least 1")
    if args.points_per_link < 16 or args.num_points < 16:
        raise SystemExit("point counts must be at least 16")
    if args.image_size < 16:
        raise SystemExit("--image-size must be at least 16")
    if args.splat_radius < 0:
        raise SystemExit("--splat-radius cannot be negative")
    if args.orientations < 2:
        raise SystemExit("--orientations must be at least 2")
    if args.pixels_per_cell < 1 or args.cells_per_block < 1:
        raise SystemExit("HOG cell/block sizes must be positive")
    if args.joint_signature_weight is not None:
        if args.joint_signature_weight < 0:
            raise SystemExit("--joint-signature-weight cannot be negative")
        args.kinematic_weight = args.joint_signature_weight
    if min(args.part_weight, args.kinematic_weight, args.rest_hog_weight) < 0:
        raise SystemExit("clustering weights cannot be negative")
    if args.part_weight + args.kinematic_weight + args.rest_hog_weight <= 0:
        raise SystemExit("at least one clustering weight must be positive")
    if args.shape_hist_bins < 2 or args.d2_pairs < 16:
        raise SystemExit("--shape-hist-bins must be >= 2 and --d2-pairs >= 16")
    if args.n_init < 1 or args.max_iterations < 1:
        raise SystemExit("--n-init and --max-iterations must be positive")
    cells = args.image_size // args.pixels_per_cell
    if cells < args.cells_per_block:
        raise SystemExit(
            "HOG configuration has fewer cells than --cells-per-block; "
            "increase --image-size or reduce the cell/block sizes"
        )


def _normalise_cloud(points: np.ndarray, pose_normalization: str) -> np.ndarray:
    """Centre and isotropically scale a cloud, optionally aligning PCA axes."""

    points = np.asarray(points, dtype=np.float64)
    centered = points - points.mean(axis=0, keepdims=True)
    if pose_normalization == "pca":
        _, axes, _ = geo.pca_axes(centered)
        centered = centered @ axes
    extent = np.ptp(centered, axis=0)
    scale = float(np.max(extent))
    if scale <= 1e-12:
        raise ValueError("degenerate point cloud with zero spatial extent")
    return centered / scale


def _occupancy_projection(
    points: np.ndarray,
    plane: tuple[int, int],
    *,
    image_size: int,
    splat_radius: int,
) -> np.ndarray:
    """Project points to a binary image without a depth buffer or renderer."""

    margin = max(2, int(round(0.06 * image_size)))
    usable = max(1, image_size - 1 - 2 * margin)
    uv = points[:, plane]
    # The cloud has maximum 3D extent 1 and is centred at zero. Preserve that
    # common isotropic scale in every view rather than fitting views separately.
    px = np.rint((uv[:, 0] + 0.5) * usable + margin).astype(np.int32)
    py = np.rint((0.5 - uv[:, 1]) * usable + margin).astype(np.int32)
    px = np.clip(px, 0, image_size - 1)
    py = np.clip(py, 0, image_size - 1)

    image = np.zeros((image_size, image_size), dtype=np.float32)
    for dy in range(-splat_radius, splat_radius + 1):
        yy = np.clip(py + dy, 0, image_size - 1)
        for dx in range(-splat_radius, splat_radius + 1):
            xx = np.clip(px + dx, 0, image_size - 1)
            image[yy, xx] = 1.0
    return image


def _hog_vector(image: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    return np.asarray(
        hog(
            image,
            orientations=args.orientations,
            pixels_per_cell=(args.pixels_per_cell, args.pixels_per_cell),
            cells_per_block=(args.cells_per_block, args.cells_per_block),
            block_norm="L2-Hys",
            feature_vector=True,
        ),
        dtype=np.float32,
    )


def _cloud_descriptor(
    points: np.ndarray, args: argparse.Namespace
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return pose-normalized multi-view HOG and display projections."""

    normalized = _normalise_cloud(points, args.pose_normalization)
    display_projections = [
        _occupancy_projection(
            normalized,
            plane,
            image_size=args.image_size,
            splat_radius=args.splat_radius,
        )
        for plane in _VIEW_PLANES
    ]

    if args.pose_normalization == "pca":
        # PCA eigenvectors have arbitrary signs. Averaging the descriptor over
        # all eight axis sign choices makes the result independent of those
        # signs (and invariant to mirroring caused solely by PCA ambiguity).
        variants: list[np.ndarray] = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    signed = normalized * np.array([sx, sy, sz])
                    view_features = []
                    for plane in _VIEW_PLANES:
                        image = _occupancy_projection(
                            signed,
                            plane,
                            image_size=args.image_size,
                            splat_radius=args.splat_radius,
                        )
                        view_features.append(_hog_vector(image, args))
                    variants.append(np.concatenate(view_features))
        feature = np.mean(np.stack(variants), axis=0)
    else:
        feature = np.concatenate(
            [_hog_vector(image, args) for image in display_projections]
        )

    norm = float(np.linalg.norm(feature))
    if norm > 1e-12:
        feature = feature / norm
    return feature.astype(np.float32), display_projections


def _histogram01(values: np.ndarray, bins: int) -> np.ndarray:
    """Return a unit-sum histogram on [0, 1], including degenerate inputs."""

    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.zeros(bins, dtype=np.float32)
    maximum = float(np.max(values))
    if maximum > 1e-12:
        values = values / maximum
    else:
        values = np.zeros_like(values)
    counts, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    counts = counts.astype(np.float64)
    counts /= max(float(counts.sum()), 1.0)
    return counts.astype(np.float32)


def _segment_shape_descriptor(
    points: np.ndarray,
    *,
    area_fraction: float,
    object_scale: float,
    args: argparse.Namespace,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Describe a rigid segment independently of its current world pose.

    PCA variance/extent ratios capture coarse shape, while radial and D2
    histograms retain more of the sampled surface distribution.  Only the
    final two scalars retain relative scale within the complete object.
    """

    bins = args.shape_hist_bins
    dimension = 3 + 3 + bins + bins + 2
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.zeros(dimension, dtype=np.float32)

    centered = points - points.mean(axis=0, keepdims=True)
    _, axes, eigvals = geo.pca_axes(centered)
    eigvals = np.maximum(eigvals, 0.0)
    variance_ratios = eigvals / max(float(eigvals.sum()), 1e-12)

    aligned = centered @ axes
    extents = np.sort(np.ptp(aligned, axis=0))[::-1]
    max_extent = float(np.max(extents))
    extent_ratios = extents / max(max_extent, 1e-12)

    radial = np.linalg.norm(centered, axis=1)
    radial_hist = _histogram01(radial, bins)

    pair_count = min(args.d2_pairs, max(len(points), 1) * 4)
    first = rng.randint(0, len(points), size=pair_count)
    second = rng.randint(0, len(points), size=pair_count)
    d2 = np.linalg.norm(points[first] - points[second], axis=1)
    d2_hist = _histogram01(d2, bins)

    # RMS radius is invariant to translation and rotation, unlike a world-axis
    # bounding box.  object_scale is also intrinsic to the complete object.
    segment_scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    relative_scale = segment_scale / max(object_scale, 1e-12)
    return np.concatenate(
        [
            variance_ratios,
            extent_ratios,
            radial_hist,
            d2_hist,
            np.asarray([relative_scale, area_fraction]),
        ]
    ).astype(np.float32)


def _articulated_descriptors(
    model: parse.ObjectModel,
    *,
    mimic_joints: Sequence[str],
    args: argparse.Namespace,
    object_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ordered rigid-segment and joint-relation descriptors.

    Links connected by fixed joints are merged into one rigid segment.  The
    segment clouds are assembled at q=0 only to resolve arbitrary URDF link
    frames; their descriptors are subsequently translation/rotation invariant.
    """

    world = geo.link_world_transforms(model.base_link, model.joints, {})
    segment_for_link = {model.base_link: 0}
    next_segment = 1
    for joint in model.joints:
        if joint.is_moving:
            segment_for_link[joint.child] = next_segment
            next_segment += 1
        else:
            segment_for_link[joint.child] = segment_for_link[joint.parent]

    segment_clouds: list[list[np.ndarray]] = [[] for _ in range(next_segment)]
    segment_areas = np.zeros(next_segment, dtype=np.float64)
    for link_name, link in model.links.items():
        segment_index = segment_for_link.get(link_name)
        transform = world.get(link_name)
        if segment_index is None or transform is None:
            continue
        if link.points.size:
            segment_clouds[segment_index].append(
                geo.transform_points(transform, link.points)
            )
        if link.mesh is not None:
            segment_areas[segment_index] += max(float(link.mesh.area), 0.0)

    clouds_by_segment = [
        np.concatenate(clouds, axis=0) if clouds else np.zeros((0, 3))
        for clouds in segment_clouds
    ]
    segment_scales = []
    for cloud in clouds_by_segment:
        if len(cloud):
            centered = cloud - cloud.mean(axis=0, keepdims=True)
            segment_scales.append(
                float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
            )
        else:
            segment_scales.append(0.0)
    total_area = float(segment_areas.sum())
    # sqrt(surface area) is a length scale that is invariant to articulation
    # and to how geometry is distributed among fixed-connected URDF links.
    # Fall back to summed segment RMS radii for point-only geometry.
    object_scale = (
        float(np.sqrt(total_area))
        if total_area > 1e-12
        else float(sum(segment_scales))
    )
    rng = np.random.RandomState(object_seed + 104729)
    shape_features = []
    for index, cloud in enumerate(clouds_by_segment):
        area_fraction = (
            float(segment_areas[index] / total_area) if total_area > 1e-12 else 0.0
        )
        shape_features.append(
            _segment_shape_descriptor(
                cloud,
                area_fraction=area_fraction,
                object_scale=object_scale,
                args=args,
                rng=rng,
            )
        )

    # Joint axes and origins are expressed in the base frame at q=0.  Absolute
    # directions are intentionally discarded; consecutive |dot| values survive
    # global rotation and joint-axis sign conventions.
    mimic_set = set(mimic_joints)
    joint_features = []
    previous_axis: np.ndarray | None = None
    previous_pivot: np.ndarray | None = None
    previous_was_prismatic = False
    for joint in model.moving_joints:
        child_transform = world[joint.child]
        axis = child_transform[:3, :3] @ joint.axis
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        pivot = child_transform[:3, 3]
        has_previous = previous_axis is not None
        axis_relation = (
            abs(float(np.dot(previous_axis, axis))) if has_previous else 0.0
        )
        pivot_delta = pivot - previous_pivot if previous_pivot is not None else None
        # Sliding the previous prismatic joint changes only the component along
        # its axis, so discard that component to keep the spacing descriptor
        # independent of extension state.  A revolute joint preserves the full
        # distance from its pivot under articulation.
        if pivot_delta is not None and previous_was_prismatic:
            pivot_delta = (
                pivot_delta - np.dot(pivot_delta, previous_axis) * previous_axis
            )
        pivot_spacing = (
            float(np.linalg.norm(pivot_delta)) / max(object_scale, 1e-12)
            if pivot_delta is not None
            else 0.0
        )
        joint_features.append(
            np.asarray(
                [
                    float(joint.type in ("revolute", "continuous")),
                    float(joint.type == "prismatic"),
                    float(has_previous),
                    axis_relation,
                    pivot_spacing,
                    float(joint.name in mimic_set),
                ],
                dtype=np.float32,
            )
        )
        previous_axis = axis
        previous_pivot = pivot
        previous_was_prismatic = joint.type == "prismatic"

    joint_array = (
        np.stack(joint_features)
        if joint_features
        else np.zeros((0, 6), dtype=np.float32)
    )
    return np.stack(shape_features), joint_array


def _load_feature(
    ref: ArchiveRef,
    *,
    work_dir: Path,
    args: argparse.Namespace,
    object_seed: int,
) -> ObjectFeature:
    urdf_path = dataset.extract_archive(ref, work_dir)
    if urdf_path is None:
        raise ValueError("archive contains no URDF")
    model = parse.load_object(
        urdf_path,
        object_id=ref.object_id,
        source_archive=ref.archive,
        category=ref.category,
        sub_category=ref.sub_category,
        split=ref.split,
        n_points_per_link=args.points_per_link,
        seed=object_seed,
        allow_fixed_branches=not args.strict_serial,
        # At source q=0, mimic joints need no special FK treatment: their
        # default value is zero just like the driving joint. Keeping them in
        # the chain ensures that all descendant link geometry is sampled.
        allow_mimic_joints=True,
        # canonical-zero requires independent variables.  Conversion affects
        # only this extracted temporary URDF and mirrors articraft_verify's
        # save-time policy; the source dataset archive is never modified.
        convert_mimic_to_independent=args.rest_state == "canonical-zero",
    )
    if model.skip_reason:
        raise ValueError(model.skip_reason)

    mimic_joints = tuple(
        [
            joint.name
            for joint in model.urdf.robot.joints
            if joint.mimic is not None
        ]
        + [str(record["joint"]) for record in model.converted_mimic_joints]
    )
    joint_signature = tuple(
        "R" if joint.type in ("revolute", "continuous") else "P"
        for joint in model.moving_joints
    )
    fallback = False
    if args.rest_state == "canonical-zero":
        result = canonical_zero.canonicalize_zero(
            model,
            check_collision=False,
            unreachable_policy=args.unreachable,
            apply=True,
        )
        fallback = result.fallback

    all_points = model.object_points()
    if len(all_points) == 0:
        raise ValueError("object has no sampled visual geometry")
    segment_features, joint_features = _articulated_descriptors(
        model,
        mimic_joints=mimic_joints,
        args=args,
        object_seed=object_seed,
    )
    points = geo.farthest_point_subsample(
        all_points, args.num_points, seed=object_seed
    )
    rest_hog, projections = _cloud_descriptor(points, args)
    return ObjectFeature(
        ref,
        len(points),
        rest_hog,
        segment_features,
        joint_features,
        projections,
        fallback,
        mimic_joints,
        joint_signature,
    )


def _sequence_edit_distance(left: Sequence[str], right: Sequence[str]) -> float:
    """Normalized Levenshtein distance for ordered R/P joint signatures."""

    if not left and not right:
        return 0.0
    previous = np.arange(len(right) + 1, dtype=np.float64)
    for row, left_symbol in enumerate(left, start=1):
        current = np.empty(len(right) + 1, dtype=np.float64)
        current[0] = row
        for column, right_symbol in enumerate(right, start=1):
            current[column] = min(
                previous[column] + 1.0,
                current[column - 1] + 1.0,
                previous[column - 1] + float(left_symbol != right_symbol),
            )
        previous = current
    return float(previous[-1] / max(len(left), len(right), 1))


def _dtw_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Average-cost dynamic time warping for two feature sequences."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) == 0 and len(right) == 0:
        return 0.0
    if len(left) == 0 or len(right) == 0:
        return 1.0

    costs = np.full((len(left) + 1, len(right) + 1), np.inf, dtype=np.float64)
    steps = np.zeros((len(left) + 1, len(right) + 1), dtype=np.int32)
    costs[0, 0] = 0.0
    for row in range(1, len(left) + 1):
        for column in range(1, len(right) + 1):
            candidates = (
                (costs[row - 1, column], steps[row - 1, column]),
                (costs[row, column - 1], steps[row, column - 1]),
                (costs[row - 1, column - 1], steps[row - 1, column - 1]),
            )
            previous_cost, previous_steps = min(candidates, key=lambda value: value[0])
            local = float(np.linalg.norm(left[row - 1] - right[column - 1]))
            costs[row, column] = previous_cost + local
            steps[row, column] = previous_steps + 1
    return float(costs[-1, -1] / max(int(steps[-1, -1]), 1))


def _pairwise_distances(
    items: Sequence[ObjectFeature],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute part, kinematic, and rest-layout distance matrices."""

    count = len(items)
    part = np.zeros((count, count), dtype=np.float64)
    kinematic = np.zeros((count, count), dtype=np.float64)
    rest_hog = np.zeros((count, count), dtype=np.float64)
    for left_index in range(count):
        left = items[left_index]
        for right_index in range(left_index + 1, count):
            right = items[right_index]
            part_distance = _dtw_distance(
                left.segment_features, right.segment_features
            )
            signature_distance = _sequence_edit_distance(
                left.joint_signature, right.joint_signature
            )
            relation_distance = _dtw_distance(
                left.joint_features, right.joint_features
            )
            kinematic_distance = 0.6 * signature_distance + 0.4 * relation_distance
            hog_distance = float(np.linalg.norm(left.rest_hog - right.rest_hog))
            part[left_index, right_index] = part[
                right_index, left_index
            ] = part_distance
            kinematic[left_index, right_index] = kinematic[
                right_index, left_index
            ] = kinematic_distance
            rest_hog[left_index, right_index] = rest_hog[
                right_index, left_index
            ] = hog_distance
    return part, kinematic, rest_hog


def _median_scaled(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale nonzero pair distances to median 1 for meaningful cue weights."""

    nonzero = matrix[matrix > 1e-12]
    scale = float(np.median(nonzero)) if nonzero.size else 1.0
    return matrix / max(scale, 1e-12), scale


def _initial_medoids(
    distances: np.ndarray, clusters: int, rng: np.random.RandomState
) -> np.ndarray:
    """Distance-weighted initialization analogous to k-means++."""

    count = len(distances)
    medoids = [int(rng.randint(count))]
    while len(medoids) < clusters:
        nearest = np.min(distances[:, medoids], axis=1)
        nearest[np.asarray(medoids)] = 0.0
        weights = nearest * nearest
        candidates = np.setdiff1d(np.arange(count), np.asarray(medoids))
        if float(weights.sum()) <= 1e-15:
            choice = int(rng.choice(candidates))
        else:
            weights /= weights.sum()
            choice = int(rng.choice(count, p=weights))
            if choice in medoids:  # Numerical protection for tiny probabilities.
                choice = int(rng.choice(candidates))
        medoids.append(choice)
    return np.asarray(medoids, dtype=np.int32)


def _assign_to_medoids(distances: np.ndarray, medoids: np.ndarray) -> np.ndarray:
    labels = np.argmin(distances[:, medoids], axis=1).astype(np.int32)
    # Exact duplicate objects can otherwise leave a later medoid empty.
    for cluster, medoid in enumerate(medoids):
        labels[int(medoid)] = cluster
    return labels


def _kmedoids(
    distances: np.ndarray,
    *,
    clusters: int,
    random_state: int,
    n_init: int,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Cluster a precomputed mixed-distance matrix by alternate medoid updates."""

    best: tuple[np.ndarray, np.ndarray, float] | None = None
    master_rng = np.random.RandomState(random_state)
    for _ in range(n_init):
        rng = np.random.RandomState(int(master_rng.randint(0, 2**31 - 1)))
        medoids = _initial_medoids(distances, clusters, rng)
        for _iteration in range(max_iterations):
            labels = _assign_to_medoids(distances, medoids)
            updated = medoids.copy()
            for cluster in range(clusters):
                members = np.flatnonzero(labels == cluster)
                if len(members):
                    within = distances[np.ix_(members, members)]
                    updated[cluster] = int(members[np.argmin(within.sum(axis=1))])
            if np.array_equal(updated, medoids):
                break
            medoids = updated
        labels = _assign_to_medoids(distances, medoids)
        objective = float(
            np.sum(distances[np.arange(len(distances)), medoids[labels]])
        )
        if best is None or objective < best[2]:
            best = labels.copy(), medoids.copy(), objective
    assert best is not None
    return best


def _pad_sequences(
    items: Sequence[ObjectFeature], attribute: str
) -> tuple[np.ndarray, np.ndarray]:
    """Pad variable-length feature sequences for storage in features.npz."""

    sequences = [np.asarray(getattr(item, attribute)) for item in items]
    lengths = np.asarray([len(sequence) for sequence in sequences], dtype=np.int32)
    max_length = int(max(lengths, default=0))
    dimension = (
        int(sequences[0].shape[1])
        if sequences and sequences[0].ndim == 2
        else 0
    )
    padded = np.zeros((len(items), max_length, dimension), dtype=np.float32)
    for index, sequence in enumerate(sequences):
        padded[index, : len(sequence)] = sequence
    return padded, lengths


def _projection_strip(item: ObjectFeature, *, scale: int = 1) -> Image.Image:
    size = item.projections[0].shape[0]
    images = []
    for name, projection in zip(_VIEW_NAMES, item.projections):
        gray = np.uint8(np.clip(1.0 - projection, 0.0, 1.0) * 255)
        image = Image.fromarray(gray, mode="L").convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 22, 12), fill="white")
        draw.text((2, 1), name, fill="black", font=ImageFont.load_default())
        if scale != 1:
            image = image.resize((size * scale, size * scale), Image.Resampling.NEAREST)
        images.append(image)
    strip = Image.new("RGB", (sum(im.width for im in images), images[0].height), "white")
    x = 0
    for image in images:
        strip.paste(image, (x, 0))
        x += image.width
    return strip


def _save_projection_images(items: Sequence[ObjectFeature], output_dir: Path) -> None:
    projection_dir = output_dir / "projections"
    projection_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        _projection_strip(item).save(projection_dir / f"{item.ref.object_id}.png")


def _wrapped_label(item: ObjectFeature, distance: float, is_medoid: bool) -> list[str]:
    object_id = item.ref.object_id
    if object_id.startswith("rec_"):
        object_id = object_id[4:]
    prefix = "MEDOID  " if is_medoid else ""
    return [
        f"{prefix}{object_id[:44]}",
        f"{item.ref.sub_category} / {item.ref.split}  d={distance:.4f}",
        f"joints={'-'.join(item.joint_signature) or 'NONE'}",
    ]


def _save_cluster_sheet(
    cluster_id: int,
    members: Sequence[int],
    items: Sequence[ObjectFeature],
    pairwise_distances: np.ndarray,
    distances: np.ndarray,
    medoid_index: int,
    output_dir: Path,
    *,
    columns: int = 3,
) -> None:
    ordered = [medoid_index] + sorted(
        (idx for idx in members if idx != medoid_index),
        key=lambda idx: float(pairwise_distances[idx, medoid_index]),
    )
    sample_strip = _projection_strip(items[ordered[0]])
    cell_width = sample_strip.width + 16
    cell_height = sample_strip.height + 54
    rows = (len(ordered) + columns - 1) // columns
    header = 38
    sheet = Image.new(
        "RGB", (columns * cell_width, header + rows * cell_height), "white"
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    medoid_id = items[medoid_index].ref.object_id
    draw.text(
        (8, 8),
        f"Cluster {cluster_id:02d} | n={len(ordered)} | medoid={medoid_id}",
        fill="black",
        font=font,
    )
    for position, idx in enumerate(ordered):
        row, col = divmod(position, columns)
        x = col * cell_width + 8
        y = header + row * cell_height
        strip = _projection_strip(items[idx])
        sheet.paste(strip, (x, y))
        label_y = y + strip.height + 2
        for line_no, line in enumerate(
            _wrapped_label(items[idx], float(distances[idx]), idx == medoid_index)
        ):
            draw.text((x, label_y + line_no * 12), line, fill="black", font=font)
        if idx == medoid_index:
            draw.rectangle(
                (x - 2, y - 2, x + strip.width + 1, y + strip.height + 1),
                outline=(220, 40, 40),
                width=2,
            )
    sheet.save(output_dir / f"cluster_{cluster_id:02d}.png")


def _write_outputs(
    items: Sequence[ObjectFeature],
    combined_matrix: np.ndarray,
    part_matrix: np.ndarray,
    kinematic_matrix: np.ndarray,
    rest_hog_matrix: np.ndarray,
    labels: np.ndarray,
    medoids: np.ndarray,
    objective: float,
    component_scales: dict[str, float],
    failures: Sequence[dict[str, str]],
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_projection_images(items, output_dir)

    assigned_medoids = medoids[labels]
    row_indices = np.arange(len(items))
    distances = combined_matrix[row_indices, assigned_medoids]
    part_distances = part_matrix[row_indices, assigned_medoids]
    kinematic_distances = kinematic_matrix[row_indices, assigned_medoids]
    rest_hog_distances = rest_hog_matrix[row_indices, assigned_medoids]
    for cluster_id in range(args.clusters):
        members = np.flatnonzero(labels == cluster_id).tolist()
        medoid = int(medoids[cluster_id])
        _save_cluster_sheet(
            cluster_id,
            members,
            items,
            combined_matrix,
            distances,
            medoid,
            output_dir,
        )

    with (output_dir / "assignments.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "object_id",
                "category",
                "sub_category",
                "split",
                "cluster",
                "distance_to_medoid",
                "part_geometry_distance_to_medoid",
                "kinematic_distance_to_medoid",
                "rest_hog_distance_to_medoid",
                "is_medoid",
                "points_used",
                "rigid_segment_count",
                "canonical_fallback",
                "joint_signature",
                "mimic_joints",
                "archive",
            ]
        )
        for idx, item in enumerate(items):
            cluster_id = int(labels[idx])
            writer.writerow(
                [
                    item.ref.object_id,
                    item.ref.category,
                    item.ref.sub_category,
                    item.ref.split,
                    cluster_id,
                    f"{float(distances[idx]):.8f}",
                    f"{float(part_distances[idx]):.8f}",
                    f"{float(kinematic_distances[idx]):.8f}",
                    f"{float(rest_hog_distances[idx]):.8f}",
                    idx == int(medoids[cluster_id]),
                    item.points_used,
                    len(item.segment_features),
                    item.canonical_fallback,
                    "-".join(item.joint_signature) or "NONE",
                    ";".join(item.mimic_joints),
                    str(item.ref.archive),
                ]
            )

    weight_sum = args.part_weight + args.kinematic_weight + args.rest_hog_weight
    config = {
        "category": args.category,
        "sub_categories": args.sub_category,
        "splits": args.split,
        "fit_only": not args.include_not_fit,
        "clusters": args.clusters,
        "rest_state": args.rest_state,
        "unreachable": args.unreachable,
        "pose_normalization": args.pose_normalization,
        "points_per_link": args.points_per_link,
        "num_points": args.num_points,
        "image_size": args.image_size,
        "splat_radius": args.splat_radius,
        "hog_orientations": args.orientations,
        "hog_pixels_per_cell": args.pixels_per_cell,
        "hog_cells_per_block": args.cells_per_block,
        "random_state": args.random_state,
        "n_init": args.n_init,
        "max_iterations": args.max_iterations,
        "weights": {
            "part_geometry": args.part_weight / weight_sum,
            "kinematic": args.kinematic_weight / weight_sum,
            "rest_hog": args.rest_hog_weight / weight_sum,
        },
        "component_median_scales": component_scales,
        "shape_hist_bins": args.shape_hist_bins,
        "d2_pairs": args.d2_pairs,
        "segment_encoding": (
            "Fixed-connected links are merged; each rigid segment uses PCA "
            "variance ratios, sorted bbox ratios, radial and D2 histograms, "
            "relative intrinsic RMS scale, and surface-area fraction."
        ),
        "kinematic_encoding": (
            "Normalized R/P edit distance plus DTW over consecutive absolute "
            "axis dot products, articulation-invariant normalized pivot "
            "spacing, and mimic flags."
        ),
        "strict_serial": args.strict_serial,
        "objective": objective,
        "projection_note": (
            "Binary orthographic occupancy from sampled surface points; "
            "no depth, lighting, texture, or mesh rasterization."
        ),
    }
    clusters = []
    for cluster_id in range(args.clusters):
        member_indices = np.flatnonzero(labels == cluster_id).tolist()
        medoid_idx = int(medoids[cluster_id])
        clusters.append(
            {
                "cluster": cluster_id,
                "size": len(member_indices),
                "medoid": items[medoid_idx].ref.object_id,
                "members": [items[idx].ref.object_id for idx in member_indices],
            }
        )
    payload = {
        "schema_version": 2,
        "method": "articulated segment/kinematic/rest-HOG distance + k-medoids",
        "config": config,
        "clusters": clusters,
        "objects": [
            {
                "object_id": item.ref.object_id,
                "category": item.ref.category,
                "sub_category": item.ref.sub_category,
                "split": item.ref.split,
                "cluster": int(labels[idx]),
                "distance_to_medoid": float(distances[idx]),
                "part_geometry_distance_to_medoid": float(part_distances[idx]),
                "kinematic_distance_to_medoid": float(kinematic_distances[idx]),
                "rest_hog_distance_to_medoid": float(rest_hog_distances[idx]),
                "is_medoid": idx == int(medoids[int(labels[idx])]),
                "points_used": item.points_used,
                "rigid_segment_count": len(item.segment_features),
                "canonical_fallback": item.canonical_fallback,
                "joint_signature": "-".join(item.joint_signature) or "NONE",
                "mimic_joints": list(item.mimic_joints),
                "archive": str(item.ref.archive),
            }
            for idx, item in enumerate(items)
        ],
        "failures": list(failures),
    }
    (output_dir / "assignments.json").write_text(json.dumps(payload, indent=2) + "\n")
    segment_features, segment_counts = _pad_sequences(items, "segment_features")
    joint_features, joint_counts = _pad_sequences(items, "joint_features")
    np.savez_compressed(
        output_dir / "features.npz",
        object_ids=np.asarray([item.ref.object_id for item in items]),
        rest_hog_features=np.stack([item.rest_hog for item in items]).astype(
            np.float32
        ),
        segment_features=segment_features,
        segment_counts=segment_counts,
        joint_features=joint_features,
        joint_counts=joint_counts,
        joint_signatures=np.asarray(
            ["-".join(item.joint_signature) or "NONE" for item in items]
        ),
        combined_distance_matrix=combined_matrix.astype(np.float32),
        part_geometry_distance_matrix=part_matrix.astype(np.float32),
        kinematic_distance_matrix=kinematic_matrix.astype(np.float32),
        rest_hog_distance_matrix=rest_hog_matrix.astype(np.float32),
        labels=labels.astype(np.int32),
        medoid_indices=medoids.astype(np.int32),
        distances_to_medoid=distances.astype(np.float32),
        part_geometry_distances_to_medoid=part_distances.astype(np.float32),
        kinematic_distances_to_medoid=kinematic_distances.astype(np.float32),
        rest_hog_distances_to_medoid=rest_hog_distances.astype(np.float32),
    )


def _cluster_summary(
    labels: np.ndarray, items: Sequence[ObjectFeature]
) -> Iterable[str]:
    for cluster_id in sorted(set(map(int, labels))):
        indices = np.flatnonzero(labels == cluster_id)
        leaves: dict[str, int] = {}
        signatures: dict[str, int] = {}
        for idx in indices:
            item = items[int(idx)]
            leaf = item.ref.sub_category
            leaves[leaf] = leaves.get(leaf, 0) + 1
            signature = "-".join(item.joint_signature) or "NONE"
            signatures[signature] = signatures.get(signature, 0) + 1
        leaf_text = ", ".join(
            f"{name}:{count}" for name, count in sorted(leaves.items())
        )
        signature_text = ", ".join(
            f"{name}:{count}" for name, count in sorted(signatures.items())
        )
        yield (
            f"cluster {cluster_id:02d}: n={len(indices):3d}  "
            f"{leaf_text}  joints=[{signature_text}]"
        )


def _default_output_dir(
    args: argparse.Namespace, refs: Sequence[ArchiveRef]
) -> Path:
    """Build the default path for a selection containing exactly one leaf."""

    selected_leaves = sorted({ref.sub_category for ref in refs})
    if len(selected_leaves) != 1:
        leaves = ", ".join(selected_leaves)
        raise SystemExit(
            "The default output layout requires exactly one sub-category, but "
            f"this selection contains {len(selected_leaves)}: {leaves}. Repeat "
            "the command with --sub-category <leaf>, or provide --output-dir "
            "when intentionally clustering several leaves together."
        )
    return (
        _REPO_ROOT
        / "data/urdfs/HOGClusters"
        / args.category
        / selected_leaves[0]
        / f"k_{args.clusters}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose >= 2 else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.verbose < 2:
        logging.getLogger("trimesh").setLevel(logging.WARNING)

    refs = dataset.discover_archives(
        args.dataset_root,
        fit_only=not args.include_not_fit,
        categories=[args.category],
        sub_categories=args.sub_category,
        splits=args.split,
    )
    if args.limit is not None:
        refs = refs[: args.limit]
    if not refs:
        raise SystemExit("No matching URDF archives were found")

    output_dir = args.output_dir or _default_output_dir(args, refs)
    items: list[ObjectFeature] = []
    failures: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="articraft_hog_") as temp_dir:
        work_dir = Path(temp_dir)
        for index, ref in enumerate(refs, start=1):
            LOGGER.info("[%d/%d] sampling %s", index, len(refs), ref.object_id)
            try:
                item = _load_feature(
                    ref,
                    work_dir=work_dir,
                    args=args,
                    object_seed=args.random_state + index,
                )
            except Exception as exc:  # noqa: BLE001 - record bad dataset members
                LOGGER.warning("skipping %s: %s", ref.object_id, exc)
                failures.append(
                    {
                        "object_id": ref.object_id,
                        "archive": str(ref.archive),
                        "reason": str(exc),
                    }
                )
                continue
            items.append(item)

    if len(items) < args.clusters:
        raise SystemExit(
            f"Only {len(items)} objects loaded successfully; cannot form "
            f"{args.clusters} clusters"
        )

    LOGGER.info("computing mixed pairwise distances for %d objects", len(items))
    part_raw, kinematic_raw, rest_hog_raw = _pairwise_distances(items)
    part_matrix, part_scale = _median_scaled(part_raw)
    kinematic_matrix, kinematic_scale = _median_scaled(kinematic_raw)
    rest_hog_matrix, rest_hog_scale = _median_scaled(rest_hog_raw)

    weight_sum = args.part_weight + args.kinematic_weight + args.rest_hog_weight
    part_weight = args.part_weight / weight_sum
    kinematic_weight = args.kinematic_weight / weight_sum
    rest_hog_weight = args.rest_hog_weight / weight_sum
    combined_matrix = (
        part_weight * part_matrix
        + kinematic_weight * kinematic_matrix
        + rest_hog_weight * rest_hog_matrix
    )
    LOGGER.info(
        "k-medoids weights: part=%.3f kinematic=%.3f rest-HOG=%.3f",
        part_weight,
        kinematic_weight,
        rest_hog_weight,
    )
    labels, medoids, objective = _kmedoids(
        combined_matrix,
        clusters=args.clusters,
        random_state=args.random_state,
        n_init=args.n_init,
        max_iterations=args.max_iterations,
    )

    # Stable cluster numbering makes repeated runs and GUI inspection easier.
    order = np.asarray(
        sorted(
            range(args.clusters),
            key=lambda cluster: items[int(medoids[cluster])].ref.object_id,
        ),
        dtype=np.int32,
    )
    remap = np.empty(args.clusters, dtype=np.int32)
    remap[order] = np.arange(args.clusters, dtype=np.int32)
    labels = remap[labels]
    medoids = medoids[order]

    _write_outputs(
        items,
        combined_matrix,
        part_matrix,
        kinematic_matrix,
        rest_hog_matrix,
        labels,
        medoids,
        objective,
        {
            "part_geometry": part_scale,
            "kinematic": kinematic_scale,
            "rest_hog": rest_hog_scale,
        },
        failures,
        args,
        output_dir,
    )

    print(f"Wrote articulated k-medoids clustering results to {output_dir}")
    for line in _cluster_summary(labels, items):
        print(line)
    if failures:
        print(f"Skipped {len(failures)} object(s); see assignments.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
