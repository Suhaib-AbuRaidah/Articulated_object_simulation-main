#!/usr/bin/env python3
"""Cluster Articraft URDFs by rest-state geometry using HOG + k-means.

The geometric descriptor is built only from surface points sampled from the
URDF visual meshes/primitives.  No depth renderer, lighting, texture, or RGB
appearance is used.  HOG still requires a 2D input, so the sampled 3D points
are projected to three binary orthographic occupancy images (XY, XZ, and YZ).
An optional weighted feature block additionally represents the ordered moving-
joint signature (for example R-R-R or R-P-R).

By default, the rest cloud is centred, isotropically scaled, PCA-aligned, and
averaged over all PCA-axis sign choices.  This makes clustering insensitive to
the source URDF's global rotation and to PCA sign ambiguity while retaining the
geometry of its q=0 articulation state.

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

Dependencies beyond the Articraft canonicalization stack are scikit-image,
scikit-learn, and Pillow.
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
    from sklearn.cluster import KMeans
    from skimage.feature import hog
except ImportError as exc:  # pragma: no cover - only exercised in incomplete envs
    raise SystemExit(
        "cluster_urdf_hog.py requires Pillow, scikit-image, and scikit-learn. "
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
    """HOG descriptor and inspection images for one dataset object."""

    ref: ArchiveRef
    points_used: int
    feature: np.ndarray
    projections: list[np.ndarray]
    canonical_fallback: bool = False
    mimic_joints: tuple[str, ...] = ()
    joint_signature: tuple[str, ...] = ()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster a category's URDFs by rest-state mesh geometry using "
            "surface-point occupancy projections, HOG, and k-means."
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
        "--clusters", "-k", type=int, required=True, help="Number of k-means clusters."
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
        help="Seed for point subsampling and k-means (default: %(default)s).",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=30,
        help="Independent k-means initializations (default: %(default)s).",
    )
    parser.add_argument(
        "--joint-signature-weight",
        type=float,
        default=0.5,
        help=(
            "Weight of the ordered moving-joint signature relative to the "
            "unit-normalized HOG block; 0 gives geometry-only clustering "
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
    if args.joint_signature_weight < 0:
        raise SystemExit("--joint-signature-weight cannot be negative")
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
        feature = np.concatenate([_hog_vector(image, args) for image in display_projections])

    norm = float(np.linalg.norm(feature))
    if norm > 1e-12:
        feature = feature / norm
    return feature.astype(np.float32), display_projections


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
    )
    if model.skip_reason:
        raise ValueError(model.skip_reason)

    mimic_joints = tuple(
        joint.name
        for joint in model.urdf.robot.joints
        if joint.mimic is not None
    )
    joint_signature = tuple(
        "R" if joint.type in ("revolute", "continuous") else "P"
        for joint in model.moving_joints
    )
    if mimic_joints and args.rest_state == "canonical-zero":
        names = ", ".join(mimic_joints)
        raise ValueError(
            "canonical-zero cannot independently solve coupled mimic joint(s): "
            f"{names}; use --rest-state source"
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

    points = model.object_points()
    if len(points) == 0:
        raise ValueError("object has no sampled visual geometry")
    points = geo.farthest_point_subsample(points, args.num_points, seed=object_seed)
    feature, projections = _cloud_descriptor(points, args)
    return ObjectFeature(
        ref,
        len(points),
        feature,
        projections,
        fallback,
        mimic_joints,
        joint_signature,
    )


def _joint_signature_matrix(
    items: Sequence[ObjectFeature],
) -> tuple[np.ndarray, int]:
    """One-hot encode ordered R/P signatures, padding with an explicit NONE."""

    max_joints = max((len(item.joint_signature) for item in items), default=0)
    if max_joints == 0:
        return np.zeros((len(items), 0), dtype=np.float32), 0

    symbols = ("R", "P", "NONE")
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
    encoded = np.zeros((len(items), max_joints * len(symbols)), dtype=np.float32)
    for row, item in enumerate(items):
        padded = item.joint_signature + ("NONE",) * (
            max_joints - len(item.joint_signature)
        )
        for position, symbol in enumerate(padded):
            column = position * len(symbols) + symbol_index[symbol]
            encoded[row, column] = 1.0

    # Give every object's complete signature equal total magnitude regardless
    # of chain length or the maximum chain length in the current selection.
    norms = np.linalg.norm(encoded, axis=1, keepdims=True)
    encoded /= np.maximum(norms, 1e-12)
    return encoded, max_joints


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
    features: np.ndarray,
    distances: np.ndarray,
    medoid_index: int,
    output_dir: Path,
    *,
    columns: int = 3,
) -> None:
    medoid_feature = features[medoid_index]
    ordered = [medoid_index] + sorted(
        (idx for idx in members if idx != medoid_index),
        key=lambda idx: float(np.linalg.norm(features[idx] - medoid_feature)),
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


def _medoid_for_cluster(
    features: np.ndarray, members: Sequence[int]
) -> int:
    """Return the real member nearest to all other members in feature space."""

    if len(members) == 1:
        return int(members[0])
    local = features[np.asarray(members)]
    norms = np.sum(local * local, axis=1)
    squared = norms[:, None] + norms[None, :] - 2.0 * (local @ local.T)
    squared = np.maximum(squared, 0.0)
    return int(members[int(np.argmin(squared.sum(axis=1)))])


def _write_outputs(
    items: Sequence[ObjectFeature],
    features: np.ndarray,
    geometry_features: np.ndarray,
    joint_signature_features: np.ndarray,
    labels: np.ndarray,
    distances: np.ndarray,
    geometry_distances: np.ndarray,
    joint_signature_distances: np.ndarray,
    centers: np.ndarray,
    failures: Sequence[dict[str, str]],
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_projection_images(items, output_dir)

    medoids: dict[int, int] = {}
    for cluster_id in range(args.clusters):
        members = np.flatnonzero(labels == cluster_id).tolist()
        medoid = _medoid_for_cluster(features, members)
        medoids[cluster_id] = medoid
        _save_cluster_sheet(
            cluster_id,
            members,
            items,
            features,
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
                "distance_to_centroid",
                "geometry_distance_to_centroid",
                "joint_signature_distance_to_centroid",
                "is_medoid",
                "points_used",
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
                    f"{float(geometry_distances[idx]):.8f}",
                    f"{float(joint_signature_distances[idx]):.8f}",
                    idx == medoids[cluster_id],
                    item.points_used,
                    item.canonical_fallback,
                    "-".join(item.joint_signature) or "NONE",
                    ";".join(item.mimic_joints),
                    str(item.ref.archive),
                ]
            )

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
        "joint_signature_weight": args.joint_signature_weight,
        "joint_signature_encoding": (
            "Position-wise one-hot R/P/NONE; continuous joints map to R; "
            "fixed joints are omitted; each signature block is L2-normalized."
        ),
        "strict_serial": args.strict_serial,
        "geometry_feature_dimension": int(geometry_features.shape[1]),
        "joint_signature_feature_dimension": int(joint_signature_features.shape[1]),
        "combined_feature_dimension": int(features.shape[1]),
        "projection_note": (
            "Binary orthographic occupancy from sampled surface points; "
            "no depth, lighting, texture, or mesh rasterization."
        ),
    }
    clusters = []
    for cluster_id in range(args.clusters):
        member_indices = np.flatnonzero(labels == cluster_id).tolist()
        medoid_idx = medoids[cluster_id]
        clusters.append(
            {
                "cluster": cluster_id,
                "size": len(member_indices),
                "medoid": items[medoid_idx].ref.object_id,
                "members": [items[idx].ref.object_id for idx in member_indices],
            }
        )
    payload = {
        "schema_version": 1,
        "method": "surface-point occupancy projections + HOG + k-means",
        "config": config,
        "clusters": clusters,
        "objects": [
            {
                "object_id": item.ref.object_id,
                "category": item.ref.category,
                "sub_category": item.ref.sub_category,
                "split": item.ref.split,
                "cluster": int(labels[idx]),
                "distance_to_centroid": float(distances[idx]),
                "geometry_distance_to_centroid": float(geometry_distances[idx]),
                "joint_signature_distance_to_centroid": float(
                    joint_signature_distances[idx]
                ),
                "is_medoid": idx == medoids[int(labels[idx])],
                "points_used": item.points_used,
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
    np.savez_compressed(
        output_dir / "features.npz",
        object_ids=np.asarray([item.ref.object_id for item in items]),
        features=features.astype(np.float32),
        geometry_features=geometry_features.astype(np.float32),
        joint_signature_features=joint_signature_features.astype(np.float32),
        joint_signatures=np.asarray(
            ["-".join(item.joint_signature) or "NONE" for item in items]
        ),
        labels=labels.astype(np.int32),
        distances_to_centroid=distances.astype(np.float32),
        geometry_distances_to_centroid=geometry_distances.astype(np.float32),
        joint_signature_distances_to_centroid=joint_signature_distances.astype(
            np.float32
        ),
        centers=centers.astype(np.float32),
    )


def _cluster_summary(labels: np.ndarray, items: Sequence[ObjectFeature]) -> Iterable[str]:
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
        leaf_text = ", ".join(f"{name}:{count}" for name, count in sorted(leaves.items()))
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

    geometry_features = np.stack([item.feature for item in items])
    joint_signature_features, max_joints = _joint_signature_matrix(items)
    weighted_signature_features = (
        args.joint_signature_weight * joint_signature_features
    )
    features = np.concatenate(
        [geometry_features, weighted_signature_features], axis=1
    )
    LOGGER.info(
        "clustering %d objects with %d HOG + %d joint-signature features "
        "(max joints=%d, signature weight=%.3f)",
        len(items),
        geometry_features.shape[1],
        joint_signature_features.shape[1],
        max_joints,
        args.joint_signature_weight,
    )
    kmeans = KMeans(
        n_clusters=args.clusters,
        random_state=args.random_state,
        n_init=args.n_init,
    )
    labels = kmeans.fit_predict(features)
    distinct_labels = np.unique(labels)
    if len(distinct_labels) != args.clusters:
        raise SystemExit(
            f"k-means produced only {len(distinct_labels)} distinct clusters from "
            f"the requested {args.clusters}. Lower --clusters or adjust the HOG "
            "parameters."
        )
    distances_all = kmeans.transform(features)
    distances = distances_all[np.arange(len(items)), labels]
    geometry_dimension = geometry_features.shape[1]
    assigned_centers = kmeans.cluster_centers_[labels]
    geometry_distances = np.linalg.norm(
        geometry_features - assigned_centers[:, :geometry_dimension], axis=1
    )
    joint_signature_distances = np.linalg.norm(
        weighted_signature_features - assigned_centers[:, geometry_dimension:],
        axis=1,
    )
    _write_outputs(
        items,
        features,
        geometry_features,
        joint_signature_features,
        labels,
        distances,
        geometry_distances,
        joint_signature_distances,
        kmeans.cluster_centers_,
        failures,
        args,
        output_dir,
    )

    print(f"Wrote HOG clustering results to {output_dir}")
    for line in _cluster_summary(labels, items):
        print(line)
    if failures:
        print(f"Skipped {len(failures)} object(s); see assignments.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
