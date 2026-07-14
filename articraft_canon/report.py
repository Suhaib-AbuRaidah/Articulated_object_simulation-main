"""Quality-control reporting.

Two products:

  * a **per-object dry-run line** summarising what canonicalization *would* do
    (q* per joint, fallback flag, chosen symmetry order, Umeyama residual,
    predicted Chamfer to the sub-category reference);
  * a **per-sub-category QC report**: the pairwise Chamfer matrix between all
    aligned objects at ``q = 0`` (which should have near-zero variance once the
    frames are consistent) plus outlier flags for objects that align poorly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from . import geometry as geo
from .subcat_frame import Alignment


@dataclass
class SubCategoryQC:
    """QC summary for one sub-category."""

    sub_category: str
    n_objects: int
    reference_id: str
    chamfer_mean: float
    chamfer_std: float
    chamfer_max: float
    outliers: List[str]
    fallback_ids: List[str]


def format_object_dryrun(
    object_id: str,
    q_star: Dict[str, float],
    fallback: bool,
    fallback_reason: str,
    symmetry_order: int,
    alignment: Alignment | None,
    widened_joints: List[str] | None = None,
) -> str:
    """One human-readable line describing the planned canonicalization."""
    q_txt = ", ".join(f"{n}={v:+.3f}" for n, v in q_star.items()) or "(no moving joints)"
    parts = [f"{object_id}"]
    parts.append(f"q*=[{q_txt}]")
    parts.append(f"sym_order={symmetry_order}")
    if alignment is not None:
        parts.append(f"umeyama_rms={alignment.residual:.4f}")
        parts.append(f"pred_chamfer={alignment.predicted_chamfer:.4f}")
        if alignment.is_reference:
            parts.append("[REFERENCE]")
    if widened_joints:
        parts.append(f"[WIDENED: {', '.join(widened_joints)}]")
    if fallback:
        parts.append(f"[FALLBACK: {fallback_reason}]")
    return "  ".join(parts)


def sub_category_qc(
    sub_category: str,
    object_ids: List[str],
    aligned_clouds: List[np.ndarray],
    reference_id: str,
    fallback_ids: List[str],
    *,
    subsample: int = 512,
    outlier_sigma: float = 2.0,
) -> tuple[SubCategoryQC, np.ndarray]:
    """Compute the pairwise-Chamfer QC summary and matrix for a sub-category."""
    reduced = [geo.farthest_point_subsample(c, subsample, seed=6) for c in aligned_clouds]
    n = len(reduced)
    M = np.zeros((n, n))
    for i in range(n):
        for k in range(i + 1, n):
            d = geo.chamfer_distance(reduced[i], reduced[k])
            M[i, k] = M[k, i] = d

    if n > 1:
        per_object_mean = M.sum(axis=1) / (n - 1)
        upper = M[np.triu_indices(n, k=1)]
        mean, std, mx = float(upper.mean()), float(upper.std()), float(upper.max())
        thresh = mean + outlier_sigma * std
        outliers = [object_ids[i] for i in range(n) if per_object_mean[i] > thresh]
    else:
        mean = std = mx = 0.0
        outliers = []

    qc = SubCategoryQC(
        sub_category=sub_category,
        n_objects=n,
        reference_id=reference_id,
        chamfer_mean=mean,
        chamfer_std=std,
        chamfer_max=mx,
        outliers=outliers,
        fallback_ids=fallback_ids,
    )
    return qc, M


def format_sub_category_qc(qc: SubCategoryQC) -> str:
    """Multi-line human-readable rendering of a sub-category QC summary."""
    lines = [
        f"[{qc.sub_category}] n={qc.n_objects}  ref={qc.reference_id}",
        f"    pairwise Chamfer @ q=0:  mean={qc.chamfer_mean:.4f}  "
        f"std={qc.chamfer_std:.4f}  max={qc.chamfer_max:.4f}",
    ]
    if qc.fallback_ids:
        lines.append(f"    mid-limit fallbacks ({len(qc.fallback_ids)}): "
                     f"{', '.join(qc.fallback_ids)}")
    if qc.outliers:
        lines.append(f"    OUTLIERS ({len(qc.outliers)}): {', '.join(qc.outliers)}")
    else:
        lines.append("    outliers: none")
    return "\n".join(lines)
