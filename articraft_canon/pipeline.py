"""Pipeline orchestration.

Ties the five stages together, grouped by sub-category (Stage 4 needs the whole
sub-category in memory at once).  Honours the ``--dry-run`` / ``--write``
contract: dry-run computes and reports everything but writes nothing; write
additionally emits canonical URDFs and JSON sidecars to a separate output tree.

Flow per sub-category::

    for each archive:  extract -> parse (S1) -> canonical zero (S2) -> root frame (S3)
    align the whole sub-category to a reference (S4) and fold frames
    for each object:   build NOCS/NPCS sidecar (S5)
    compute QC report; optionally write outputs
"""

from __future__ import annotations

import json
import logging
import shutil
import tarfile
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yourdfpy

from . import canonical_zero, dataset, nocs, parse, report, root_frame, subcat_frame
from .dataset import ArchiveRef

logger = logging.getLogger(__name__)


@dataclass
class ObjectRun:
    """Per-object bundle of everything the stages produced."""

    ref: ArchiveRef
    model: parse.ObjectModel
    canonical: Optional[canonical_zero.CanonicalResult] = None
    frame: Optional[root_frame.RootFrame] = None
    alignment: Optional[subcat_frame.Alignment] = None
    sidecar: Optional[dict] = None


@dataclass
class PipelineConfig:
    dataset_root: Path
    output_dir: Path
    work_dir: Path
    write: bool = False
    check_collision: bool = False   # opt-in: AABB test false-positives on nesting
    first_moving_is_reference: bool = False  # see canonical_zero._solve_extended_config
    unreachable_policy: str = "mid-limit"   # 'mid-limit' | 'clamp' | 'extend-limits'
    align_link_frames: bool = True          # bake base-aligned (Option B) link frames
    n_points_per_link: int = 2048
    exemplars: Dict[str, str] = None       # sub_category -> object_id override
    fit_only: bool = True


# --------------------------------------------------------------------------- #
# Per-object front half (Stages 1-3)
# --------------------------------------------------------------------------- #
def _process_object(ref: ArchiveRef, cfg: PipelineConfig) -> Optional[ObjectRun]:
    """Extract + Stages 1-3 for one object.  None if it could not be loaded."""
    urdf_path = dataset.extract_archive(ref, cfg.work_dir)
    if urdf_path is None:
        return None

    model = parse.load_object(
        urdf_path,
        object_id=ref.object_id,
        source_archive=ref.archive,
        category=ref.category,
        sub_category=ref.sub_category,
        split=ref.split,
        n_points_per_link=cfg.n_points_per_link,
    )
    run = ObjectRun(ref=ref, model=model)
    if model.skip_reason:
        return run  # keep it so we can report the skip

    run.canonical = canonical_zero.canonicalize_zero(
        model,
        check_collision=cfg.check_collision,
        first_moving_is_reference=cfg.first_moving_is_reference,
        unreachable_policy=cfg.unreachable_policy,
        apply=True,
    )
    run.frame = root_frame.compute_root_frame(model)
    return run


# --------------------------------------------------------------------------- #
# Writing outputs
# --------------------------------------------------------------------------- #
def _write_outputs(run: ObjectRun, cfg: PipelineConfig) -> None:
    """Emit one canonical object as a ``.tar.gz`` mirroring the original dataset.

    Archive layout (top-level folder = object id, exactly like the source
    archives)::

        <object_id>/model.urdf        # canonical (origins/limits baked)
        <object_id>/assets/...        # meshes copied so the URDF stays valid
        <object_id>/canonical.json    # NOCS/NPCS + frames sidecar
        <object_id>/compile_report.json  # copied from source if present

    The originals are only read from; nothing under the dataset is modified.
    """
    object_id = run.model.object_id
    out_dir = (cfg.output_dir / run.ref.category / run.ref.sub_category
               / run.ref.split)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / f"{object_id}.tar.gz"

    src_dir = run.model.urdf_path.parent
    stage = Path(tempfile.mkdtemp(prefix="articraft_stage_", dir=cfg.work_dir))
    try:
        # Option B: re-orient every link frame to the base orientation (a
        # write-time, kinematically-identical reparameterisation).
        if cfg.align_link_frames:
            nocs.bake_base_aligned_link_frames(run.model)

        # Canonical URDF.  yourdfpy's write re-applies the load-time "magic"
        # filename handler, which would bake absolute work-dir mesh paths into
        # the file; swap in the null handler so the stored *relative* mesh paths
        # (matching the bundled assets/) survive.
        urdf_out = stage / "model.urdf"
        original_handler = run.model.urdf._filename_handler
        run.model.urdf._filename_handler = yourdfpy.filename_handler_null
        try:
            run.model.urdf.write_xml_file(str(urdf_out))
        finally:
            run.model.urdf._filename_handler = original_handler

        # Sidecar.
        with open(stage / "canonical.json", "w") as fh:
            json.dump(run.sidecar, fh, indent=2)

        # Bundle into a .tar.gz with the object id as the top-level folder.
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(urdf_out, arcname=f"{object_id}/model.urdf")
            tar.add(stage / "canonical.json", arcname=f"{object_id}/canonical.json")
            assets = src_dir / "assets"
            if assets.is_dir():
                tar.add(assets, arcname=f"{object_id}/assets")
            report = src_dir / "compile_report.json"
            if report.is_file():
                tar.add(report, arcname=f"{object_id}/compile_report.json")
    finally:
        shutil.rmtree(stage, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Sub-category driver (Stages 4-5 + QC)
# --------------------------------------------------------------------------- #
def _process_sub_category(
    sub_category: str, refs: List[ArchiveRef], cfg: PipelineConfig
) -> Optional[report.SubCategoryQC]:
    """Run the whole pipeline for one sub-category and print/report results."""
    print(f"\n=== sub-category: {sub_category} ({len(refs)} object(s)) ===")

    runs: List[ObjectRun] = []
    for ref in refs:
        run = _process_object(ref, cfg)
        if run is not None:
            runs.append(run)

    usable = [r for r in runs if r.frame is not None]
    for r in runs:
        if r.model.skip_reason:
            print(f"  SKIP {r.model.object_id}: {r.model.skip_reason}")

    if not usable:
        print("  (no usable objects in this sub-category)")
        return None

    # Stage 4 -- align frames to a reference.
    object_ids = [r.model.object_id for r in usable]
    frames = [r.frame for r in usable]
    exemplar = (cfg.exemplars or {}).get(sub_category)
    ref_idx, alignments = subcat_frame.align_sub_category(
        object_ids, frames, exemplar=exemplar
    )
    reference_id = object_ids[ref_idx]

    folded_frames: List[root_frame.RootFrame] = []
    for r, aln in zip(usable, alignments):
        r.alignment = aln
        folded = subcat_frame.fold_into_root(r.frame, aln)
        folded_frames.append(folded)

        # Stage 5 -- sidecar (uses the folded, sub-category-consistent frame).
        r.sidecar = nocs.build_sidecar(r.model, folded, aln.transform[:3, 3])

    # Per-object dry-run lines.
    for r, aln in zip(usable, alignments):
        line = report.format_object_dryrun(
            r.model.object_id,
            r.canonical.q_star if r.canonical else {},
            r.canonical.fallback if r.canonical else False,
            r.canonical.reason if r.canonical else "",
            r.frame.order,
            aln,
            r.canonical.widened_joints if r.canonical else None,
        )
        print(f"  {line}")

    # QC report over the aligned clouds.
    aligned_clouds = [f.points_canonical for f in folded_frames]
    fallback_ids = [r.model.object_id for r in usable
                    if r.canonical and r.canonical.fallback]
    qc, _matrix = report.sub_category_qc(
        sub_category, object_ids, aligned_clouds, reference_id, fallback_ids
    )
    print(report.format_sub_category_qc(qc))

    # Write outputs only in write mode.
    if cfg.write:
        for r in usable:
            _write_outputs(r, cfg)
        # Persist the QC report as JSON alongside the outputs.
        qc_dir = cfg.output_dir / "_qc"
        qc_dir.mkdir(parents=True, exist_ok=True)
        with open(qc_dir / f"{sub_category}.json", "w") as fh:
            json.dump(qc.__dict__, fh, indent=2)

    return qc


# --------------------------------------------------------------------------- #
# Top-level entry
# --------------------------------------------------------------------------- #
def run_pipeline(cfg: PipelineConfig, refs: List[ArchiveRef]) -> List[report.SubCategoryQC]:
    """Group ``refs`` by sub-category and run the full pipeline over each."""
    mode = "WRITE" if cfg.write else "DRY-RUN"
    print(f"canonicalization pipeline [{mode}]  ->  {cfg.output_dir}")

    by_sub: Dict[str, List[ArchiveRef]] = defaultdict(list)
    for ref in refs:
        by_sub[ref.sub_category].append(ref)

    reports: List[report.SubCategoryQC] = []
    for sub_category in sorted(by_sub):
        qc = _process_sub_category(sub_category, by_sub[sub_category], cfg)
        if qc is not None:
            reports.append(qc)

    print(f"\nprocessed {len(refs)} object(s) across {len(by_sub)} sub-category(ies).")
    if cfg.write:
        print(f"canonical outputs written under: {cfg.output_dir}")
    else:
        print("dry-run only: no files written (pass --write to emit).")
    return reports
