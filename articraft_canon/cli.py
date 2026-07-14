"""Command-line interface for the Articraft canonicalization pipeline.

Examples
--------
    # Dry-run over one sub-category (default: writes nothing)
    python -m articraft_canon --sub-category serial_elbow_arm

    # Dry-run over a whole category
    python -m articraft_canon --category robotic_arm

    # Emit canonical URDFs + sidecars for the full fit set
    python -m articraft_canon --write --output-dir data/urdfs/Canonical

    # Force a specific reference object in a sub-category
    python -m articraft_canon --category articulated_task_lamp \
        --exemplar articulated_task_lamp=rec_articulated_task_lamp_0001
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path
from typing import Dict, List

from . import dataset
from .pipeline import PipelineConfig, run_pipeline

_DEFAULT_DATASET = Path("data/urdfs/Dataset")
_DEFAULT_OUTPUT = Path("data/urdfs/Canonical")


def _parse_exemplars(items: List[str]) -> Dict[str, str]:
    """Parse ``--exemplar sub_category=object_id`` overrides into a dict."""
    out: Dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--exemplar expects sub_category=object_id, got '{item}'")
        sub, obj = item.split("=", 1)
        out[sub.strip()] = obj.strip()
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="articraft_canon",
        description="Canonicalize the Articraft URDF dataset (analytic, no learning).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset-root", type=Path, default=_DEFAULT_DATASET,
                   help=f"Dataset root (default: {_DEFAULT_DATASET}).")
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT,
                   help=f"Where canonical outputs go (default: {_DEFAULT_OUTPUT}).")
    p.add_argument("--work-dir", type=Path, default=None,
                   help="Scratch dir for extracted archives (default: a temp dir).")

    # Selection filters.
    p.add_argument("--category", action="append", default=None,
                   help="Restrict to a top-level category (repeatable).")
    p.add_argument("--sub-category", action="append", default=None,
                   help="Restrict to a sub-category leaf (repeatable).")
    p.add_argument("--split", action="append", default=None,
                   choices=["train", "val", "test"],
                   help="Restrict to a split (repeatable).")
    p.add_argument("--include-not-fit", action="store_true",
                   help="Also process 'not-fit' archives (default: fit only).")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N objects (handy for smoke tests).")

    # Behaviour.
    p.add_argument("--write", action="store_true",
                   help="Emit canonical URDFs + sidecars. Default is dry-run.")
    p.add_argument("--no-align-link-frames", action="store_true",
                   help="Do not re-orient emitted URDF link frames to the base "
                        "(Option B). By default every link frame is rewritten to "
                        "the base orientation, translated to the link origin.")
    p.add_argument("--check-collision", action="store_true",
                   help="Enable the Stage 2 self-collision test (off by default: "
                        "the conservative AABB test false-positives on the "
                        "concentric/nested designs common in this dataset).")
    p.add_argument("--first-link-reference", action="store_true",
                   help="Stage 2: seed the straightening from the first moving "
                        "link instead of the base (leave the first moving joint "
                        "at q*=0). Default aligns from the base, i.e. the first "
                        "moving link is straightened relative to the base too.")
    p.add_argument("--unreachable", choices=["mid-limit", "clamp", "extend-limits"],
                   default="mid-limit",
                   help="Stage 2 policy when the extended q* exceeds a joint "
                        "limit: 'mid-limit' (spec default) drops to the mid-limit "
                        "pose; 'clamp' keeps the most-extended reachable pose; "
                        "'extend-limits' widens the limits to admit the straight "
                        "pose when it is collision-free (real mesh collision), "
                        "else falls back to mid-limit.")
    p.add_argument("--points-per-link", type=int, default=2048,
                   help="Surface samples per link for geometry stages.")
    p.add_argument("--exemplar", action="append", default=None,
                   help="Reference override: sub_category=object_id (repeatable).")
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="-v for INFO, -vv for DEBUG logging.")
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    refs = dataset.discover_archives(
        args.dataset_root,
        fit_only=not args.include_not_fit,
        categories=args.category,
        sub_categories=args.sub_category,
        splits=args.split,
    )
    if args.limit is not None:
        refs = refs[: args.limit]
    if not refs:
        print("No matching archives found.")
        return 1

    # Use a managed temp dir unless the user pinned one (so we can clean up).
    work_ctx = (
        tempfile.TemporaryDirectory(prefix="articraft_work_")
        if args.work_dir is None else None
    )
    work_dir = Path(work_ctx.name) if work_ctx else args.work_dir

    cfg = PipelineConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        work_dir=work_dir,
        write=args.write,
        check_collision=args.check_collision,
        first_moving_is_reference=args.first_link_reference,
        unreachable_policy=args.unreachable,
        align_link_frames=not args.no_align_link_frames,
        n_points_per_link=args.points_per_link,
        exemplars=_parse_exemplars(args.exemplar),
        fit_only=not args.include_not_fit,
    )

    try:
        run_pipeline(cfg, refs)
    finally:
        if work_ctx is not None:
            work_ctx.cleanup()
    return 0
