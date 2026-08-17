"""Command-line entry point for the Articraft verification GUI.

    python -m articraft_verify --dataset-root data/urdfs/Dataset \
        --category robotic_arm --output-dir data/urdfs/Verified
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from . import catalog as catalog_mod
from .app import VerifierApp
from .store import DecisionStore

_DEFAULT_DATASET = Path("data/urdfs/Dataset")
_DEFAULT_OUTPUT = Path("data/urdfs/Verified")
_DEFAULT_REFERENCE = Path("data/urdfs/reference_images")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="articraft_verify",
        description="Viser GUI to human-verify and correct the Articraft canonicalization.",
    )
    p.add_argument("--dataset-root", type=Path, default=_DEFAULT_DATASET,
                   help=f"Dataset root (default: {_DEFAULT_DATASET}).")
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT,
                   help=f"Where verified outputs + decisions go (default: {_DEFAULT_OUTPUT}).")
    p.add_argument("--reference-dir", type=Path, default=_DEFAULT_REFERENCE,
                   help=f"Folder of per-category reference images (default: {_DEFAULT_REFERENCE}). "
                        "Created empty; drop up to 3 images in <reference-dir>/<category>/.")
    p.add_argument("--category", action="append", default=None,
                   help="Restrict to a top-level category (repeatable).")
    p.add_argument("--sub-category", action="append", default=None,
                   help="Restrict to a sub-category (repeatable).")
    p.add_argument("--split", action="append", default=None,
                   choices=["train", "val", "test"], help="Restrict to a split (repeatable).")
    p.add_argument("--include-not-fit", action="store_true",
                   help="Also include 'not-fit' archives (default: fit only).")
    p.add_argument(
        "--raw-urdf",
        action="store_true",
        help=(
            "Load the source URDF at its original q=0 state with its original "
            "joint origins, limits, and link coordinate frames. Skip automatic "
            "canonical-zero processing before interactive editing."
        ),
    )
    p.add_argument(
        "--allow-fixed-branches",
        action="store_true",
        help=(
            "Accept URDF trees with fixed-only side branches when the moving-"
            "joint skeleton is still serial. Fixed child links remain loaded, "
            "rendered, transformed, and saved with their parents."
        ),
    )
    p.add_argument(
        "--counter-rotate-link-frames",
        action="store_true",
        help=(
            "Start with frame counter-rotation enabled separately for every "
            "rotational joint. Each joint also has its own GUI checkbox. "
            "Geometry and physical joint axes still follow the full pose."
        ),
    )
    p.add_argument("--no-skip-done", action="store_true",
                   help="Do not jump past already-decided (unchanged) objects.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * min(args.verbose, 2),
                        format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("articraft_verify").setLevel(logging.INFO)

    catalog = catalog_mod.Catalog(
        dataset_root=args.dataset_root,
        work_dir=catalog_mod.make_work_dir(),
        fit_only=not args.include_not_fit,
        categories=args.category,
        sub_categories_filter=args.sub_category,
        splits=args.split,
        allow_fixed_branches=args.allow_fixed_branches,
    )
    if not catalog.refs:
        print(f"No objects found under {args.dataset_root} with the given filters.")
        return 1

    store = DecisionStore(args.output_dir)
    app = VerifierApp(
        catalog, store, args.reference_dir,
        host=args.host, port=args.port,
        skip_done=not args.no_skip_done,
        raw_urdf=args.raw_urdf,
        default_counter_rotate_link_frames=args.counter_rotate_link_frames,
    )
    app.run()
    return 0
