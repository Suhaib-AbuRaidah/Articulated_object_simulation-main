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
from .objectstate import MODE_FRAME_FIX, MODE_REST_POSE
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
    p.add_argument("--mode", choices=[MODE_FRAME_FIX, MODE_REST_POSE], default=MODE_FRAME_FIX,
                   help="Default edit mode: frame-fix (frames only) or rest-pose (re-bake).")
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
    )
    if not catalog.refs:
        print(f"No objects found under {args.dataset_root} with the given filters.")
        return 1

    store = DecisionStore(args.output_dir)
    app = VerifierApp(
        catalog, store, args.reference_dir,
        host=args.host, port=args.port,
        default_mode=args.mode, skip_done=not args.no_skip_done,
    )
    app.run()
    return 0
