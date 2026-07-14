"""Articraft URDF dataset canonicalization pipeline (analytic, no learning).

A modular CLI that rewrites ~392 serial articulated objects into a consistent
canonical rest state and consistent link frames for NOCS/NPCS supervision.

Stages (one module each):
    1. parse            -- kinematic tree + per-link geometry
    2. canonical_zero   -- fully-extended canonical q = 0
    3. root_frame       -- per-object PCA frame, symmetry, scale
    4. subcat_frame     -- sub-category frame consistency (medoid + Umeyama)
    5. nocs             -- Option B link frames + NOCS/NPCS sidecar

Run ``python -m articraft_canon --help``.
"""

__all__ = [
    "geometry",
    "dataset",
    "parse",
    "canonical_zero",
    "root_frame",
    "subcat_frame",
    "nocs",
    "report",
    "pipeline",
    "cli",
]

__version__ = "0.1.0"
