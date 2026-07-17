"""Articraft canonicalization verification GUI (Viser).

A human-in-the-loop web tool to verify and correct the frames/rest-pose produced
by ``articraft_canon``, one object at a time, and re-bake accepted corrections
into a separate output tree (originals untouched).

Run ``python -m articraft_verify --help``.
"""

__all__ = ["catalog", "coloring", "render", "references", "objectstate", "store", "app", "cli"]
__version__ = "0.1.0"
