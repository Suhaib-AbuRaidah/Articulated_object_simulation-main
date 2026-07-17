"""Reference images loaded from a per-category folder.

Layout (created on demand; you drop images in yourself):

    <reference-dir>/<category>/*.png        # up to 3 reference pictures
                                            # e.g. reference URDF view, NOCS, NPCS

The folder is created empty if missing -- the tool never writes images, it only
reads whatever you place there.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def category_dir(reference_dir: Path, category: str) -> Path:
    """Return (and create) the reference folder for ``category``."""
    d = Path(reference_dir) / category
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_reference_images(
    reference_dir: Path, category: str, max_images: int = 3
) -> List[Tuple[str, np.ndarray]]:
    """Load up to ``max_images`` images from ``<reference-dir>/<category>/``."""
    from PIL import Image

    d = category_dir(reference_dir, category)
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in _EXTS)
    out: List[Tuple[str, np.ndarray]] = []
    for path in files[:max_images]:
        try:
            out.append((path.name, np.asarray(Image.open(path).convert("RGB"))))
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not load reference image %s: %s", path, exc)
    return out
