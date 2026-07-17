"""Tiny dependency-free renderer: colour a point cloud into a 2D image.

Used to draw the object's live NOCS / NPCS as flat images in the GUI side panel,
so they can be eyeballed against the uploaded reference pictures.  Orthographic
3/4 view, painter's algorithm (far points first), small square splats.
"""

from __future__ import annotations

import numpy as np

# Fixed 3/4 view direction and up vector.
_VIEW = np.array([1.0, -1.0, 0.55])
_UP = np.array([0.0, 0.0, 1.0])


def _basis():
    view = _VIEW / np.linalg.norm(_VIEW)
    right = np.cross(view, _UP)
    right /= np.linalg.norm(right)
    up = np.cross(right, view)
    return right, up, view


def render_points_image(
    points: np.ndarray,
    colors: np.ndarray,
    size: int = 320,
    splat: int = 2,
    background=(255, 255, 255),
) -> np.ndarray:
    """Rasterise ``(N,3)`` ``points`` with ``(N,3)`` uint8 ``colors`` to an image."""
    img = np.full((size, size, 3), background, dtype=np.uint8)
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return img
    right, up, view = _basis()

    u = pts @ right
    v = pts @ up
    depth = pts @ view

    # Fit the projection to the image with a small margin.
    lo = np.array([u.min(), v.min()])
    hi = np.array([u.max(), v.max()])
    span = float(max(hi - lo).max() if np.ndim(hi - lo) else max(hi - lo))
    span = max(span, 1e-6)
    margin = 0.08 * size
    scale = (size - 2 * margin) / span
    px = (margin + (u - lo[0]) * scale).astype(int)
    py = (size - margin - (v - lo[1]) * scale).astype(int)  # flip y for image coords

    order = np.argsort(depth)  # far -> near, so near overwrites
    cols = np.asarray(colors, dtype=np.uint8)
    for i in order:
        x, y = px[i], py[i]
        x0, x1 = max(0, x), min(size, x + splat)
        y0, y1 = max(0, y), min(size, y + splat)
        if x1 > x0 and y1 > y0:
            img[y0:y1, x0:x1] = cols[i]
    return img
