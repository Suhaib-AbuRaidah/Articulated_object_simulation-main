"""Per-object state for the single-object verifier.

The editable inputs are **per-link coordinate frames** and the **joint states**.

  * ``link_euler[name]`` -- an extra world-frame rotation of *that link's* frame
    (default 0 = the Stage-3 canonical orientation).  Rotating a link's frame
    rotates its triad and re-derives its NPCS; rotating the **base** link's frame
    (= the object root frame) re-derives NOCS.  Nothing rotates the object as a
    whole against a global frame.
  * ``cfg`` -- joint states.

NOCS is computed **in the root (base-link) frame**; NPCS is computed **in each
link's own frame** -- never in the viewer's global frame.  Geometry for the 3D
view is only centred + unit-scaled (``V``, no rotation), so the object keeps its
natural orientation while the editable frames float on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from articraft_canon import canonical_zero, geometry as geo, parse, root_frame
from articraft_canon.parse import JointSpec, ObjectModel
from articraft_canon.root_frame import RootFrame

from .coloring import _to_rgb

MODE_FRAME_FIX = "frame-fix"
MODE_REST_POSE = "rest-pose"


def _mat_to_wxyz(R: np.ndarray):
    x, y, z, w = Rotation.from_matrix(_orthonormalize(R)).as_quat()
    return (float(w), float(x), float(y), float(z))


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(R, float))
    M = U @ Vt
    if np.linalg.det(M) < 0:
        U[:, -1] *= -1
        M = U @ Vt
    return M


def _build_link_meshes(model: ObjectModel) -> Dict[str, trimesh.Trimesh]:
    urdf_dir = model.urdf_path.parent
    out: Dict[str, trimesh.Trimesh] = {}
    for link in model.urdf.robot.links:
        pieces = [parse._visual_to_mesh(v, urdf_dir) for v in link.visuals]
        pieces = [m for m in pieces if m is not None]
        if pieces:
            out[link.name] = trimesh.util.concatenate(pieces)
    return out


@dataclass
class ObjectState:
    model: ObjectModel
    frame: RootFrame                             # Stage-3 canonical orientation R0
    link_euler: Dict[str, np.ndarray] = field(default_factory=dict)  # per-link world-euler (deg)
    cfg: Dict[str, float] = field(default_factory=dict)
    mode: str = MODE_FRAME_FIX
    _meshes: Dict[str, trimesh.Trimesh] = field(default_factory=dict)
    _view: np.ndarray = field(default_factory=lambda: np.eye(4))     # centre+scale, no rotation

    # ------------------------------------------------------------------ #
    @classmethod
    def from_model(cls, model: ObjectModel, *, unreachable_policy: str = "extend-limits"):
        canonical_zero.canonicalize_zero(
            model, check_collision=False, unreachable_policy=unreachable_policy, apply=True)
        frame = root_frame.compute_root_frame(model)
        st = cls(model=model, frame=frame, _meshes=_build_link_meshes(model))
        st.link_euler = {name: np.zeros(3) for name in model.chain}
        st._recompute_view()
        return st

    def _recompute_view(self) -> None:
        pts = self.model.object_points(self.cfg)
        if pts.size == 0:
            self._view = np.eye(4)
            return
        centroid = pts.mean(axis=0)
        diag = geo.bbox_diagonal(pts) or 1.0
        s = 1.0 / diag
        V = np.eye(4)
        V[:3, :3] = s * np.eye(3)
        V[:3, 3] = -s * centroid
        self._view = V

    # ------------------------------------------------------------------ #
    # Per-link frames
    # ------------------------------------------------------------------ #
    def base_link(self) -> str:
        return self.model.base_link

    def link_R(self, name: str) -> np.ndarray:
        """World orientation of ``name``'s frame = extra-euler ∘ Stage-3 orientation."""
        delta = self.link_euler.get(name, np.zeros(3))
        extra = Rotation.from_euler("xyz", delta, degrees=True).as_matrix()
        return extra @ np.asarray(self.frame.rotation, float)

    def _world(self) -> Dict[str, np.ndarray]:
        return geo.link_world_transforms(self.model.base_link, self.model.joints, self.cfg)

    # ------------------------------------------------------------------ #
    # 3D view geometry (centred + scaled only; no global rotation)
    # ------------------------------------------------------------------ #
    def link_meshes_view(self) -> Dict[str, trimesh.Trimesh]:
        world = self._world()
        out = {}
        for name, mesh in self._meshes.items():
            if name not in world:
                continue
            m = mesh.copy()
            m.apply_transform(self._view @ world[name])
            out[name] = m
        return out

    def link_frames_view(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(wxyzs, positions)`` for each link's *own* frame (edited orientation)."""
        world = self._world()
        wxyzs, positions = [], []
        for name in self.model.chain:
            if name not in world:
                continue
            positions.append(geo.transform_points(self._view, world[name][:3, 3][None, :])[0])
            wxyzs.append(_mat_to_wxyz(self.link_R(name)))   # V has no rotation
        return np.asarray(wxyzs, float), np.asarray(positions, float)

    def joint_axes_view(self, length: float = 0.18) -> List[Tuple[np.ndarray, np.ndarray]]:
        world = self._world()
        segs = []
        for j in self.model.moving_joints:
            if j.child not in world:
                continue
            pivot = geo.transform_points(self._view, world[j.child][:3, 3][None, :])[0]
            d = geo.rotate_vectors(world[j.child], j.axis)
            d = d / (np.linalg.norm(d) + 1e-9) * length
            segs.append((pivot - d, pivot + d))
        return segs

    # ------------------------------------------------------------------ #
    # NOCS (root frame) / NPCS (per-link frame)
    # ------------------------------------------------------------------ #
    def _world_points(self, max_points: int = 12000):
        world = self._world()
        per = max(256, max_points // max(len(self.model.chain), 1))
        chunks, ids = [], []
        for name in self.model.chain:
            link = self.model.links.get(name)
            if link is None or link.points.size == 0 or name not in world:
                continue
            pts = geo.farthest_point_subsample(link.points, per, seed=0)
            chunks.append(geo.transform_points(world[name], pts))
            ids.append(np.full(len(pts), self.model.chain.index(name), np.int32))
        if not chunks:
            return np.zeros((0, 3)), np.zeros((0,), np.int32)
        return np.concatenate(chunks), np.concatenate(ids)

    def nocs_npcs_render(self):
        """Return ``(display_points, nocs_colors, npcs_colors)`` for the images.

        ``display_points`` are the view-normalised world positions (object shape);
        colours are NOCS (root frame) and NPCS (each part in its own link frame).
        """
        world_pts, ids = self._world_points()
        display = geo.transform_points(self._view, world_pts) if len(world_pts) else world_pts

        # NOCS: whole object expressed in the base-link (root) frame.
        R_base = self.link_R(self.base_link())
        nocs_local = world_pts @ R_base            # == (R_base^T @ p^T)^T
        nocs_cols = _nocs_unit(nocs_local)

        # NPCS: each part expressed in *its own* link frame.
        npcs_cols = np.zeros((len(world_pts), 3), np.uint8)
        for i, name in enumerate(self.model.chain):
            mask = ids == i
            if not np.any(mask):
                continue
            local = world_pts[mask] @ self.link_R(name)
            npcs_cols[mask] = _nocs_unit(local)
        return display, nocs_cols, npcs_cols, ids

    # ------------------------------------------------------------------ #
    # Edits
    # ------------------------------------------------------------------ #
    def set_link_euler(self, name: str, euler) -> None:
        self.link_euler[name] = np.asarray(euler, float)

    def snap_link_90(self, name: str) -> None:
        e = self.link_euler.get(name, np.zeros(3))
        self.link_euler[name] = np.round(e / 90.0) * 90.0

    def reset_link(self, name: str) -> None:
        self.link_euler[name] = np.zeros(3)

    def reset_all_links(self) -> None:
        self.link_euler = {name: np.zeros(3) for name in self.model.chain}

    def set_current_pose_as_zero(self) -> None:
        canonical_zero._bake(self.model, dict(self.cfg))
        self.cfg = {}
        self.frame = root_frame.compute_root_frame(self.model)
        self._meshes = _build_link_meshes(self.model)
        self._recompute_view()

    # ------------------------------------------------------------------ #
    # Sidecar (NOCS root frame, NPCS per-link frame)
    # ------------------------------------------------------------------ #
    def build_sidecar(self) -> Dict:
        world = self._world()  # q = 0 after any rest-pose re-bake
        R_base = self.link_R(self.base_link())

        part_world: Dict[str, np.ndarray] = {}
        for name in self.model.chain:
            link = self.model.links.get(name)
            if link is None or link.points.size == 0 or name not in world:
                continue
            part_world[name] = geo.transform_points(world[name], link.points)

        obj_pts = (np.concatenate(list(part_world.values())) if part_world
                   else np.zeros((1, 3)))
        obj_local = obj_pts @ R_base
        obj_bmin, obj_bmax, obj_factor = geo.unit_cube_normalization(obj_local)

        inbound: Dict[str, JointSpec] = {j.child: j for j in self.model.joints}
        parts: List[Dict] = []
        for name in self.model.chain:
            if name not in part_world:
                continue
            R_L = self.link_R(name)
            local = part_world[name] @ R_L
            pmin, pmax, pfactor = geo.unit_cube_normalization(local)
            # NPCS (part frame) -> NOCS (root frame): rotation + scale + translation.
            rot = R_base.T @ R_L
            scale = float(obj_factor / pfactor)
            translation = obj_factor * (rot @ pmin - obj_bmin)
            entry = {
                "link": name,
                "frame_rotation_world": R_L.tolist(),
                "frame_euler_deg": list(map(float, self.link_euler.get(name, np.zeros(3)))),
                "npcs_bbox_min": pmin.tolist(),
                "npcs_bbox_max": pmax.tolist(),
                "npcs_factor": float(pfactor),
                "npcs_to_nocs_rotation": rot.tolist(),
                "npcs_to_nocs_scale": scale,
                "npcs_to_nocs_translation": translation.tolist(),
            }
            j = inbound.get(name)
            if j is not None and j.is_moving:
                axis_world = geo.rotate_vectors(world[name], j.axis)
                entry["joint"] = {
                    "name": j.name, "type": j.type,
                    "axis_in_link_frame": (R_L.T @ axis_world).tolist(),
                    "limit_lower": float(j.lower), "limit_upper": float(j.upper),
                }
            else:
                entry["joint"] = None
            parts.append(entry)

        return {
            "object_id": self.model.object_id,
            "category": self.model.category,
            "sub_category": self.model.sub_category,
            "split": self.model.split,
            "base_link": self.model.base_link,
            "chain": self.model.chain,
            "canonical_fallback": self.model.canonical_fallback,
            "root_frame_rotation_world": R_base.tolist(),
            "nocs": {
                "bbox_min": obj_bmin.tolist(), "bbox_max": obj_bmax.tolist(),
                "factor": float(obj_factor),
                "note": "nocs = ((p @ R_base) - bbox_min) * factor  (root/base-link frame)",
            },
            "parts": parts,
            "verification": {
                "mode": self.mode,
                "link_euler_deg": {n: list(map(float, e)) for n, e in self.link_euler.items()},
            },
        }


def _nocs_unit(local: np.ndarray) -> np.ndarray:
    """Unit-cube-normalise ``local`` coords and map to uint8 RGB."""
    if len(local) == 0:
        return np.zeros((0, 3), np.uint8)
    bmin, _bmax, factor = geo.unit_cube_normalization(local)
    return _to_rgb((local - bmin) * factor)
