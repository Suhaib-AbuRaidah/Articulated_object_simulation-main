"""Per-object state for the single-object verifier.

The editable inputs are **per-link coordinate frames**, **joint states**, and
**joint limits**, plus a physical **geometry/axis orientation**. All are committed
to the emitted URDF when accepted.

  * ``link_euler[name]`` -- an extra world-frame rotation of *that link's* frame
    (default 0 = the Stage-3 canonical orientation, or the original URDF frame
    in raw mode).  Rotating a link's frame
    rotates its triad and re-derives its NPCS; rotating the **base** link's frame
    (= the object root frame) re-derives NOCS.  Nothing rotates the object as a
    whole; physical geometry/axis correction is handled by ``object_euler``.
  * ``cfg`` -- joint states.
  * ``joint_limits`` -- editable lower/upper limits in the current joint
    coordinates. When ``cfg`` is baked as the new zero, these limits shift by
    ``-cfg`` with it.
  * ``object_euler`` -- a physical orientation correction applied to every
    link's visual/collision/inertial data and every movable joint axis. Link-
    frame orientations stay fixed, while frame/pivot positions rotate about
    the object base so adjacent links remain connected.

NOCS is computed **in the root (base-link) frame**; NPCS is computed **in each
link's own frame** -- never in the viewer's global frame.  Geometry for the 3D
view is only centred + unit-scaled (``V``, no rotation), leaving the explicit
physical orientation correction visible against the fixed global axes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Set, Tuple

import numpy as np
import trimesh
import yourdfpy
from scipy.spatial.transform import Rotation

from articraft_canon import canonical_zero, geometry as geo, parse, root_frame
from articraft_canon.parse import JointSpec, ObjectModel
from articraft_canon.root_frame import RootFrame

from .coloring import _to_rgb
from .normalization import (
    factor_and_corners,
    normalize,
    npcs_to_nocs_similarity,
)


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


def _identity_root_frame() -> RootFrame:
    """Neutral root frame used when automatic canonicalization is disabled."""
    return RootFrame(
        centroid=np.zeros(3),
        rotation=np.eye(3),
        scale=1.0,
        symmetry_group=[np.eye(3)],
    )


def _build_link_meshes(model: ObjectModel) -> Dict[str, trimesh.Trimesh]:
    urdf_dir = model.urdf_path.parent
    out: Dict[str, trimesh.Trimesh] = {}
    for link in model.urdf.robot.links:
        pieces = [parse._visual_to_mesh(v, urdf_dir) for v in link.visuals]
        pieces = [m for m in pieces if m is not None]
        if pieces:
            out[link.name] = trimesh.util.concatenate(pieces)
    return out


def _bake_link_frame_rotations(
    model: ObjectModel,
    old_world: Mapping[str, np.ndarray],
    desired_world_rotations: Mapping[str, np.ndarray],
) -> None:
    """Bake desired q=0 link-frame rotations into the URDF.

    Each link origin stays at its current q=0 position. Visual, collision,
    inertial, point-cloud, mesh, joint-origin, and joint-axis data are
    re-expressed in the edited frames. Thus geometry and kinematics are
    preserved, while a normal URDF viewer sees the edited frames directly.

    The base link has no parent transform in URDF, so its edited orientation is
    used as the new object-coordinate gauge. Geometry is counter-rotated into
    that gauge, making the edited base axes visible relative to the object.
    """
    target_world: Dict[str, np.ndarray] = {}
    for name in model.chain:
        target = np.eye(4)
        target[:3, :3] = _orthonormalize(desired_world_rotations[name])
        target[:3, 3] = np.asarray(old_world[name], float)[:3, 3]
        target_world[name] = target

    base_inverse = np.linalg.inv(target_world[model.base_link])
    new_world = {
        name: base_inverse @ target_world[name]
        for name in model.chain
    }
    # Maps coordinates in each old local frame into its edited local frame.
    old_to_new_local = {
        name: np.linalg.inv(target_world[name]) @ np.asarray(old_world[name], float)
        for name in model.chain
    }

    urdf_joints = {joint.name: joint for joint in model.urdf.robot.joints}
    for joint in model.joints:
        new_origin = (
            np.linalg.inv(new_world[joint.parent]) @ new_world[joint.child]
        )
        local_change = old_to_new_local[joint.child][:3, :3]
        new_axis = local_change @ np.asarray(joint.axis, float)
        axis_norm = np.linalg.norm(new_axis)
        if axis_norm > 1e-12:
            new_axis = new_axis / axis_norm

        joint.origin = new_origin
        joint.axis = new_axis
        urdf_joint = urdf_joints[joint.name]
        urdf_joint.origin = new_origin
        if urdf_joint.axis is not None:
            urdf_joint.axis = new_axis

    urdf_links = {link.name: link for link in model.urdf.robot.links}
    for name in model.chain:
        local_change = old_to_new_local[name]
        urdf_link = urdf_links.get(name)
        if urdf_link is not None:
            for group in ("visuals", "collisions"):
                for item in getattr(urdf_link, group, None) or []:
                    origin = (
                        np.asarray(item.origin, float)
                        if item.origin is not None
                        else np.eye(4)
                    )
                    item.origin = local_change @ origin
            inertial = getattr(urdf_link, "inertial", None)
            if inertial is not None:
                origin = (
                    np.asarray(inertial.origin, float)
                    if inertial.origin is not None
                    else np.eye(4)
                )
                inertial.origin = local_change @ origin

        link = model.links.get(name)
        if link is not None:
            if link.points.size:
                link.points = geo.transform_points(local_change, link.points)
            if link.mesh is not None:
                link.mesh.apply_transform(local_change)


def _coordinate_base_link(model: ObjectModel) -> str:
    """Return the original object base behind our optional global wrapper."""
    for joint in model.joints:
        if (
            joint.parent == model.base_link
            and joint.type == "fixed"
            and joint.name.startswith("articraft_global_to_base")
        ):
            return joint.child
    return model.base_link


def _bake_geometry_axis_and_pivot_rotation(
    model: ObjectModel,
    rotation_world: np.ndarray,
) -> None:
    """Correct geometry/axes while preserving link-frame orientations.

    ``rotation_world`` is conjugated into every link/joint frame independently.
    Visual, collision, inertial, sampled-point, and mesh data are rotated about
    their link origins. Link origins and joint pivots rotate about the object
    base, so joint-origin translations change but joint-origin rotations do not.
    """
    correction_world = _orthonormalize(rotation_world)
    if np.allclose(correction_world, np.eye(3), rtol=0.0, atol=1e-12):
        return

    world = geo.link_world_transforms(model.base_link, model.joints, {})
    object_base = _coordinate_base_link(model)
    if object_base not in world:
        raise ValueError(f"object base link '{object_base}' has no world transform")
    pivot_world = np.asarray(world[object_base], dtype=float)[:3, 3]
    affected = set(model.chain[model.chain.index(object_base):])
    target_world: Dict[str, np.ndarray] = {}
    for name, transform in world.items():
        target = np.asarray(transform, dtype=float).copy()
        if name in affected:
            offset = target[:3, 3] - pivot_world
            target[:3, 3] = pivot_world + correction_world @ offset
        # Deliberately retain target[:3, :3]: frame orientation is unchanged.
        target_world[name] = target

    urdf_links = {link.name: link for link in model.urdf.robot.links}
    for name in affected:
        if name not in world:
            continue
        rotation_world_link = np.asarray(world[name], dtype=float)[:3, :3]
        correction_local = np.eye(4)
        correction_local[:3, :3] = _orthonormalize(
            rotation_world_link.T @ correction_world @ rotation_world_link
        )

        urdf_link = urdf_links.get(name)
        if urdf_link is not None:
            for group in ("visuals", "collisions"):
                for item in getattr(urdf_link, group, None) or []:
                    origin = (
                        np.asarray(item.origin, dtype=float)
                        if item.origin is not None
                        else np.eye(4)
                    )
                    item.origin = correction_local @ origin
            inertial = getattr(urdf_link, "inertial", None)
            if inertial is not None:
                origin = (
                    np.asarray(inertial.origin, dtype=float)
                    if inertial.origin is not None
                    else np.eye(4)
                )
                # The inertia-frame axes rotate with the rigid body, so its
                # numeric tensor remains valid in that carried frame.
                inertial.origin = correction_local @ origin

        link = model.links.get(name)
        if link is not None:
            if link.points.size:
                link.points = geo.transform_points(correction_local, link.points)
            if link.mesh is not None:
                link.mesh.apply_transform(correction_local)

    urdf_joints = {joint.name: joint for joint in model.urdf.robot.joints}
    for joint in model.moving_joints:
        if joint.child not in affected or joint.child not in world:
            continue
        rotation_world_joint = np.asarray(world[joint.child], dtype=float)[:3, :3]
        correction_joint = _orthonormalize(
            rotation_world_joint.T @ correction_world @ rotation_world_joint
        )
        axis = correction_joint @ np.asarray(joint.axis, dtype=float)
        norm = np.linalg.norm(axis)
        if norm <= 1e-12:
            raise ValueError(f"joint '{joint.name}' has a degenerate corrected axis")
        axis = axis / norm
        joint.axis = axis
        urdf_joint = urdf_joints[joint.name]
        if urdf_joint.axis is not None:
            urdf_joint.axis = axis.copy()

    # Move every child pivot with the corrected parent geometry. Computing the
    # translation from target world poses handles arbitrarily oriented parent
    # frames. Keep the original origin rotation exactly, rather than deriving
    # an equivalent matrix and introducing Euler/round-off drift.
    for joint in model.joints:
        if joint.child not in affected:
            continue
        relative_target = (
            np.linalg.inv(target_world[joint.parent]) @ target_world[joint.child]
        )
        joint.origin[:3, 3] = relative_target[:3, 3]
        urdf_joints[joint.name].origin = joint.origin.copy()


@dataclass
class ObjectState:
    model: ObjectModel
    frame: RootFrame                             # Stage-3 canonical orientation R0
    raw_urdf: bool = False                       # preserve source joints/link frames
    frames_baked: bool = False                   # edited frames now live in the URDF
    counter_rotate_joint_frames: Set[str] = field(default_factory=set)
    coordinate_base_link: Optional[str] = None   # NOCS/original object base
    object_euler: np.ndarray = field(default_factory=lambda: np.zeros(3))
    link_euler: Dict[str, np.ndarray] = field(default_factory=dict)  # per-link world-euler (deg)
    cfg: Dict[str, float] = field(default_factory=dict)
    joint_limits: Dict[str, np.ndarray] = field(default_factory=dict)
    applied_link_euler: Dict[str, np.ndarray] = field(default_factory=dict)
    applied_object_euler: np.ndarray = field(default_factory=lambda: np.zeros(3))
    applied_joint_cfg: Dict[str, float] = field(default_factory=dict)
    applied_counter_rotate_joint_frames: List[str] = field(default_factory=list)
    requested_joint_limits: Dict[str, np.ndarray] = field(default_factory=dict)
    applied_joint_limits: Dict[str, np.ndarray] = field(default_factory=dict)
    _meshes: Dict[str, trimesh.Trimesh] = field(default_factory=dict)
    _view: np.ndarray = field(default_factory=lambda: np.eye(4))     # centre+scale, no rotation

    # ------------------------------------------------------------------ #
    @classmethod
    def from_model(
        cls,
        model: ObjectModel,
        *,
        unreachable_policy: str = "extend-limits",
        raw_urdf: bool = False,
    ):
        if raw_urdf:
            # Parsing is read-only: retain the source joint origins, limits, zero
            # state, and link frames exactly as they appear in the URDF.
            frame = _identity_root_frame()
        else:
            canonical_zero.canonicalize_zero(
                model,
                check_collision=False,
                unreachable_policy=unreachable_policy,
                apply=True,
            )
            frame = root_frame.compute_root_frame(model)
        st = cls(
            model=model,
            frame=frame,
            raw_urdf=raw_urdf,
            coordinate_base_link=_coordinate_base_link(model),
            _meshes=_build_link_meshes(model),
        )
        st.link_euler = {name: np.zeros(3) for name in model.chain}
        st.joint_limits = {
            joint.name: np.array([joint.lower, joint.upper], dtype=float)
            for joint in model.moving_joints
        }
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
        return self.coordinate_base_link or self.model.base_link

    def object_link_names(self) -> List[str]:
        """Editable/displayable object links (excluding our global wrapper)."""
        if self.base_link() == self.model.base_link:
            return list(self.model.chain)
        return [name for name in self.model.chain if name != self.model.base_link]

    def link_R(
        self,
        name: str,
        model_world: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """Displayed orientation of an editable canonical or URDF link frame."""
        return self._link_R_model(name, model_world)

    def _link_R_model(
        self,
        name: str,
        model_world: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """Editable frame orientation; physical correction does not alter it."""
        delta = self.link_euler.get(name, np.zeros(3))
        extra = Rotation.from_euler("xyz", delta, degrees=True).as_matrix()
        if (self.raw_urdf or self.frames_baked) and self.counter_rotate_joint_frames:
            # Remove only the selected joints' rotation contributions from the
            # displayed frames. A selected joint therefore counter-rotates the
            # coordinate frames of its child and all descendants, while
            # unselected joint rotations remain visible. Geometry and physical
            # joint axes always use the full configuration in ``model_world``.
            frame_cfg = {
                joint_name: value
                for joint_name, value in self.cfg.items()
                if joint_name not in self.counter_rotate_joint_frames
            }
            frame_world = geo.link_world_transforms(
                self.model.base_link,
                self.model.joints,
                frame_cfg,
            )
            base_rotation = frame_world[name][:3, :3]
        elif self.raw_urdf or self.frames_baked:
            transforms = self._model_world() if model_world is None else model_world
            base_rotation = transforms[name][:3, :3]
        else:
            base_rotation = np.asarray(self.frame.rotation, float)
        return extra @ base_rotation

    def object_rotation(self) -> np.ndarray:
        return Rotation.from_euler(
            "xyz",
            np.asarray(self.object_euler, dtype=float),
            degrees=True,
        ).as_matrix()

    def _model_world(self) -> Dict[str, np.ndarray]:
        return geo.link_world_transforms(self.model.base_link, self.model.joints, self.cfg)

    def _world(
        self,
        model_world: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """Frames with corrected positions and unchanged orientations."""
        transforms = self._model_world() if model_world is None else model_world
        correction = self.object_rotation()
        base = self.base_link()
        pivot = np.asarray(transforms[base], dtype=float)[:3, 3]
        affected = set(self.object_link_names())
        corrected: Dict[str, np.ndarray] = {}
        for name, transform in transforms.items():
            frame = np.asarray(transform, dtype=float).copy()
            if name in affected:
                frame[:3, 3] = pivot + correction @ (frame[:3, 3] - pivot)
            corrected[name] = frame
        return corrected

    def _geometry_world(
        self,
        model_world: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """Rigidly corrected geometry with stationary frame orientations."""
        transforms = self._model_world() if model_world is None else model_world
        corrected_frames = self._world(transforms)
        correction = self.object_rotation()
        corrected: Dict[str, np.ndarray] = {}
        for name, transform in transforms.items():
            transform = np.asarray(transform, dtype=float)
            geometry_transform = corrected_frames[name].copy()
            geometry_transform[:3, :3] = correction @ transform[:3, :3]
            corrected[name] = geometry_transform
        return corrected

    # ------------------------------------------------------------------ #
    # 3D view geometry (centred + scaled only; no global rotation)
    # ------------------------------------------------------------------ #
    def global_frame_position_view(self) -> np.ndarray:
        """Position of the fixed global/base origin in viewer coordinates."""
        return geo.transform_points(self._view, np.zeros((1, 3)))[0]

    def link_meshes_view(self) -> Dict[str, trimesh.Trimesh]:
        world = self._geometry_world()
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
        model_world = self._model_world()
        world = self._world(model_world)
        wxyzs, positions = [], []
        for name in self.object_link_names():
            if name not in world:
                continue
            positions.append(geo.transform_points(self._view, world[name][:3, 3][None, :])[0])
            wxyzs.append(_mat_to_wxyz(self.link_R(name, model_world)))
        return np.asarray(wxyzs, float), np.asarray(positions, float)

    def joint_axes_view(self, length: float = 0.18) -> List[Tuple[np.ndarray, np.ndarray]]:
        world = self._world()
        correction = self.object_rotation()
        segs = []
        for j in self.model.moving_joints:
            if j.child not in world:
                continue
            pivot = geo.transform_points(self._view, world[j.child][:3, 3][None, :])[0]
            d = correction @ geo.rotate_vectors(world[j.child], j.axis)
            d = d / (np.linalg.norm(d) + 1e-9) * length
            segs.append((pivot - d, pivot + d))
        return segs

    # ------------------------------------------------------------------ #
    # NOCS (root frame) / NPCS (per-link frame)
    # ------------------------------------------------------------------ #
    def _world_points(self, max_points: int = 12000):
        world = self._geometry_world()
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
        model_world = self._model_world()

        # NOCS: whole object expressed in the base-link (root) frame.
        R_base = self.link_R(self.base_link(), model_world)
        nocs_local = world_pts @ R_base            # == (R_base^T @ p^T)^T
        nocs_cols = _nocs_unit(nocs_local)

        # NPCS: each part expressed in *its own* link frame.
        npcs_cols = np.zeros((len(world_pts), 3), np.uint8)
        for i, name in enumerate(self.model.chain):
            mask = ids == i
            if not np.any(mask):
                continue
            local = world_pts[mask] @ self.link_R(name, model_world)
            npcs_cols[mask] = _nocs_unit(local)
        return display, nocs_cols, npcs_cols, ids

    # ------------------------------------------------------------------ #
    # Edits
    # ------------------------------------------------------------------ #
    def set_object_euler(self, euler) -> None:
        self.object_euler = np.asarray(euler, dtype=float)

    def set_counter_rotate_joint_frames(self, name: str, enabled: bool) -> None:
        """Toggle frame compensation for one rotational joint's child subtree."""
        joint = next((item for item in self.model.moving_joints if item.name == name), None)
        if joint is None:
            raise KeyError(f"unknown moving joint: {name}")
        if joint.type not in ("revolute", "continuous"):
            raise ValueError(
                f"joint '{name}' is {joint.type}; it has no rotational state to compensate"
            )
        if enabled:
            self.counter_rotate_joint_frames.add(name)
        else:
            self.counter_rotate_joint_frames.discard(name)

    def snap_object_90(self) -> None:
        self.object_euler = np.round(self.object_euler / 90.0) * 90.0

    def reset_object_orientation(self) -> None:
        self.object_euler = np.zeros(3)

    def set_link_euler(self, name: str, euler) -> None:
        self.link_euler[name] = np.asarray(euler, float)

    def snap_link_90(self, name: str) -> None:
        e = self.link_euler.get(name, np.zeros(3))
        self.link_euler[name] = np.round(e / 90.0) * 90.0

    def reset_link(self, name: str) -> None:
        self.link_euler[name] = np.zeros(3)

    def reset_all_links(self) -> None:
        self.link_euler = {name: np.zeros(3) for name in self.model.chain}

    def set_joint_limits(self, name: str, lower: float, upper: float) -> None:
        joint = next((item for item in self.model.moving_joints if item.name == name), None)
        if joint is None:
            raise KeyError(f"unknown moving joint: {name}")
        if joint.type == "continuous":
            raise ValueError(f"continuous joint '{name}' has no finite URDF range")
        values = np.asarray([lower, upper], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"joint limits must be finite: {name}")
        if values[0] > values[1]:
            raise ValueError(f"joint lower limit exceeds upper limit: {name}")
        self.joint_limits[name] = values

    def joint_limit_edits(self) -> Dict[str, np.ndarray]:
        edits: Dict[str, np.ndarray] = {}
        for joint in self.model.moving_joints:
            pending = self.joint_limits.get(
                joint.name,
                np.array([joint.lower, joint.upper], dtype=float),
            )
            current = np.array([joint.lower, joint.upper], dtype=float)
            if not np.allclose(pending, current, rtol=0.0, atol=1e-12):
                edits[joint.name] = np.asarray(pending, dtype=float).copy()
        return edits

    def _apply_joint_limits(self) -> None:
        """Write pending ranges into JointSpecs and the serializable URDF."""
        urdf_joints = {joint.name: joint for joint in self.model.urdf.robot.joints}
        for joint in self.model.moving_joints:
            limits = self.joint_limits.get(joint.name)
            if limits is None:
                continue
            lower, upper = map(float, limits)
            if joint.type == "continuous":
                current = np.array([joint.lower, joint.upper], dtype=float)
                if not np.allclose(limits, current, rtol=0.0, atol=1e-12):
                    raise ValueError(
                        f"continuous joint '{joint.name}' has no finite URDF range"
                    )
                continue

            joint.lower = lower
            joint.upper = upper
            urdf_joint = urdf_joints[joint.name]
            if urdf_joint.limit is None:
                urdf_joint.limit = yourdfpy.Limit(
                    effort=1.0,
                    velocity=1.0,
                    lower=lower,
                    upper=upper,
                )
            else:
                urdf_joint.limit.lower = lower
                urdf_joint.limit.upper = upper

    def has_pending_edits(self) -> bool:
        joint_changed = any(abs(float(value)) > 1e-12 for value in self.cfg.values())
        frame_changed = any(np.any(np.abs(e) > 1e-12) for e in self.link_euler.values())
        object_changed = np.any(np.abs(self.object_euler) > 1e-12)
        return (
            joint_changed
            or frame_changed
            or object_changed
            or bool(self.joint_limit_edits())
        )

    def commit_edits(self) -> None:
        """Bake object orientation, joint state/ranges, and frames into URDF."""
        current_world = self._model_world()
        desired_rotations = {
            name: self._link_R_model(name, current_world).copy()
            for name in self.model.chain
        }
        object_rotation = self.object_rotation()
        self.applied_object_euler = np.asarray(self.object_euler, float).copy()
        self.applied_joint_cfg = {
            name: float(value) for name, value in self.cfg.items()
        }
        self.applied_counter_rotate_joint_frames = sorted(
            self.counter_rotate_joint_frames
        )
        self.applied_link_euler = {
            name: np.asarray(value, float).copy()
            for name, value in self.link_euler.items()
        }
        self.requested_joint_limits = self.joint_limit_edits()

        # Limits are expressed in the current joint coordinates. Apply them
        # before the pose bake so _bake shifts both endpoints by -cfg together
        # with the new q=0 origin.
        self._apply_joint_limits()
        # Make the currently displayed articulation the new q=0 pose.
        canonical_zero._bake(self.model, dict(self.cfg))
        # Then reparameterize all actual URDF link frames to match the edited
        # triads that were displayed at that pose.
        _bake_link_frame_rotations(
            self.model,
            current_world,
            desired_rotations,
        )
        # Finally correct physical geometry and motion axes, moving each pivot
        # with the connected geometry while retaining frame/origin rotations.
        _bake_geometry_axis_and_pivot_rotation(self.model, object_rotation)

        self.cfg = {}
        self.object_euler = np.zeros(3)
        self.link_euler = {name: np.zeros(3) for name in self.model.chain}
        self.joint_limits = {
            joint.name: np.array([joint.lower, joint.upper], dtype=float)
            for joint in self.model.moving_joints
        }
        self.applied_joint_limits = {
            joint.name: self.joint_limits[joint.name].copy()
            for joint in self.model.moving_joints
            if joint.type != "continuous"
        }
        self.frames_baked = True
        self._meshes = _build_link_meshes(self.model)
        self._recompute_view()

    # ------------------------------------------------------------------ #
    # Sidecar (NOCS root frame, NPCS per-link frame)
    # ------------------------------------------------------------------ #
    def build_sidecar(self) -> Dict:
        world = self._model_world()  # accepted state/orientation are now q = 0
        R_base = self.link_R(self.base_link(), world)

        part_world: Dict[str, np.ndarray] = {}
        for name in self.model.chain:
            link = self.model.links.get(name)
            if link is None or link.points.size == 0 or name not in world:
                continue
            part_world[name] = geo.transform_points(world[name], link.points)

        obj_pts = (np.concatenate(list(part_world.values())) if part_world
                   else np.zeros((1, 3)))
        obj_local = obj_pts @ R_base
        obj_factor, obj_bmin, obj_bmax = factor_and_corners(obj_local)

        inbound: Dict[str, JointSpec] = {j.child: j for j in self.model.joints}
        parts: List[Dict] = []
        for name in self.model.chain:
            if name not in part_world:
                continue
            R_L = self.link_R(name, world)
            local = part_world[name] @ R_L
            pfactor, pmin, pmax = factor_and_corners(local)
            # NPCS (part frame) -> NOCS (root frame): rotation + scale + translation.
            rot = R_base.T @ R_L
            scale, translation = npcs_to_nocs_similarity(
                pfactor,
                pmin,
                pmax,
                obj_factor,
                obj_bmin,
                obj_bmax,
                rot,
            )
            entry = {
                "link": name,
                "frame_rotation_world": R_L.tolist(),
                "frame_euler_deg": list(
                    map(float, self.applied_link_euler.get(name, np.zeros(3)))
                ),
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
            "base_link": self.base_link(),
            "object_coordinate_base_link": self.base_link(),
            "urdf_root_link": self.model.base_link,
            "chain": self.model.chain,
            "canonical_fallback": self.model.canonical_fallback,
            "root_frame_rotation_world": R_base.tolist(),
            "nocs": {
                "bbox_min": obj_bmin.tolist(), "bbox_max": obj_bmax.tolist(),
                "factor": float(obj_factor),
                "note": (
                    "nocs = 0.5 + factor * ((p @ R_base) - bbox_center); "
                    "factor = 1 / ||bbox_max - bbox_min||_2"
                ),
            },
            "parts": parts,
            "verification": {
                "raw_urdf": self.raw_urdf,
                "fixed_branches_allowed": self.model.fixed_branches_allowed,
                "converted_mimic_joints": list(
                    self.model.converted_mimic_joints
                ),
                "frames_baked_into_urdf": self.frames_baked,
                "applied_joint_state": dict(self.applied_joint_cfg),
                "counter_rotated_link_frames_for_joint_state": (
                    bool(self.applied_counter_rotate_joint_frames)
                ),
                "counter_rotated_link_frames_for_joints": list(
                    self.applied_counter_rotate_joint_frames
                ),
                "applied_object_euler_deg": list(
                    map(float, self.applied_object_euler)
                ),
                "requested_joint_limits": {
                    name: list(map(float, limits))
                    for name, limits in self.requested_joint_limits.items()
                },
                "baked_joint_limits": {
                    name: list(map(float, limits))
                    for name, limits in self.applied_joint_limits.items()
                },
                "applied_link_euler_deg": {
                    name: list(map(float, e))
                    for name, e in self.applied_link_euler.items()
                },
            },
        }


def _nocs_unit(local: np.ndarray) -> np.ndarray:
    """Paper-compatible diagonal-normalise ``local`` and map to uint8 RGB."""
    if len(local) == 0:
        return np.zeros((0, 3), np.uint8)
    factor, bmin, bmax = factor_and_corners(local)
    return _to_rgb(normalize(local, factor, bmin, bmax))
