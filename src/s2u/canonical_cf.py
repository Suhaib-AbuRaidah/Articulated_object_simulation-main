"""
PyTorch implementation of an optimized canonical coordinate frame for 3D object normalization.

This is a practical GPU implementation of the main method from:
"Optimized Canonical Coordinate Frames for 3D Object Normalization".

Input:
    points:  Tensor [N, 3]
    normals: optional Tensor [N, 3]

Output:
    canonical center c, orthonormal axes A, normalized/local points.
"""

from __future__ import annotations

from dataclasses import dataclass
import glob
import os
from typing import Optional, Tuple, Dict, List
import math

import numpy as np
import torch
import torch.nn.functional as F

import open3d as o3d

@dataclass
class CanonicalFrameResult:
    center: torch.Tensor          # [3], world-space origin of canonical frame
    axes: torch.Tensor            # [3, 3], columns are world-space canonical axes
    normalized_points: torch.Tensor  # [N, 3], points in canonical frame after scale normalization
    local_points: torch.Tensor       # [N, 3], points in canonical frame before scale normalization
    scale: torch.Tensor              # scalar scale used for normalized_points
    energy: float
    info: Dict[str, float]


def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def _skew(w: torch.Tensor) -> torch.Tensor:
    """Return skew-symmetric matrix for a 3-vector."""
    wx, wy, wz = w[0], w[1], w[2]
    z = torch.zeros((), device=w.device, dtype=w.dtype)
    return torch.stack([
        torch.stack([z, -wz, wy]),
        torch.stack([wz, z, -wx]),
        torch.stack([-wy, wx, z]),
    ])


def so3_exp(rotvec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Exponential map from axis-angle vector [3] to SO(3) matrix [3, 3]."""
    theta2 = (rotvec * rotvec).sum()
    theta = torch.sqrt(theta2 + eps)
    K = _skew(rotvec)
    I = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype)

    small = theta < 1e-4
    A = torch.where(
        small,
        1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
        torch.sin(theta) / theta,
    )
    B = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - torch.cos(theta)) / (theta2 + eps),
    )
    return I + A * K + B * (K @ K)


def axis_rotation_matrix(axis_index: int, angle: float, device, dtype) -> torch.Tensor:
    """Column-vector rotation matrix around local x/y/z axis."""
    c = torch.tensor(math.cos(angle), device=device, dtype=dtype)
    s = torch.tensor(math.sin(angle), device=device, dtype=dtype)
    z = torch.zeros((), device=device, dtype=dtype)
    o = torch.ones((), device=device, dtype=dtype)

    if axis_index == 0:
        return torch.stack([
            torch.stack([o, z, z]),
            torch.stack([z, c, -s]),
            torch.stack([z, s, c]),
        ])
    if axis_index == 1:
        return torch.stack([
            torch.stack([c, z, s]),
            torch.stack([z, o, z]),
            torch.stack([-s, z, c]),
        ])
    if axis_index == 2:
        return torch.stack([
            torch.stack([c, -s, z]),
            torch.stack([s, c, z]),
            torch.stack([z, z, o]),
        ])
    raise ValueError("axis_index must be 0, 1, or 2")


def pca_frame(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return PCA center and right-handed PCA axes. Axes are columns of A."""
    center = points.mean(dim=0)
    X = points - center
    cov = (X.T @ X) / max(points.shape[0], 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    A = eigvecs[:, order]
    if torch.det(A) < 0:
        A = A.clone()
        A[:, 2] *= -1.0
    return center, A


def estimate_normals_knn(
    points: torch.Tensor,
    k: int = 24,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """
    Simple GPU KNN PCA normal estimation for point clouds.

    This is useful when normals are not available. It is O(N^2), so for large objects it is
    better to estimate normals with Open3D or your dataset preprocessing pipeline.
    """
    device, dtype = points.device, points.dtype
    N = points.shape[0]
    normals = torch.empty((N, 3), device=device, dtype=dtype)
    kk = min(k + 1, N)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        d = torch.cdist(points[start:end], points)
        idx = d.topk(kk, largest=False).indices[:, 1:]  # remove self
        nbrs = points[idx]                              # [B, k, 3]
        mu = nbrs.mean(dim=1, keepdim=True)
        X = nbrs - mu
        cov = X.transpose(1, 2) @ X / max(k, 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        normals[start:end] = eigvecs[:, :, 0]

    return _normalize(normals)


class CanonicalFrameEnergy:
    def __init__(
        self,
        points: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        grid_resolution: int = 96,
        distance_layers: int = 10,
        field_padding: float = 0.15,
        sample_count: int = 10_000,
        lambda_sym: float = 0.75,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        if device is None:
            device = points.device if points.is_cuda else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.dtype = dtype
        self.points = points.to(device=device, dtype=dtype).contiguous()
        self.N = self.points.shape[0]
        self.grid_resolution = int(grid_resolution)
        self.M = int(distance_layers)
        self.lambda_sym = float(lambda_sym)

        if normals is not None:
            self.normals = _normalize(normals.to(device=device, dtype=dtype).contiguous())
        else:
            self.normals = None

        if weights is not None:
            w = weights.to(device=device, dtype=dtype).reshape(-1)
            self.weights = w / w.sum().clamp_min(1e-8)
        else:
            self.weights = None

        if self.N > sample_count:
            perm = torch.randperm(self.N, device=device)[:sample_count]
            self.sample_points = self.points[perm]
        else:
            self.sample_points = self.points

        self._build_distance_field(field_padding)

    def _build_distance_field(self, padding: float) -> None:
        """Build integer Manhattan distance-layer field on the GPU."""
        pts = self.points
        mn = pts.min(dim=0).values
        mx = pts.max(dim=0).values
        self.points_min = mn
        self.points_max = mx
        extent = (mx - mn).clamp_min(1e-6)
        pad = padding * extent.max()
        self.bb_min = mn - pad
        self.bb_max = mx + pad
        size = (self.bb_max - self.bb_min).clamp_min(1e-6)

        R = self.grid_resolution
        idx = ((pts - self.bb_min) / size * (R - 1)).round().long()
        idx = idx.clamp(0, R - 1)
        x, y, z = idx[:, 0], idx[:, 1], idx[:, 2]
        flat = z * R * R + y * R + x

        occ_flat = torch.zeros((R * R * R,), device=self.device, dtype=torch.bool)
        occ_flat[flat] = True
        visited = occ_flat.view(1, 1, R, R, R)  # [B, C, D=z, H=y, W=x]

        field = torch.full((1, 1, R, R, R), float(self.M), device=self.device, dtype=self.dtype)
        field[visited] = 0.0

        kernel = torch.zeros((1, 1, 3, 3, 3), device=self.device, dtype=self.dtype)
        kernel[0, 0, 1, 1, 1] = 1.0
        kernel[0, 0, 0, 1, 1] = 1.0
        kernel[0, 0, 2, 1, 1] = 1.0
        kernel[0, 0, 1, 0, 1] = 1.0
        kernel[0, 0, 1, 2, 1] = 1.0
        kernel[0, 0, 1, 1, 0] = 1.0
        kernel[0, 0, 1, 1, 2] = 1.0

        for layer in range(1, self.M + 1):
            expanded = F.conv3d(visited.to(self.dtype), kernel, padding=1) > 0.5
            new = expanded & (~visited)
            field[new] = float(layer)
            visited = expanded

        self.distance_field = field.contiguous()

    def sample_distance_layers(self, query_points: torch.Tensor) -> torch.Tensor:
        """
        Trilinearly sample the distance layer l(p) for query points.
        Returns values in [0, M].
        """
        size = (self.bb_max - self.bb_min).clamp_min(1e-6)
        grid = (query_points - self.bb_min) / size * 2.0 - 1.0
        grid = grid.view(1, -1, 1, 1, 3)  # grid_sample order: x, y, z
        vals = F.grid_sample(
            self.distance_field,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return vals.view(-1).clamp(0.0, float(self.M))

    @staticmethod
    def local_coordinates(points: torch.Tensor, A: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """World points -> local canonical coordinates. A columns are world axes."""
        return (points - c) @ A

    @staticmethod
    def world_coordinates(local_points: torch.Tensor, A: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Local canonical coordinates -> world points. A columns are world axes."""
        return local_points @ A.T + c

    def symmetry_measures(self, A: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return rotational and plane-reflective symmetry distances for the 3 frame axes.

        rs[k]  = rotational symmetry distance around axis k.
        prs[k] = plane reflective symmetry distance across plane normal to axis k.
        Lower is better.
        """
        x = self.local_coordinates(self.sample_points, A, c)
        rs_values: List[torch.Tensor] = []
        prs_values: List[torch.Tensor] = []

        rotation_angles = [math.pi, 2.0 * math.pi / 3.0, 4.0 * math.pi / 5.0]

        for k in range(3):
            # Plane-reflective symmetry: x_k -> -x_k
            xr = x.clone()
            xr[:, k] = -xr[:, k]
            reflected = self.world_coordinates(xr, A, c)
            prs = self.sample_distance_layers(reflected).mean() / float(self.M)
            prs_values.append(prs)

            # Rotational symmetry: choose best order per sample among 2, 3, 5.
            per_order = []
            for angle in rotation_angles:
                Rk = axis_rotation_matrix(k, angle, self.device, self.dtype)
                xrot = x @ Rk.T
                rotated = self.world_coordinates(xrot, A, c)
                per_order.append(self.sample_distance_layers(rotated))
            stacked = torch.stack(per_order, dim=0)  # [3 orders, N]
            rs = stacked.min(dim=0).values.mean() / float(self.M)
            rs_values.append(rs)

        return torch.stack(rs_values), torch.stack(prs_values)

    def symmetry_energy(self, A: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Hybrid symmetry energy.

        We select the best PRS axis and the best RS axis among the remaining axes, then
        combine the dominant and secondary symmetry terms.
        """
        rs, prs = self.symmetry_measures(A, c)

        prs_axis = int(torch.argmin(prs).detach().item())
        remaining = [i for i in range(3) if i != prs_axis]
        rs_axis = remaining[int(torch.argmin(rs[remaining]).detach().item())]

        main = 0.5 * (prs[prs_axis] + rs[rs_axis])
        secondary = 0.5 * (rs[prs_axis] + prs[rs_axis])
        return self.lambda_sym * main + (1.0 - self.lambda_sym) * secondary

    def rectilinearity_energy(self, A: torch.Tensor) -> torch.Tensor:
        """Rectilinearity energy from normal-axis alignment. Lower is better."""
        if self.normals is None:
            return torch.zeros((), device=self.device, dtype=self.dtype)

        dots = torch.abs(self.normals @ A)  # [N, 3]
        best = dots.max(dim=1).values

        if self.weights is None:
            Rval = best.mean()
        else:
            Rval = (best * self.weights).sum()

        worst = 1.0 / math.sqrt(3.0)  # cos(atan(sqrt(2)))
        return ((1.0 - Rval) / (1.0 - worst)).clamp_min(0.0)

    def center_energy(self, A: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """AABB-center consistency energy. Lower is better."""
        local = self.local_coordinates(self.points, A, c)
        mn = local.min(dim=0).values
        mx = local.max(dim=0).values
        center_bb_local = 0.5 * (mn + mx)
        center_bb_world = self.world_coordinates(center_bb_local[None, :], A, c)[0]
        diag = (mx - mn).norm().clamp_min(1e-8)
        return (2.0 * (c - center_bb_world).norm() / diag).clamp_min(0.0)

    def total_energy(
        self,
        A: torch.Tensor,
        c: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        Esym = self.symmetry_energy(A, c)
        ER = self.rectilinearity_energy(A)
        EC = self.center_energy(A, c)
        E = alpha * Esym + beta * ER + gamma * EC
        return E, {"E": E, "Esym": Esym, "ER": ER, "EC": EC}

    @torch.no_grad()
    def sort_and_sign_axes(
        self,
        A: torch.Tensor,
        c: torch.Tensor,
        axis_order: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Sort axes using the requested deterministic ordering.

        If axis_order is None, sort similarly to the paper:
            a2: best PRS axis, a1: best RS axis among remaining, a3: leftover.
        Then fix signs with a simple mass-skewness heuristic and enforce right-handedness.
        """
        if axis_order is None:
            rs, prs = self.symmetry_measures(A, c)
            prs_axis = int(torch.argmin(prs).item())
            remaining = [i for i in range(3) if i != prs_axis]
            rs_axis = remaining[int(torch.argmin(rs[remaining]).item())]
            leftover = [i for i in range(3) if i not in (prs_axis, rs_axis)][0]

            # Columns: a1 = best RS, a2 = best PRS, a3 = remaining
            A2 = torch.stack([A[:, rs_axis], A[:, prs_axis], A[:, leftover]], dim=1).clone()
        else:
            local0 = self.local_coordinates(self.points, A, c)
            if axis_order == "extent_desc":
                scores = local0.max(dim=0).values - local0.min(dim=0).values
                order = torch.argsort(scores, descending=True)
            elif axis_order == "extent_asc":
                scores = local0.max(dim=0).values - local0.min(dim=0).values
                order = torch.argsort(scores)
            elif axis_order == "variance_desc":
                scores = local0.var(dim=0, unbiased=False)
                order = torch.argsort(scores, descending=True)
            else:
                raise ValueError(
                    "axis_order must be None, 'extent_desc', 'extent_asc', or 'variance_desc'"
                )
            A2 = A[:, order].clone()

        local = self.local_coordinates(self.points, A2, c)
        for k in range(3):
            # Prefer the sign where the farthest side of the object lies in positive direction.
            if torch.abs(local[:, k].min()) > torch.abs(local[:, k].max()):
                A2[:, k] *= -1.0
                local[:, k] *= -1.0

        if torch.det(A2) < 0:
            A2[:, 2] *= -1.0

        return A2


def optimize_canonical_frame(
    points: torch.Tensor,
    normals: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
    grid_resolution: int = 96,
    distance_layers: int = 10,
    sample_count: int = 10_000,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.5,
    lr: float = 3e-2,
    steps: int = 150,
    restarts: int = 5,
    random_rotation_degrees: float = 15.0,
    use_lbfgs: bool = True,
    normalize_scale: str = "max_extent",
    axis_order: Optional[str] = 'variance_desc',
    device: Optional[str | torch.device] = None,
    dtype: torch.dtype = torch.float32,
    verbose: bool = True,
) -> CanonicalFrameResult:
    """
    Optimize a canonical frame and return normalized points.

    Args:
        points: [N, 3] point cloud.
        normals: optional [N, 3] normals. If omitted, rectilinearity is disabled unless beta=0.
        weights: optional [N] point/normal weights.
        grid_resolution: distance field resolution. 64-128 is typical.
        distance_layers: number of Manhattan distance layers M. Paper used M=10.
        sample_count: number of sample points for symmetry energy. Paper used N=10,000.
        alpha, beta, gamma: weights for symmetry, rectilinearity, center energy.
        lr, steps, restarts: Adam optimization settings.
        use_lbfgs: run a short L-BFGS refinement after Adam.
        normalize_scale: "max_extent", "diag", or "none".
        axis_order: optional final axis renaming: "extent_desc", "extent_asc", or
            "variance_desc". If omitted, use symmetry-based ordering.
    """
    if device is not None:
        device = torch.device(device)

    energy_model = CanonicalFrameEnergy(
        points=points,
        normals=normals,
        weights=weights,
        grid_resolution=grid_resolution,
        distance_layers=distance_layers,
        sample_count=sample_count,
        device=device,
        dtype=dtype,
    )

    c0, A0 = pca_frame(energy_model.points)
    best = {
        "loss": float("inf"),
        "center": None,
        "axes": None,
        "info": None,
    }

    max_angle = math.radians(random_rotation_degrees)
    init_rotvecs = [torch.zeros(3, device=energy_model.device, dtype=dtype)]
    for _ in range(max(0, restarts - 1)):
        rv = torch.randn(3, device=energy_model.device, dtype=dtype)
        rv = rv / rv.norm().clamp_min(1e-8) * (torch.rand((), device=energy_model.device, dtype=dtype) * max_angle)
        init_rotvecs.append(rv)

    for restart_id, init_rv in enumerate(init_rotvecs):
        rotvec = init_rv.clone().detach().requires_grad_(True)
        center = c0.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([rotvec, center], lr=lr)

        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            R = so3_exp(rotvec)
            A = A0 @ R
            loss, parts = energy_model.total_energy(A, center, alpha=alpha, beta=beta, gamma=gamma)
            loss.backward()
            opt.step()
            with torch.no_grad():
                center.clamp_(energy_model.points_min, energy_model.points_max)

            if verbose and (step % 50 == 0 or step == steps - 1):
                parts_print = {k: float(v.detach().cpu()) for k, v in parts.items()}
                print(
                    f"restart={restart_id:02d} step={step:04d} "
                    f"E={parts_print['E']:.5f} Esym={parts_print['Esym']:.5f} "
                    f"ER={parts_print['ER']:.5f} EC={parts_print['EC']:.5f}"
                )

        if use_lbfgs:
            opt2 = torch.optim.LBFGS([rotvec, center], lr=0.5, max_iter=35, line_search_fn="strong_wolfe")

            def closure():
                opt2.zero_grad(set_to_none=True)
                R = so3_exp(rotvec)
                A = A0 @ R
                loss, _ = energy_model.total_energy(A, center, alpha=alpha, beta=beta, gamma=gamma)
                loss.backward()
                return loss

            try:
                opt2.step(closure)
            except RuntimeError:
                # L-BFGS can fail if the piecewise distance field gives a bad line search.
                pass
            with torch.no_grad():
                center.clamp_(energy_model.points_min, energy_model.points_max)

        with torch.no_grad():
            R = so3_exp(rotvec)
            A = A0 @ R
            loss, parts = energy_model.total_energy(A, center, alpha=alpha, beta=beta, gamma=gamma)
            current = float(loss.detach().cpu())
            if current < best["loss"]:
                best["loss"] = current
                best["center"] = center.detach().clone()
                best["axes"] = A.detach().clone()
                best["info"] = {k: float(v.detach().cpu()) for k, v in parts.items()}

    center = best["center"]
    axes = energy_model.sort_and_sign_axes(best["axes"], center, axis_order=axis_order)

    local = energy_model.local_coordinates(energy_model.points, axes, center)
    mn = local.min(dim=0).values
    mx = local.max(dim=0).values
    extent = mx - mn

    if normalize_scale == "max_extent":
        scale = extent.max().clamp_min(1e-8)
    elif normalize_scale == "diag":
        scale = extent.norm().clamp_min(1e-8)
    elif normalize_scale == "none":
        scale = torch.ones((), device=energy_model.device, dtype=dtype)
    else:
        raise ValueError("normalize_scale must be 'max_extent', 'diag', or 'none'")

    normalized = local / scale

    return CanonicalFrameResult(
        center=center,
        axes=axes,
        normalized_points=normalized,
        local_points=local,
        scale=scale,
        energy=best["loss"],
        info=best["info"],
    )


def _to_numpy(points: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(points, torch.Tensor):
        return points.detach().cpu().numpy()
    return np.asarray(points)


def visualize_canonical_frames(
    points_full: torch.Tensor | np.ndarray,
    parts: List[Tuple[torch.Tensor | np.ndarray, CanonicalFrameResult]],
    frame_size: Optional[float] = None,
    window_name: str = "Canonical Coordinate Frames",
) -> None:
    """Draw a full object and the calculated canonical frame for each segmented part."""
    points_full_np = _to_numpy(points_full)

    if frame_size is None:
        extent = np.ptp(points_full_np, axis=0)
        frame_size = max(float(np.max(extent)) * 0.15, 1e-6)

    pcd_full = o3d.geometry.PointCloud()
    pcd_full.points = o3d.utility.Vector3dVector(points_full_np)
    pcd_full.paint_uniform_color((0.75, 0.75, 0.75))
    geometries = []

    part_colors = [
        (0.85, 0.15, 0.15),
        (0.10, 0.65, 0.20),
        (0.15, 0.35, 0.90),
        (0.90, 0.60, 0.10),
        (0.65, 0.20, 0.80),
        (0.20, 0.80, 0.65),
        (0.80, 0.10, 0.50),
        (0.50, 0.50, 0.90),
        (0.90, 0.30, 0.30),
    ]

    for part_idx, (points, result) in enumerate(parts):
        center = result.center.detach().cpu().numpy()
        axes = result.axes.detach().cpu().numpy()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(_to_numpy(points))
        pcd.paint_uniform_color(part_colors[part_idx % len(part_colors)])
        geometries.append(pcd)

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        transform = np.eye(4)
        transform[:3, :3] = axes
        transform[:3, 3] = center
        frame.transform(transform)
        geometries.append(frame)

        center_marker = o3d.geometry.TriangleMesh.create_sphere(radius=frame_size * 0.04)
        center_marker.paint_uniform_color((0.1, 0.1, 0.1))
        center_marker.translate(center)
        geometries.append(center_marker)

    o3d.visualization.draw_geometries(geometries, window_name=window_name)