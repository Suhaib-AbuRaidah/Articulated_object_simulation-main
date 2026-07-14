# articraft_canon

A modular, **analytic (no-learning)** CLI that canonicalizes the Articraft URDF
dataset (~392 serial articulated objects across 16 sub-category leaves) into a
consistent canonical rest state and consistent link frames for NOCS/NPCS
supervision.

Stack: `yourdfpy` + `trimesh` + `numpy`/`scipy` (+ `python-fcl` for accurate
self-collision). Originals are **never modified** — archives are read, extracted
to a scratch dir, and canonical outputs are written to a separate tree as
`.tar.gz` archives mirroring the source dataset. Dry-run is the default;
`--write` emits archives.

```bash
pip install yourdfpy trimesh numpy scipy python-fcl   # dependencies
python -m articraft_canon --help
```

## Pipeline (one module per stage)

| Stage | Module | What it does |
|------:|--------|--------------|
| 1 | [`parse.py`](parse.py) | Load a URDF; extract the serial kinematic tree base→tip; per-joint type/axis/origin/limits; surface-sample a point cloud per link in link-local frame. Branching / multi-DOF / mimic / unsupported joints are **logged and skipped**. |
| 2 | [`canonical_zero.py`](canonical_zero.py) | **Fully-extended canonical q=0.** Per link a canonical direction = joint-to-next-joint vector (fallback: mesh PCA long axis). Base→tip, closed-form `q*` makes each child's direction collinear with its parent's (revolute: signed angle ⊥ axis; prismatic: travel to the reach-maximising limit). Bake `new_origin = old ∘ motion(q*)`, shift limits by `-q*`. Unreachable/self-colliding → mid-limit fallback. |
| 3 | [`root_frame.py`](root_frame.py) | Per-object root frame: PCA up/front chosen from the 6 signed-axis candidates; discrete symmetry group via self-Chamfer; scale to unit bbox diagonal. |
| 4 | [`subcat_frame.py`](subcat_frame.py) | Sub-category frame consistency: reference = Chamfer **medoid** (`--exemplar` override); Umeyama/ICP rigid-align every object to it; for symmetric objects pick the group element with least residual; fold the transform into the object frame so the whole sub-category shares one frame. |
| 5 | [`nocs.py`](nocs.py) | **Option B frames + NOCS/NPCS.** Link frames share the object's canonical orientation (identity relative rotations at q=0 — parts are *not* oriented by joint axes). This is baked into the **emitted URDF** too (`bake_base_aligned_link_frames`): every link frame is rewritten to the base orientation, translated to the link origin (joint origins become pure translations; rotations move into the joint axis and the visual/collision origins; inertial origins are neutralised — see note below — kinematically identical). Joint geometry is also stored as explicit params (axis, type, limits, pivot) in the canonical frame, and it emits NOCS (whole object → unit cube), NPCS (per-part → unit cube) and per-part NPCS→NOCS scale+translation into a JSON sidecar. |

Support modules: [`geometry.py`](geometry.py) (SE(3), FK, PCA, Chamfer, Umeyama),
[`dataset.py`](dataset.py) (discovery + safe extraction), [`report.py`](report.py)
(dry-run lines + per-sub-category QC), [`pipeline.py`](pipeline.py) (orchestration),
[`cli.py`](cli.py).

## Usage

```bash
# Dry-run one sub-category (writes nothing; prints per-object plan + QC)
python -m articraft_canon --sub-category serial_elbow_arm

# Dry-run a whole category
python -m articraft_canon --category robotic_arm

# Emit canonical URDFs + sidecars for the full fit set
python -m articraft_canon --write --output-dir data/urdfs/Canonical

# Pin the reference object for a sub-category
python -m articraft_canon --category articulated_task_lamp \
    --exemplar articulated_task_lamp=rec_articulated_task_lamp_0001
```

Key flags: `--write`, `--category/--sub-category/--split` (repeatable filters),
`--include-not-fit`, `--limit N`, `--exemplar sub=obj`, `--check-collision`,
`--unreachable {mid-limit,clamp,extend-limits}`, `--first-link-reference`,
`--no-align-link-frames`, `-v/-vv`.

### Output layout (`--write`)

Each object is emitted as a `.tar.gz` mirroring the source dataset (top-level
folder = object id):

```
<output-dir>/<category>/<sub_category>/<split>/<object_id>.tar.gz
    <object_id>/model.urdf          # canonical (extended rest, baked limits, relative mesh paths)
    <object_id>/assets/...          # meshes bundled so the URDF stays valid
    <object_id>/canonical.json      # frames + NOCS/NPCS sidecar (incl. limit provenance)
    <object_id>/compile_report.json # copied from source if present
<output-dir>/_qc/<sub_category>.json # QC report (not archived)
```

## Interpretation notes / decisions

These are the places where the spec left a judgement call; each is exposed as a
flag so you can reproduce the literal reading.

- **"Fully extended" = straight chain, aligned from the base.** Straightening
  starts at the base link: the first moving link is made collinear with the
  base's own direction, then each subsequent link with its parent (base-outward,
  as specified). Under fixed joint limits this is often unreachable, but the
  `extend-limits` policy resolves that by widening limits, so base alignment is
  the default. Pass `--first-link-reference` to instead seed the chain from the
  first moving link (leave the first moving joint at q*=0) — the older behaviour,
  useful with `mid-limit`/`clamp`.
- **Per-category canonical axis** (`_CATEGORY_UP_AXIS` in `canonical_zero.py`).
  The base's direction defaults to its mount vector (origin → first joint), but
  a small off-axis mount offset can tilt the whole straightened chain (e.g. one
  lamp's arm came out 44° off vertical). Where a whole category mounts along one
  consistent axis (measured: +Z for lamp/telescope/cartesian/dish/robotic-arm/
  spotlight at 94–100%, +X for telescoping slide at 74%), that axis is **forced**
  as the base direction so every object in the category extends the same way.
  Genuinely mixed categories (cctv_mast, laptop_stand, wall_mounts) are omitted
  and keep their per-object mount vector. Edit the dict to add/adjust categories.
  (This only affects revolute/continuous first joints; a swivel or prismatic
  first joint still can't be reoriented — see below.)
- **Leaf-link direction is oriented outward.** Interior links point to their next
  joint; a leaf (tip) link falls back to its mesh PCA long axis, whose sign is
  ambiguous. We orient it away from the link's inbound joint (toward its geometry
  centroid) so the last link extends *along* the chain rather than folding back
  on it. Flat/disc tips (dishes, LED bars) and pan joints whose axis is parallel
  to the chain direction can still end up perpendicular/opposed — that is
  inherent to the geometry/kinematics, not a sign error.
- **Unreachable policy** (`--unreachable`). When the extended `q*` exceeds a
  joint limit:
  - `mid-limit` (spec default) drops the whole object to its per-joint mid-limit
    pose;
  - `clamp` keeps the most-extended *reachable* pose (clamp to nearest limit,
    original limits untouched);
  - `extend-limits` **widens** the offending joint limits just enough to admit
    the straight `q = 0` pose — but only if that pose is self-collision-free.
    The collision gate here is **mandatory** and uses real mesh–mesh collision
    (`python-fcl`); if the straight pose collides, the object falls back to
    mid-limit. Original limits are preserved in the sidecar
    (`limit_*_original`, `limit_widened`, object-level `limits_widened`). On the
    full fit set this straightens ~139 objects that limits alone would block,
    leaves ~186 already-reachable, and correctly refuses ~58 whose straight pose
    would interpenetrate.
- **`continuous` joints** are treated as limit-free revolute joints (single-DOF),
  not skipped.
- **Inertial frames are neutralised during link-frame alignment.** PyBullet
  frames each link by its inertial (CoM) frame, so if a link's `<inertial>`
  origin carries a rotation, the base-aligned link frame and the frame PyBullet
  actually uses diverge — and parts visibly **detach at their joints under
  motion** (yourdfpy ignores inertial, so it looks fine there). Because this
  pipeline is purely kinematic (fixed base, zero gravity, direct joint resets),
  `bake_base_aligned_link_frames` zeroes each chain link's inertial origin so
  PyBullet's link frame coincides with the base-aligned URDF frame.
- **Frame + scale live in the sidecar, not baked into meshes.** The canonical
  URDF is canonicalized for *articulation* (Stage 2); the object/sub-category
  rigid frame, scale, symmetry and NOCS/NPCS are recorded as transforms in
  `canonical.json` (non-destructive and reversible). The canonical transform is
  `canonical = scale · R_root · (p − centroid) + align_translation`.
- **Self-collision** uses real non-adjacent mesh–mesh collision via
  `python-fcl` when available, falling back to a coarse AABB-overlap test (with a
  warning) otherwise. For the `mid-limit`/`clamp` policies it is an **optional**
  fallback trigger, off by default (`--check-collision`), because the coarse
  AABB test false-positives on concentric/nested designs. For `extend-limits` it
  is **always on** and should be backed by fcl — do not widen limits on the
  strength of the AABB test alone.

## Dry-run & QC output

Per object: `q*` per joint, chosen symmetry order, Umeyama residual, predicted
Chamfer to the reference, and `[FALLBACK …]` when applicable. Per sub-category:
pairwise Chamfer at q=0 (mean/std/max — expect near-zero variance once frames are
consistent), mid-limit fallbacks, and outlier flags (objects whose mean Chamfer
to the rest exceeds mean + 2σ).
