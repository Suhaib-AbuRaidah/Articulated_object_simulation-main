# articraft_verify

A **Viser web GUI** to human-verify and correct the `articraft_canon`
canonicalization — one object at a time, no reference object and no algorithmic
alignment. You load an object as a **mesh** with its link coordinate frames and
joint axes (pybullet-viewer style), correct the physical geometry and motion-axis
orientation while preserving frame orientations, adjust the joints and **each link's own coordinate frame**
with sliders, and eyeball the object's **own** NOCS/NPCS
(rendered live in the side panel) against reference pictures you drop in a folder.

**NOCS is computed in the object root (base-link) frame; NPCS is computed in each
link's own frame** — not the viewer's global frame. Object orientation rotates
link geometry, motion axes, and pivot positions about the object base while
preserving frame orientations; link-frame edits remain independent corrections.

```bash
pip install viser yourdfpy trimesh numpy scipy pillow
python -m articraft_verify --dataset-root data/urdfs/Dataset --category robotic_arm
# open the printed http://…:8080 URL
```

To inspect the source URDF without automatically changing its zero state, joint
origins/limits, or link coordinate frames before editing, add ``--raw-urdf``:

```bash
python -m articraft_verify --dataset-root data/urdfs/Dataset \
    --category robotic_arm --raw-urdf --counter-rotate-link-frames \
    --allow-fixed-branches
```

## What you see

- **3D scene:** the loaded object as a mesh (in canonical coordinates), with a
  fixed global-frame triad, per-link coordinate-frame triads, the object
  root-frame triad, and the joint axes (yellow).
- **Side panel:**
  - **NOCS (root frame)** and **NPCS (link frames)** — the object's own
    normalised coordinate renders, recomputed on every joint/frame change (never
    hand-editable). NOCS uses the base-link frame; NPCS colours each part in its
    own link frame.
  - **Reference images** — up to 3 pictures loaded from
    `<reference-dir>/<category>/` (the folder is created empty; drop your
    reference NOCS/NPCS/URDF pictures there and reload the object).

## Controls

| Control | Effect |
|---|---|
| **Object orientation X/Y/Z°** | rotate every link's visual/collision/inertial data, joint axis, and pivot position; link-frame orientations and joint-origin RPY remain fixed while joint-origin XYZ follows the connected geometry |
| **Snap object to 90° / Reset object orientation** | snap or clear the physical orientation correction |
| **Joint state sliders** | change the pose; the mesh, frames and NOCS/NPCS follow live, and Accept bakes this pose as the new `q=0` |
| **`<joint>`: counter-rotate child frames** | independently remove that joint's rotation from the coordinate frames of its child and descendants; meshes and physical joint axes still follow the full articulated pose, and Accept bakes the compensated frames into the URDF |
| **Joint lower / upper inputs** | edit finite URDF ranges in the current joint coordinates; Accept shifts them with the selected state and writes the resulting limits into `model.urdf` |
| **Edit a link's frame** → **Link** dropdown + rot X/Y/Z° | rotate the *selected link's own* frame; its triad follows and its NPCS re-renders (rotating the base link re-renders NOCS) |
| **Snap this link to 90° / Reset this link / Reset all links** | 90°-lattice snap or clear the selected link (or all) |
| **Link frames / Joint axes** checkboxes | toggle the triads / axes |
| **Accept & save / Skip / Prev / Next** | Accept bakes object orientation, joint-state/range, and frame edits into the URDF; Skip records the decision without writing or overwriting an archive |

There is one edit workflow: geometry/axis orientation, joint states, finite joint
ranges, and link-frame sliders are always active and are committed together by
**Accept & save**.
`--counter-rotate-link-frames` starts with every rotational joint's individual
GUI checkbox enabled. You can then disable or enable compensation separately for
each joint. For a single +45° revolute motion about Y, that joint's child subtree
receives the equivalent -45° frame compensation. For a chain, forward kinematics
removes only the selected joints' rotation contributions and retains all others.
`--raw-urdf` only chooses whether loading starts from the source URDF or the
automatic canonical-zero result; it does not change what can be edited or saved.
`--allow-fixed-branches` accepts rigid side attachments such as covers, rails,
and brackets while still rejecting objects whose moving-joint skeleton branches.
The fixed links remain part of rendering, frame transforms, NOCS/NPCS, and the
saved URDF; they are not discarded or flattened.

Mimic joints are accepted automatically by the verifier and converted to
ordinary independent joints in memory. Their joint type, axis, origin, and
limits are retained, but the emitted `model.urdf` omits the `<mimic>` element.
The source archive is never changed, and the original mimic target, multiplier,
and offset are recorded under `verification.converted_mimic_joints` in
`canonical.json` and in the verifier decision log.

## Output (never touches originals)

```
<output>/<category>/<sub_category>/<split>/<object_id>.tar.gz   # edited URDF + canonical.json
<output>/_verify/decisions.json                                # append-only log (resume by input hash)
```

`canonical.json` carries NOCS/NPCS + frames + a `verification` block containing
the applied object orientation, joint state, whether link frames were
counter-rotated, requested/baked joint limits, and frame Euler edits.
Each decision is keyed by an input hash
(archive path/size/mtime), so re-runs jump to new/changed objects
(`--no-skip-done` to disable).

## Modules

| File | Role |
|---|---|
| `catalog.py` | discovery, flat queue, input hashing |
| `objectstate.py` | canonicalization + editable frame (Euler) & joints, meshes, NOCS/NPCS points, re-bake, sidecar |
| `coloring.py` | NOCS / NPCS colours |
| `render.py` | rasterise a coloured cloud into a 2D image (the side-panel NOCS/NPCS) |
| `references.py` | per-category reference-image folder (create + load) |
| `store.py` | decision log + write edited `.tar.gz` outputs |
| `app.py` | Viser scene + GUI |
| `cli.py` | launch |

Built on `articraft_canon` (parse / canonical_zero / root_frame / nocs /
geometry). viser 1.0.30 has no global keyboard API, so navigation is buttons.
