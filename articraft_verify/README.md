# articraft_verify

A **Viser web GUI** to human-verify and correct the `articraft_canon`
canonicalization — one object at a time, no reference object and no algorithmic
alignment. You load an object as a **mesh** with its link coordinate frames and
joint axes (pybullet-viewer style), adjust the joints and **each link's own
coordinate frame** with sliders, and eyeball the object's **own** NOCS/NPCS
(rendered live in the side panel) against reference pictures you drop in a folder.

**NOCS is computed in the object root (base-link) frame; NPCS is computed in each
link's own frame** — not the viewer's global frame. The object is never rotated
against a global frame: only the per-link frames move.

```bash
pip install viser yourdfpy trimesh numpy scipy pillow
python -m articraft_verify --dataset-root data/urdfs/Dataset --category robotic_arm
# open the printed http://…:8080 URL
```

## What you see

- **3D scene:** the loaded object as a mesh (in canonical coordinates), with
  per-link coordinate-frame triads, the object root-frame triad, and the joint
  axes (yellow). Only the loaded object — nothing else.
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
| **Joint sliders** | drive `update_cfg`; the mesh, frames and NOCS/NPCS follow live |
| **Edit a link's frame** → **Link** dropdown + rot X/Y/Z° | rotate the *selected link's own* frame; its triad follows and its NPCS re-renders (rotating the base link re-renders NOCS) |
| **Snap this link to 90° / Reset this link / Reset all links** | 90°-lattice snap or clear the selected link (or all) |
| **Link frames / Joint axes** checkboxes | toggle the triads / axes |
| **Accept & save / Skip / Prev / Next** | queue navigation; Accept writes the re-baked object |

## Modes (`--mode`)

- **frame-fix** (default): joint sliders only *inspect*; the frame sliders are
  the edit.
- **rest-pose**: joint sliders *edit* the pose; **Accept** does "set current as
  canonical zero → re-bake" (shifts joint origins/limits so the current pose
  becomes `q = 0`).

## Output (never touches originals)

```
<output>/<category>/<sub_category>/<split>/<object_id>.tar.gz   # re-baked URDF + canonical.json
<output>/_verify/decisions.json                                # append-only log (resume by input hash)
```

`canonical.json` carries NOCS/NPCS + frames + a `verification` block (mode, frame
Euler, final base→canonical transform). Each decision is keyed by an input hash
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
| `store.py` | decision log + write re-baked `.tar.gz` outputs |
| `app.py` | Viser scene + GUI |
| `cli.py` | launch |

Built on `articraft_canon` (parse / canonical_zero / root_frame / nocs /
geometry). viser 1.0.30 has no global keyboard API, so navigation is buttons.
