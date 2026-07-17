"""Single-object Viser verifier.

Loads one object as a **mesh** (natural orientation, only centred+scaled for the
view) with each link's coordinate frame and the joint axes.  You:

  * drive **joint sliders** (``update_cfg``);
  * select a link and rotate **its own coordinate frame** -- the triad follows and
    that part's **NPCS** re-renders; rotating the base link re-renders **NOCS**;
  * eyeball the object's live **NOCS** (root-frame) and **NPCS** (per-link-frame)
    images against reference pictures dropped in ``<reference-dir>/<category>/``.

The object is never rotated against a global frame; only per-link frames move,
and NOCS/NPCS are computed in those frames -- not the viewer's world frame.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import viser

from articraft_canon.dataset import ArchiveRef

from . import catalog as catalog_mod
from . import references, render
from .objectstate import MODE_FRAME_FIX, MODE_REST_POSE, ObjectState
from .store import DecisionStore

logger = logging.getLogger(__name__)


class VerifierApp:
    def __init__(
        self,
        catalog: catalog_mod.Catalog,
        store: DecisionStore,
        reference_dir: Path,
        *,
        host: str = "0.0.0.0",
        port: int = 8080,
        default_mode: str = MODE_FRAME_FIX,
        skip_done: bool = True,
    ) -> None:
        self.catalog = catalog
        self.store = store
        self.reference_dir = Path(reference_dir)
        self.skip_done = skip_done
        self.default_mode = default_mode
        self.server = viser.ViserServer(host=host, port=port)

        self.queue: List[ArchiveRef] = list(catalog.refs)
        self.index = 0
        self.state: Optional[ObjectState] = None
        self._joint_sliders: List = []
        self._ref_image_handles: List = []
        self._mesh_handles: List = []

        self._build_gui()
        if self.queue:
            self._load(self._first_index())

    def _first_index(self) -> int:
        if self.skip_done:
            for i, r in enumerate(self.queue):
                if not self.store.is_done(r.object_id, catalog_mod.input_hash(r)):
                    return i
        return 0

    # ================================================================== #
    # GUI
    # ================================================================== #
    def _build_gui(self) -> None:
        gui = self.server.gui
        gui.configure_theme(control_layout="collapsible")
        self.status = gui.add_markdown("Loading…")

        with gui.add_folder("Object"):
            self.obj_dropdown = gui.add_dropdown(
                "Object", [r.object_id for r in self.queue] or ["(none)"])
            self.obj_dropdown.on_update(lambda _: self._select(self.obj_dropdown.value))
            gui.add_button("◀ Prev").on_click(lambda _: self._advance(-1))
            gui.add_button("Next ▶").on_click(lambda _: self._advance(+1))
            gui.add_button("✓ Accept & save", color="green").on_click(lambda _: self._accept())
            gui.add_button("Skip", color="gray").on_click(lambda _: self._skip())

        with gui.add_folder("Mode & view"):
            self.mode_dropdown = gui.add_dropdown(
                "Edit mode", (MODE_FRAME_FIX, MODE_REST_POSE), initial_value=self.default_mode)
            self.mode_dropdown.on_update(lambda _: self._on_mode())
            self.show_link_frames = gui.add_checkbox("Link frames", True)
            self.show_joint_axes = gui.add_checkbox("Joint axes", True)
            self.show_link_frames.on_update(lambda _: self._render_scene())
            self.show_joint_axes.on_update(lambda _: self._render_scene())

        self.joints_folder = gui.add_folder("Joint states")

        with gui.add_folder("Edit a link's frame"):
            self.link_dropdown = gui.add_dropdown("Link", ["(none)"])
            self.link_dropdown.on_update(lambda _: self._on_link_select())
            self.link_sliders = {
                "rx": gui.add_slider("rot X°", -180, 180, 1, 0),
                "ry": gui.add_slider("rot Y°", -180, 180, 1, 0),
                "rz": gui.add_slider("rot Z°", -180, 180, 1, 0),
            }
            for h in self.link_sliders.values():
                h.on_update(lambda _: self._on_link_frame_change())
            gui.add_button("Snap this link to 90°").on_click(lambda _: self._snap90())
            gui.add_button("Reset this link").on_click(lambda _: self._reset_link())
            gui.add_button("Reset all links").on_click(lambda _: self._reset_all())

        with gui.add_folder("NOCS / NPCS (live)"):
            self.nocs_img = gui.add_image(np.full((8, 8, 3), 240, np.uint8), label="NOCS (root frame)")
            self.npcs_img = gui.add_image(np.full((8, 8, 3), 240, np.uint8), label="NPCS (link frames)")

        self.ref_folder = gui.add_folder("Reference images")
        with self.ref_folder:
            self.ref_folder_note = gui.add_markdown("_(per-category folder)_")

    # ================================================================== #
    # Loading
    # ================================================================== #
    def _select(self, label: str) -> None:
        for i, r in enumerate(self.queue):
            if r.object_id == label and i != self.index:
                self._load(i)
                return

    def _advance(self, step: int) -> None:
        if self.queue:
            self._load(self.index + step)

    def _load(self, index: int) -> None:
        if not self.queue:
            self.status.content = "### No objects."
            return
        self.index = index % len(self.queue)
        ref = self.queue[self.index]
        model = self.catalog.load_model(ref)
        if model is None or model.skip_reason:
            reason = model.skip_reason if model else "could not load"
            self.status.content = f"### {ref.object_id}\n**skipped:** {reason}"
            self.state = None
            return
        self.state = ObjectState.from_model(model)
        self.state.mode = self.mode_dropdown.value
        self.obj_dropdown.value = ref.object_id
        self.link_dropdown.options = list(self.state.model.chain)
        self.link_dropdown.value = self.state.base_link()
        self._sync_link_sliders()
        self._rebuild_joint_sliders()
        self._load_reference_images(ref.category)
        self._render_scene()
        self._render_nocs_npcs()
        self._update_status()

    def _rebuild_joint_sliders(self) -> None:
        for h in self._joint_sliders:
            h.remove()
        self._joint_sliders = []
        if self.state is None:
            return
        with self.joints_folder:
            for j in self.state.model.moving_joints:
                lo, hi = float(j.lower), float(j.upper)
                if hi <= lo:
                    lo, hi = -np.pi, np.pi
                step = (hi - lo) / 200.0 or 0.01
                s = self.server.gui.add_slider(j.name, lo, hi, step, float(np.clip(0.0, lo, hi)))
                s.on_update(self._make_joint_cb(j.name))
                self._joint_sliders.append(s)

    def _make_joint_cb(self, name: str):
        def cb(_):
            if self.state is None:
                return
            for s in self._joint_sliders:
                if s.label == name:
                    self.state.cfg[name] = float(s.value)
            self._render_scene()
            self._render_nocs_npcs()
        return cb

    def _load_reference_images(self, category: str) -> None:
        for h in self._ref_image_handles:
            h.remove()
        self._ref_image_handles = []
        imgs = references.load_reference_images(self.reference_dir, category)
        folder = references.category_dir(self.reference_dir, category)
        self.ref_folder_note.content = (
            f"_{len(imgs)} image(s) in_ `{folder}`\n\n"
            "_drop up to 3 reference pictures there and reload the object._")
        for name, img in imgs:
            with self.ref_folder:
                self._ref_image_handles.append(self.server.gui.add_image(img, label=name))

    # ================================================================== #
    # Per-link editing
    # ================================================================== #
    def _selected_link(self) -> Optional[str]:
        if self.state is None:
            return None
        v = self.link_dropdown.value
        return v if v in self.state.model.chain else None

    def _sync_link_sliders(self) -> None:
        name = self._selected_link()
        if name is None:
            return
        e = self.state.link_euler.get(name, np.zeros(3))
        for k, i in (("rx", 0), ("ry", 1), ("rz", 2)):
            self.link_sliders[k].value = float(e[i])

    def _on_link_select(self) -> None:
        self._sync_link_sliders()

    def _on_link_frame_change(self) -> None:
        name = self._selected_link()
        if name is None:
            return
        self.state.set_link_euler(name, [self.link_sliders["rx"].value,
                                         self.link_sliders["ry"].value,
                                         self.link_sliders["rz"].value])
        self._render_scene()
        self._render_nocs_npcs()
        self._update_status()

    def _snap90(self) -> None:
        name = self._selected_link()
        if name is None:
            return
        self.state.snap_link_90(name)
        self._sync_link_sliders()
        self._render_scene()
        self._render_nocs_npcs()

    def _reset_link(self) -> None:
        name = self._selected_link()
        if name is None:
            return
        self.state.reset_link(name)
        self._sync_link_sliders()
        self._render_scene()
        self._render_nocs_npcs()

    def _reset_all(self) -> None:
        if self.state is None:
            return
        self.state.reset_all_links()
        self._sync_link_sliders()
        self._render_scene()
        self._render_nocs_npcs()

    def _on_mode(self) -> None:
        if self.state is not None:
            self.state.mode = self.mode_dropdown.value
        self._update_status()

    # ================================================================== #
    # Rendering
    # ================================================================== #
    def _render_scene(self) -> None:
        if self.state is None:
            return
        scene = self.server.scene
        for h in self._mesh_handles:
            h.remove()
        self._mesh_handles = []
        for name, mesh in self.state.link_meshes_view().items():
            self._mesh_handles.append(scene.add_mesh_trimesh(f"/object/mesh/{name}", mesh))

        wxyzs, positions = self.state.link_frames_view()
        if self.show_link_frames.value and len(wxyzs):
            scene.add_batched_axes("/object/link_frames", wxyzs, positions,
                                   axes_length=0.16, axes_radius=0.007)
        else:
            scene.add_batched_axes("/object/link_frames", np.zeros((0, 4)), np.zeros((0, 3)))

        segs = self.state.joint_axes_view() if self.show_joint_axes.value else []
        if segs:
            pts = np.array([[s, e] for s, e in segs], float)
            cols = np.tile(np.array([[[255, 220, 0], [255, 220, 0]]], np.uint8), (len(segs), 1, 1))
            scene.add_line_segments("/object/joint_axes", pts, cols, line_width=4.0)
        else:
            scene.add_line_segments("/object/joint_axes", np.zeros((1, 2, 3)),
                                    np.zeros((1, 2, 3), np.uint8), visible=False)

    def _render_nocs_npcs(self) -> None:
        if self.state is None:
            return
        display, nocs_cols, npcs_cols, _ids = self.state.nocs_npcs_render()
        self.nocs_img.image = render.render_points_image(display, nocs_cols)
        self.npcs_img.image = render.render_points_image(display, npcs_cols)

    def _update_status(self, extra: str = "") -> None:
        if self.state is None:
            return
        ref = self.queue[self.index]
        done = "✓" if self.store.is_done(ref.object_id, catalog_mod.input_hash(ref)) else "•"
        sel = self._selected_link()
        edited = [n for n, e in self.state.link_euler.items() if np.any(e)]
        lines = [
            f"### {ref.object_id}  {done}",
            f"{ref.category} / {ref.sub_category}  ·  object {self.index + 1}/{len(self.queue)}",
            f"mode: **{self.state.mode}**  ·  editing frame: `{sel}`",
            f"edited link frames: `{edited or 'none'}`",
        ]
        if self.state.model.canonical_fallback:
            lines.append("⚠️ Stage-2 mid-limit fallback (extension unreachable)")
        if extra:
            lines.append(f"_{extra}_")
        self.status.content = "\n\n".join(lines)

    # ================================================================== #
    # Save
    # ================================================================== #
    def _accept(self) -> None:
        if self.state is None:
            return
        if self.state.mode == MODE_REST_POSE and self.state.cfg:
            self.state.set_current_pose_as_zero()
            self.link_dropdown.value = self.state.base_link()
            self._sync_link_sliders()
        ref = self.queue[self.index]
        corrected = any(np.any(e) for e in self.state.link_euler.values())
        outcome = "corrected" if corrected else "accepted"
        path = self.store.record(self.state, ref, catalog_mod.input_hash(ref), outcome)
        logger.info("wrote %s", path)
        self._update_status(extra=f"saved → {path.name}")
        self._advance(+1)

    def _skip(self) -> None:
        if self.state is None:
            return
        ref = self.queue[self.index]
        self.store.record(self.state, ref, catalog_mod.input_hash(ref), "skipped")
        self._advance(+1)

    # ================================================================== #
    def run(self) -> None:
        logger.info("Viser server running; open the printed URL. Ctrl+C to stop.")
        try:
            self.server.sleep_forever()
        except KeyboardInterrupt:
            pass
