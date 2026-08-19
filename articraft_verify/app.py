"""Single-object Viser verifier.

Loads one object as a **mesh** (natural orientation, only centred+scaled for the
view) with each link's coordinate frame and the joint axes.  You:

  * correct physical geometry and motion-axis orientation while keeping link-
    frame orientations fixed and moving pivots with the connected geometry;
  * drive **joint sliders** (``update_cfg``);
  * select a link and rotate **its own coordinate frame** -- the triad follows and
    that part's **NPCS** re-renders; rotating the base link re-renders **NOCS**;
  * eyeball the object's live **NOCS** (root-frame) and **NPCS** (per-link-frame)
    images against reference pictures dropped in ``<reference-dir>/<category>/``.

Object orientation rotates physical contents, motion axes, and pivot positions
about the object base while preserving link-frame orientations. Per-link frame
edits independently change the coordinate frames used for NOCS/NPCS.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import viser

from articraft_canon.dataset import ArchiveRef

from . import catalog as catalog_mod
from . import references, render
from .objectstate import ObjectState
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
        skip_done: bool = True,
        raw_urdf: bool = False,
        default_counter_rotate_link_frames: bool = False,
    ) -> None:
        self.catalog = catalog
        self.store = store
        self.reference_dir = Path(reference_dir)
        self.skip_done = skip_done
        self.raw_urdf = raw_urdf
        self.default_counter_rotate_link_frames = bool(
            default_counter_rotate_link_frames
        )
        self.server = viser.ViserServer(host=host, port=port)

        self.queue: List[ArchiveRef] = list(catalog.refs)
        self.index = 0
        self.state: Optional[ObjectState] = None
        self._joint_sliders: Dict[str, object] = {}
        self._joint_frame_checkboxes: Dict[str, object] = {}
        self._joint_limit_inputs: Dict[str, Dict[str, object]] = {}
        self._joint_controls: List = []
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

        with gui.add_folder("View"):
            self.show_global_frame = gui.add_checkbox("Global frame", True)
            self.show_link_frames = gui.add_checkbox("Link frames", True)
            self.show_joint_axes = gui.add_checkbox("Joint axes", True)
            self.show_global_frame.on_update(lambda _: self._render_scene())
            self.show_link_frames.on_update(lambda _: self._render_scene())
            self.show_joint_axes.on_update(lambda _: self._render_scene())

        with gui.add_folder("Object orientation"):
            gui.add_markdown(
                "Rotate every link's visual, collision, inertial data, and "
                "motion axis. Link-frame orientations and joint-origin RPY "
                "stay fixed; pivot positions and joint-origin XYZ rotate with "
                "the connected geometry."
            )
            self.object_sliders = {
                "rx": gui.add_slider("object rot X°", -180, 180, 1, 0),
                "ry": gui.add_slider("object rot Y°", -180, 180, 1, 0),
                "rz": gui.add_slider("object rot Z°", -180, 180, 1, 0),
            }
            for handle in self.object_sliders.values():
                handle.on_update(lambda _: self._on_object_orientation_change())
            gui.add_button("Snap object to 90°").on_click(lambda _: self._snap_object90())
            gui.add_button("Reset object orientation").on_click(
                lambda _: self._reset_object_orientation()
            )

        self.joints_folder = gui.add_folder("Joint states & ranges")
        with self.joints_folder:
            gui.add_markdown(
                "Ranges use the current joint coordinates. On save, the "
                "displayed state becomes `q=0`, so both saved endpoints shift "
                "by `-state`."
            )
            gui.add_markdown(
                "Each rotational joint has its own frame-compensation checkbox. "
                "It removes that joint's rotation from its child and descendant "
                "coordinate frames, without changing geometry or physical axes."
            )

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
        self.state = ObjectState.from_model(model, raw_urdf=self.raw_urdf)
        if self.default_counter_rotate_link_frames:
            for joint in self.state.model.moving_joints:
                if joint.type in ("revolute", "continuous"):
                    self.state.set_counter_rotate_joint_frames(joint.name, True)
        self.obj_dropdown.value = ref.object_id
        self.link_dropdown.options = self.state.object_link_names()
        self.link_dropdown.value = self.state.base_link()
        self._sync_object_sliders()
        self._sync_link_sliders()
        self._rebuild_joint_sliders()
        self._load_reference_images(ref.category)
        self._render_scene()
        self._render_nocs_npcs()
        self._update_status()

    def _rebuild_joint_sliders(self) -> None:
        for h in self._joint_controls:
            h.remove()
        self._joint_sliders = {}
        self._joint_frame_checkboxes = {}
        self._joint_limit_inputs = {}
        self._joint_controls = []
        if self.state is None:
            return
        with self.joints_folder:
            for j in self.state.model.moving_joints:
                limits = self.state.joint_limits.get(
                    j.name,
                    np.array([j.lower, j.upper], dtype=float),
                )
                lo, hi = map(float, limits)
                if hi <= lo:
                    lo, hi = -np.pi, np.pi
                step = (hi - lo) / 200.0 or 0.01
                state_value = float(np.clip(self.state.cfg.get(j.name, 0.0), lo, hi))
                s = self.server.gui.add_slider(
                    f"{j.name} state",
                    lo,
                    hi,
                    step,
                    state_value,
                )
                s.on_update(self._make_joint_cb(j.name))
                self._joint_sliders[j.name] = s
                self._joint_controls.append(s)

                if j.type in ("revolute", "continuous"):
                    counter_rotate = self.server.gui.add_checkbox(
                        f"{j.name}: counter-rotate child frames",
                        j.name in self.state.counter_rotate_joint_frames,
                        hint=(
                            f"Remove {j.name}'s rotation from the coordinate "
                            f"frames of {j.child} and its descendants."
                        ),
                    )
                    counter_rotate.on_update(
                        self._make_joint_frame_compensation_cb(j.name)
                    )
                    self._joint_frame_checkboxes[j.name] = counter_rotate
                    self._joint_controls.append(counter_rotate)

                continuous = j.type == "continuous"
                hint = (
                    "Continuous URDF joints have no finite lower/upper range."
                    if continuous
                    else "Limit in the current, pre-bake joint coordinates."
                )
                lower = self.server.gui.add_number(
                    f"{j.name} lower",
                    float(limits[0]),
                    step=step,
                    disabled=continuous,
                    hint=hint,
                )
                upper = self.server.gui.add_number(
                    f"{j.name} upper",
                    float(limits[1]),
                    step=step,
                    disabled=continuous,
                    hint=hint,
                )
                self._joint_limit_inputs[j.name] = {
                    "lower": lower,
                    "upper": upper,
                }
                self._joint_controls.extend((lower, upper))
                if not continuous:
                    lower.on_update(self._make_joint_limit_cb(j.name, "lower"))
                    upper.on_update(self._make_joint_limit_cb(j.name, "upper"))

    def _make_joint_cb(self, name: str):
        def cb(_):
            if self.state is None:
                return
            slider = self._joint_sliders.get(name)
            if slider is None:
                return
            value = float(slider.value)
            limits = self.state.joint_limits.get(name)
            if limits is not None:
                value = float(np.clip(value, limits[0], limits[1]))
                if value != float(slider.value):
                    slider.value = value
            self.state.cfg[name] = value
            self._render_scene()
            self._render_nocs_npcs()
            self._update_status()
        return cb

    def _make_joint_frame_compensation_cb(self, name: str):
        def cb(_):
            if self.state is None:
                return
            checkbox = self._joint_frame_checkboxes.get(name)
            if checkbox is None:
                return
            self.state.set_counter_rotate_joint_frames(name, checkbox.value)
            self._render_scene()
            self._render_nocs_npcs()
            self._update_status()
        return cb

    def _make_joint_limit_cb(self, name: str, changed: str):
        def cb(_):
            if self.state is None:
                return
            inputs = self._joint_limit_inputs.get(name)
            slider = self._joint_sliders.get(name)
            if inputs is None or slider is None:
                return

            lower = float(inputs["lower"].value)
            upper = float(inputs["upper"].value)
            state_value = float(self.state.cfg.get(name, slider.value))
            # Keep the currently displayed pose reachable. The state callback
            # separately clamps future slider changes to the edited interval.
            if changed == "lower" and lower > state_value:
                lower = state_value
                inputs["lower"].value = lower
            if changed == "upper" and upper < state_value:
                upper = state_value
                inputs["upper"].value = upper
            if lower > upper:
                if changed == "lower":
                    lower = upper
                    inputs["lower"].value = lower
                else:
                    upper = lower
                    inputs["upper"].value = upper

            self.state.set_joint_limits(name, lower, upper)
            self._update_status()
            # Viser slider bounds are immutable after creation. Rebuild the
            # controls so an expanded/narrowed range immediately becomes the
            # state slider's actual selectable interval.
            self._rebuild_joint_sliders()

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
    # Physical geometry/axis orientation
    # ================================================================== #
    def _sync_object_sliders(self) -> None:
        if self.state is None:
            return
        for key, index in (("rx", 0), ("ry", 1), ("rz", 2)):
            self.object_sliders[key].value = float(self.state.object_euler[index])

    def _on_object_orientation_change(self) -> None:
        if self.state is None:
            return
        self.state.set_object_euler(
            [
                self.object_sliders["rx"].value,
                self.object_sliders["ry"].value,
                self.object_sliders["rz"].value,
            ]
        )
        self._render_scene()
        self._render_nocs_npcs()
        self._update_status()

    def _snap_object90(self) -> None:
        if self.state is None:
            return
        self.state.snap_object_90()
        self._sync_object_sliders()
        self._render_scene()
        self._render_nocs_npcs()
        self._update_status()

    def _reset_object_orientation(self) -> None:
        if self.state is None:
            return
        self.state.reset_object_orientation()
        self._sync_object_sliders()
        self._render_scene()
        self._render_nocs_npcs()
        self._update_status()

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

        if self.show_global_frame.value:
            scene.add_batched_axes(
                "/world/global_frame",
                np.array([[1.0, 0.0, 0.0, 0.0]]),
                self.state.global_frame_position_view()[None, :],
                axes_length=0.23,
                axes_radius=0.009,
            )
        else:
            scene.add_batched_axes(
                "/world/global_frame",
                np.zeros((0, 4)),
                np.zeros((0, 3)),
            )

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
        edited_ranges = list(self.state.joint_limit_edits())
        lines = [
            f"### {ref.object_id}  {done}",
            f"{ref.category} / {ref.sub_category}  ·  object {self.index + 1}/{len(self.queue)}",
            f"editing frame: `{sel}`",
            f"URDF loading: **{'raw/preserved' if self.state.raw_urdf else 'canonicalized'}**",
            f"object orientation XYZ°: `{self.state.object_euler.tolist()}`",
            "counter-rotated frame joints: "
            f"`{sorted(self.state.counter_rotate_joint_frames) or 'none'}`",
            f"edited link frames: `{edited or 'none'}`",
            f"edited joint ranges: `{edited_ranges or 'none'}`",
        ]
        if self.state.model.canonical_fallback:
            lines.append("⚠️ Stage-2 mid-limit fallback (extension unreachable)")
        if self.state.model.converted_mimic_joints:
            names = [
                conversion["joint"]
                for conversion in self.state.model.converted_mimic_joints
            ]
            lines.append(
                "⚠️ mimic joint(s) converted to independent joints: "
                f"`{names}`"
            )
        if extra:
            lines.append(f"_{extra}_")
        self.status.content = "\n\n".join(lines)

    # ================================================================== #
    # Save
    # ================================================================== #
    def _accept(self) -> None:
        if self.state is None:
            return
        ref = self.queue[self.index]
        corrected = self.state.has_pending_edits()
        self.state.commit_edits()
        outcome = "corrected" if corrected else "accepted"
        path = self.store.record(self.state, ref, catalog_mod.input_hash(ref), outcome)
        assert path is not None
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
