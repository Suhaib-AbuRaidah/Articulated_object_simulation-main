"""Decision log + output writer (never touches the originals).

Outputs mirror the ``articraft_canon`` layout so downstream tooling is unchanged:

    <output>/<category>/<sub_category>/<split>/<object_id>.tar.gz
        <object_id>/model.urdf       # accepted joint pose + link frames baked in
        <object_id>/canonical.json   # NOCS/NPCS + frames + verification provenance
        <object_id>/assets/...       # copied so the URDF stays valid
    <output>/_verify/decisions.json  # append-only decision log (resume by hash)
"""

from __future__ import annotations

import datetime as _dt
import json
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yourdfpy

from articraft_canon.dataset import ArchiveRef

from .objectstate import ObjectState


@dataclass
class DecisionStore:
    output_dir: Path

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self._log_dir = self.output_dir / "_verify"
        self._log_path = self._log_dir / "decisions.json"
        self._decisions: Dict[str, dict] = {}
        if self._log_path.exists():
            self._decisions = json.loads(self._log_path.read_text())

    # ------------------------------------------------------------------ #
    def is_done(self, object_id: str, input_hash: str) -> bool:
        """True if a decision for this object exists with a matching input hash."""
        rec = self._decisions.get(object_id)
        return bool(rec and rec.get("input_hash") == input_hash)

    def decision(self, object_id: str) -> Optional[dict]:
        return self._decisions.get(object_id)

    # ------------------------------------------------------------------ #
    def record(
        self,
        state: ObjectState,
        ref: ArchiveRef,
        input_hash: str,
        outcome: str,
    ) -> Optional[Path]:
        """Record a decision, writing an archive only for accepted objects."""
        archive_path = (
            None if outcome == "skipped" else self._write_object(state, ref)
        )
        entry = {
            "object_id": ref.object_id,
            "category": ref.category,
            "sub_category": ref.sub_category,
            "split": ref.split,
            "input_hash": input_hash,
            "outcome": outcome,                       # accepted | corrected | skipped
            "raw_urdf": state.raw_urdf,
            "fixed_branches_allowed": state.model.fixed_branches_allowed,
            "converted_mimic_joints": list(
                state.model.converted_mimic_joints
            ),
            "frames_baked_into_urdf": state.frames_baked,
            "applied_joint_state": dict(state.applied_joint_cfg),
            "counter_rotated_link_frames_for_joint_state": (
                bool(state.applied_counter_rotate_joint_frames)
            ),
            "counter_rotated_link_frames_for_joints": list(
                state.applied_counter_rotate_joint_frames
            ),
            "applied_object_euler_deg": list(
                map(float, state.applied_object_euler)
            ),
            "requested_joint_limits": {
                name: list(map(float, limits))
                for name, limits in state.requested_joint_limits.items()
            },
            "baked_joint_limits": {
                name: list(map(float, limits))
                for name, limits in state.applied_joint_limits.items()
            },
            "applied_link_euler_deg": {
                name: list(map(float, e))
                for name, e in state.applied_link_euler.items()
            },
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        }
        if archive_path is not None:
            entry["output_archive"] = str(archive_path.relative_to(self.output_dir))
        self._decisions[ref.object_id] = entry
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path.write_text(json.dumps(self._decisions, indent=2))
        return archive_path

    # ------------------------------------------------------------------ #
    def _write_object(self, state: ObjectState, ref: ArchiveRef) -> Path:
        model = state.model
        out_dir = self.output_dir / ref.category / ref.sub_category / ref.split
        out_dir.mkdir(parents=True, exist_ok=True)
        archive_path = out_dir / f"{ref.object_id}.tar.gz"

        sidecar = state.build_sidecar()

        src_dir = model.urdf_path.parent
        stage = Path(tempfile.mkdtemp(prefix="articraft_verify_out_"))
        try:
            urdf_out = stage / "model.urdf"
            handler = model.urdf._filename_handler
            model.urdf._filename_handler = yourdfpy.filename_handler_null
            try:
                model.urdf.write_xml_file(str(urdf_out))
            finally:
                model.urdf._filename_handler = handler
            (stage / "canonical.json").write_text(json.dumps(sidecar, indent=2))

            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(urdf_out, arcname=f"{ref.object_id}/model.urdf")
                tar.add(stage / "canonical.json", arcname=f"{ref.object_id}/canonical.json")
                assets = src_dir / "assets"
                if assets.is_dir():
                    tar.add(assets, arcname=f"{ref.object_id}/assets")
                report = src_dir / "compile_report.json"
                if report.is_file():
                    tar.add(report, arcname=f"{ref.object_id}/compile_report.json")
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        return archive_path
