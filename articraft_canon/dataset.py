"""Dataset discovery and safe extraction.

The Articraft dataset ships as ``.tar.gz`` archives laid out as::

    data/urdfs/Dataset/<category>/sub_categories/<sub_category>/<fit|not-fit>/<split>/<object>.tar.gz

Each archive contains a ``model.urdf`` (plus optional ``assets/meshes``).  We
only ever *read* the originals: archives are extracted into a scratch working
directory, and canonical outputs are written to a separate output tree.  The
original ``data/urdfs/Dataset`` is never touched.
"""

from __future__ import annotations

import logging
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArchiveRef:
    """A located dataset archive together with its taxonomy coordinates."""

    archive: Path
    category: str
    sub_category: str
    split: str          # train | val | test | (unknown)
    fit: str            # fit | not-fit | (unknown)

    @property
    def object_id(self) -> str:
        # Strip the .tar.gz suffix to get a stable id.
        name = self.archive.name
        for suffix in (".tar.gz", ".tgz"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return self.archive.stem


def _classify(archive: Path, dataset_root: Path) -> ArchiveRef:
    """Infer (category, sub_category, split, fit) from the archive's path."""
    try:
        rel_parts = archive.relative_to(dataset_root).parts
    except ValueError:
        rel_parts = archive.parts
    category = rel_parts[0] if rel_parts else "unknown"

    parts = list(rel_parts)
    sub_category = "unknown"
    if "sub_categories" in parts:
        i = parts.index("sub_categories")
        if i + 1 < len(parts):
            sub_category = parts[i + 1]

    fit = next((p for p in parts if p in ("fit", "not-fit")), "unknown")
    split = next((p for p in parts if p in ("train", "val", "test")), "unknown")
    return ArchiveRef(archive, category, sub_category, split, fit)


def discover_archives(
    dataset_root: Path,
    *,
    fit_only: bool = True,
    categories: Optional[List[str]] = None,
    sub_categories: Optional[List[str]] = None,
    splits: Optional[List[str]] = None,
) -> List[ArchiveRef]:
    """Find dataset archives under ``dataset_root`` with optional filters.

    ``fit_only`` restricts to the curated ``fit/`` objects (the ~392 canonical
    objects); set it False to include ``not-fit`` too.
    """
    dataset_root = Path(dataset_root)
    refs: List[ArchiveRef] = []
    for archive in sorted(dataset_root.rglob("*.tar.gz")):
        ref = _classify(archive, dataset_root)
        if fit_only and ref.fit != "fit":
            continue
        if categories and ref.category not in categories:
            continue
        if sub_categories and ref.sub_category not in sub_categories:
            continue
        if splits and ref.split not in splits:
            continue
        refs.append(ref)
    logger.info("discovered %d archive(s) under %s", len(refs), dataset_root)
    return refs


def _is_within(base: Path, target: Path) -> bool:
    """True if ``target`` resolves to a path inside ``base`` (tar-slip guard)."""
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def extract_archive(ref: ArchiveRef, work_dir: Path) -> Optional[Path]:
    """Safely extract ``ref`` under ``work_dir`` and return its ``model.urdf``.

    Members escaping the destination (absolute paths, ``..``) are rejected.
    Returns None if no URDF is found.
    """
    dest = Path(work_dir) / ref.category / ref.sub_category / ref.object_id
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ref.archive, "r:gz") as tar:
        safe_members = []
        for member in tar.getmembers():
            member_path = dest / member.name
            if not _is_within(dest, member_path):
                logger.warning("skipping unsafe tar member %s in %s",
                               member.name, ref.archive)
                continue
            safe_members.append(member)
        tar.extractall(dest, members=safe_members)

    urdfs = sorted(dest.rglob("*.urdf"))
    if not urdfs:
        logger.warning("no .urdf inside %s", ref.archive)
        return None
    for u in urdfs:
        if u.name == "model.urdf":
            return u
    return urdfs[0]
