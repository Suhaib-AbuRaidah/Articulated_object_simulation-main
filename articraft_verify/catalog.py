"""Object discovery and loading (no reference / no alignment).

Provides a flat, ordered queue of ``fit`` archives plus a stable per-object input
hash so re-runs can jump past unchanged objects.
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from articraft_canon import dataset, parse
from articraft_canon.dataset import ArchiveRef
from articraft_canon.parse import ObjectModel


def input_hash(ref: ArchiveRef) -> str:
    """Stable hash of an object's input (archive path + size + mtime)."""
    st = ref.archive.stat()
    h = hashlib.sha1()
    h.update(str(ref.archive).encode())
    h.update(str(st.st_size).encode())
    h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()[:16]


@dataclass
class Catalog:
    dataset_root: Path
    work_dir: Path
    fit_only: bool = True
    categories: Optional[List[str]] = None
    sub_categories_filter: Optional[List[str]] = None
    splits: Optional[List[str]] = None
    refs: List[ArchiveRef] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.refs = dataset.discover_archives(
            self.dataset_root,
            fit_only=self.fit_only,
            categories=self.categories,
            sub_categories=self.sub_categories_filter,
            splits=self.splits,
        )
        self.refs.sort(key=lambda r: (r.category, r.sub_category, r.object_id))

    def load_model(self, ref: ArchiveRef) -> Optional[ObjectModel]:
        urdf = dataset.extract_archive(ref, self.work_dir)
        if urdf is None:
            return None
        return parse.load_object(
            urdf, object_id=ref.object_id, source_archive=ref.archive,
            category=ref.category, sub_category=ref.sub_category, split=ref.split,
        )


def make_work_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="articraft_verify_"))
