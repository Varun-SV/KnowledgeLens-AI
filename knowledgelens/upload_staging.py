from __future__ import annotations

import io
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

_MIB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class StagedUpload:
    """One upload retained only after it fits the aggregate build envelope."""

    name: str
    data: bytes

    @property
    def size(self) -> int:
        return len(self.data)


class BufferedUpload(io.BytesIO):
    """Fresh file-like view compatible with the ingestion helpers."""

    def __init__(self, staged: StagedUpload):
        super().__init__(staged.data)
        self.name = staged.name
        self.size = staged.size


def staged_upload_total(staged: Iterable[StagedUpload]) -> int:
    return sum(item.size for item in staged)


def remaining_upload_bytes(staged: Iterable[StagedUpload], max_bytes: int) -> int:
    if max_bytes < 0:
        raise ValueError("Upload byte limit must be non-negative.")
    return max(0, max_bytes - staged_upload_total(staged))


def next_upload_limit_mb(staged: Iterable[StagedUpload], max_bytes: int, max_files: int) -> int | None:
    """Return a conservative per-next-file limit that cannot exceed the remaining aggregate budget.

    Streamlit's ``max_upload_size`` is integer MiB and applies to each selected file.
    The UI intentionally accepts one file at a time; after each successful stage the
    uploader is recreated with a limit derived from the remaining aggregate bytes.
    """
    items = list(staged)
    if max_files < 1:
        raise ValueError("Upload file limit must be positive.")
    if len(items) >= max_files:
        return None

    remaining = remaining_upload_bytes(items, max_bytes)
    whole_mib = remaining // _MIB
    return int(whole_mib) if whole_mib >= 1 else None


def _uploaded_bytes(uploaded_file: Any) -> bytes:
    if uploaded_file is None:
        raise ValueError("No source file was selected.")

    size = getattr(uploaded_file, "size", None)
    if isinstance(size, int) and not isinstance(size, bool) and size < 0:
        raise ValueError("Uploaded file reported an invalid size.")

    if hasattr(uploaded_file, "getvalue"):
        raw = uploaded_file.getvalue()
    else:
        uploaded_file.seek(0)
        raw = uploaded_file.read()

    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    data = bytes(raw)
    if isinstance(size, int) and not isinstance(size, bool) and size != len(data):
        raise ValueError("Uploaded file size changed while it was being staged.")
    return data


def stage_uploaded_file(
    staged: Iterable[StagedUpload],
    uploaded_file: Any,
    *,
    max_bytes: int,
    max_files: int,
) -> list[StagedUpload]:
    """Return a new queue only if the selected file fits both aggregate limits."""
    items = list(staged)
    if len(items) >= max_files:
        raise ValueError(f"At most {max_files} source files can be staged for one build.")

    name = str(getattr(uploaded_file, "name", "") or "").strip()
    if not name:
        raise ValueError("Uploaded source file must have a filename.")

    # The Streamlit widget already bounds the pending file to the remaining budget,
    # but this independent check keeps sibling callers from bypassing the invariant.
    data = _uploaded_bytes(uploaded_file)
    if staged_upload_total(items) + len(data) > max_bytes:
        raise ValueError(
            f"Combined staged sources exceed the {max_bytes / _MIB:g} MiB per-build upload limit."
        )

    return [*items, StagedUpload(name=name, data=data)]


def materialize_staged_uploads(staged: Iterable[StagedUpload]) -> list[BufferedUpload]:
    """Create fresh independent file objects only when a build actually starts."""
    return [BufferedUpload(item) for item in staged]
