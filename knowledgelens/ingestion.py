from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from pypdf import PdfReader

from .models import DocumentChunk


@dataclass(frozen=True, slots=True)
class IngestionLimits:
    max_files: int = 24
    max_upload_bytes: int = 24 * 1024 * 1024
    max_extracted_chars: int = 1_000_000
    max_chunks: int = 320


@dataclass(slots=True)
class IngestionResult:
    chunks: list[DocumentChunk]
    warnings: list[str]
    fatal_error: str | None = None

    def __iter__(self) -> Iterator[list[DocumentChunk] | list[str]]:
        """Preserve the existing `chunks, warnings = prepare_chunks(...)` API."""
        yield self.chunks
        yield self.warnings


DEFAULT_INGESTION_LIMITS = IngestionLimits()


def _format_pages(pages: list[int]) -> str:
    return ", ".join(str(page) for page in pages)


def _extract_sections_with_warnings(uploaded_file) -> tuple[list[tuple[int | None, str]], list[str]]:
    """Extract sections plus coverage warnings while preserving PDF page provenance."""
    uploaded_file.seek(0)
    name = str(uploaded_file.name)
    lower_name = name.lower()

    if lower_name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        sections: list[tuple[int | None, str]] = []
        failed_pages: list[int] = []
        empty_pages: list[int] = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                failed_pages.append(page_number)
                continue
            if text.strip():
                sections.append((page_number, text))
            else:
                empty_pages.append(page_number)

        warnings: list[str] = []
        if failed_pages:
            warnings.append(f"{name}: PDF text extraction failed on page(s) {_format_pages(failed_pages)}")
        if empty_pages:
            warnings.append(
                f"{name}: PDF page(s) {_format_pages(empty_pages)} had no extractable text "
                "(possibly scanned/image-only)"
            )
        return sections, warnings

    raw = uploaded_file.read()
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return [(None, raw.decode(encoding))], []
        except UnicodeDecodeError:
            continue
    return [], []


def extract_sections_from_file(uploaded_file) -> list[tuple[int | None, str]]:
    """Compatibility wrapper returning extracted sections without warning metadata."""
    sections, _warnings = _extract_sections_with_warnings(uploaded_file)
    return sections


def _split_blocks(text: str) -> list[str]:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks: list[str] = []
    current: list[str] = []

    for line in cleaned.split("\n"):
        line = line.rstrip()
        if line.strip():
            current.append(line)
        elif current:
            blocks.append("\n".join(current))
            current = []

    if current:
        blocks.append("\n".join(current))
    return blocks


def _split_oversized(text: str, max_chars: int, overlap: int) -> Iterable[str]:
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        safe_boundary: str | None = None
        if end < len(text):
            newline_boundary = text.rfind("\n", start, end)
            sentence_boundary = text.rfind(". ", start, end)
            boundary = max(newline_boundary, sentence_boundary)
            if boundary > start + max_chars // 2:
                # The piece ends at a semantic boundary, so copying overlap would
                # repeat already-complete lines/sentences into the next LLM request.
                end = boundary + 1
                safe_boundary = "newline" if boundary == newline_boundary else "sentence"

        piece = text[start:end].strip("\n")
        if piece:
            yield piece
        if end >= len(text):
            break

        if safe_boundary == "newline":
            # `end` already points at the first character of the next line. Do not
            # consume any following spaces/tabs: they may be meaningful indentation.
            start = end
            continue
        if safe_boundary == "sentence":
            # The boundary includes the period but not its separator. Consume only
            # horizontal prose spacing, never a newline/indentation sequence.
            start = end
            while start < len(text) and text[start] in {" ", "\t"}:
                start += 1
            continue

        # Only retain overlap when we truly had to cut mid-content.
        next_start = max(start + 1, end - overlap)
        newline = text.find("\n", next_start, end)
        start = newline + 1 if newline != -1 else next_start


def chunk_section(text: str, max_chars: int = 3200, overlap: int = 240) -> list[str]:
    """Chunk text while preserving structure and avoiding duplication at safe block boundaries."""
    blocks = _split_blocks(text)
    if not blocks:
        stripped = text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
        blocks = [stripped] if stripped else []

    chunks: list[str] = []
    current = ""
    for block in blocks:
        if len(block) > max_chars:
            if current:
                chunks.append(current.strip("\n"))
                current = ""
            chunks.extend(_split_oversized(block, max_chars, overlap))
            continue

        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current.strip("\n"))

        # A blank-line/block boundary is already a safe semantic split. Do not
        # copy the prior chunk's suffix into this block: doing so can cause a
        # complete claim to be extracted twice from adjacent LLM requests.
        current = block

    if current:
        chunks.append(current.strip("\n"))
    return chunks


def _uploaded_file_size(uploaded_file) -> int:
    size = getattr(uploaded_file, "size", None)
    if isinstance(size, int) and not isinstance(size, bool) and size >= 0:
        return size
    if hasattr(uploaded_file, "getbuffer"):
        return uploaded_file.getbuffer().nbytes

    try:
        position = uploaded_file.tell()
    except (AttributeError, OSError):
        position = None
    uploaded_file.seek(0, 2)
    total = uploaded_file.tell()
    if position is not None:
        uploaded_file.seek(position)
    else:
        uploaded_file.seek(0)
    return total


def _workload_limit_warning(files, limits: IngestionLimits) -> str | None:
    if len(files) > limits.max_files:
        return f"Ingestion limit exceeded: at most {limits.max_files} files can be processed in one build."

    total_bytes = sum(_uploaded_file_size(uploaded_file) for uploaded_file in files)
    if total_bytes > limits.max_upload_bytes:
        mib = limits.max_upload_bytes / (1024 * 1024)
        return f"Ingestion limit exceeded: combined uploads must be at most {mib:g} MiB per build."
    return None


def _uploaded_file_digest(uploaded_file) -> str:
    """Return a stable short content digest without changing the file's read position."""
    if hasattr(uploaded_file, "getvalue"):
        raw = uploaded_file.getvalue()
    else:
        try:
            position = uploaded_file.tell()
        except (AttributeError, OSError):
            position = None
        uploaded_file.seek(0)
        raw = uploaded_file.read()
        if position is not None:
            uploaded_file.seek(position)
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    return hashlib.sha256(bytes(raw)).hexdigest()[:10]


def _source_labels(uploaded_files) -> list[str]:
    """Keep normal filenames readable while disambiguating duplicate basenames stably."""
    files = list(uploaded_files)
    name_counts = Counter(str(uploaded_file.name).casefold() for uploaded_file in files)
    digests = [_uploaded_file_digest(uploaded_file) for uploaded_file in files]
    duplicate_content_counts = Counter(
        (str(uploaded_file.name).casefold(), digest)
        for uploaded_file, digest in zip(files, digests, strict=True)
        if name_counts[str(uploaded_file.name).casefold()] > 1
    )
    occurrence: defaultdict[tuple[str, str], int] = defaultdict(int)

    labels: list[str] = []
    for uploaded_file, digest in zip(files, digests, strict=True):
        name = str(uploaded_file.name)
        folded_name = name.casefold()
        if name_counts[folded_name] == 1:
            labels.append(name)
            continue

        key = (folded_name, digest)
        if duplicate_content_counts[key] > 1:
            occurrence[key] += 1
            labels.append(f"{name} · {digest}-{occurrence[key]}")
        else:
            labels.append(f"{name} · {digest}")
    return labels


def prepare_chunks(
    uploaded_files,
    limits: IngestionLimits = DEFAULT_INGESTION_LIMITS,
) -> IngestionResult:
    """Prepare bounded chunks and distinguish fatal budget rejection from empty extraction."""
    files = list(uploaded_files)
    limit_warning = _workload_limit_warning(files, limits)
    if limit_warning:
        return IngestionResult([], [], fatal_error=limit_warning)

    source_labels = _source_labels(files)
    chunks: list[DocumentChunk] = []
    warnings: list[str] = []
    global_index = 0
    extracted_chars = 0

    for uploaded_file, source_label in zip(files, source_labels, strict=True):
        try:
            sections, extraction_warnings = _extract_sections_with_warnings(uploaded_file)
        except Exception as exc:
            warnings.append(f"{uploaded_file.name}: {exc}")
            continue
        warnings.extend(extraction_warnings)

        if not sections:
            if not extraction_warnings:
                warnings.append(f"{uploaded_file.name}: no extractable text found")
            continue

        for page, text in sections:
            extracted_chars += len(text)
            if extracted_chars > limits.max_extracted_chars:
                return IngestionResult(
                    [],
                    warnings,
                    fatal_error=(
                        "Ingestion limit exceeded: extracted text is too large for one build "
                        f"(maximum {limits.max_extracted_chars:,} characters)."
                    ),
                )

            pieces = chunk_section(text)
            if global_index + len(pieces) > limits.max_chunks:
                return IngestionResult(
                    [],
                    warnings,
                    fatal_error=(
                        "Ingestion limit exceeded: the build would require too many model requests "
                        f"(maximum {limits.max_chunks} chunks)."
                    ),
                )

            for piece in pieces:
                global_index += 1
                chunks.append(
                    DocumentChunk(
                        source=source_label,
                        text=piece,
                        chunk_index=global_index,
                        page=page,
                    )
                )
    return IngestionResult(chunks, warnings)
