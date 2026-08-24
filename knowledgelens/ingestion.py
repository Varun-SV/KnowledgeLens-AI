from __future__ import annotations

from collections.abc import Iterable

from pypdf import PdfReader

from .models import DocumentChunk


def extract_sections_from_file(uploaded_file) -> list[tuple[int | None, str]]:
    """Extract source sections while preserving PDF page provenance."""
    uploaded_file.seek(0)
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        sections: list[tuple[int | None, str]] = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if text.strip():
                sections.append((page_number, text))
        return sections

    raw = uploaded_file.read()
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return [(None, raw.decode(encoding))]
        except UnicodeDecodeError:
            continue
    return []


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
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if boundary > start + max_chars // 2:
                end = boundary + 1

        piece = text[start:end].strip("\n")
        if piece:
            yield piece
        if end >= len(text):
            break

        next_start = max(start + 1, end - overlap)
        newline = text.find("\n", next_start, end)
        start = newline + 1 if newline != -1 else next_start


def chunk_section(text: str, max_chars: int = 3200, overlap: int = 240) -> list[str]:
    """Chunk text without flattening source-code/YAML line structure or indentation."""
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
        prefix = current[-overlap:] if current and overlap else ""
        current = f"{prefix}\n\n{block}" if prefix else block
        if len(current) > max_chars:
            chunks.extend(_split_oversized(current, max_chars, overlap))
            current = ""

    if current:
        chunks.append(current.strip("\n"))
    return chunks


def prepare_chunks(uploaded_files) -> tuple[list[DocumentChunk], list[str]]:
    chunks: list[DocumentChunk] = []
    warnings: list[str] = []
    global_index = 0

    for uploaded_file in uploaded_files:
        try:
            sections = extract_sections_from_file(uploaded_file)
        except Exception as exc:
            warnings.append(f"{uploaded_file.name}: {exc}")
            continue

        if not sections:
            warnings.append(f"{uploaded_file.name}: no extractable text found")
            continue

        for page, text in sections:
            for piece in chunk_section(text):
                global_index += 1
                chunks.append(
                    DocumentChunk(
                        source=uploaded_file.name,
                        text=piece,
                        chunk_index=global_index,
                        page=page,
                    )
                )
    return chunks, warnings
