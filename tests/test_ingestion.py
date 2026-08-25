from io import BytesIO

from knowledgelens.ingestion import chunk_section, prepare_chunks


class UploadedBytes(BytesIO):
    def __init__(self, name: str, data: bytes):
        super().__init__(data)
        self.name = name


def test_chunking_preserves_line_structure_and_indentation():
    source = "root:\n  child: value\n  nested:\n    item: 1\n\nprint('hello')\n  indented = True"
    chunks = chunk_section(source, max_chars=200, overlap=20)
    joined = "\n\n".join(chunks)
    assert "root:\n  child: value" in joined
    assert "  nested:\n    item: 1" in joined
    assert "print('hello')\n  indented = True" in joined


def test_oversized_newline_split_preserves_next_line_indentation():
    source = (
        "def configure():\n"
        "    alpha = '" + ("x" * 55) + "'\n"
        "    nested_call(\n"
        "        first_argument,\n"
        "        second_argument,\n"
        "    )\n"
        "    return alpha"
    )
    chunks = chunk_section(source, max_chars=90, overlap=25)
    joined = "\n".join(chunks)

    assert len(chunks) >= 2
    assert "    nested_call(" in joined
    assert "        first_argument," in joined
    assert "        second_argument," in joined
    assert "    return alpha" in joined


def test_safe_block_boundaries_do_not_repeat_overlap_text():
    first = "Alpha fully supports Beta."
    second = "Gamma independently supports Delta."
    chunks = chunk_section(f"{first}\n\n{second}", max_chars=40, overlap=20)

    assert chunks == [first, second]
    assert first[-20:] not in chunks[1]


def test_safe_sentence_boundary_in_oversized_block_does_not_repeat_completed_sentence():
    completed = "Claim Alpha supports Beta."
    source = (
        "Introductory words fill the oversized paragraph. "
        f"{completed} "
        "Gamma continues with enough additional text to require another chunk and preserve the remaining content."
    )
    chunks = chunk_section(source, max_chars=82, overlap=30)

    assert len(chunks) >= 2
    assert sum(completed in chunk for chunk in chunks) == 1
    assert chunks[0].endswith(".")
    assert not chunks[1].startswith(chunks[0][-30:])


def test_mid_content_oversized_split_retains_overlap():
    source = "A" * 150
    chunks = chunk_section(source, max_chars=80, overlap=20)

    assert len(chunks) >= 2
    assert chunks[0][-20:] == chunks[1][:20]


def test_unique_filename_keeps_readable_source_name():
    chunks, warnings = prepare_chunks([UploadedBytes("notes.md", b"Alpha connects to Beta")])
    assert warnings == []
    assert chunks[0].source == "notes.md"


def test_duplicate_filenames_get_stable_content_disambiguators():
    chunks, warnings = prepare_chunks(
        [
            UploadedBytes("config.yaml", b"service: alpha"),
            UploadedBytes("config.yaml", b"service: beta"),
        ]
    )
    assert warnings == []
    sources = {chunk.source for chunk in chunks}
    assert len(sources) == 2
    assert all(source.startswith("config.yaml · ") for source in sources)


def test_identical_duplicate_filenames_still_get_unique_source_labels():
    chunks, warnings = prepare_chunks(
        [
            UploadedBytes("report.txt", b"same content"),
            UploadedBytes("report.txt", b"same content"),
        ]
    )
    assert warnings == []
    sources = [chunk.source for chunk in chunks]
    assert len(set(sources)) == 2
    assert sources[0].endswith("-1")
    assert sources[1].endswith("-2")
