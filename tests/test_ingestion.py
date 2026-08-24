from knowledgelens.ingestion import chunk_section


def test_chunking_preserves_line_structure_and_indentation():
    source = "root:\n  child: value\n  nested:\n    item: 1\n\nprint('hello')\n  indented = True"
    chunks = chunk_section(source, max_chars=200, overlap=20)
    joined = "\n\n".join(chunks)
    assert "root:\n  child: value" in joined
    assert "  nested:\n    item: 1" in joined
    assert "print('hello')\n  indented = True" in joined
