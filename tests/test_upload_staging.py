from types import SimpleNamespace

import pytest

from knowledgelens.upload_staging import (
    StagedUpload,
    materialize_staged_uploads,
    next_upload_limit_mb,
    remaining_upload_bytes,
    stage_uploaded_file,
    staged_upload_total,
)

MIB = 1024 * 1024


def _upload(name: str, size: int):
    data = b"x" * size
    return SimpleNamespace(name=name, size=size, getvalue=lambda: data)


def test_next_single_file_limit_shrinks_with_remaining_aggregate_budget():
    staged = [StagedUpload("a.txt", b"a" * (10 * MIB)), StagedUpload("b.txt", b"b" * (5 * MIB))]

    assert staged_upload_total(staged) == 15 * MIB
    assert remaining_upload_bytes(staged, 24 * MIB) == 9 * MIB
    assert next_upload_limit_mb(staged, 24 * MIB, 24) == 9


def test_staging_rejects_a_file_that_would_cross_the_aggregate_budget():
    staged = [StagedUpload("existing.txt", b"a" * (23 * MIB))]

    accepted = stage_uploaded_file(staged, _upload("last.txt", MIB), max_bytes=24 * MIB, max_files=24)
    assert staged_upload_total(accepted) == 24 * MIB

    with pytest.raises(ValueError, match="Combined staged sources exceed"):
        stage_uploaded_file(staged, _upload("too-large.txt", MIB + 1), max_bytes=24 * MIB, max_files=24)

    # Failed staging is functional: the already retained queue is unchanged.
    assert staged_upload_total(staged) == 23 * MIB


def test_staging_stops_exposing_an_uploader_when_file_count_or_whole_mib_budget_is_exhausted():
    full_count = [StagedUpload(f"{index}.txt", b"x") for index in range(24)]
    almost_full_bytes = [StagedUpload("large.bin", b"x" * (24 * MIB - 512 * 1024))]

    assert next_upload_limit_mb(full_count, 24 * MIB, 24) is None
    assert next_upload_limit_mb(almost_full_bytes, 24 * MIB, 24) is None


def test_materialized_build_streams_are_fresh_and_preserve_name_and_size():
    staged = [StagedUpload("notes.md", b"alpha")]
    first = materialize_staged_uploads(staged)[0]
    second = materialize_staged_uploads(staged)[0]

    assert first is not second
    assert first.name == "notes.md"
    assert first.size == 5
    assert first.read() == b"alpha"
    assert second.read() == b"alpha"
