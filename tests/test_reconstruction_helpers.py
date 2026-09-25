"""Tests for reconstruction file helpers."""

import os
from datetime import datetime, timezone
from pathlib import Path

from app.reconstruction_helpers import (
    FUSED_GLB,
    fused_glb_path,
    list_finished_reconstructions,
    parse_job_id,
    reconstructed_glb_path,
)

FINISHED_JOB = "11111111-1111-1111-1111-111111111111"
OLDER_JOB = "22222222-2222-2222-2222-222222222222"


def _write_glb(
    root: Path,
    job_id: str,
    content: bytes = b"glb-bytes",
    filename: str = "reconstructed.glb",
) -> Path:
    glb_path = root / job_id / "colmap" / filename
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    glb_path.write_bytes(content)
    return glb_path


def test_parse_job_id_accepts_uuid():
    assert parse_job_id(FINISHED_JOB) == FINISHED_JOB


def test_parse_job_id_rejects_non_uuid():
    try:
        parse_job_id("../secret")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_reconstructed_glb_path_uses_colmap_filename():
    path = reconstructed_glb_path(Path("data"), FINISHED_JOB)
    assert path == Path("data") / FINISHED_JOB / "colmap" / "reconstructed.glb"


def test_fused_glb_path_uses_colmap_filename():
    path = fused_glb_path(Path("data"), FINISHED_JOB)
    assert path == Path("data") / FINISHED_JOB / "colmap" / "fused.glb"


def test_list_finished_reconstructions_empty_when_missing_dir(tmp_path):
    assert list_finished_reconstructions(tmp_path / "missing") == []


def test_list_finished_reconstructions_skips_incomplete_and_non_uuid(tmp_path):
    _write_glb(tmp_path, FINISHED_JOB)
    (tmp_path / "not-a-uuid" / "colmap").mkdir(parents=True)
    (tmp_path / "not-a-uuid" / "colmap" / "reconstructed.glb").write_bytes(b"x")
    incomplete = tmp_path / OLDER_JOB
    incomplete.mkdir()
    (incomplete / "video.mp4").write_bytes(b"video")

    items = list_finished_reconstructions(tmp_path)

    assert [item["job_id"] for item in items] == [FINISHED_JOB]
    assert items[0]["glb_url"] == f"/api/reconstructions/{FINISHED_JOB}/glb"
    datetime.fromisoformat(items[0]["modified_at"])


def test_list_finished_reconstructions_lists_fused_models(tmp_path):
    _write_glb(tmp_path, FINISHED_JOB, filename=FUSED_GLB)
    _write_glb(tmp_path, OLDER_JOB)

    items = list_finished_reconstructions(
        tmp_path,
        filename=FUSED_GLB,
        url_name="fused",
    )

    assert [item["job_id"] for item in items] == [FINISHED_JOB]
    assert items[0]["glb_url"] == f"/api/reconstructions/{FINISHED_JOB}/fused"


def test_list_finished_reconstructions_sorts_newest_first(tmp_path):
    older = _write_glb(tmp_path, OLDER_JOB, b"old")
    newer = _write_glb(tmp_path, FINISHED_JOB, b"new")
    older_mtime = 1_700_000_000
    newer_mtime = 1_800_000_000
    os.utime(older, (older_mtime, older_mtime))
    os.utime(newer, (newer_mtime, newer_mtime))

    items = list_finished_reconstructions(tmp_path)

    assert [item["job_id"] for item in items] == [FINISHED_JOB, OLDER_JOB]
    assert items[0]["modified_at"] == datetime.fromtimestamp(
        newer_mtime,
        tz=timezone.utc,
    ).isoformat()
