"""Tests for ordered S3 and DynamoDB artifact publication."""

import json
import logging
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from app.queue.publish import publish_artifacts
from app.queue.retry_publish import main
from app.settings import Settings


@pytest.fixture
def publish_workspace(tmp_path, monkeypatch):
    """Create a local summary and mock the configured AWS clients."""
    artifacts = tmp_path / "job-1" / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "database.db").write_bytes(b"database")
    summary_path = artifacts / "summary.json"
    summary_path.write_text(json.dumps({
        "job_id": "job-1",
        "database_snapshot": "artifacts/database.db",
        "feature_archive": None,
        "reconstruction_duration_seconds": 1.25,
        "median_triangulation_angle_degrees": 8.5,
    }))
    settings = Settings(dynamodb_table_name="jobs", s3_bucket="bucket")
    monkeypatch.setattr("app.queue.publish.get_settings", lambda: settings)
    resource = MagicMock()
    session = MagicMock()
    session.resource.return_value = resource
    monkeypatch.setattr("app.queue.publish.get_aws_session", lambda configured: session)
    return summary_path, resource


@pytest.mark.parametrize("with_archive", [False, True])
def test_publish_uploads_in_order_then_writes_summary(
    publish_workspace, monkeypatch, with_archive, caplog,
):
    """Store exact S3 keys and versions only after all uploads succeed."""
    caplog.set_level(logging.INFO, logger="app.queue.publish")
    summary_path, resource = publish_workspace
    if with_archive:
        (summary_path.parent / "features.tar").write_bytes(b"tar")
        summary = json.loads(summary_path.read_text())
        summary["feature_archive"] = "artifacts/features.tar"
        summary_path.write_text(json.dumps(summary))
    events = []

    def upload(key, source, content_type):
        """Record an artifact upload and return a distinct object version."""
        events.append(("s3", key, source, content_type))
        return {"VersionId": f"version-{len(events)}"}

    def write_item(*, Item):
        """Record the DynamoDB write after the S3 uploads."""
        events.append(("dynamodb", Item))

    monkeypatch.setattr("app.queue.publish.stream_file_to_s3", upload)
    resource.Table.return_value.put_item.side_effect = write_item
    publish_artifacts(summary_path)

    assert events[0][:2] == ("s3", "job-1/database.db")
    assert events[0][2] == summary_path.parent / "database.db"
    assert events[0][3] == "application/vnd.sqlite3"
    if with_archive:
        assert events[1][:2] == ("s3", "job-1/features.tar")
        assert events[1][2] == summary_path.parent / "features.tar"
        assert events[1][3] == "application/x-tar"
    assert events[-1][0] == "dynamodb"
    item = events[-1][1]
    assert item["database_s3_key"] == "job-1/database.db"
    assert item["database_s3_version_id"] == "version-1"
    assert item["feature_archive_s3_key"] == (
        "job-1/features.tar" if with_archive else None
    )
    assert item["feature_archive_s3_version_id"] == (
        "version-2" if with_archive else None
    )
    assert "database_snapshot" not in item
    assert "summary_path" not in item
    assert item["reconstruction_duration_seconds"] == Decimal("1.25")
    assert item["median_triangulation_angle_degrees"] == Decimal("8.5")
    resource.Table.assert_called_once_with("jobs")
    resource.meta.client.close.assert_called_once_with()
    assert "Finished uploading artifacts and saving summary for job job-1" in (
        caplog.text
    )


def test_publish_keeps_local_files_when_archive_upload_fails(
    publish_workspace, monkeypatch,
):
    """Skip DynamoDB when the second upload fails and retain retry files."""
    summary_path, resource = publish_workspace
    archive = summary_path.parent / "features.tar"
    archive.write_bytes(b"tar")
    summary = json.loads(summary_path.read_text())
    summary["feature_archive"] = "artifacts/features.tar"
    summary_path.write_text(json.dumps(summary))
    calls = []

    def upload(key, source, content_type):
        """Fail the archive upload after the database succeeds."""
        calls.append(key)
        if key.endswith("features.tar"):
            raise RuntimeError("upload failed")
        return {"VersionId": "database-version"}

    monkeypatch.setattr("app.queue.publish.stream_file_to_s3", upload)
    with pytest.raises(RuntimeError, match="upload failed"):
        publish_artifacts(summary_path)

    assert calls == ["job-1/database.db", "job-1/features.tar"]
    resource.Table.return_value.put_item.assert_not_called()
    assert summary_path.is_file()
    assert archive.is_file()
    assert (summary_path.parent / "database.db").is_file()


def test_publish_requires_s3_version_id(publish_workspace, monkeypatch):
    """Reject unversioned uploads before writing the job record."""
    summary_path, resource = publish_workspace
    monkeypatch.setattr("app.queue.publish.stream_file_to_s3", lambda *args: {})
    with pytest.raises(ValueError, match="version ID"):
        publish_artifacts(summary_path)
    resource.Table.return_value.put_item.assert_not_called()


def test_republish_replaces_summary_with_new_object_versions(
    publish_workspace, monkeypatch,
):
    """Write the same job key again with each successful upload's version ID."""
    summary_path, resource = publish_workspace
    versions = iter(["version-1", "version-2"])
    monkeypatch.setattr(
        "app.queue.publish.stream_file_to_s3",
        lambda *args: {"VersionId": next(versions)},
    )

    publish_artifacts(summary_path)
    publish_artifacts(summary_path)

    first, second = resource.Table.return_value.put_item.call_args_list
    assert first.kwargs["Item"]["job_id"] == "job-1"
    assert second.kwargs["Item"]["job_id"] == "job-1"
    assert first.kwargs["Item"]["database_s3_version_id"] == "version-1"
    assert second.kwargs["Item"]["database_s3_version_id"] == "version-2"
    assert summary_path.is_file()


def test_retry_command_uses_saved_summary(tmp_path, monkeypatch):
    """Publish a saved summary and clean the job's local copies on success."""
    job_id = "123e4567-e89b-12d3-a456-426614174000"
    settings = Settings(data_directory=tmp_path)
    publish = MagicMock()
    job_root = tmp_path / job_id
    summary = job_root / "artifacts/summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({
        "job_id": job_id, "feature_extractor_algorithm": "SIFT",
    }))
    database = job_root / "colmap/database.db"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"database")
    monkeypatch.setattr("app.queue.retry_publish.get_settings", lambda: settings)
    monkeypatch.setattr("app.queue.retry_publish.publish_artifacts", publish)

    main([job_id])

    publish.assert_called_once_with(summary)
    assert not summary.parent.exists()
    assert not database.exists()


def test_retry_command_keeps_files_when_publication_fails(tmp_path, monkeypatch):
    """Leave the saved summary and database available after retry failure."""
    job_id = "123e4567-e89b-12d3-a456-426614174000"
    summary = tmp_path / job_id / "artifacts/summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({
        "job_id": job_id, "feature_extractor_algorithm": "SIFT",
    }))
    settings = Settings(data_directory=tmp_path)
    monkeypatch.setattr("app.queue.retry_publish.get_settings", lambda: settings)
    monkeypatch.setattr(
        "app.queue.retry_publish.publish_artifacts",
        MagicMock(side_effect=RuntimeError("publication failed")),
    )

    with pytest.raises(RuntimeError, match="publication failed"):
        main([job_id])

    assert summary.is_file()


def test_retry_command_rejects_invalid_job_id(monkeypatch):
    """Reject path traversal before resolving the saved summary."""
    publish = MagicMock()
    monkeypatch.setattr("app.queue.retry_publish.publish_artifacts", publish)

    with pytest.raises(SystemExit) as exit_info:
        main(["../another-job"])

    assert exit_info.value.code == 2
    publish.assert_not_called()
