"""Tests for uploading prepared local artifacts to S3."""

from unittest.mock import MagicMock

from app.s3 import stream_file_to_s3
from app.settings import Settings


def test_stream_file_to_s3_streams_local_artifact(tmp_path, monkeypatch):
    """Send the file body, key, and content type to the configured bucket."""
    source = tmp_path / "database.db"
    source.write_bytes(b"sqlite-snapshot")
    settings = Settings(s3_bucket="bucket")
    client = MagicMock()
    client.put_object.side_effect = lambda **kwargs: {
        "VersionId": "version-1",
        "body_contents": kwargs["Body"].read(),
    }
    monkeypatch.setattr("app.s3.get_settings", lambda: settings)
    monkeypatch.setattr("app.s3._get_s3_client", lambda configured: client)

    response = stream_file_to_s3(
        "job-1/database.db", source, "application/vnd.sqlite3",
    )

    assert response["VersionId"] == "version-1"
    assert response["body_contents"] == b"sqlite-snapshot"
    assert client.put_object.call_args.kwargs["Bucket"] == "bucket"
    assert client.put_object.call_args.kwargs["Key"] == (
        "job-1/database.db"
    )
    assert client.put_object.call_args.kwargs["ContentType"] == (
        "application/vnd.sqlite3"
    )
    assert client.put_object.call_args.kwargs["Body"].closed
