"""Tests for DynamoDB metadata and local model availability."""

from decimal import Decimal
from unittest.mock import MagicMock, call

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError
from fastapi import HTTPException

from app.reconstruction_catalog import (
    METADATA_FIELDS,
    get_reconstruction_catalog,
    read_published_jobs,
)
from app.settings import Settings

JOB_ID = "11111111-1111-1111-1111-111111111111"


@pytest.fixture
def dynamodb(monkeypatch):
    """Provide a configured jobs table without making any AWS requests."""
    settings = Settings(dynamodb_table_name="jobs")
    session = MagicMock()
    resource = session.resource.return_value
    monkeypatch.setattr(
        "app.reconstruction_catalog.get_settings", lambda: settings,
    )
    monkeypatch.setattr(
        "app.reconstruction_catalog.get_aws_session",
        lambda configured: session,
    )
    return resource


def test_scan_reads_all_pages_and_closes_client(dynamodb):
    """Follow continuation keys even when an intermediate page is empty."""
    table = dynamodb.Table.return_value
    table.scan.side_effect = [
        {
            "Items": [{"job_id": JOB_ID}],
            "LastEvaluatedKey": {"job_id": "first"},
        },
        {"Items": [], "LastEvaluatedKey": {"job_id": "second"}},
        {"Items": [{"job_id": "last"}]},
    ]
    assert read_published_jobs() == [{"job_id": JOB_ID}, {"job_id": "last"}]
    assert table.scan.call_args_list == [
        call(),
        call(ExclusiveStartKey={"job_id": "first"}),
        call(ExclusiveStartKey={"job_id": "second"}),
    ]
    dynamodb.Table.assert_called_once_with("jobs")
    dynamodb.meta.client.close.assert_called_once_with()


@pytest.mark.parametrize("error", [
    ClientError({"Error": {"Code": "AccessDeniedException"}}, "Scan"),
    EndpointConnectionError(endpoint_url="https://dynamodb.invalid"),
])
def test_scan_failure_is_unavailable_and_closes_client(dynamodb, error):
    """Surface service failures while releasing the resource client."""
    dynamodb.Table.return_value.scan.side_effect = error
    with pytest.raises(HTTPException) as raised:
        read_published_jobs()
    assert raised.value.status_code == 503
    assert "DynamoDB" in raised.value.detail
    dynamodb.meta.client.close.assert_called_once_with()


def test_missing_table_does_not_create_aws_resource(monkeypatch):
    """Report a missing table configuration before accessing AWS."""
    session = MagicMock()
    monkeypatch.setattr(
        "app.reconstruction_catalog.get_settings",
        lambda: Settings(dynamodb_table_name=""),
    )
    monkeypatch.setattr("app.reconstruction_catalog.get_aws_session", session)
    with pytest.raises(HTTPException, match="DYNAMODB_TABLE_NAME"):
        read_published_jobs()
    session.assert_not_called()


def test_invalid_aws_configuration_is_unavailable(monkeypatch):
    """Convert credential/profile setup failures into a catalog 503."""
    monkeypatch.setattr(
        "app.reconstruction_catalog.get_settings",
        lambda: Settings(dynamodb_table_name="jobs"),
    )
    monkeypatch.setattr(
        "app.reconstruction_catalog.get_aws_session",
        MagicMock(side_effect=ValueError("Missing credentials")),
    )
    with pytest.raises(HTTPException) as raised:
        read_published_jobs()
    assert raised.value.status_code == 503


@pytest.mark.parametrize("sparse,dense", [
    (False, False), (True, False), (False, True), (True, True),
])
def test_model_urls_only_exist_for_available_files(
    tmp_path, monkeypatch, sparse, dense,
):
    """Keep every published job and advertise only GLBs present on disk."""
    monkeypatch.setattr(
        "app.reconstruction_catalog.read_published_jobs",
        lambda: [{"job_id": JOB_ID, "reconstruction": "dense"}],
    )
    colmap = tmp_path / JOB_ID / "colmap"
    colmap.mkdir(parents=True)
    if sparse:
        (colmap / "reconstructed.glb").write_bytes(b"sparse")
    if dense:
        (colmap / "fused.glb").write_bytes(b"dense")
    job, = get_reconstruction_catalog(tmp_path)["jobs"]
    assert job["sparse_glb_url"] == (
        f"/api/reconstructions/{JOB_ID}/glb" if sparse else None
    )
    assert job["dense_glb_url"] == (
        f"/api/reconstructions/{JOB_ID}/fused" if dense else None
    )
    assert set(job) == set(METADATA_FIELDS) | {
        "sparse_glb_url", "dense_glb_url",
    }
    assert "reconstruction" not in job
    assert "database_snapshot" not in job
    assert "summary_path" not in job
    assert "feature_archive" not in job


def test_metadata_is_json_ready_and_sorted_by_actual_completion(
    tmp_path, monkeypatch,
):
    """Omit local paths and preserve numbers, sorting dates across offsets."""
    records = [
        {"job_id": "missing-time"},
        {
            "job_id": JOB_ID,
            "completed_at": "2026-11-19T12:00:00+02:00",
            "image_count": Decimal("0"),
            "reconstruction_duration_seconds": Decimal("1.25"),
            "median_triangulation_angle_degrees": Decimal("0"),
            "database_snapshot": "artifacts/database.db",
            "summary_path": "artifacts/summary.json",
            "feature_archive": "artifacts/features.tar",
            "database_s3_key": "not-exposed",
        },
        {"job_id": "newest", "completed_at": "2026-11-19T10:30:00Z"},
        {"job_id": "invalid-time", "completed_at": "invalid"},
        {"job_id": "naive-time", "completed_at": "2026-11-19T09:00:00"},
    ]
    monkeypatch.setattr(
        "app.reconstruction_catalog.read_published_jobs", lambda: records,
    )
    jobs = get_reconstruction_catalog(tmp_path)["jobs"]
    assert [job["job_id"] for job in jobs] == [
        "newest", JOB_ID, "naive-time", "missing-time", "invalid-time",
    ]
    job = jobs[1]
    assert type(job["image_count"]) is int
    assert job["image_count"] == 0
    assert type(job["reconstruction_duration_seconds"]) is float
    assert job["reconstruction_duration_seconds"] == 1.25
    assert job["median_triangulation_angle_degrees"] == 0
    assert "database_snapshot" not in job
    assert "summary_path" not in job
    assert "feature_archive" not in job
    assert "database_s3_key" not in job


@pytest.mark.parametrize("job_id", [None, "../outside", "not-a-uuid"])
def test_legacy_invalid_identifiers_have_no_model_urls(
    tmp_path, monkeypatch, job_id,
):
    """Never resolve paths outside the UUID job directories."""
    monkeypatch.setattr(
        "app.reconstruction_catalog.read_published_jobs",
        lambda: [{"job_id": job_id}],
    )
    job, = get_reconstruction_catalog(tmp_path)["jobs"]
    assert job["job_id"] == job_id
    assert job["sparse_glb_url"] is None
    assert job["dense_glb_url"] is None


def test_catalog_keeps_local_models_without_published_metadata(tmp_path, monkeypatch):
    """Keep the legacy model lists when DynamoDB has no matching job row."""
    monkeypatch.setattr(
        "app.reconstruction_catalog.read_published_jobs", lambda: [],
    )
    colmap = tmp_path / JOB_ID / "colmap"
    colmap.mkdir(parents=True)
    (colmap / "reconstructed.glb").write_bytes(b"sparse")
    (colmap / "fused.glb").write_bytes(b"dense")

    catalog = get_reconstruction_catalog(tmp_path)

    assert catalog["jobs"] == []
    assert catalog["reconstructions"][0]["glb_url"] == (
        f"/api/reconstructions/{JOB_ID}/glb"
    )
    assert catalog["fused"][0]["glb_url"] == (
        f"/api/reconstructions/{JOB_ID}/fused"
    )
