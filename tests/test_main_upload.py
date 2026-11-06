"""Tests for the video upload endpoint queue integration."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


@pytest.fixture
def put_object_mock(monkeypatch):
    """Mock the S3 upload so endpoint tests do not need AWS credentials."""
    mock = MagicMock()
    monkeypatch.setattr("app.main.put_object", mock)

    async def run_inline(function, *args):
        """Run the mocked blocking call inline for deterministic endpoint tests."""
        return function(*args)

    monkeypatch.setattr("app.main.run_in_threadpool", run_inline)
    return mock


def test_video_upload_enqueues_sparse_job(tmp_path, monkeypatch, put_object_mock):
    """Upload the video to S3 and enqueue sparse reconstruction with SIFT."""
    monkeypatch.chdir(tmp_path)
    Path("data").mkdir()
    Path("static").mkdir()
    (Path("static") / "index.html").write_text("<html></html>")

    fake_job = MagicMock()
    with patch("app.main.enqueue_reconstruction", return_value=fake_job) as enqueue:
        response = client.post(
            "/video-upload",
            files={"file": ("clip.mp4", b"video-bytes", "video/mp4")},
            data={"strides": "10", "reconstruction": "sparse"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["message"] == "Video uploaded successfully"
    assert "job_id" in payload

    job_id = payload["job_id"]
    put_object_mock.assert_called_once_with(
        f"{job_id}/video.mp4", b"video-bytes",
    )
    enqueue.assert_called_once_with(job_id, 10, "sparse", "SIFT")


@pytest.mark.parametrize("algorithm", ["SIFT", "ORB", "AKAZE"])
def test_video_upload_enqueues_dense_job(
    tmp_path, monkeypatch, algorithm, put_object_mock,
):
    """Upload to S3 and enqueue dense reconstruction with the selected algorithm."""
    monkeypatch.chdir(tmp_path)
    Path("data").mkdir()
    Path("static").mkdir()
    (Path("static") / "index.html").write_text("<html></html>")

    fake_job = MagicMock()
    with patch("app.main.enqueue_reconstruction", return_value=fake_job) as enqueue:
        response = client.post(
            "/video-upload",
            files={"file": ("clip.mp4", b"video-bytes", "video/mp4")},
            data={
                "strides": "5", "reconstruction": "dense",
                "feature_extractor_algorithm": algorithm,
            },
        )

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    put_object_mock.assert_called_once_with(
        f"{job_id}/video.mp4", b"video-bytes",
    )
    enqueue.assert_called_once_with(
        job_id, 5, "dense", algorithm,
    )


def test_video_upload_rejects_invalid_reconstruction(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("data").mkdir()
    Path("static").mkdir()
    (Path("static") / "index.html").write_text("<html></html>")

    with patch("app.main.enqueue_reconstruction") as enqueue:
        response = client.post(
            "/video-upload",
            files={"file": ("clip.mp4", b"video-bytes", "video/mp4")},
            data={"strides": "5", "reconstruction": "mesh"},
        )

    assert response.status_code == 422
    enqueue.assert_not_called()


def test_video_upload_requires_form_fields(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("data").mkdir()
    Path("static").mkdir()
    (Path("static") / "index.html").write_text("<html></html>")

    with patch("app.main.enqueue_reconstruction") as enqueue:
        response = client.post(
            "/video-upload",
            files={"file": ("clip.mp4", b"video-bytes", "video/mp4")},
        )

    assert response.status_code == 422
    enqueue.assert_not_called()


@pytest.mark.parametrize("algorithm", ["SURF", "orb"])
def test_video_upload_rejects_invalid_algorithm(tmp_path, monkeypatch, algorithm):
    """Reject unsupported algorithm values before saving or enqueuing uploads."""
    monkeypatch.chdir(tmp_path)
    with patch("app.main.enqueue_reconstruction") as enqueue:
        response = client.post(
            "/video-upload",
            files={"file": ("clip.mp4", b"video", "video/mp4")},
            data={
                "strides": "5", "reconstruction": "sparse",
                "feature_extractor_algorithm": algorithm,
            },
        )
    assert response.status_code == 422
    enqueue.assert_not_called()
    assert not Path("data").exists()
