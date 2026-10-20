"""Tests for the video upload endpoint queue integration."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_video_upload_enqueues_sparse_job(tmp_path, monkeypatch):
    """Save the uploaded video and enqueue sparse reconstruction with SIFT."""
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
    video_path = Path("data") / job_id / "video.mp4"
    assert video_path.is_file()
    assert video_path.read_bytes() == b"video-bytes"
    enqueue.assert_called_once_with(job_id, 10, "sparse", "SIFT")


@pytest.mark.parametrize("algorithm", ["SIFT", "ORB", "AKAZE"])
def test_video_upload_enqueues_dense_job(tmp_path, monkeypatch, algorithm):
    """Enqueue dense reconstruction with the selected feature algorithm."""
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
    enqueue.assert_called_once_with(
        response.json()["job_id"], 5, "dense", algorithm,
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
