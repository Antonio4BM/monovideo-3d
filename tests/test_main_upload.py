"""Tests for the video upload endpoint queue integration."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_video_upload_enqueues_sparse_job(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("data").mkdir()
    Path("static").mkdir()
    (Path("static") / "index.html").write_text("<html></html>")

    fake_job = MagicMock()
    with patch("app.main.enqueue_sparse_reconstruction", return_value=fake_job) as enqueue:
        response = client.post(
            "/video-upload",
            files={"file": ("clip.mp4", b"video-bytes", "video/mp4")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["message"] == "Video uploaded successfully"
    assert "job_id" in payload

    job_id = payload["job_id"]
    video_path = Path("data") / job_id / "video.mp4"
    assert video_path.is_file()
    assert video_path.read_bytes() == b"video-bytes"
    enqueue.assert_called_once_with(job_id)
