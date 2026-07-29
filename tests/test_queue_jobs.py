"""Tests for reconstruction queue jobs."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.queue.jobs import (
    build_sparse_paths,
    enqueue_sparse_reconstruction,
    run_sparse_reconstruction,
)
from app.queue.settings import DEFAULT_STRIDES


def test_build_sparse_paths():
    video_path, output_path = build_sparse_paths("abc-123")

    assert video_path == "data/abc-123/video.mp4"
    assert output_path == "data/abc-123/frames"


def test_run_sparse_reconstruction_missing_video(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    job_id = "missing-job"

    with pytest.raises(FileNotFoundError, match="Video not found"):
        run_sparse_reconstruction(job_id)


def test_run_sparse_reconstruction_invokes_cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    job_id = "job-1"
    video_dir = Path("data") / job_id
    video_dir.mkdir(parents=True)
    (video_dir / "video.mp4").write_bytes(b"fake-video")

    with patch("app.queue.jobs.subprocess.run") as run_mock:
        run_mock.return_value = MagicMock(returncode=0)
        result = run_sparse_reconstruction(job_id, strides=15)

    assert result == {
        "job_id": job_id,
        "video_path": "data/job-1/video.mp4",
        "output_path": "data/job-1/frames",
    }
    args = run_mock.call_args.args[0]
    assert args[1:3] == ["-m", "app.reconstruction"]
    assert "colmap-sparse" in args
    assert "--video_path" in args
    assert "data/job-1/video.mp4" in args
    assert "--output_path" in args
    assert "data/job-1/frames" in args
    assert "--strides" in args
    assert "15" in args
    assert run_mock.call_args.kwargs["check"] is True


def test_enqueue_sparse_reconstruction_uses_queue():
    fake_queue = MagicMock()
    fake_job = MagicMock(name="job")
    fake_queue.enqueue.return_value = fake_job

    job = enqueue_sparse_reconstruction("uuid-1", strides=10, queue=fake_queue)

    assert job is fake_job
    fake_queue.enqueue.assert_called_once_with(
        run_sparse_reconstruction,
        "uuid-1",
        10,
    )


def test_enqueue_sparse_reconstruction_creates_queue_when_omitted():
    fake_queue = MagicMock()
    fake_job = MagicMock(name="job")
    fake_queue.enqueue.return_value = fake_job

    with patch("app.queue.connection.get_queue", return_value=fake_queue) as get_queue:
        job = enqueue_sparse_reconstruction("uuid-2")

    get_queue.assert_called_once_with()
    fake_queue.enqueue.assert_called_once_with(
        run_sparse_reconstruction,
        "uuid-2",
        DEFAULT_STRIDES,
    )
    assert job is fake_job
