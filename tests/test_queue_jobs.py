"""Tests for reconstruction queue jobs."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.queue.jobs import (
    build_sparse_paths,
    enqueue_reconstruction,
    run_dense_reconstruction,
    run_sparse_reconstruction,
)
from app.queue.settings import JOB_TIMEOUT


def test_build_sparse_paths():
    video_path, output_path = build_sparse_paths("abc-123")

    assert video_path == "data/abc-123/video.mp4"
    assert output_path == "data/abc-123/frames"


def test_run_sparse_reconstruction_missing_video(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    job_id = "missing-job"

    with pytest.raises(FileNotFoundError, match="Video not found"):
        run_sparse_reconstruction(job_id, strides=5)


@pytest.mark.parametrize("algorithm", ["SIFT", "ORB", "AKAZE"])
def test_run_sparse_reconstruction_invokes_cli(tmp_path, monkeypatch, algorithm):
    monkeypatch.chdir(tmp_path)
    job_id = "job-1"
    video_dir = Path("data") / job_id
    video_dir.mkdir(parents=True)
    (video_dir / "video.mp4").write_bytes(b"fake-video")

    with patch("app.queue.jobs.subprocess.run") as run_mock:
        run_mock.return_value = MagicMock(returncode=0)
        result = run_sparse_reconstruction(
            job_id, strides=15, feature_extractor_algorithm=algorithm
        )

    assert result == {
        "job_id": job_id,
        "video_path": "data/job-1/video.mp4",
        "output_path": "data/job-1/frames",
        "reconstruction": "sparse",
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
    assert args[args.index("--feature_extractor_algorithm") + 1] == algorithm
    assert run_mock.call_args.kwargs["check"] is True


@pytest.mark.parametrize("algorithm", ["SIFT", "ORB", "AKAZE"])
def test_run_dense_reconstruction_runs_sparse_then_dense(
    tmp_path, monkeypatch, algorithm
):
    monkeypatch.chdir(tmp_path)
    job_id = "job-dense"
    video_dir = Path("data") / job_id
    video_dir.mkdir(parents=True)
    (video_dir / "video.mp4").write_bytes(b"fake-video")

    with patch("app.queue.jobs.subprocess.run") as run_mock:
        run_mock.return_value = MagicMock(returncode=0)
        result = run_dense_reconstruction(
            job_id, strides=8, feature_extractor_algorithm=algorithm
        )

    assert result == {
        "job_id": job_id,
        "video_path": "data/job-dense/video.mp4",
        "output_path": "data/job-dense/frames",
        "reconstruction": "dense",
    }
    assert run_mock.call_count == 2
    sparse_args = run_mock.call_args_list[0].args[0]
    dense_args = run_mock.call_args_list[1].args[0]
    assert "colmap-sparse" in sparse_args
    assert "8" in sparse_args
    assert sparse_args[
        sparse_args.index("--feature_extractor_algorithm") + 1
    ] == algorithm
    assert "colmap-dense" in dense_args
    assert "--images_path" in dense_args
    assert "data/job-dense/frames" in dense_args


def test_enqueue_reconstruction_sparse_uses_queue():
    fake_queue = MagicMock()
    fake_job = MagicMock(name="job")
    fake_queue.enqueue.return_value = fake_job

    job = enqueue_reconstruction(
        "uuid-1",
        strides=10,
        reconstruction="sparse",
        queue=fake_queue,
    )

    assert job is fake_job
    fake_queue.enqueue.assert_called_once_with(
        run_sparse_reconstruction,
        "uuid-1",
        10,
        feature_extractor_algorithm="SIFT",
        job_timeout=JOB_TIMEOUT,
    )


def test_enqueue_reconstruction_dense_uses_queue():
    fake_queue = MagicMock()
    fake_job = MagicMock(name="job")
    fake_queue.enqueue.return_value = fake_job

    job = enqueue_reconstruction(
        "uuid-dense",
        strides=5,
        reconstruction="dense",
        queue=fake_queue,
    )

    assert job is fake_job
    fake_queue.enqueue.assert_called_once_with(
        run_dense_reconstruction,
        "uuid-dense",
        5,
        feature_extractor_algorithm="SIFT",
        job_timeout=JOB_TIMEOUT,
    )


def test_enqueue_reconstruction_rejects_invalid_mode():
    with pytest.raises(ValueError, match="Unsupported reconstruction"):
        enqueue_reconstruction("uuid-x", strides=5, reconstruction="mesh")


def test_enqueue_reconstruction_creates_queue_when_omitted():
    fake_queue = MagicMock()
    fake_job = MagicMock(name="job")
    fake_queue.enqueue.return_value = fake_job

    with patch("app.queue.connection.get_queue", return_value=fake_queue) as get_queue:
        job = enqueue_reconstruction("uuid-2", strides=5, reconstruction="sparse")

    get_queue.assert_called_once_with()
    fake_queue.enqueue.assert_called_once_with(
        run_sparse_reconstruction,
        "uuid-2",
        5,
        feature_extractor_algorithm="SIFT",
        job_timeout=JOB_TIMEOUT,
    )
    assert job is fake_job


@pytest.mark.parametrize("algorithm", ["SIFT", "ORB", "AKAZE"])
@pytest.mark.parametrize("mode", ["sparse", "dense"])
def test_enqueue_passes_algorithm_to_worker(algorithm, mode):
    """Preserve the selected algorithm in the serialized worker arguments."""
    queue = MagicMock()
    enqueue_reconstruction(
        "job", 5, mode, queue=queue, feature_extractor_algorithm=algorithm
    )
    assert queue.enqueue.call_args.kwargs["feature_extractor_algorithm"] == algorithm
    assert queue.enqueue.call_args.kwargs["job_timeout"] == JOB_TIMEOUT


def test_enqueue_rejects_invalid_algorithm():
    """Reject invalid feature algorithms before creating a queued job."""
    queue = MagicMock()
    with pytest.raises(ValueError, match="Unsupported feature algorithm"):
        enqueue_reconstruction(
            "job", 5, "sparse", queue=queue, feature_extractor_algorithm="SURF"
        )
    queue.enqueue.assert_not_called()
