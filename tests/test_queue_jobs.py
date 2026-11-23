"""Tests for reconstruction queue jobs."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.queue.jobs import (
    build_sparse_paths,
    enqueue_reconstruction,
    process_dense_job,
    run_dense_reconstruction,
    run_sparse_reconstruction,
)
from app.settings import Settings

DEFAULT_SETTINGS = Settings(data_directory=Path("data"), job_timeout=3300)


@pytest.fixture(autouse=True)
def configure_job_settings(monkeypatch):
    """Use explicit shared Pydantic settings and mock the S3 video download."""
    monkeypatch.setattr("app.queue.jobs.get_settings", lambda: DEFAULT_SETTINGS)

    def download_video(key, destination):
        assert key == f"{destination.parent.name}/video.mp4"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fake-video")

    monkeypatch.setattr("app.queue.jobs.get_object", download_video)


def test_build_sparse_paths():
    video_path, output_path = build_sparse_paths("abc-123")

    assert video_path == "data/abc-123/video.mp4"
    assert output_path == "data/abc-123/frames"


def test_run_sparse_reconstruction_missing_video(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    job_id = "missing-job"
    monkeypatch.setattr("app.queue.jobs.get_object", lambda key, destination: None)

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

    with patch("app.queue.jobs.time.monotonic", side_effect=[10.0, 12.5]), patch(
        "app.queue.jobs.subprocess.run"
    ) as run_mock, patch(
        "app.queue.jobs.prepare_artifacts"
    ) as prepare_mock, patch(
        "app.queue.jobs.publish_artifacts"
    ) as publish_mock:
        run_mock.return_value = MagicMock(returncode=0)
        prepare_mock.return_value = Path("data/job-dense/artifacts/summary.json")
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
    assert not (tmp_path / video_dir / "video.mp4").exists()
    artifact_args = prepare_mock.call_args.args
    assert artifact_args[:4] == (video_dir, job_id, algorithm, 8)
    assert datetime.fromisoformat(artifact_args[4]).tzinfo == timezone.utc
    assert artifact_args[5] == 2.5
    publish_mock.assert_called_once_with(prepare_mock.return_value)


def test_sparse_job_prepares_artifacts_after_pipeline(tmp_path, monkeypatch):
    """Prepare sparse artifacts only after the pipeline returns successfully."""
    monkeypatch.chdir(tmp_path)
    with patch("app.queue.jobs.time.monotonic", side_effect=[2.0, 5.0]), patch(
        "app.queue.jobs.run_sparse_reconstruction"
    ) as run_mock, patch(
        "app.queue.jobs.prepare_artifacts"
    ) as prepare_mock, patch(
        "app.queue.jobs.publish_artifacts"
    ) as publish_mock:
        run_mock.return_value = {"job_id": "job-sparse"}
        prepare_mock.return_value = Path("data/job-sparse/artifacts/summary.json")
        result = process_dense_job("job-sparse", 5, "ORB")

    assert result == {"job_id": "job-sparse"}
    artifact_args = prepare_mock.call_args.args
    assert artifact_args[:4] == (
        Path("data/job-sparse"), "job-sparse", "ORB", 5,
    )
    assert datetime.fromisoformat(artifact_args[4]).tzinfo == timezone.utc
    assert artifact_args[5] == 3.0
    publish_mock.assert_called_once_with(prepare_mock.return_value)


@pytest.mark.parametrize("worker", [process_dense_job, run_dense_reconstruction])
@pytest.mark.parametrize("failed_step", ["prepare", "publish"])
def test_job_failure_retains_local_files(
    tmp_path, monkeypatch, worker, failed_step,
):
    """Propagate preparation or publication failures without local cleanup."""
    monkeypatch.chdir(tmp_path)
    job_root = Path("data/job-failed")
    files = [
        job_root / "video.mp4",
        job_root / "colmap/database.db",
        job_root / "frames/frame.jpg",
        job_root / "colmap/reconstructed.glb",
        job_root / "artifacts/database.db",
        job_root / "artifacts/features.tar",
        job_root / "artifacts/summary.json",
    ]
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"retained")
    sparse_result = {
        "job_id": "job-failed",
        "output_path": str(job_root / "frames"),
    }

    with patch("app.queue.jobs.run_sparse_reconstruction") as sparse, patch(
        "app.queue.jobs.subprocess.run"
    ), patch("app.queue.jobs.prepare_artifacts") as prepare, patch(
        "app.queue.jobs.publish_artifacts"
    ) as publish:
        sparse.return_value = sparse_result
        prepare.return_value = job_root / "artifacts/summary.json"
        if failed_step == "prepare":
            prepare.side_effect = RuntimeError("preparation failed")
        else:
            publish.side_effect = RuntimeError("publication failed")

        with pytest.raises(RuntimeError, match="failed"):
            worker("job-failed", 5, "ORB")

    assert all(path.is_file() for path in files)
    if failed_step == "prepare":
        publish.assert_not_called()
    else:
        publish.assert_called_once_with(job_root / "artifacts/summary.json")


@pytest.mark.parametrize("worker", [process_dense_job, run_dense_reconstruction])
@pytest.mark.parametrize("algorithm", ["SIFT", "ORB", "AKAZE"])
def test_success_removes_published_files_and_keeps_outputs(
    tmp_path, monkeypatch, worker, algorithm,
):
    """Clean local database, feature files, and artifacts after publication."""
    monkeypatch.chdir(tmp_path)
    job_root = Path("data/job-complete")
    video = job_root / "video.mp4"
    database = job_root / "colmap/database.db"
    model = job_root / "colmap/reconstructed.glb"
    frame = job_root / "frames/frame.jpg"
    summary = job_root / "artifacts/summary.json"
    snapshot = job_root / "artifacts/database.db"
    archive = job_root / "artifacts/features.tar"
    feature = job_root / "colmap/features" / algorithm.lower() / "frame.npz"
    for path in (video, database, model, frame, summary, snapshot, archive):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"local")
    if algorithm != "SIFT":
        feature.parent.mkdir(parents=True)
        feature.write_bytes(b"npz")

    with patch("app.queue.jobs.run_sparse_reconstruction") as sparse, patch(
        "app.queue.jobs.subprocess.run"
    ), patch("app.queue.jobs.prepare_artifacts", return_value=summary), patch(
        "app.queue.jobs.publish_artifacts"
    ) as publish:
        sparse.return_value = {
            "job_id": "job-complete", "output_path": str(job_root / "frames"),
        }
        worker("job-complete", 5, algorithm)

    publish.assert_called_once_with(summary)
    assert not video.exists()
    assert not database.exists()
    assert not (job_root / "artifacts").exists()
    assert not feature.exists()
    assert model.is_file()
    assert frame.is_file()


def test_enqueue_reconstruction_sparse_uses_queue():
    """Enqueue a SIFT sparse job on the supplied queue."""
    fake_queue = MagicMock()
    fake_job = MagicMock(name="job")
    fake_queue.enqueue.return_value = fake_job

    job = enqueue_reconstruction(
        "uuid-1",
        strides=10,
        reconstruction="sparse",
        feature_extractor_algorithm="SIFT",
        queue=fake_queue,
    )

    assert job is fake_job
    fake_queue.enqueue.assert_called_once_with(
        process_dense_job,
        "uuid-1",
        10,
        feature_extractor_algorithm="SIFT",
        job_timeout=DEFAULT_SETTINGS.job_timeout,
    )


def test_enqueue_reconstruction_dense_uses_queue():
    """Enqueue a SIFT dense job on the supplied queue."""
    fake_queue = MagicMock()
    fake_job = MagicMock(name="job")
    fake_queue.enqueue.return_value = fake_job

    job = enqueue_reconstruction(
        "uuid-dense",
        strides=5,
        reconstruction="dense",
        feature_extractor_algorithm="SIFT",
        queue=fake_queue,
    )

    assert job is fake_job
    fake_queue.enqueue.assert_called_once_with(
        run_dense_reconstruction,
        "uuid-dense",
        5,
        feature_extractor_algorithm="SIFT",
        job_timeout=DEFAULT_SETTINGS.job_timeout,
    )


def test_enqueue_reconstruction_rejects_invalid_mode():
    """Reject an unsupported mode with a valid feature algorithm."""
    with pytest.raises(ValueError, match="Unsupported reconstruction"):
        enqueue_reconstruction(
            "uuid-x", strides=5, reconstruction="mesh",
            feature_extractor_algorithm="SIFT",
        )


def test_enqueue_reconstruction_creates_queue_when_omitted():
    """Create a queue when enqueuing a SIFT job without one."""
    fake_queue = MagicMock()
    fake_job = MagicMock(name="job")
    fake_queue.enqueue.return_value = fake_job

    with patch("app.queue.jobs.get_queue", return_value=fake_queue) as get_queue:
        job = enqueue_reconstruction(
            "uuid-2", strides=5, reconstruction="sparse",
            feature_extractor_algorithm="SIFT",
        )

    get_queue.assert_called_once_with()
    fake_queue.enqueue.assert_called_once_with(
        process_dense_job,
        "uuid-2",
        5,
        feature_extractor_algorithm="SIFT",
        job_timeout=DEFAULT_SETTINGS.job_timeout,
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
    assert queue.enqueue.call_args.kwargs["job_timeout"] == DEFAULT_SETTINGS.job_timeout


def test_enqueue_rejects_invalid_algorithm():
    """Reject invalid feature algorithms before creating a queued job."""
    queue = MagicMock()
    with pytest.raises(ValueError, match="Unsupported feature algorithm"):
        enqueue_reconstruction(
            "job", 5, "sparse", queue=queue, feature_extractor_algorithm="SURF"
        )
    queue.enqueue.assert_not_called()
