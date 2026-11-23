"""Tests for locally staged reconstruction artifacts."""

import json
import logging
import sqlite3
import tarfile

import numpy as np
import pytest

from app.queue.artifacts import cleanup_published_artifacts, prepare_artifacts


@pytest.mark.parametrize("algorithm", ["SIFT", "ORB", "AKAZE"])
def test_prepare_artifacts_snapshots_database_and_archives_features(
    tmp_path, algorithm,
):
    """Back up WAL data and retain the expected feature directory in the tar."""
    job_root = tmp_path / "job-1"
    colmap = job_root / "colmap"
    colmap.mkdir(parents=True)
    source = sqlite3.connect(colmap / "database.db")
    source.execute("PRAGMA journal_mode=WAL")
    source.execute("CREATE TABLE samples (value TEXT)")
    source.execute("INSERT INTO samples VALUES ('complete')")
    source.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY)")
    source.executemany("INSERT INTO images VALUES (?)", [(1,), (2,)])
    source.execute("CREATE TABLE keypoints (rows INTEGER)")
    source.executemany("INSERT INTO keypoints VALUES (?)", [(3,), (4,)])
    source.execute("CREATE TABLE matches (rows INTEGER)")
    source.executemany("INSERT INTO matches VALUES (?)", [(5,), (6,)])
    source.execute("CREATE TABLE descriptors (rows INTEGER)")
    descriptor_rows = [(3,), (4,)] if algorithm == "SIFT" else [(99,)]
    source.executemany("INSERT INTO descriptors VALUES (?)", descriptor_rows)
    source.execute("CREATE TABLE metadata (key TEXT, value REAL)")
    source.execute(
        "INSERT INTO metadata VALUES (?, ?)",
        ("median_triangulation_angle_degrees", 12.5),
    )
    source.commit()

    if algorithm != "SIFT":
        feature_dir = colmap / "features" / algorithm.lower()
        feature_dir.mkdir(parents=True)
        np.savez(
            feature_dir / "frame.jpg.npz",
            descriptors=np.zeros((3, 32), dtype=np.uint8),
        )
        np.savez(
            feature_dir / "next.jpg.npz",
            descriptors=np.zeros((4, 32), dtype=np.uint8),
        )

    try:
        summary_path = prepare_artifacts(
            job_root, "job-1", algorithm, 10,
            "2026-11-12T12:00:00+00:00", 4.25,
        )
        summary = json.loads(summary_path.read_text())
        with sqlite3.connect(job_root / summary["database_snapshot"]) as backup:
            assert backup.execute("SELECT value FROM samples").fetchone() == (
                "complete",
            )
        assert summary["job_id"] == "job-1"
        assert "reconstruction" not in summary
        assert summary["feature_extractor_algorithm"] == algorithm
        assert summary["strides"] == 10
        assert summary["completed_at"] == "2026-11-12T12:00:00+00:00"
        assert summary["reconstruction_duration_seconds"] == 4.25
        assert summary["summary_path"] == "artifacts/summary.json"
        assert summary["image_count"] == 2
        assert summary["total_keypoints"] == 7
        assert summary["total_matches"] == 11
        assert summary["total_descriptors"] == 7
        assert summary["median_triangulation_angle_degrees"] == 12.5

        if algorithm == "SIFT":
            assert summary["feature_archive"] is None
            assert not (job_root / "artifacts/features.tar").exists()
        else:
            archive = job_root / summary["feature_archive"]
            with tarfile.open(archive, "r") as tar:
                assert "colmap/features/" + algorithm.lower() in tar.getnames()
                assert (
                    f"colmap/features/{algorithm.lower()}/frame.jpg.npz"
                    in tar.getnames()
                )
                assert (
                    f"colmap/features/{algorithm.lower()}/next.jpg.npz"
                    in tar.getnames()
                )
    finally:
        source.close()


def test_prepare_artifacts_requires_database(tmp_path):
    """Do not publish a summary if the source database is absent."""
    with pytest.raises(FileNotFoundError, match="COLMAP database not found"):
        prepare_artifacts(tmp_path, "job", "SIFT", 5, "now", 1.0)
    assert not (tmp_path / "artifacts/summary.json").exists()


def test_prepare_artifacts_requires_binary_feature_files(tmp_path):
    """Reject an ORB job without feature files before saving artifacts."""
    colmap = tmp_path / "colmap"
    colmap.mkdir()
    with sqlite3.connect(colmap / "database.db"):
        pass

    with pytest.raises(FileNotFoundError, match="Feature archives not found"):
        prepare_artifacts(tmp_path, "job", "ORB", 5, "now", 1.0)
    assert not (tmp_path / "artifacts").exists()


def test_prepare_artifacts_handles_missing_angle_and_empty_counts(tmp_path):
    """Write null angle and zero totals for empty SQLite metric tables."""
    colmap = tmp_path / "colmap"
    colmap.mkdir()
    with sqlite3.connect(colmap / "database.db") as database:
        database.execute("CREATE TABLE images (image_id INTEGER)")
        database.execute("CREATE TABLE keypoints (rows INTEGER)")
        database.execute("CREATE TABLE matches (rows INTEGER)")
        database.execute("CREATE TABLE descriptors (rows INTEGER)")

    summary_path = prepare_artifacts(
        tmp_path, "job", "SIFT", 5, "now", 1.0,
    )
    summary = json.loads(summary_path.read_text())
    assert summary["median_triangulation_angle_degrees"] is None
    assert summary["image_count"] == 0
    assert summary["total_keypoints"] == 0
    assert summary["total_matches"] == 0
    assert summary["total_descriptors"] == 0


@pytest.mark.parametrize("algorithm", ["SIFT", "ORB"])
def test_preparation_failure_keeps_completed_local_artifacts(tmp_path, algorithm):
    """Retain the snapshot and any archive if summary creation fails."""
    colmap = tmp_path / "colmap"
    colmap.mkdir()
    with sqlite3.connect(colmap / "database.db") as database:
        database.execute("CREATE TABLE incomplete (value INTEGER)")
    if algorithm == "ORB":
        features = colmap / "features/orb"
        features.mkdir(parents=True)
        np.savez(
            features / "frame.jpg.npz",
            descriptors=np.zeros((1, 32), dtype=np.uint8),
        )

    with pytest.raises(sqlite3.OperationalError, match="no such table: images"):
        prepare_artifacts(
            tmp_path, "job", algorithm, 5, "now", 1.0,
        )

    assert (colmap / "database.db").is_file()
    assert (tmp_path / "artifacts/database.db").is_file()
    assert (tmp_path / "artifacts/features.tar").is_file() == (
        algorithm == "ORB"
    )
    assert not (tmp_path / "artifacts/summary.json").exists()


def test_cleanup_logs_completed_local_artifact_removal(tmp_path, caplog):
    """Log the local path once successful cleanup finishes."""
    caplog.set_level(logging.INFO, logger="app.queue.artifacts")
    (tmp_path / "video.mp4").write_bytes(b"video")
    (tmp_path / "colmap").mkdir()
    (tmp_path / "colmap/database.db").write_bytes(b"database")
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts/summary.json").write_text("{}")

    cleanup_published_artifacts(tmp_path, "SIFT")

    assert "Finished cleaning up published local artifacts" in caplog.text
    assert not (tmp_path / "artifacts").exists()
