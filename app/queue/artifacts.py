"""Prepare local reconstruction artifacts for later publication."""

import json
import logging
import os
import shutil
import sqlite3
import tarfile
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def prepare_artifacts(
    job_root: Path,
    job_id: str,
    reconstruction: str,
    algorithm: str,
    strides: int,
    completed_at: str,
    reconstruction_duration_seconds: float,
) -> Path:
    """Save a consistent database snapshot, feature archive, and summary.

    The summary is written last, so its presence marks a complete local set.

    Args:
        job_root (Path): Local directory containing the job's ``colmap`` data.
        job_id (str): Reconstruction job identifier.
        reconstruction (str): Completed pipeline, ``sparse`` or ``dense``.
        algorithm (str): Feature algorithm, SIFT, ORB, or AKAZE.
        strides (int): Frame extraction interval used for the job.
        completed_at (str): UTC timestamp captured when reconstruction ended.
        reconstruction_duration_seconds (float): Elapsed time through the
            requested pipeline, measured before artifact preparation.

    Returns:
        Path: Path to the local JSON summary.

    Raises:
        FileNotFoundError: If the database or required feature files are absent.
        OSError: If an artifact cannot be saved.
        sqlite3.Error: If the database cannot be backed up.
    """
    if algorithm not in {"SIFT", "ORB", "AKAZE"}:
        raise ValueError(f"Unsupported feature algorithm: {algorithm}")
    if reconstruction not in {"sparse", "dense"}:
        raise ValueError(f"Unsupported reconstruction: {reconstruction}")

    source_database = job_root / "colmap" / "database.db"
    if not source_database.is_file():
        raise FileNotFoundError(f"COLMAP database not found: {source_database}")

    feature_files = []
    feature_dir = job_root / "colmap" / "features" / algorithm.lower()
    if algorithm != "SIFT":
        feature_files = sorted(feature_dir.glob("*.npz"))
        if not feature_files:
            raise FileNotFoundError(f"Feature archives not found: {feature_dir}")

    artifacts_dir = job_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    snapshot = artifacts_dir / "database.db"
    archive = artifacts_dir / "features.tar"
    summary_path = artifacts_dir / "summary.json"
    summary_path.unlink(missing_ok=True)

    with tempfile.NamedTemporaryFile(dir=artifacts_dir, delete=False) as staged:
        staged_path = Path(staged.name)
    try:
        with sqlite3.connect(
            f"{source_database.resolve().as_uri()}?mode=ro", uri=True,
        ) as source:
            with sqlite3.connect(staged_path) as target:
                source.backup(target)
        os.replace(staged_path, snapshot)
    finally:
        staged_path.unlink(missing_ok=True)

    if feature_files:
        with tempfile.NamedTemporaryFile(dir=artifacts_dir, delete=False) as staged:
            staged_path = Path(staged.name)
        try:
            with tarfile.open(staged_path, "w") as tar:
                archive_root = Path("colmap") / "features" / algorithm.lower()
                tar.add(feature_dir, arcname=archive_root, recursive=False)
                for feature_file in feature_files:
                    tar.add(feature_file, arcname=archive_root / feature_file.name)
            os.replace(staged_path, archive)
        finally:
            staged_path.unlink(missing_ok=True)
    else:
        archive.unlink(missing_ok=True)

    with sqlite3.connect(snapshot) as database:
        image_count = database.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        total_keypoints = database.execute(
            "SELECT COALESCE(SUM(rows), 0) FROM keypoints"
        ).fetchone()[0]
        total_matches = database.execute(
            "SELECT COALESCE(SUM(rows), 0) FROM matches"
        ).fetchone()[0]
        if algorithm == "SIFT":
            total_descriptors = database.execute(
                "SELECT COALESCE(SUM(rows), 0) FROM descriptors"
            ).fetchone()[0]
        else:
            total_descriptors = 0
            for feature_file in feature_files:
                with np.load(feature_file, allow_pickle=False) as features:
                    descriptors = features["descriptors"]
                    if descriptors.ndim != 2:
                        raise ValueError(
                            f"Invalid descriptor array in {feature_file}"
                        )
                    total_descriptors += descriptors.shape[0]
        metadata_exists = database.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'metadata'"
        ).fetchone()
        angle = None
        if metadata_exists:
            row = database.execute(
                "SELECT value FROM metadata WHERE key = ?",
                ("median_triangulation_angle_degrees",),
            ).fetchone()
            if row is not None:
                angle = row[0]

    summary = {
        "job_id": job_id,
        "reconstruction": reconstruction,
        "feature_extractor_algorithm": algorithm,
        "strides": strides,
        "completed_at": completed_at,
        "reconstruction_duration_seconds": reconstruction_duration_seconds,
        "database_snapshot": "artifacts/database.db",
        "feature_archive": "artifacts/features.tar" if feature_files else None,
        "summary_path": "artifacts/summary.json",
        "image_count": image_count,
        "total_keypoints": total_keypoints,
        "total_matches": total_matches,
        "total_descriptors": total_descriptors,
        "median_triangulation_angle_degrees": angle,
    }
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=artifacts_dir, delete=False,
    ) as staged:
        staged_path = Path(staged.name)
    try:
        with staged_path.open("w", encoding="utf-8") as staged:
            json.dump(summary, staged, indent=2)
            staged.write("\n")
        os.replace(staged_path, summary_path)
    finally:
        staged_path.unlink(missing_ok=True)
    return summary_path


def cleanup_published_artifacts(job_root: Path, algorithm: str) -> None:
    """Remove local inputs and publication copies after a successful write.

    Reconstructed models and frames stay in place for the gallery and viewer.

    Args:
        job_root (Path): Local directory for the published job.
        algorithm (str): Feature algorithm used for the reconstruction.

    Raises:
        ValueError: If the feature algorithm is unsupported.
        OSError: If a local file or directory cannot be removed.
    """
    if algorithm not in {"SIFT", "ORB", "AKAZE"}:
        raise ValueError(f"Unsupported feature algorithm: {algorithm}")

    logger.info("Cleaning up published local artifacts in %s", job_root)
    (job_root / "video.mp4").unlink(missing_ok=True)
    (job_root / "colmap" / "database.db").unlink(missing_ok=True)
    if algorithm in {"ORB", "AKAZE"}:
        feature_dir = job_root / "colmap" / "features" / algorithm.lower()
        for feature_file in feature_dir.glob("*.npz"):
            feature_file.unlink()

    artifacts_dir = job_root / "artifacts"
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)
    logger.info("Finished cleaning up published local artifacts in %s", job_root)
