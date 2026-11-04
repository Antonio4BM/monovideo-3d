"""Tests for the COLMAP sparse mapper command."""

import logging
import subprocess
from unittest.mock import patch

from app.pipeline.colmap_mapper import colmap_mapper


def test_colmap_mapper_runs_with_expected_paths_and_options(tmp_path):
    """Create the sparse directory and map features from the COLMAP workspace."""
    colmap = tmp_path / "colmap"
    colmap.mkdir()
    database = colmap / "database.db"
    database.touch()
    frames = tmp_path / "frames"

    with patch("app.pipeline.colmap_mapper.subprocess.run") as run:
        result = colmap_mapper(frames)

    assert result is None
    assert (colmap / "sparse").is_dir()
    run.assert_called_once_with(
        [
            "colmap",
            "mapper",
            "--database_path", database,
            "--image_path", colmap / "images",
            "--output_path", colmap / "sparse",
            "--Mapper.filter_max_reproj_error", "2.0",
            "--Mapper.filter_min_tri_angle", "2.0",
        ],
        check=True,
    )


def test_colmap_mapper_skips_missing_database(tmp_path, caplog):
    """Report a missing database without starting COLMAP."""
    with patch("app.pipeline.colmap_mapper.subprocess.run") as run:
        with caplog.at_level(logging.ERROR):
            result = colmap_mapper(tmp_path / "frames")

    assert result is None
    assert (tmp_path / "colmap" / "sparse").is_dir()
    run.assert_not_called()
    assert "Database file not found" in caplog.text


def test_colmap_mapper_logs_subprocess_failure(tmp_path, caplog):
    """Return after a failed COLMAP process and log the error."""
    colmap = tmp_path / "colmap"
    colmap.mkdir()
    (colmap / "database.db").touch()
    error = subprocess.CalledProcessError(1, ["colmap", "mapper"])

    with patch("app.pipeline.colmap_mapper.subprocess.run", side_effect=error) as run:
        with caplog.at_level(logging.ERROR):
            result = colmap_mapper(tmp_path / "frames")

    assert result is None
    run.assert_called_once()
    assert "Failed to map features" in caplog.text
