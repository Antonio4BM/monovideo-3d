"""Verify that sparse exports select the same best model as dense processing."""

import struct
import sqlite3
from unittest.mock import patch

import pytest

from app.pipeline.colmap_reconstructor import (
    _camera_centers,
    _triangulation_angle,
    reconstructor,
)


def _write_sparse_geometry(model):
    """Write two tracked points seen by cameras with a two-unit baseline.

    Args:
        model (Path): Sparse COLMAP model directory.
    """
    with (model / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 2))
        for image_id, translation in ((1, -1.0), (2, 1.0)):
            handle.write(struct.pack(
                "<i7di", image_id, 1, 0, 0, 0,
                translation, 0, 0, 1,
            ))
            handle.write(b"frame.jpg\0")
            handle.write(struct.pack("<Q", 0))
    with (model / "points3D.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 2))
        for point_id, height in ((1, 1.0), (2, 3 ** 0.5)):
            handle.write(struct.pack("<Q3d", point_id, 0, height, 0))
            handle.write(struct.pack("<3BdQ", 255, 255, 255, 0.0, 2))
            handle.write(struct.pack("<iiii", 1, 0, 2, 0))


@pytest.mark.parametrize(
    "counts, expected",
    [
        ([(2, 10000), (128, 19666)], "1"),
        ([(128, 19666), (2, 30000)], "0"),
        ([(128, 100), (128, 200)], "1"),
    ],
)
def test_reconstructor_selects_best_model(tmp_path, counts, expected):
    """Rank models by registered images first, then by point count."""
    sparse = tmp_path / "colmap" / "sparse"
    for index, (images, points) in enumerate(counts):
        model = sparse / str(index)
        model.mkdir(parents=True)
        (model / "images.bin").write_bytes(struct.pack("<Q", images))
        (model / "points3D.bin").write_bytes(struct.pack("<Q", points))

    with patch("app.pipeline.colmap_reconstructor.subprocess.run") as run:
        reconstructor(tmp_path / "frames")

    run.assert_called_once()
    args = run.call_args.args[0]
    assert args[args.index("--input_path") + 1] == sparse / expected
    assert args[args.index("--output_path") + 1] == (
        tmp_path / "colmap" / "reconstructed.ply"
    )
    assert run.call_args.kwargs["check"] is True


def test_reconstructor_skips_missing_models(tmp_path):
    """Do not attempt conversion when no usable sparse model exists."""
    with patch("app.pipeline.colmap_reconstructor.subprocess.run") as run:
        reconstructor(tmp_path / "frames")
    run.assert_not_called()


def test_reconstructor_saves_median_triangulation_angle(tmp_path):
    """Persist the median of two point angles in the COLMAP database."""
    colmap = tmp_path / "colmap"
    model = colmap / "sparse" / "0"
    model.mkdir(parents=True)
    _write_sparse_geometry(model)
    with sqlite3.connect(colmap / "database.db"):
        pass

    with patch("app.pipeline.colmap_reconstructor.subprocess.run"):
        reconstructor(tmp_path / "frames")

    with sqlite3.connect(colmap / "database.db") as connection:
        rows = connection.execute("SELECT key, value FROM metadata").fetchall()
    assert rows[0][0] == "median_triangulation_angle_degrees"
    assert rows[0][1] == pytest.approx(75.0)


def test_triangulation_angle_skips_zero_length_ray():
    """Exclude a point located at a camera center."""
    assert _triangulation_angle((0, 0, 0), (0, 0, 0), (1, 0, 0)) is None


def test_camera_center_uses_inverse_colmap_pose(tmp_path):
    """Invert rotation and translation when finding a camera center."""
    images = tmp_path / "images.bin"
    root_half = 2 ** -0.5
    with images.open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack(
            "<i7di", 7, root_half, 0, 0, root_half, 0, -2, 0, 1,
        ))
        handle.write(b"rotated.jpg\0")
        handle.write(struct.pack("<Q", 0))

    assert _camera_centers(images)[7] == pytest.approx((2, 0, 0))


def test_reconstructor_does_not_save_metadata_after_conversion_error(tmp_path):
    """Leave the database unchanged when PLY conversion fails."""
    colmap = tmp_path / "colmap"
    model = colmap / "sparse" / "0"
    model.mkdir(parents=True)
    _write_sparse_geometry(model)
    with sqlite3.connect(colmap / "database.db"):
        pass

    with patch("app.pipeline.colmap_reconstructor.subprocess.run") as run:
        from subprocess import CalledProcessError

        run.side_effect = CalledProcessError(1, "colmap")
        reconstructor(tmp_path / "frames")

    with sqlite3.connect(colmap / "database.db") as connection:
        metadata_table = connection.execute(
            "SELECT name FROM sqlite_master WHERE name = 'metadata'"
        ).fetchone()
    assert metadata_table is None
