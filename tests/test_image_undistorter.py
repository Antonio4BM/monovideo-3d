"""Tests for sparse model inspection and image undistorter wiring."""

import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.pipeline.image_undistorter import image_undistorter, inspector


def _write_bin_count(path: Path, count: int) -> None:
    path.write_bytes(struct.pack("<Q", count))


def _make_sparse_model(sparse_root: Path, model_index: int, images: int, points: int) -> None:
    model_dir = sparse_root / str(model_index)
    model_dir.mkdir(parents=True)
    _write_bin_count(model_dir / "images.bin", images)
    _write_bin_count(model_dir / "points3D.bin", points)


def test_inspector_returns_model_with_most_images(tmp_path):
    sparse_root = tmp_path / "sparse"
    sparse_root.mkdir()
    _make_sparse_model(sparse_root, 0, images=2, points=82)
    _make_sparse_model(sparse_root, 1, images=185, points=17993)

    assert inspector(sparse_root) == 1


def test_inspector_uses_points_as_tiebreaker(tmp_path):
    sparse_root = tmp_path / "sparse"
    sparse_root.mkdir()
    _make_sparse_model(sparse_root, 0, images=10, points=100)
    _make_sparse_model(sparse_root, 2, images=10, points=500)

    assert inspector(sparse_root) == 2


def test_inspector_returns_none_when_sparse_missing(tmp_path):
    assert inspector(tmp_path / "missing") is None


def test_inspector_skips_incomplete_models(tmp_path):
    sparse_root = tmp_path / "sparse"
    sparse_root.mkdir()
    incomplete = sparse_root / "0"
    incomplete.mkdir()
    _write_bin_count(incomplete / "images.bin", 5)
    _make_sparse_model(sparse_root, 1, images=3, points=10)

    assert inspector(sparse_root) == 1


def test_image_undistorter_uses_best_model_index(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    colmap = tmp_path / "colmap"
    (colmap / "images").mkdir(parents=True)
    sparse_root = colmap / "sparse"
    sparse_root.mkdir()
    _make_sparse_model(sparse_root, 0, images=2, points=10)
    _make_sparse_model(sparse_root, 1, images=50, points=1000)

    with patch("app.pipeline.image_undistorter.subprocess.run") as run_mock:
        run_mock.return_value = MagicMock(returncode=0)
        image_undistorter(frames)

    args = run_mock.call_args.args[0]
    assert "--input_path" in args
    input_path = Path(args[args.index("--input_path") + 1])
    assert input_path == sparse_root / "1"
