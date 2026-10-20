"""Verify that sparse exports select the same best model as dense processing."""

import struct
from unittest.mock import patch

import pytest

from app.pipeline.colmap_reconstructor import reconstructor


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
