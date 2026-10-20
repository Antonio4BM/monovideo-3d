"""Shared tools for inspecting COLMAP sparse reconstruction models."""

import logging
import struct
from pathlib import Path

logger = logging.getLogger(__name__)


def _read_bin_count(bin_path: Path) -> int:
    """Read the leading little-endian uint64 count from a COLMAP binary file.

    Args:
        bin_path (Path): Path to ``images.bin`` or ``points3D.bin``.

    Returns:
        int: Element count stored in the first 8 bytes.

    Raises:
        OSError: If the file cannot be read.
        struct.error: If the file is shorter than 8 bytes.
    """
    with open(bin_path, "rb") as handle:
        return struct.unpack("<Q", handle.read(8))[0]


def inspector(sparse_path) -> int | None:
    """Inspect sparse model bins and return the best model index.

    Each subdirectory under ``sparse_path`` is scored by registered image
    count from ``images.bin``, then by ``points3D.bin`` count as a
    tiebreaker.

    Args:
        sparse_path (str | Path): COLMAP ``sparse`` directory containing
            numbered model folders (``0``, ``1``, ...).

    Returns:
        int | None: Best model index, or ``None`` when no valid models exist.
    """
    sparse_root = Path(sparse_path)
    if not sparse_root.is_dir():
        logger.error(f"Sparse path not found: {sparse_root}")
        return None

    best_index = None
    best_score = (-1, -1)

    for model_dir in sorted(sparse_root.iterdir()):
        if not model_dir.is_dir():
            continue
        try:
            model_index = int(model_dir.name)
        except ValueError:
            continue

        images_bin = model_dir / "images.bin"
        points_bin = model_dir / "points3D.bin"
        if not images_bin.is_file() or not points_bin.is_file():
            logger.warning(f"Skipping incomplete sparse model: {model_dir}")
            continue

        try:
            image_count = _read_bin_count(images_bin)
            point_count = _read_bin_count(points_bin)
        except (OSError, struct.error) as exc:
            logger.warning(f"Failed to inspect sparse model {model_dir}: {exc}")
            continue

        score = (image_count, point_count)
        logger.info(
            f"Sparse model {model_index}: {image_count} images, {point_count} points",
        )
        if score > best_score:
            best_score = score
            best_index = model_index

    if best_index is None:
        logger.error(f"No valid sparse models found in {sparse_root}")
        return None

    logger.info(
        f"Selected sparse model {best_index} "
        f"({best_score[0]} images, {best_score[1]} points)",
    )
    return best_index
