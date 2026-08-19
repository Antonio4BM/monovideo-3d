"""COLMAP image undistorter and sparse-model selection helpers."""

import logging
import struct
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)
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


def image_undistorter(images_path):
    """Undistort images for dense reconstruction using the best sparse model.

    Args:
        images_path (str | Path): Frames directory used to locate the sibling
            ``colmap`` workspace.

    Returns:
        None: Always returns ``None``. Returns early on missing inputs or
        COLMAP failure (errors are logged).
    """
    colmap_path = Path(images_path).parent / "colmap"
    images_path = colmap_path / "images"
    if not images_path.exists():
        logger.error("Images path not found, please run colmap-features first")
        return

    sparse_root = colmap_path / "sparse"
    model_index = inspector(sparse_root)
    if model_index is None:
        logger.error("Sparse model not found, please run colmap-mapper first")
        return

    input_path = sparse_root / str(model_index)
    logger.info(f"Undistorting images in {images_path} with sparse model {model_index}")
    try:
        subprocess.run(
            [
                "colmap",
                "image_undistorter",
                "--image_path",
                images_path,
                "--input_path",
                input_path,
                "--output_path",
                colmap_path / "dense",
                "--output_type",
                "COLMAP",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to undistort images: {e}")
        return
    logger.info(f"Undistorted images in {colmap_path / 'dense'}")
