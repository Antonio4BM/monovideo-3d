"""Undistort images using the best COLMAP sparse model."""

import logging
import subprocess
from pathlib import Path

from app.pipeline.tools import inspector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
