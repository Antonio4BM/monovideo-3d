import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def image_undistorter(images_path):
    colmap_path = Path(images_path).parent / "colmap"
    images_path = colmap_path / "images"
    if not images_path.exists():
        logger.error("Images path not found, please run colmap-features first")
        return
    logger.info(f"Undistorting images in {images_path}")
    try:
        subprocess.run([
            "colmap",
            "image_undistorter",
            "--image_path", images_path,
            "--input_path", colmap_path / "sparse/0",
            "--output_path", colmap_path / "dense",
            "--output_type", "COLMAP",
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to undistort images: {e}")
        return
    logger.info(f"Undistorted images in {colmap_path / 'dense'}")