import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def features_extractor(images_path):

    colmap_path = Path(images_path).parent / "colmap"
    colmap_path.mkdir(parents=True, exist_ok=True)
    frames_links = Path(colmap_path) / "images"
    frames_links.mkdir(parents=True, exist_ok=True)

    frames = Path(images_path).glob("*.jpg")
    frames_list = list(frames)
    logger.info(f"Creating links for {len(frames_list)} frames")
    for frame in frames_list:
        link_path = frames_links / frame.name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(frame.resolve())


    logger.info(f"Extracting features from {frames_links}")
    try:
        subprocess.run([
            "colmap",
            "feature_extractor",
            "--image_path", frames_links,
            "--database_path", colmap_path / "database.db",
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", "OPENCV"
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract features: {e}")
        return
    logger.info(f"Features extracted to {colmap_path / 'database.db'}")