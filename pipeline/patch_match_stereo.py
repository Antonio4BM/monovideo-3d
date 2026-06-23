import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def patch_match_stereo(images_path):
    colmap_path = Path(images_path).parent / "colmap"
    images_path = colmap_path / "images"
    dense_path = colmap_path / "dense"
    if not images_path.exists() or not dense_path.exists():
        logger.error("Images path or dense path not found, please run image_undistorter first")
        return
    try:
        subprocess.run([
            "colmap",
            "patch_match_stereo",
            "--workspace_path", dense_path,
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.gpu_index", "0",
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run patch match stereo: {e}")
        return
    logger.info(f"Patch match stereo completed in {dense_path}")