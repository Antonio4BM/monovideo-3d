import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def stereo_fusion(images_path):
    colmap_path = Path(images_path).parent / "colmap"
    dense_path = colmap_path / "dense"
    if not dense_path.exists():
        logger.error("Dense path not found, please run patch_match_stereo first")
        return
    logger.info(f"Fusing stereo images in {dense_path}")
    try:
        subprocess.run([
            "colmap",
            "stereo_fusion",
            "--workspace_path", dense_path,
            "--workspace_format", "COLMAP",
            "--input_type", "geometric",
            "--output_path", colmap_path / "fused.ply",
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to fuse stereo images: {e}")
        return
    logger.info(f"Stereo fusion completed in {colmap_path / 'fused.ply'}")