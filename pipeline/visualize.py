import logging
from pathlib import Path

import trimesh
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize(images_path, filename="reconstructed"):
    colmap_path = Path(images_path).parent / "colmap"

    if not (colmap_path / f"{filename}.ply").exists():
        logger.error("Reconstructed ply file not found, please run colmap-reconstructor first")
        return

    mesh = trimesh.load(colmap_path / f"{filename}.ply")
    mesh.export(colmap_path / f"{filename}.glb")

    logger.info(f"Visualizing 3D model in {colmap_path / f"{filename}.glb"}")