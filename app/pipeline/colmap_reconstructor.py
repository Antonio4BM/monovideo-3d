import subprocess
import logging
from pathlib import Path

from app.pipeline.tools import inspector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reconstructor(images_path):
    """Convert a COLMAP sparse model to a PLY point cloud.

    Selects the sparse model with the most registered images, breaking ties
    by its number of 3D points, and writes ``colmap/reconstructed.ply`` via
    ``colmap model_converter``. Uses the same selection as dense reconstruction.

    Args:
        images_path (str | Path): Directory of source frames; used to locate
            the sibling ``colmap`` workspace.

    Returns:
        None: Always returns ``None``. Returns early if the sparse model is
        missing or conversion fails (errors are logged).
    """
    colmap_path = Path(images_path).parent / "colmap"
    sparse_root = colmap_path / "sparse"
    model_index = inspector(sparse_root)
    if model_index is None:
        logger.error("Sparse model not found, please run colmap-mapper first")
        return
    sparse_path = sparse_root / str(model_index)
    logger.info(f"Reconstructing 3D model from {sparse_path}")
    try:
        subprocess.run([
            "colmap",
            "model_converter",
            "--input_path", sparse_path,
            "--output_path", colmap_path / "reconstructed.ply",
            "--output_type", "PLY",
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to reconstruct 3D model: {e}")
        return