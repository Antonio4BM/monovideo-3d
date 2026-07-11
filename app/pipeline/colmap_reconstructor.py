import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reconstructor(images_path):
    """Convert a COLMAP sparse model to a PLY point cloud.

    Reads ``colmap/sparse/0`` and writes ``colmap/reconstructed.ply`` via
    ``colmap model_converter``.

    Args:
        images_path (str | Path): Directory of source frames; used to locate
            the sibling ``colmap`` workspace.

    Returns:
        None: Always returns ``None``. Returns early if the sparse model is
        missing or conversion fails (errors are logged).
    """
    colmap_path = Path(images_path).parent / "colmap"
    sparse_path = colmap_path / "sparse/0"
    if not sparse_path.exists():
        logger.error("Sparse path not found, please run colmap-mapper first")
        return
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