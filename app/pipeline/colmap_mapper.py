import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def colmap_mapper(images_path):
    """Run COLMAP sparse reconstruction (mapper) for a feature database.

    Expects ``colmap/database.db`` next to ``images_path``'s parent. Writes
    the sparse model under ``colmap/sparse``.

    Args:
        images_path (str | Path): Directory of source frames; used to locate
            the sibling ``colmap`` workspace.

    Returns:
        None: Always returns ``None``. Returns early if the database is missing
        or the mapper subprocess fails (errors are logged).

    Raises:
        OSError: If the sparse output directory cannot be created.
    """
    colmap_path = Path(images_path).parent / "colmap"
    database_path = colmap_path / "database.db"
    sparse_path = colmap_path / "sparse"
    sparse_path.mkdir(parents=True, exist_ok=True)
    if not database_path.exists():
        logger.error("Database file not found, please run colmap-features first")
        return
    logger.info(f"Mapping features between images in {colmap_path / 'images'}")
    try:
        subprocess.run([
            "colmap",
            "mapper",
            "--database_path", database_path,
            "--image_path", colmap_path / "images",
            "--output_path", sparse_path,
            "--Mapper.filter_max_reproj_error", "2.0",
            "--Mapper.filter_min_tri_angle", "2.0",
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to map features: {e}")
        return