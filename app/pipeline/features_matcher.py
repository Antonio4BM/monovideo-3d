import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def features_matcher(images_path):
    """Match COLMAP features sequentially across frames in the database.

    Runs ``colmap sequential_matcher`` on ``colmap/database.db`` with overlap
    12, loop detection, and SIFT brute-force matching.

    Args:
        images_path (str | Path): Directory of source frames; used to locate
            the sibling ``colmap`` workspace.

    Returns:
        None: Always returns ``None``. Returns early if the database is missing
        or matching fails (errors are logged).
    """
    colmap_path = Path(images_path).parent / "colmap"
    database_path = colmap_path / "database.db"
    if not database_path.exists():
        logger.error("Database file not found, please run colmap-features first")
        return
    logger.info(f"Matching features between images in {colmap_path / 'images'}")
    try:
        subprocess.run([
            "colmap",
            "sequential_matcher",
            "--database_path", database_path,
            "--SequentialMatching.overlap", "12",
            "--SequentialMatching.loop_detection", "1",
            "--FeatureMatching.type", "SIFT_BRUTEFORCE",
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to match features: {e}")
        return