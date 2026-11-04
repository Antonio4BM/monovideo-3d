import logging
import math
import sqlite3
import statistics
import struct
import subprocess
from pathlib import Path

from app.pipeline.tools import inspector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _read_exact(handle, size):
    """Read a fixed number of bytes from a COLMAP binary model file.

    Args:
        handle (BinaryIO): Open model file.
        size (int): Number of bytes to read.

    Returns:
        bytes: The requested bytes.

    Raises:
        ValueError: If the model file ends unexpectedly.
    """
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("Truncated COLMAP sparse model")
    return data


def _camera_centers(images_path):
    """Read image poses and calculate world-space camera centers.

    Args:
        images_path (Path): Path to COLMAP ``images.bin``.

    Returns:
        dict[int, tuple[float, float, float]]: Centers keyed by image ID.
    """
    centers = {}
    with images_path.open("rb") as handle:
        image_count, = struct.unpack("<Q", _read_exact(handle, 8))
        for _ in range(image_count):
            image_id, *pose, _camera_id = struct.unpack(
                "<i7di", _read_exact(handle, 64)
            )
            qw, qx, qy, qz, tx, ty, tz = pose
            # COLMAP stores world-to-camera rotation and translation.
            rotation = (
                (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz),
                 2 * (qx * qz + qw * qy)),
                (2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz),
                 2 * (qy * qz - qw * qx)),
                (2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx),
                 1 - 2 * (qx * qx + qy * qy)),
            )
            translation = (tx, ty, tz)
            centers[image_id] = tuple(
                -sum(rotation[row][column] * translation[row]
                     for row in range(3))
                for column in range(3)
            )
            while _read_exact(handle, 1) != b"\0":
                pass
            point_count, = struct.unpack("<Q", _read_exact(handle, 8))
            _read_exact(handle, 24 * point_count)
    return centers


def _triangulation_angle(point, center_a, center_b):
    """Calculate the angle in degrees between two camera-to-point rays.

    Args:
        point (tuple[float, float, float]): World-space 3D point.
        center_a (tuple[float, float, float]): First camera center.
        center_b (tuple[float, float, float]): Second camera center.

    Returns:
        float | None: Angle in degrees, or ``None`` for a zero-length ray.
    """
    ray_a = tuple(point[i] - center_a[i] for i in range(3))
    ray_b = tuple(point[i] - center_b[i] for i in range(3))
    length_a = math.sqrt(sum(value * value for value in ray_a))
    length_b = math.sqrt(sum(value * value for value in ray_b))
    if not length_a or not length_b:
        return None
    cosine = sum(a * b for a, b in zip(ray_a, ray_b)) / (
        length_a * length_b
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _median_triangulation_angle(sparse_path):
    """Find the median point angle across a COLMAP sparse reconstruction.

    For each point, use the median angle of all distinct observing camera
    pairs. The model median then gives each triangulated point equal weight.

    Args:
        sparse_path (Path): Selected COLMAP model directory.

    Returns:
        float | None: Median angle in degrees, or ``None`` if no valid
            camera pair observes a point.
    """
    centers = _camera_centers(sparse_path / "images.bin")
    point_angles = []
    with (sparse_path / "points3D.bin").open("rb") as handle:
        point_count, = struct.unpack("<Q", _read_exact(handle, 8))
        for _ in range(point_count):
            _point_id, x, y, z = struct.unpack("<Q3d", _read_exact(handle, 32))
            _read_exact(handle, 11)  # RGB (3 bytes) and reprojection error (8).
            track_length, = struct.unpack("<Q", _read_exact(handle, 8))
            image_ids = set()
            for _ in range(track_length):
                image_id, _point_index = struct.unpack(
                    "<ii", _read_exact(handle, 8)
                )
                if image_id in centers:
                    image_ids.add(image_id)
            point = (x, y, z)
            angles = []
            ordered_ids = sorted(image_ids)
            for index, first_id in enumerate(ordered_ids):
                for second_id in ordered_ids[index + 1:]:
                    angle = _triangulation_angle(
                        point, centers[first_id], centers[second_id]
                    )
                    if angle is not None:
                        angles.append(angle)
            if angles:
                point_angles.append(statistics.median(angles))
    return statistics.median(point_angles) if point_angles else None


def _save_triangulation_metadata(database_path, angle):
    """Store the sparse model's median triangulation angle in SQLite.

    Args:
        database_path (Path): Existing COLMAP feature database.
        angle (float): Median angle in degrees.
    """
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata "
            "(key TEXT PRIMARY KEY, value REAL NOT NULL)"
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("median_triangulation_angle_degrees", angle),
        )


def reconstructor(images_path):
    """Convert a COLMAP sparse model to a PLY point cloud.

    Selects the sparse model with the most registered images, breaking ties
    by its number of 3D points, and writes ``colmap/reconstructed.ply`` via
    ``colmap model_converter``. Stores its median triangulation angle in the
    existing COLMAP database when valid point tracks are available.

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
    database_path = colmap_path / "database.db"
    if database_path.is_file():
        try:
            angle = _median_triangulation_angle(sparse_path)
            if angle is not None:
                _save_triangulation_metadata(database_path, angle)
            else:
                logger.warning("No valid triangulation angles in %s", sparse_path)
        except (OSError, ValueError, struct.error, sqlite3.Error) as exc:
            logger.warning("Could not save triangulation metadata: %s", exc)
