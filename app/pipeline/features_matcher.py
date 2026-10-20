"""Match SIFT or binary features and verify their geometry with COLMAP."""

import logging
import re
import sqlite3
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

RATIO_THRESHOLD = 0.75
FRAME_OVERLAP = 12

logger = logging.getLogger(__name__)


def _frame_order(path):
    """Return a natural sort key for numbered frame filenames.

    Args:
        path (Path): Frame path.

    Returns:
        list: Text and numeric filename components in temporal sort order.
    """
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", path.name)
    ]


def _load_features(path, descriptor_size):
    """Load and validate binary descriptors and their keypoint coordinates.

    Args:
        path (Path): Feature archive produced by the extractor.
        descriptor_size (int): Expected number of bytes per descriptor.

    Returns:
        tuple: Contiguous coordinate and descriptor arrays with matching rows.

    Raises:
        ValueError: If feature arrays are malformed or contain invalid values.
        OSError: If the archive cannot be read.
    """
    with np.load(path, allow_pickle=False) as archive:
        keypoints = np.asarray(archive["keypoints"], dtype=np.float32)
        descriptors = archive["descriptors"]
    if (
        keypoints.ndim != 2
        or keypoints.shape[1] != 2
        or not np.isfinite(keypoints).all()
        or descriptors.ndim != 2
        or descriptors.shape != (len(keypoints), descriptor_size)
        or descriptors.dtype != np.uint8
    ):
        raise ValueError(f"Invalid binary features: {path}")
    return np.ascontiguousarray(keypoints), np.ascontiguousarray(descriptors)


def _ratio_matches(matcher, descriptors_a, descriptors_b):
    """Find descriptor correspondences passing the fixed nearest-neighbor ratio.

    Args:
        matcher (cv2.BFMatcher): Hamming matcher with cross-check disabled.
        descriptors_a (np.ndarray): Query binary descriptors.
        descriptors_b (np.ndarray): Train binary descriptors.

    Returns:
        list[tuple[int, int]]: Query/train indices into the original keypoints.
    """
    if len(descriptors_a) == 0 or len(descriptors_b) < 2:
        return []
    neighbors = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    return [
        (pair[0].queryIdx, pair[0].trainIdx)
        for pair in neighbors
        if len(pair) == 2
        and pair[0].distance < RATIO_THRESHOLD * pair[1].distance
    ]


def _match_binary_features(images_path, colmap_path, algorithm):
    """Import binary keypoints, match nearby frames, and verify correspondences.

    COLMAP's text importer requires SIFT-shaped descriptors. Temporary zero
    placeholders allow its image reader to initialize cameras and rigs; they
    are never matched and are deleted after geometric verification. Actual
    binary descriptors remain in the extractor's NPZ archives.

    Args:
        images_path (Path): Source JPEG frame directory.
        colmap_path (Path): Workspace containing image links and NPZ archives.
        algorithm (str): Normalized ORB or AKAZE name.

    Raises:
        FileExistsError: If a reconstruction database already exists.
        ValueError: If input features or verified geometry are insufficient.
        OSError: If input or output files cannot be accessed.
        subprocess.CalledProcessError: If a COLMAP import command fails.
    """
    database_path = colmap_path / "database.db"
    if database_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite {database_path}; use a fresh workspace"
        )
    frames = sorted(images_path.glob("*.jpg"), key=_frame_order)
    if len(frames) < 2:
        raise ValueError("At least two extracted frames are required")
    descriptor_size = (
        cv2.ORB_create().descriptorSize()
        if algorithm == "ORB"
        else cv2.AKAZE_create().descriptorSize()
    )
    features_path = colmap_path / "features" / algorithm.lower()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    colmap_path.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="binary-matching-", dir=colmap_path) as temp:
        staging = Path(temp)
        import_path = staging / "features"
        import_path.mkdir()
        staged_database = staging / "database.db"
        image_list = staging / "images.txt"
        matches_path = staging / "matches.txt"
        placeholder = " " + " ".join(["0"] * 128) + "\n"
        for frame in frames:
            if any(character.isspace() for character in frame.name):
                raise ValueError("COLMAP match import requires names without spaces")
            if not (colmap_path / "images" / frame.name).is_file():
                raise FileNotFoundError(f"Missing extracted image link: {frame.name}")
            keypoints, _ = _load_features(
                features_path / f"{frame.name}.npz", descriptor_size
            )
            with (import_path / f"{frame.name}.txt").open("w") as output:
                output.write(f"{len(keypoints)} 128\n")
                for x, y in keypoints:
                    output.write(f"{float(x) + 0.5} {float(y) + 0.5} 1 0")
                    output.write(placeholder)
        image_list.write_text("\n".join(frame.name for frame in frames) + "\n")

        # Cache only the current window instead of all descriptors in the video.
        cache = {}
        pair_count = 0
        with matches_path.open("w") as output:
            for index, frame in enumerate(frames):
                window = frames[index:index + FRAME_OVERLAP + 1]
                for candidate in window:
                    if candidate not in cache:
                        _, cache[candidate] = _load_features(
                            features_path / f"{candidate.name}.npz", descriptor_size
                        )
                for candidate in window[1:]:
                    matches = _ratio_matches(matcher, cache[frame], cache[candidate])
                    if not matches:
                        continue
                    output.write(f"{frame.name} {candidate.name}\n")
                    for query_index, train_index in matches:
                        output.write(f"{query_index} {train_index}\n")
                    output.write("\n")
                    pair_count += 1
                    logger.info(
                        "%s - %s: %s ratio-filtered matches",
                        frame.name, candidate.name, len(matches),
                    )
                del cache[frame]
        if pair_count == 0:
            raise ValueError("No binary feature matches passed the ratio filter")

        subprocess.run(
            [
                "colmap", "feature_importer",
                "--database_path", str(staged_database),
                "--image_path", str(colmap_path / "images"),
                "--import_path", str(import_path),
                "--image_list_path", str(image_list),
                "--ImageReader.single_camera", "1",
                "--ImageReader.camera_model", "OPENCV",
            ],
            check=True,
        )
        with sqlite3.connect(staged_database) as connection:
            imported = dict(connection.execute(
                "SELECT images.name, keypoints.rows FROM images "
                "JOIN keypoints USING (image_id)"
            ))
            for frame in frames:
                keypoints, _ = _load_features(
                    features_path / f"{frame.name}.npz", descriptor_size
                )
                if imported.get(frame.name) != len(keypoints):
                    raise ValueError(f"COLMAP did not import all features: {frame.name}")

        subprocess.run(
            [
                "colmap", "matches_importer",
                "--database_path", str(staged_database),
                "--match_list_path", str(matches_path),
                "--match_type", "raw",
                "--FeatureMatching.use_gpu", "0",
            ],
            check=True,
        )
        with sqlite3.connect(staged_database) as connection:
            verified_pairs = connection.execute(
                "SELECT COUNT(*) FROM two_view_geometries "
                "WHERE rows > 0 AND config >= 2"
            ).fetchone()[0]
            if verified_pairs == 0:
                raise ValueError("COLMAP found no geometrically verified image pairs")
            connection.execute("DELETE FROM descriptors")
            connection.execute(
                "CREATE TABLE monovideo_features (algorithm TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT INTO monovideo_features VALUES (?)", (algorithm,)
            )
            connection.commit()
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        # Hard-link publication is atomic and will not replace an existing DB.
        database_path.hardlink_to(staged_database)
        logger.info("Verified %s image pairs in %s", verified_pairs, database_path)


def features_matcher(images_path, algorithm="SIFT"):
    """Match features and geometrically verify their image correspondences.

    All algorithms pair each frame with the next 12 extracted frames in
    natural filename order, without loop detection. SIFT passes an explicit
    pair list to COLMAP for descriptor matching and geometric verification.
    ORB and AKAZE read NPZ archives, use Hamming KNN with a 0.75 ratio filter,
    and import raw matches into a new database for COLMAP verification.

    Args:
        images_path (str | Path): Directory containing source JPEG frames.
        algorithm (str): Case-insensitive SIFT (default), ORB, or AKAZE.

    Raises:
        ValueError: If the algorithm, input features, or geometry are invalid.
        OSError: If required files are missing or the binary database exists.
        subprocess.CalledProcessError: If binary import or verification fails.
    """
    if not isinstance(algorithm, str):
        raise ValueError("Algorithm must be SIFT, ORB, or AKAZE")
    algorithm = algorithm.strip().upper()
    if algorithm not in {"SIFT", "ORB", "AKAZE"}:
        raise ValueError(f"Unsupported feature matching algorithm: {algorithm}")
    colmap_path = Path(images_path).parent / "colmap"
    if algorithm != "SIFT":
        _match_binary_features(Path(images_path), colmap_path, algorithm)
        return

    database_path = colmap_path / "database.db"
    if not database_path.exists():
        logger.error("Database file not found, please run colmap-features first")
        return
    with sqlite3.connect(database_path) as connection:
        if connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'monovideo_features'"
        ).fetchone():
            raise ValueError("Cannot run SIFT matching on a binary-feature database")
    frames = sorted(Path(images_path).glob("*.jpg"), key=_frame_order)
    if len(frames) < 2:
        raise ValueError("At least two extracted frames are required")
    if any(character.isspace() for frame in frames for character in frame.name):
        raise ValueError("COLMAP match import requires names without spaces")

    # COLMAP's sequential matcher sorts names lexicographically internally.
    # Explicit pairs preserve the same temporal window as binary matching.
    with TemporaryDirectory(prefix="sift-matching-", dir=colmap_path) as temp:
        pairs_path = Path(temp) / "pairs.txt"
        with pairs_path.open("w") as output:
            for index, frame in enumerate(frames):
                for candidate in frames[index + 1:index + FRAME_OVERLAP + 1]:
                    output.write(f"{frame.name} {candidate.name}\n")
        try:
            subprocess.run(
                [
                    "colmap", "matches_importer",
                    "--database_path", database_path,
                    "--match_list_path", str(pairs_path),
                    "--match_type", "pairs",
                    "--FeatureMatching.type", "SIFT_BRUTEFORCE",
                ],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            logger.error("Failed to match features: %s", exc)
