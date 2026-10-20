"""Extract image features with COLMAP SIFT or OpenCV binary detectors."""

import logging
import subprocess
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def features_extractor(images_path, algorithm="SIFT"):
    """Extract features from JPEG frames using the selected algorithm.

    Creates a sibling ``colmap`` workspace and links frames into its
    ``images`` directory. SIFT uses COLMAP's default extractor and writes
    ``colmap/database.db`` with a single OPENCV camera model.

    ORB and AKAZE use OpenCV and write one ``<image_name>.npz`` file per
    frame under ``colmap/features/<algorithm>`` (lowercase algorithm name).
    Each archive contains ``keypoints`` (N x 2 OpenCV pixel coordinates),
    ``sizes``, ``angles`` (degrees), and binary ``descriptors``. Descriptor
    rows correspond to keypoint rows. These archives require separate
    matching and import into COLMAP before reconstruction.

    Args:
        images_path (str | Path): Directory containing source ``*.jpg`` frames.
        algorithm (str): Case-insensitive SIFT, ORB, or AKAZE. Defaults to
            SIFT, preserving the existing COLMAP extraction behavior.

    Returns:
        None: SIFT subprocess failures are logged and return early.

    Raises:
        ValueError: If the algorithm is unsupported or a frame cannot be read.
        OSError: If directories, frame links, or feature files cannot be created.
        cv2.error: If OpenCV feature extraction fails.
    """
    if not isinstance(algorithm, str):
        raise ValueError("Algorithm must be SIFT, ORB, or AKAZE")
    algorithm = algorithm.strip().upper()
    if algorithm not in {"SIFT", "ORB", "AKAZE"}:
        raise ValueError(f"Unsupported feature extraction algorithm: {algorithm}")

    colmap_path = Path(images_path).parent / "colmap"
    colmap_path.mkdir(parents=True, exist_ok=True)
    frames_links = colmap_path / "images"
    frames_links.mkdir(parents=True, exist_ok=True)

    frames_list = sorted(Path(images_path).glob("*.jpg"))
    logger.info("Creating links for %s frames", len(frames_list))
    for frame in frames_list:
        link_path = frames_links / frame.name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(frame.resolve())

    logger.info("Extracting %s features from %s", algorithm, frames_links)
    if algorithm == "SIFT":
        try:
            subprocess.run(
                [
                    "colmap",
                    "feature_extractor",
                    "--image_path", frames_links,
                    "--database_path", colmap_path / "database.db",
                    "--ImageReader.single_camera", "1",
                    "--ImageReader.camera_model", "OPENCV",
                ],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            logger.error("Failed to extract features: %s", exc)
            return
        logger.info("Features extracted to %s", colmap_path / "database.db")
        return
    elif algorithm == "ORB":
        detector = cv2.ORB_create(nfeatures=9000)
    else:
        detector = cv2.AKAZE_create(
            descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
        )

    features_path = colmap_path / "features" / algorithm.lower()
    features_path.mkdir(parents=True, exist_ok=True)
    for frame in frames_list:
        image = cv2.imread(str(frames_links / frame.name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Unable to read image: {frame}")

        keypoints, descriptors = detector.detectAndCompute(image, None)
        if descriptors is None:
            descriptors = np.empty((0, detector.descriptorSize()), dtype=np.uint8)
        coordinates = np.asarray(
            [keypoint.pt for keypoint in keypoints], dtype=np.float32
        ).reshape(-1, 2)
        sizes = np.asarray(
            [keypoint.size for keypoint in keypoints], dtype=np.float32
        )
        angles = np.asarray(
            [keypoint.angle for keypoint in keypoints], dtype=np.float32
        )
        np.savez_compressed(
            features_path / f"{frame.name}.npz",
            keypoints=coordinates,
            sizes=sizes,
            angles=angles,
            descriptors=descriptors,
        )
        logger.info("Extracted %s features from %s", len(keypoints), frame.name)

    logger.info("%s features extracted to %s", algorithm, features_path)
