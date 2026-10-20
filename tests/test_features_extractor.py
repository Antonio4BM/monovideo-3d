"""Tests for COLMAP SIFT and OpenCV binary feature extraction."""

import subprocess
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from app.pipeline.features_extractor import features_extractor


@pytest.fixture
def frames(tmp_path):
    """Create a JPEG frame with reproducible texture for real detectors."""
    path = tmp_path / "frames"
    path.mkdir()
    pixels = np.random.default_rng(42).integers(
        0, 256, size=(256, 256), dtype=np.uint8
    )
    assert cv2.imwrite(str(path / "frame_0.jpg"), pixels)
    return path


@pytest.mark.parametrize("algorithm", [None, "SIFT", " sift "])
def test_sift_runs_colmap_with_shared_opencv_camera(frames, algorithm):
    """Use COLMAP for default and explicit SIFT and link the source frame."""
    workspace = frames.parent / "colmap"
    with patch("app.pipeline.features_extractor.subprocess.run") as run:
        if algorithm is None:
            features_extractor(frames)
        else:
            features_extractor(frames, algorithm)
    run.assert_called_once_with(
        [
            "colmap", "feature_extractor",
            "--image_path", workspace / "images",
            "--database_path", workspace / "database.db",
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", "OPENCV",
        ],
        check=True,
    )
    link = workspace / "images" / "frame_0.jpg"
    assert link.is_symlink()
    assert link.resolve() == frames / "frame_0.jpg"
    assert not (workspace / "features").exists()


def test_sift_logs_colmap_failure(frames, caplog):
    """Report COLMAP subprocess failures without raising to the caller."""
    with patch(
        "app.pipeline.features_extractor.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "colmap"),
    ):
        features_extractor(frames)
    assert "Failed to extract features" in caplog.text


@pytest.mark.parametrize("algorithm,width", [("ORB", 32), ("AKAZE", 61)])
@pytest.mark.parametrize("blank", [False, True])
def test_binary_detectors_write_feature_archives(frames, algorithm, width, blank):
    """Save aligned binary descriptors and handle frames without features."""
    frame = frames / "frame_0.jpg"
    if blank:
        assert cv2.imwrite(str(frame), np.zeros((256, 256), dtype=np.uint8))
    with patch("app.pipeline.features_extractor.subprocess.run") as run:
        features_extractor(frames, f" {algorithm.lower()} ")
    run.assert_not_called()
    workspace = frames.parent / "colmap"
    assert (workspace / "images" / frame.name).resolve() == frame
    archive = workspace / "features" / algorithm.lower() / f"{frame.name}.npz"
    detector = cv2.ORB_create(nfeatures=9000) if algorithm == "ORB" else (
        cv2.AKAZE_create(descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB)
    )
    keypoints, descriptors = detector.detectAndCompute(
        cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE), None
    )
    with np.load(archive) as features:
        count = len(keypoints)
        assert count == 0 if blank else count > 0
        assert features["keypoints"].shape == (count, 2)
        assert features["descriptors"].shape == (count, width)
        assert features["descriptors"].dtype == np.uint8
        np.testing.assert_allclose(
            features["keypoints"],
            np.asarray([point.pt for point in keypoints]).reshape(-1, 2),
        )
        np.testing.assert_allclose(features["sizes"], [p.size for p in keypoints])
        np.testing.assert_allclose(features["angles"], [p.angle for p in keypoints])
        if not blank:
            np.testing.assert_array_equal(features["descriptors"], descriptors)


@pytest.mark.parametrize("algorithm", ["ORB", "AKAZE"])
def test_binary_detectors_reject_unreadable_frames(frames, algorithm):
    """Raise a useful error when an input JPEG cannot be decoded."""
    (frames / "frame_0.jpg").write_bytes(b"invalid jpeg")
    with pytest.raises(ValueError, match="Unable to read image"):
        features_extractor(frames, algorithm)


@pytest.mark.parametrize("algorithm", [None, 123, "SURF", ""])
def test_invalid_algorithm_does_not_create_workspace(frames, algorithm):
    """Reject unsupported algorithms before creating output files."""
    with pytest.raises(ValueError, match="[Aa]lgorithm"):
        features_extractor(frames, algorithm)
    assert not (frames.parent / "colmap").exists()


def test_repeated_extraction_replaces_existing_links(frames):
    """Allow reruns when the workspace contains existing or broken links."""
    workspace = frames.parent / "colmap" / "images"
    workspace.mkdir(parents=True)
    link = workspace / "frame_0.jpg"
    link.symlink_to(frames / "missing.jpg")
    with patch("app.pipeline.features_extractor.subprocess.run") as run:
        features_extractor(frames)
        features_extractor(frames)
    assert run.call_count == 2
    assert link.resolve() == frames / "frame_0.jpg"
