"""Tests for SIFT and binary matching, validation, and database publication."""

import sqlite3
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from app.pipeline.features_matcher import (
    _load_features,
    _ratio_matches,
    features_matcher,
)


@pytest.fixture
def sift_workspace(tmp_path):
    """Create a SIFT workspace with an empty database and frame directory."""
    frames = tmp_path / "frames"
    frames.mkdir()
    colmap = tmp_path / "colmap"
    colmap.mkdir()
    with sqlite3.connect(colmap / "database.db"):
        pass
    return frames


def test_sift_pairs_follow_numeric_order_and_twelve_frame_window(
    sift_workspace
):
    """Exclude alphabetical and distant neighbors from a 319-frame run."""
    for number in reversed(range(0, 638, 2)):
        (sift_workspace / f"frame_{number}.jpg").touch()
    captured = {}

    def capture_pairs(command, check):
        """Read the pair list while COLMAP would have access to it."""
        assert check is True
        assert command[:2] == ["colmap", "matches_importer"]
        assert command[command.index("--match_type") + 1] == "pairs"
        assert command[command.index("--FeatureMatching.type") + 1] == (
            "SIFT_BRUTEFORCE"
        )
        path = Path(command[command.index("--match_list_path") + 1])
        captured["path"] = path
        captured["pairs"] = [tuple(line.split()) for line in
                             path.read_text().splitlines()]

    with patch("app.pipeline.features_matcher.subprocess.run", capture_pairs):
        features_matcher(sift_workspace)

    pairs = captured["pairs"]
    assert len(pairs) == len(set(pairs)) == 3750
    assert pairs[:12] == [("frame_0.jpg", f"frame_{n}.jpg")
                          for n in range(2, 26, 2)]
    assert ("frame_10.jpg", "frame_12.jpg") in pairs
    assert ("frame_10.jpg", "frame_100.jpg") not in pairs
    assert ("frame_0.jpg", "frame_26.jpg") not in pairs
    assert pairs[-1] == ("frame_634.jpg", "frame_636.jpg")
    assert all(0 < int(b[6:-4]) - int(a[6:-4]) <= 24 for a, b in pairs)
    assert not captured["path"].exists()


@pytest.mark.parametrize("names", [[], ["frame_0.jpg"],
                                   ["frame_0.jpg", "frame 2.jpg"]])
def test_sift_rejects_invalid_frame_lists(sift_workspace, names):
    """Reject insufficient frames and filenames the pair parser cannot read."""
    for name in names:
        (sift_workspace / name).touch()
    with patch("app.pipeline.features_matcher.subprocess.run") as run:
        with pytest.raises(ValueError):
            features_matcher(sift_workspace)
    run.assert_not_called()


def test_sift_cleans_pair_list_on_colmap_failure(sift_workspace, caplog):
    """Clean temporary files and retain existing subprocess-error reporting."""
    for name in ["frame_0.jpg", "frame_2.jpg"]:
        (sift_workspace / name).touch()
    with patch("app.pipeline.features_matcher.subprocess.run") as run:
        run.side_effect = subprocess.CalledProcessError(1, "colmap")
        features_matcher(sift_workspace)
    assert "Failed to match features" in caplog.text
    assert not list(sift_workspace.parent.glob("colmap/sift-matching-*"))


@pytest.mark.parametrize("algorithm", [None, 1, "SURF", ""])
def test_rejects_invalid_algorithm(tmp_path, algorithm):
    """Reject unsupported algorithms before creating a workspace."""
    with pytest.raises(ValueError, match="[Aa]lgorithm"):
        features_matcher(tmp_path / "frames", algorithm)
    assert not (tmp_path / "colmap").exists()


def test_sift_reports_missing_database(tmp_path, caplog):
    """Report a missing SIFT database without invoking COLMAP."""
    with patch("app.pipeline.features_matcher.subprocess.run") as run:
        features_matcher(tmp_path / "frames", " sift ")
    run.assert_not_called()
    assert "Database file not found" in caplog.text


def test_sift_rejects_binary_database(sift_workspace):
    """Prevent SIFT matching on a database containing binary features."""
    with sqlite3.connect(sift_workspace.parent / "colmap/database.db") as db:
        db.execute("CREATE TABLE monovideo_features (algorithm TEXT)")
    with pytest.raises(ValueError, match="binary-feature database"):
        features_matcher(sift_workspace)


@pytest.mark.parametrize("case", ["dimensions", "coordinates", "nan", "width",
                                  "rows", "dtype", "descriptor_dimensions"])
def test_load_features_rejects_malformed_arrays(tmp_path, case):
    """Reject invalid coordinates, descriptor shapes, and descriptor types."""
    keypoints = np.zeros((2, 2), dtype=np.float32)
    descriptors = np.zeros((2, 32), dtype=np.uint8)
    if case == "dimensions":
        keypoints = keypoints.ravel()
    elif case == "coordinates":
        keypoints = np.zeros((2, 3))
    elif case == "nan":
        keypoints[0, 0] = np.nan
    elif case == "width":
        descriptors = np.zeros((2, 31), dtype=np.uint8)
    elif case == "rows":
        descriptors = descriptors[:1]
    elif case == "dtype":
        descriptors = descriptors.astype(np.float32)
    else:
        descriptors = descriptors.ravel()
    archive = tmp_path / "features.npz"
    np.savez(archive, keypoints=keypoints, descriptors=descriptors)
    with pytest.raises(ValueError, match="Invalid binary features"):
        _load_features(archive, 32)


def test_load_features_returns_contiguous_arrays(tmp_path):
    """Preserve valid values while normalizing dtype and memory layout."""
    keypoints = np.asfortranarray([[1, 2], [3, 4]], dtype=np.float64)
    descriptors = np.asfortranarray(np.zeros((2, 32), dtype=np.uint8))
    archive = tmp_path / "features.npz"
    np.savez(archive, keypoints=keypoints, descriptors=descriptors)
    coordinates, binary = _load_features(archive, 32)
    np.testing.assert_array_equal(coordinates, keypoints)
    np.testing.assert_array_equal(binary, descriptors)
    assert coordinates.dtype == np.float32
    assert coordinates.flags.c_contiguous
    assert binary.flags.c_contiguous


@pytest.mark.parametrize("query_count,train_count", [(0, 2), (2, 0), (2, 1)])
def test_ratio_matches_skips_insufficient_descriptors(
    query_count, train_count
):
    """Avoid KNN calls when there cannot be two train neighbors."""
    matcher = MagicMock()
    assert _ratio_matches(
        matcher, np.zeros((query_count, 32)), np.zeros((train_count, 32))
    ) == []
    matcher.knnMatch.assert_not_called()


def test_ratio_matches_filters_neighbors_and_preserves_indices():
    """Require two neighbors and a strict ratio below 0.75."""
    matcher = MagicMock()
    matcher.knnMatch.return_value = [
        [cv2.DMatch(3, 7, 2), cv2.DMatch(3, 8, 4)],
        [cv2.DMatch(4, 8, 3), cv2.DMatch(4, 9, 4)],
        [cv2.DMatch(5, 9, 4), cv2.DMatch(5, 8, 4)],
        [cv2.DMatch(6, 9, 1)],
        [],
    ]
    query = np.zeros((7, 32), dtype=np.uint8)
    train = np.zeros((10, 32), dtype=np.uint8)
    assert _ratio_matches(matcher, query, train) == [(3, 7)]
    matcher.knnMatch.assert_called_once_with(query, train, k=2)


@pytest.fixture(params=["ORB", "AKAZE"])
def binary_workspace(tmp_path, request):
    """Create linked frames and deterministic binary feature archives."""
    algorithm = request.param
    width = 32 if algorithm == "ORB" else 61
    frames = tmp_path / "frames"
    frames.mkdir()
    workspace = tmp_path / "colmap"
    images = workspace / "images"
    images.mkdir(parents=True)
    archives = workspace / "features" / algorithm.lower()
    archives.mkdir(parents=True)
    for index in range(14):
        frame = frames / f"frame_{index}.jpg"
        frame.touch()
        (images / frame.name).symlink_to(frame)
        np.savez(
            archives / f"{frame.name}.npz",
            keypoints=np.array([[1, 2], [3, 4]], dtype=np.float32),
            descriptors=np.array([[0] * width, [255] * width], dtype=np.uint8),
        )
    return frames, workspace, algorithm


def _colmap_stub(command, check, failure=None):
    """Emulate COLMAP's database writes while validating its text inputs.

    Args:
        command (list): COLMAP command and arguments.
        check (bool): Whether subprocess failures should raise.
        failure (str | None): Optional import or geometry failure to simulate.
    """
    assert check is True
    stage = command[1]
    if failure == stage:
        raise subprocess.CalledProcessError(1, command)
    database = command[command.index("--database_path") + 1]
    with sqlite3.connect(database) as db:
        if stage == "feature_importer":
            assert (
                command[command.index("--ImageReader.single_camera") + 1] == "1"
            )
            assert (
                command[command.index("--ImageReader.camera_model") + 1] == "OPENCV"
            )
            names = Path(command[command.index("--image_list_path") + 1])
            names = names.read_text().splitlines()
            assert names == [f"frame_{i}.jpg" for i in range(14)]
            imports = Path(command[command.index("--import_path") + 1])
            db.execute("CREATE TABLE images (image_id INTEGER, name TEXT)")
            db.execute(
                "CREATE TABLE keypoints (image_id INTEGER, rows INTEGER)"
            )
            db.execute("CREATE TABLE descriptors (data BLOB)")
            db.execute("INSERT INTO descriptors VALUES (X'00')")
            db.execute(
                "CREATE TABLE two_view_geometries (rows INTEGER, config INTEGER)"
            )
            for index, name in enumerate(names):
                lines = (imports / f"{name}.txt").read_text().splitlines()
                assert lines[0] == "2 128"
                assert lines[1].split()[:4] == ["1.5", "2.5", "1", "0"]
                assert lines[2].split()[:4] == ["3.5", "4.5", "1", "0"]
                assert (
                    all(line.split()[4:] == ["0"] * 128
                        for line in lines[1:])
                )
                db.execute("INSERT INTO images VALUES (?, ?)", (index, name))
                rows = 1 if failure == "incomplete" else 2
                db.execute(
                    "INSERT INTO keypoints VALUES (?, ?)", (index, rows)
                )
        else:
            assert stage == "matches_importer"
            assert command[command.index("--match_type") + 1] == "raw"
            assert (
                command[command.index("--FeatureMatching.use_gpu") + 1] == "0"
            )
            matches = Path(command[command.index("--match_list_path") + 1])
            blocks = matches.read_text().strip().split("\n\n")
            expected = [f"frame_{i}.jpg frame_{j}.jpg"
                        for i in range(14)
                        for j in range(i + 1, min(i + 13, 14))]
            assert [block.splitlines()[0] for block in blocks] == expected
            assert (
                all(block.splitlines()[1:] == ["0 0", "1 1"]
                    for block in blocks)
            )
            geometry = (0, 2) if failure == "empty_geometry" else (
                (2, 1) if failure == "invalid_geometry" else (2, 2)
            )
            db.execute(
                "INSERT INTO two_view_geometries VALUES (?, ?)", geometry
            )


def test_binary_matching_publishes_verified_database(binary_workspace):
    """Match Hamming descriptors and publish verified database data."""
    frames, workspace, algorithm = binary_workspace
    with patch("app.pipeline.features_matcher.subprocess.run", _colmap_stub):
        features_matcher(frames, f" {algorithm.lower()} ")
    with sqlite3.connect(workspace / "database.db") as db:
        assert (
            db.execute("SELECT * FROM monovideo_features").fetchall() == [(algorithm,)]
        )
        assert (
            db.execute("SELECT COUNT(*) FROM descriptors").fetchone() == (0,)
        )
        assert db.execute("SELECT COUNT(*) FROM images").fetchone() == (14,)
        assert (
            db.execute("SELECT * FROM two_view_geometries").fetchall() == [(2, 2)]
        )
    assert not list(workspace.glob("binary-matching-*"))
    assert (
        len(list((workspace / "features" / algorithm.lower()).glob("*.npz"))) == 14
    )


@pytest.mark.parametrize("failure", ["feature_importer", "matches_importer",
                                     "incomplete", "empty_geometry",
                                     "invalid_geometry"])
def test_binary_matching_failure_does_not_publish_database(
    binary_workspace, failure
):
    """Discard staged databases on import failures or insufficient geometry."""
    frames, workspace, algorithm = binary_workspace

    def run(command, check):
        """Simulate the selected COLMAP failure."""
        _colmap_stub(command, check, failure)

    error = (subprocess.CalledProcessError
             if failure.endswith("importer") else ValueError)
    with patch("app.pipeline.features_matcher.subprocess.run", run):
        with pytest.raises(error):
            features_matcher(frames, algorithm)
    assert not (workspace / "database.db").exists()
    assert not list(workspace.glob("binary-matching-*"))


@pytest.mark.parametrize("case,error", [
    ("database", FileExistsError), ("few_frames", ValueError),
    ("spaces", ValueError), ("missing_link", FileNotFoundError),
    ("missing_archive", FileNotFoundError), ("no_matches", ValueError),
])
def test_binary_matching_rejects_invalid_inputs(binary_workspace, case, error):
    """Reject unsafe or insufficient inputs before running COLMAP."""
    frames, workspace, algorithm = binary_workspace
    archive = workspace / "features" / algorithm.lower() / "frame_0.jpg.npz"
    if case == "database":
        (workspace / "database.db").write_bytes(b"existing database")
    elif case == "few_frames":
        for frame in list(frames.glob("*.jpg"))[1:]:
            frame.unlink()
    elif case == "spaces":
        (frames / "frame 0.jpg").touch()
    elif case == "missing_link":
        (workspace / "images/frame_0.jpg").unlink()
    elif case == "missing_archive":
        archive.unlink()
    else:
        width = 32 if algorithm == "ORB" else 61
        for path in archive.parent.glob("*.npz"):
            np.savez(path, keypoints=np.zeros((2, 2)),
                     descriptors=np.zeros((2, width), dtype=np.uint8))
    with patch("app.pipeline.features_matcher.subprocess.run") as run:
        with pytest.raises(error):
            features_matcher(frames, algorithm)
    run.assert_not_called()
    assert not list(workspace.glob("binary-matching-*"))
    if case == "database":
        assert (workspace / "database.db").read_bytes() == b"existing database"
    else:
        assert not (workspace / "database.db").exists()
