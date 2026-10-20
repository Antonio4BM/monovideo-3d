"""Tests for frame sampling and video metadata generation."""

import json
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from app.pipeline.extract_frames import extract_frames


@pytest.fixture
def video_capture():
    """Provide five distinct frames without depending on video codecs."""
    frames = [np.full((8, 12, 3), i * 40, dtype=np.uint8) for i in range(5)]
    capture = MagicMock()
    properties = {
        cv2.CAP_PROP_FPS: 2.5,
        cv2.CAP_PROP_FRAME_COUNT: 5,
        cv2.CAP_PROP_FRAME_WIDTH: 12,
        cv2.CAP_PROP_FRAME_HEIGHT: 8,
    }
    capture.get.side_effect = properties.__getitem__
    capture.read.side_effect = [(True, frame) for frame in frames] + [(False, None)]
    with patch(
        "app.pipeline.extract_frames.cv2.VideoCapture", return_value=capture
    ) as factory:
        yield capture, factory, properties, frames


@pytest.mark.parametrize("strides,indices", [(1, [0, 1, 2, 3, 4]), (2, [0, 2, 4]),
                                            (10, [0])])
def test_extract_frames_samples_video_and_writes_metadata(
    tmp_path, video_capture, strides, indices
):
    """Save stride-selected images and describe their original frame indices."""
    capture, factory, _, frames = video_capture
    output = tmp_path / "nested" / "frames"

    extract_frames("clip.mp4", str(output), strides)

    factory.assert_called_once_with("clip.mp4")
    capture.release.assert_called_once_with()
    assert capture.read.call_count == 6
    assert {path.name for path in output.glob("*.jpg")} == {
        f"frame_{index}.jpg" for index in indices
    }
    for index in indices:
        image = cv2.imread(str(output / f"frame_{index}.jpg"))
        np.testing.assert_array_equal(image, frames[index])
    metadata = json.loads((output / "frames_info.json").read_text())
    assert metadata == {
        "fps": 2.5,
        "frame_count": 5,
        "duration": 2.0,
        "width": 12,
        "height": 8,
        "strides": strides,
        "frames": [
            {"frame_id": index, "frame_path": str(output / f"frame_{index}.jpg")}
            for index in indices
        ],
    }


def test_extract_frames_handles_empty_video(tmp_path, video_capture):
    """Write empty frame metadata when a video has no readable frames."""
    capture, _, properties, _ = video_capture
    properties[cv2.CAP_PROP_FRAME_COUNT] = 0
    capture.read.side_effect = [(False, None)]
    output = tmp_path / "frames"
    output.mkdir()

    extract_frames("empty.mp4", output, 2)

    metadata = json.loads((output / "frames_info.json").read_text())
    assert metadata["frames"] == []
    assert metadata["frame_count"] == 0
    assert metadata["duration"] == 0
    assert not list(output.glob("*.jpg"))
    capture.release.assert_called_once_with()


def test_extract_frames_stops_when_read_fails(tmp_path, video_capture):
    """Stop at the first failed read even if more frames were reported."""
    capture, _, _, frames = video_capture
    capture.read.side_effect = [(True, frames[0]), (False, None)]
    output = tmp_path / "frames"

    extract_frames("truncated.mp4", output, 1)

    metadata = json.loads((output / "frames_info.json").read_text())
    assert metadata["frame_count"] == 5
    assert metadata["frames"] == [
        {"frame_id": 0, "frame_path": str(output / "frame_0.jpg")}
    ]
    assert capture.read.call_count == 2
    capture.release.assert_called_once_with()


def test_extract_frames_reports_zero_fps(tmp_path, video_capture, caplog):
    """Log invalid FPS and return before reading frames or writing metadata."""
    capture, _, properties, _ = video_capture
    properties[cv2.CAP_PROP_FPS] = 0
    output = tmp_path / "frames"

    extract_frames("invalid.mp4", output, 1)

    assert "FPS is 0, please check the video file" in caplog.text
    capture.read.assert_not_called()
    assert not list(output.iterdir())
