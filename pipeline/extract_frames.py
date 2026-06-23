import json
import logging

import cv2
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_frames(video_path, output_path, strides):

    logger.info(f"Extracting frames from {video_path} to {output_path} with strides {strides}")
    frames_list = []
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    captured = cv2.VideoCapture(video_path)

    fps = captured.get(cv2.CAP_PROP_FPS)
    frame_count = captured.get(cv2.CAP_PROP_FRAME_COUNT)
    width, height = (captured.get(cv2.CAP_PROP_FRAME_WIDTH), 
        captured.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"FPS: {fps}, Frame count: {frame_count}, Width: {width}, Height: {height}")
    if fps == 0:
        logger.error("FPS is 0, please check the video file")
        return
    duration = frame_count / fps

    i = 0
    while True:
        ret, frame = captured.read()
        if not ret:
            break
        if i % strides == 0:
            cv2.imwrite(output_dir / f"frame_{i}.jpg", frame)
            frames_list.append({"frame_id": i, "frame_path": str(output_dir / f"frame_{i}.jpg")})
        i += 1
    metadata = {
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "width": width,
        "height": height,
        "strides": strides,
        "frames": frames_list,
    }
    with open(output_dir / "frames_info.json", "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Frames info saved to {output_dir / 'frames_info.json'}")


    captured.release()
