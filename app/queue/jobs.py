"""Background jobs that run the reconstruction pipeline."""

import subprocess
import sys
from pathlib import Path
from typing import Optional

from rq import Queue
from rq.job import Job

from app.queue.settings import DATA_DIR, JOB_TIMEOUT

RECONSTRUCTION_SPARSE = "sparse"
RECONSTRUCTION_DENSE = "dense"
VALID_RECONSTRUCTIONS = frozenset({RECONSTRUCTION_SPARSE, RECONSTRUCTION_DENSE})


def build_sparse_paths(job_id: str) -> tuple[str, str]:
    """Build relative video and frames paths for a job UUID.

    Args:
        job_id (str): Upload directory UUID under ``data/``.

    Returns:
        tuple[str, str]: Relative ``video_path`` and ``output_path``.
    """
    base = Path(DATA_DIR) / job_id
    video_path = str(base / "video.mp4")
    output_path = str(base / "frames")
    return video_path, output_path


def run_sparse_reconstruction(
    job_id: str,
    strides: int,
) -> dict[str, str]:
    """Run ``colmap-sparse`` for an uploaded video via ``app.reconstruction``.

    Args:
        job_id (str): UUID of the upload directory under ``data/``.
        strides (int): Frame skip interval when extracting frames.

    Returns:
        dict[str, str]: Job id and paths used for video and frames output.

    Raises:
        FileNotFoundError: If the video file does not exist.
        subprocess.CalledProcessError: If the reconstruction CLI exits non-zero.
    """
    video_path, output_path = build_sparse_paths(job_id)
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "app.reconstruction",
            "colmap-sparse",
            "--video_path",
            video_path,
            "--output_path",
            output_path,
            "--strides",
            str(strides),
        ],
        check=True,
    )
    return {
        "job_id": job_id,
        "video_path": video_path,
        "output_path": output_path,
        "reconstruction": RECONSTRUCTION_SPARSE,
    }


def run_dense_reconstruction(
    job_id: str,
    strides: int,
) -> dict[str, str]:
    """Run sparse then ``colmap-dense`` for an uploaded video.

    Args:
        job_id (str): UUID of the upload directory under ``data/``.
        strides (int): Frame skip interval when extracting frames.

    Returns:
        dict[str, str]: Job id, paths, and reconstruction mode.

    Raises:
        FileNotFoundError: If the video file does not exist.
        subprocess.CalledProcessError: If a reconstruction CLI exits non-zero.
    """
    sparse_result = run_sparse_reconstruction(job_id, strides)
    output_path = sparse_result["output_path"]

    subprocess.run(
        [
            sys.executable,
            "-m",
            "app.reconstruction",
            "colmap-dense",
            "--images_path",
            output_path,
        ],
        check=True,
    )
    return {
        **sparse_result,
        "reconstruction": RECONSTRUCTION_DENSE,
    }


def enqueue_reconstruction(
    job_id: str,
    strides: int,
    reconstruction: str,
    queue: Optional[Queue] = None,
) -> Job:
    """Enqueue a sparse or dense reconstruction job for an upload UUID.

    Args:
        job_id (str): UUID of the upload directory under ``data/``.
        strides (int): Frame skip interval when extracting frames.
        reconstruction (str): Either ``sparse`` or ``dense``.
        queue (Queue | None): Optional RQ queue; created if omitted.

    Returns:
        Job: The enqueued RQ job instance with a 55-minute ``job_timeout``.

    Raises:
        ValueError: If ``reconstruction`` is not a supported mode.
    """
    from app.queue.connection import get_queue

    if reconstruction not in VALID_RECONSTRUCTIONS:
        raise ValueError(f"Unsupported reconstruction: {reconstruction}")

    target_queue = queue or get_queue()
    worker = (
        run_dense_reconstruction
        if reconstruction == RECONSTRUCTION_DENSE
        else run_sparse_reconstruction
    )
    return target_queue.enqueue(
        worker,
        job_id,
        strides,
        job_timeout=JOB_TIMEOUT,
    )
