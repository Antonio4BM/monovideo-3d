import logging
import uuid
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.queue.jobs import enqueue_reconstruction
from app.reconstruction_catalog import get_reconstruction_catalog
from app.reconstruction_helpers import (
    FUSED_GLB,
    RECONSTRUCTED_GLB,
    serve_colmap_glb,
)
from app.s3 import close_s3_client, put_object
from app.settings import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Release the shared S3 client when the API shuts down.

    Args:
        app (FastAPI): Application whose lifespan is being managed.

    Yields:
        None: Control while the application serves requests.
    """
    try:
        yield
    finally:
        close_s3_client()


app = FastAPI(title="MonoVideo3D-API", lifespan=lifespan)

DATA_DIR = get_settings().data_directory
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@app.post("/video-upload")
async def video_upload(
    file: UploadFile,
    strides: int = Form(...),
    reconstruction: Literal["sparse", "dense"] = Form(...),
    feature_extractor_algorithm: Literal["SIFT", "ORB", "AKAZE"] = Form("SIFT"),
):
    """Upload a video to S3 and enqueue reconstruction.

    Args:
        file (UploadFile): Incoming video file from the multipart request.
        strides (int): Frame skip interval for reconstruction (form field).
        reconstruction (Literal["sparse", "dense"]): Reconstruction mode.
        feature_extractor_algorithm (str): Feature algorithm; defaults to SIFT.

    Returns:
        dict[str, str]: Confirmation payload with ``message`` and ``job_id``.

    Raises:
        ValueError: If the S3 bucket is not configured.
        botocore.exceptions.BotoCoreError: If AWS configuration is invalid.
        botocore.exceptions.ClientError: If S3 rejects the upload.
    """
    logger.info("reading video bytes from request")
    video_bytes = await file.read()

    job_id = str(uuid.uuid4())
    logger.info("uploading video for job %s to S3", job_id)
    start_time = time.monotonic()
    await run_in_threadpool(put_object, f"{job_id}/video.mp4", video_bytes)
    end_time = time.monotonic()
    logger.info(f"upload took {end_time - start_time:.2f} seconds")

    enqueue_reconstruction(
        job_id, strides, reconstruction, feature_extractor_algorithm,
    )

    return {"message": "Video uploaded successfully", "job_id": job_id}


@app.get("/reconstructions")
async def reconstructions_page():
    """Serve the HTML view that lists and displays finished reconstructions.

    Returns:
        FileResponse: ``static/reconstructions.html``.
    """
    return FileResponse(STATIC_DIR / "reconstructions.html", media_type="text/html")


@app.get("/api/reconstructions")
async def reconstructions():
    """List all published jobs with metadata and available local models.

    Returns:
        dict[str, list[dict]]: Metadata ``jobs`` and the existing
        ``reconstructions`` and ``fused`` model lists.

    Raises:
        HTTPException: 503 when DynamoDB configuration or access fails.
    """
    return await run_in_threadpool(get_reconstruction_catalog, DATA_DIR)


@app.get("/api/reconstructions/{job_id}/glb")
async def reconstruction_glb(job_id: str):
    """Serve the ``reconstructed.glb`` model for a finished job.

    Args:
        job_id (str): Upload directory UUID under ``data/``.

    Returns:
        FileResponse: Binary GLB file for the 3D viewer.

    Raises:
        HTTPException: 400 if ``job_id`` is not a UUID, 404 if the GLB is
            missing.
    """
    return serve_colmap_glb(DATA_DIR, job_id, RECONSTRUCTED_GLB)


@app.get("/api/reconstructions/{job_id}/fused")
async def fused_reconstruction_glb(job_id: str):
    """Serve the ``fused.glb`` model for a finished dense job.

    Args:
        job_id (str): Upload directory UUID under ``data/``.

    Returns:
        FileResponse: Binary GLB file for the 3D viewer.

    Raises:
        HTTPException: 400 if ``job_id`` is not a UUID, 404 if the GLB is
            missing.
    """
    return serve_colmap_glb(DATA_DIR, job_id, FUSED_GLB)


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
