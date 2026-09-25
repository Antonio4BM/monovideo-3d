import uuid
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.queue.jobs import enqueue_reconstruction
from app.reconstruction_helpers import (
    FUSED_GLB,
    RECONSTRUCTED_GLB,
    list_finished_reconstructions,
    serve_colmap_glb,
)

app = FastAPI(title="MonoVideo3D-API")

DATA_DIR = Path("data")
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@app.post("/video-upload")
async def video_upload(
    file: UploadFile,
    strides: int = Form(...),
    reconstruction: Literal["sparse", "dense"] = Form(...),
):
    """Upload a video, save it, and enqueue reconstruction.

    Args:
        file (UploadFile): Incoming video file from the multipart request.
        strides (int): Frame skip interval for reconstruction (form field).
        reconstruction (Literal["sparse", "dense"]): Reconstruction mode.

    Returns:
        dict[str, str]: Confirmation payload with ``message`` and ``job_id``.

    Raises:
        OSError: If the destination directory or file cannot be written.
    """
    video_bytes = await file.read()

    job_id = str(uuid.uuid4())
    video_dir = DATA_DIR / job_id
    video_dir.mkdir(parents=True, exist_ok=True)
    destination_path = video_dir / "video.mp4"
    destination_path.write_bytes(video_bytes)

    enqueue_reconstruction(job_id, strides, reconstruction)

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
    """List finished reconstructed and fused COLMAP GLB files.

    Returns:
        dict[str, list[dict[str, str]]]: Payload with ``reconstructions`` and
        ``fused`` lists of ``job_id``, ``glb_url``, and ``modified_at``.
    """
    return {
        "reconstructions": list_finished_reconstructions(DATA_DIR),
        "fused": list_finished_reconstructions(
            DATA_DIR,
            filename=FUSED_GLB,
            url_name="fused",
        ),
    }


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
