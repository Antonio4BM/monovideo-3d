import uuid
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, Form, UploadFile
from fastapi.staticfiles import StaticFiles

from app.queue.jobs import enqueue_reconstruction

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


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
