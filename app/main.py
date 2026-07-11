import uvicorn
import uuid
from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI(title="MonoVideo3D-API")

DATA_DIR = Path("data")
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

@app.post("/video-upload")
async def video_upload(file: UploadFile):
    """Upload a video and save it under a unique directory as ``video.mp4``.

    Args:
        file (UploadFile): Incoming video file from the multipart request.

    Returns:
        dict[str, str]: Confirmation payload with a ``message`` key.

    Raises:
        OSError: If the destination directory or file cannot be written.
    """
    video_bytes = await file.read()

    video_path = DATA_DIR / f"{uuid.uuid4()}"
    video_path.mkdir(parents=True, exist_ok=True)
    destination_path = video_path / "video.mp4"
    destination_path.write_bytes(video_bytes)

    return {"message": "Video uploaded successfully"}


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)