"""Helpers to resolve and list finished COLMAP reconstruction files."""

import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse

RECONSTRUCTED_GLB = "reconstructed.glb"
FUSED_GLB = "fused.glb"


def parse_job_id(job_id: str) -> str:
    """Normalize a reconstruction directory name as a UUID string.

    Args:
        job_id (str): Candidate upload directory name.

    Returns:
        str: Canonical UUID string.

    Raises:
        ValueError: If ``job_id`` is not a valid UUID.
    """
    return str(uuid.UUID(job_id))


def colmap_glb_path(data_dir: Path, job_id: str, filename: str) -> Path:
    """Resolve a COLMAP GLB path for a job under ``data_dir``.

    Args:
        data_dir (Path): Root directory that holds upload job folders.
        job_id (str): Upload directory UUID.
        filename (str): GLB filename under ``colmap/``.

    Returns:
        Path: Expected path to ``colmap/<filename>``.

    Raises:
        ValueError: If ``job_id`` is not a valid UUID.
    """
    return data_dir / parse_job_id(job_id) / "colmap" / filename


def reconstructed_glb_path(data_dir: Path, job_id: str) -> Path:
    """Resolve the ``reconstructed.glb`` path for a job under ``data_dir``.

    Args:
        data_dir (Path): Root directory that holds upload job folders.
        job_id (str): Upload directory UUID.

    Returns:
        Path: Expected path to ``colmap/reconstructed.glb``.

    Raises:
        ValueError: If ``job_id`` is not a valid UUID.
    """
    return colmap_glb_path(data_dir, job_id, RECONSTRUCTED_GLB)


def fused_glb_path(data_dir: Path, job_id: str) -> Path:
    """Resolve the ``fused.glb`` path for a job under ``data_dir``.

    Args:
        data_dir (Path): Root directory that holds upload job folders.
        job_id (str): Upload directory UUID.

    Returns:
        Path: Expected path to ``colmap/fused.glb``.

    Raises:
        ValueError: If ``job_id`` is not a valid UUID.
    """
    return colmap_glb_path(data_dir, job_id, FUSED_GLB)


def list_finished_reconstructions(
    data_dir: Path,
    filename: str = RECONSTRUCTED_GLB,
    url_name: str = "glb",
) -> list[dict[str, str]]:
    """List job directories that already have the requested COLMAP GLB.

    Args:
        data_dir (Path): Root directory that holds upload job folders.
        filename (str): GLB filename under ``colmap/``.
        url_name (str): Final path segment used in ``glb_url``.

    Returns:
        list[dict[str, str]]: Finished jobs with ``job_id``, ``glb_url``, and
        ``modified_at``, newest first.
    """
    if not data_dir.is_dir():
        return []

    items: list[dict[str, str]] = []
    for child in data_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            job_id = parse_job_id(child.name)
        except ValueError:
            continue
        glb_path = colmap_glb_path(data_dir, job_id, filename)
        if not glb_path.is_file():
            continue
        modified_at = datetime.fromtimestamp(
            glb_path.stat().st_mtime,
            tz=timezone.utc,
        ).isoformat()
        items.append(
            {
                "job_id": job_id,
                "glb_url": f"/api/reconstructions/{job_id}/{url_name}",
                "modified_at": modified_at,
            }
        )

    items.sort(key=lambda item: item["modified_at"], reverse=True)
    return items


def serve_colmap_glb(data_dir: Path, job_id: str, filename: str) -> FileResponse:
    """Return a COLMAP GLB file for ``job_id`` if it exists.

    Args:
        data_dir (Path): Root directory that holds upload job folders.
        job_id (str): Upload directory UUID under ``data/``.
        filename (str): GLB filename under ``colmap/``.

    Returns:
        FileResponse: Binary GLB file for the 3D viewer.

    Raises:
        HTTPException: 400 if ``job_id`` is not a UUID, 404 if the GLB is
            missing.
    """
    try:
        glb_path = colmap_glb_path(data_dir, job_id, filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid job id") from exc

    if not glb_path.is_file():
        raise HTTPException(status_code=404, detail="Reconstruction not found")

    return FileResponse(glb_path, media_type="model/gltf-binary")
