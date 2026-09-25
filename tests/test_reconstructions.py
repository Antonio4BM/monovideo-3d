"""Tests for reconstruction API routes."""

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.reconstruction_helpers import FUSED_GLB

client = TestClient(app)

FINISHED_JOB = "11111111-1111-1111-1111-111111111111"
OLDER_JOB = "22222222-2222-2222-2222-222222222222"


def _write_glb(
    root: Path,
    job_id: str,
    content: bytes = b"glb-bytes",
    filename: str = "reconstructed.glb",
) -> Path:
    glb_path = root / job_id / "colmap" / filename
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    glb_path.write_bytes(content)
    return glb_path


def test_reconstructions_page_serves_html():
    response = client.get("/reconstructions")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert b"Finished reconstructions" in response.content
    assert b"Reconstructed" in response.content
    assert b"Fused" in response.content


def test_api_lists_finished_reconstructions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_glb(tmp_path, FINISHED_JOB)
    _write_glb(tmp_path, OLDER_JOB, filename=FUSED_GLB)

    with patch("app.main.DATA_DIR", tmp_path):
        response = client.get("/api/reconstructions")

    assert response.status_code == 200
    payload = response.json()
    assert payload["reconstructions"][0]["job_id"] == FINISHED_JOB
    assert payload["fused"][0]["job_id"] == OLDER_JOB
    assert payload["fused"][0]["glb_url"] == f"/api/reconstructions/{OLDER_JOB}/fused"


def test_api_serves_reconstructed_glb(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_glb(tmp_path, FINISHED_JOB, b"glb-content")

    with patch("app.main.DATA_DIR", tmp_path):
        response = client.get(f"/api/reconstructions/{FINISHED_JOB}/glb")

    assert response.status_code == 200
    assert response.content == b"glb-content"
    assert response.headers["content-type"].startswith("model/gltf-binary")


def test_api_glb_rejects_invalid_job_id():
    response = client.get("/api/reconstructions/not-a-uuid/glb")
    assert response.status_code == 400


def test_api_glb_returns_404_when_missing(tmp_path):
    with patch("app.main.DATA_DIR", tmp_path):
        response = client.get(f"/api/reconstructions/{FINISHED_JOB}/glb")

    assert response.status_code == 404


def test_api_serves_fused_glb(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_glb(tmp_path, FINISHED_JOB, b"fused-content", filename=FUSED_GLB)

    with patch("app.main.DATA_DIR", tmp_path):
        response = client.get(f"/api/reconstructions/{FINISHED_JOB}/fused")

    assert response.status_code == 200
    assert response.content == b"fused-content"
    assert response.headers["content-type"].startswith("model/gltf-binary")


def test_api_fused_glb_rejects_invalid_job_id():
    response = client.get("/api/reconstructions/not-a-uuid/fused")
    assert response.status_code == 400


def test_api_fused_glb_returns_404_when_missing(tmp_path):
    with patch("app.main.DATA_DIR", tmp_path):
        response = client.get(f"/api/reconstructions/{FINISHED_JOB}/fused")

    assert response.status_code == 404
