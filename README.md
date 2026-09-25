# monovideo3d

## Overview

Demo app for video uploading and 3D reconstruction from a single (monocular) video. A FastAPI service serves a landing-page uploader that accepts short videos, then enqueues a COLMAP-based pipeline (sparse or dense) on Redis/RQ. Finished models are listed on a reconstructions page and opened in a 3D viewer (`reconstructed.glb` and, for dense jobs, `fused.glb`).

An NVIDIA GPU is required (dense reconstruction uses COLMAP `patch_match_stereo` with GPU index `0`; Docker Compose reserves NVIDIA devices).

## Project structure

```
monovideo3d/
├── app/
│   ├── main.py                     # FastAPI routes: upload, reconstructions page, GLB APIs
│   ├── reconstruction_helpers.py   # UUID/path helpers, listing, and GLB file serving
│   ├── reconstruction.py           # CLI entrypoint for COLMAP pipeline steps
│   ├── queue/
│   │   ├── jobs.py                 # Enqueue sparse/dense reconstruction workers
│   │   ├── connection.py           # Redis / RQ connection
│   │   └── settings.py             # Queue name, timeout, Redis host
│   └── pipeline/
│       ├── extract_frames.py              # Sample frames from video (OpenCV)
│       ├── colmap_features_extractor.py   # COLMAP feature extraction
│       ├── features_matcher.py            # COLMAP feature matching
│       ├── colmap_mapper.py               # Sparse SfM mapping
│       ├── colmap_reconstructor.py        # Sparse model → PLY
│       ├── image_undistorter.py           # Undistort images for dense stage
│       ├── patch_match_stereo.py          # Dense depth (GPU)
│       ├── stereo_fusion.py               # Fuse depth maps → dense cloud
│       └── visualize.py                   # PLY → GLB via trimesh
├── static/
│   ├── index.html                  # Upload landing page
│   ├── index.css                   # Upload page styles
│   ├── index.js                    # UI wiring (pick / drag-drop / upload)
│   ├── uploader.js                 # Validation + POST helpers
│   ├── reconstructions.html        # Finished reconstructions gallery
│   ├── reconstructions.css         # Gallery + viewer styles
│   ├── reconstructions.js          # Gallery page wiring
│   ├── gallery.js                  # Fetch catalog, render lists, load model-viewer
│   └── tests/                      # Vitest tests for uploader.js and gallery.js
├── tests/
│   ├── test_reconstructions.py           # Reconstruction API route tests
│   ├── test_reconstruction_helpers.py    # Path / listing helper unit tests
│   ├── test_main_upload.py               # Upload endpoint tests
│   └── test_queue_*.py                   # Queue settings, connection, and jobs
├── data/                           # Uploaded videos and COLMAP outputs (gitignored)
├── Dockerfile                      # COLMAP base image + Python venv + uvicorn
├── docker-compose.yaml             # API, Redis, and RQ worker (GPU)
├── requirements.txt                # Python dependencies
├── package.json                    # Frontend test tooling (Vitest)
└── vitest.config.js
```

## Installation

### Prerequisites

- Docker and Docker Compose with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (recommended)
- Or locally: NVIDIA GPU + drivers, [COLMAP](https://colmap.github.io/) with CUDA, Python 3, Redis, and Node.js (only for frontend tests)

### Python packages

From `requirements.txt`:

- `fastapi`, `uvicorn`, `python-multipart` — API and uploads
- `opencv-python` — frame extraction
- `open3d`, `trimesh`, `numpy` — 3D I/O / conversion
- `redis`, `rq` — job queue (upload → background reconstruction)
- `pytest`, `pytest-cov`, `httpx` — API and helper tests

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Frontend tests (optional)

```bash
npm install
```

## Usage

### Docker (recommended)

```bash
docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000). Upload a video between **20 and 40 seconds** (client-side validation), choose a frame stride and sparse or dense reconstruction. The file is saved as `data/<uuid>/video.mp4` and a worker runs the pipeline.

Finished models appear at [http://localhost:8000/reconstructions](http://localhost:8000/reconstructions). Sparse jobs produce `colmap/reconstructed.glb`; dense jobs also produce `colmap/fused.glb`.

### Local API

```bash
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Start Redis, then an RQ worker (55-minute job timeout; default worker heartbeat):

```bash
rq worker monovideo-3d --url redis://localhost:6379/0
```

### Reconstruction API

- `GET /reconstructions` — HTML gallery with reconstructed and fused lists
- `GET /api/reconstructions` — JSON catalog (`job_id`, `glb_url`, `modified_at`)
- `GET /api/reconstructions/{job_id}/glb` — serve `reconstructed.glb`
- `GET /api/reconstructions/{job_id}/fused` — serve `fused.glb`

Path helpers and listing live in `app/reconstruction_helpers.py`; `app/main.py` only defines the routes.

### Reconstruction CLI

Run steps from the project root with the venv active and COLMAP on `PATH`:

```bash
# Full sparse pipeline (frames → features → matches → map → PLY → GLB)
python -m app.reconstruction colmap-sparse \
  --video_path data/<uuid>/video.mp4 \
  --output_path data/<uuid>/frames \
  --strides 20

# Dense pipeline (after sparse; needs GPU)
python -m app.reconstruction colmap-dense --images_path data/<uuid>/frames
```

Individual commands: `extract-frames`, `colmap-features`, `colmap-matches`, `colmap-mapper`, `colmap-reconstructor`, `colmap-visualizer`, `colmap-fusion`, `colmap-dense`, `colmap-sparse`.

### Tests

```bash
# Backend (API routes, reconstruction helpers, queue)
pytest

# Frontend (uploader + gallery)
npm test
```

## Stack

- **Backend:** FastAPI, Uvicorn
- **Queue:** Redis, RQ
- **3D / vision:** COLMAP, OpenCV, Open3D, Trimesh, NumPy
- **Frontend:** Vanilla HTML / CSS / JS (uploader + reconstructions gallery, model-viewer)
- **Infra:** Docker, NVIDIA GPU
- **Tests:** Pytest (API, helpers, queue), Vitest (uploader and gallery)
