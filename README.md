# monovideo3d

## Overview

Demo app for video uploading and 3D reconstruction from a single (monocular) video. A FastAPI service serves a landing-page uploader that accepts short videos, and a COLMAP-based CLI pipeline turns frames into sparse or dense 3D point clouds (exported as PLY / GLB).

**Current status:** the upload API and the reconstruction CLI are not connected yet—uploaded videos are stored under `data/`, but reconstruction must be run manually. A Redis queue broker is planned to link upload to background reconstruction jobs; it is listed in dependencies but not wired up.

An NVIDIA GPU is required (dense reconstruction uses COLMAP `patch_match_stereo` with GPU index `0`; Docker Compose reserves NVIDIA devices).

## Project structure

```
monovideo3d/
├── app/
│   ├── main.py                 # FastAPI app: POST /video-upload + static UI
│   ├── reconstruction.py       # CLI entrypoint for COLMAP pipeline steps
│   └── pipeline/
│       ├── extract_frames.py           # Sample frames from video (OpenCV)
│       ├── colmap_features_extractor.py # COLMAP feature extraction
│       ├── features_matcher.py         # COLMAP feature matching
│       ├── colmap_mapper.py            # Sparse SfM mapping
│       ├── colmap_reconstructor.py     # Sparse model → PLY
│       ├── image_undistorter.py        # Undistort images for dense stage
│       ├── patch_match_stereo.py       # Dense depth (GPU)
│       ├── stereo_fusion.py            # Fuse depth maps → dense cloud
│       └── visualize.py                # PLY → GLB via trimesh
├── static/
│   ├── index.html              # Upload landing page
│   ├── index.css               # Page styles
│   ├── index.js                # UI wiring (pick / drag-drop / upload)
│   ├── uploader.js             # Validation + POST helpers
│   └── tests/                  # Vitest tests for uploader.js
├── data/                       # Uploaded videos (gitignored)
├── Dockerfile                  # COLMAP base image + Python venv + uvicorn
├── docker-compose.yaml         # GPU service on port 8000
├── requirements.txt            # Python dependencies
├── package.json                # Frontend test tooling (Vitest)
└── vitest.config.js
```

## Installation

### Prerequisites

- Docker and Docker Compose with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (recommended)
- Or locally: NVIDIA GPU + drivers, [COLMAP](https://colmap.github.io/) with CUDA, Python 3, and Node.js (only for frontend tests)

### Python packages

From `requirements.txt`:

- `fastapi`, `uvicorn`, `python-multipart` — API and uploads
- `opencv-python` — frame extraction
- `open3d`, `trimesh`, `numpy` — 3D I/O / conversion
- `redis` — planned job queue broker (not integrated yet)

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

Open [http://localhost:8000](http://localhost:8000). Upload a video between **20 and 40 seconds** (client-side validation). The file is saved as `data/<uuid>/video.mp4`.

### Local API

```bash
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

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

### Frontend tests

```bash
npm test
```

## Stack

- **Backend:** FastAPI, Uvicorn
- **3D / vision:** COLMAP, OpenCV, Open3D, Trimesh, NumPy
- **Frontend:** Vanilla HTML / CSS / JS (drag-and-drop uploader)
- **Infra:** Docker, NVIDIA GPU
- **Planned:** Redis (queue broker to connect upload → reconstruction)
- **Tests:** Vitest (uploader unit tests)
