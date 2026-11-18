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
│       ├── features_extractor.py          # COLMAP SIFT or OpenCV ORB/AKAZE extraction
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
├── requirements-api.txt            # FastAPI service dependencies
├── requirements-worker.txt         # Reconstruction worker dependencies
├── package.json                    # Frontend test tooling (Vitest)
└── vitest.config.js
```

## Installation

### Prerequisites

- Docker and Docker Compose with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (recommended)
- Or locally: NVIDIA GPU + drivers, [COLMAP](https://colmap.github.io/) with CUDA, Python 3, Redis, and Node.js (only for frontend tests)

### Python packages

Across `requirements-api.txt` and `requirements-worker.txt`:

- `fastapi`, `uvicorn`, `python-multipart` — API and uploads
- `boto3` — S3 video storage using the selected AWS profile
- `pydantic-settings` — typed application configuration from Compose environment variables
- `opencv-python` — frame extraction
- `open3d`, `trimesh`, `numpy` — 3D I/O / conversion
- `redis`, `rq` — job queue (upload → background reconstruction)
- `pytest`, `pytest-cov`, `httpx` — API and helper tests

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-api.txt -r requirements-worker.txt
```

## Usage

### Docker (recommended)

Set these variables in the project-root `.env` file (see `.env.example`):

```dotenv
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_PROFILE=default
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET=your-video-bucket
DYNAMODB_TABLE_NAME=your-jobs-table
DATA_DIR=data
```

The API credentials need `s3:PutObject` on the bucket's objects. The worker
credentials need `s3:GetObject` and `s3:PutObject`. See [AWS
infrastructure and worker permissions](#aws-infrastructure-and-worker-permissions)
for the table and worker IAM policy.
Compose passes these values
to the API and worker, where `app/settings.py` validates and caches them. Compose
creates the profile's region configuration inside the container. No host AWS
directory is mounted or copied. `.env` is excluded from Git and Docker builds.

```bash
docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000). The upload endpoint stores the
video only at `s3://<S3_BUCKET>/<job_id>/video.mp4`, enqueues reconstruction after
the upload succeeds, and returns its job ID. The worker downloads the video to
`data/<job_id>/video.mp4` before frame extraction. Both sparse and dense
reconstruction use this download step.
After publication succeeds, the worker removes the local `video.mp4`, the
original `colmap/database.db`, ORB or AKAZE `.npz` files, and the entire local
`artifacts/` directory. Frames and reconstructed models remain for the gallery
and viewer. If reconstruction, preparation, or publication fails, the RQ job
fails and all local files created so far are retained.

After a successful sparse or dense pipeline, the worker saves a consistent
SQLite backup at `data/<job_id>/artifacts/database.db` and a JSON summary at
`data/<job_id>/artifacts/summary.json`. ORB and AKAZE jobs also save
`data/<job_id>/artifacts/features.tar`, containing
`colmap/features/<algorithm>/` and its `.npz` files. SIFT jobs have no feature
archive. These local files remain available if publication fails.
The summary records the job ID, reconstruction mode, feature algorithm,
completion time in UTC, elapsed seconds through pipeline completion, and local
artifact paths. It counts images and sums keypoint and match rows from the
SQLite snapshot. SIFT descriptor totals come from SQLite; ORB and AKAZE totals
come from their NPZ descriptor arrays. The median triangulation angle is read
from SQLite when available and is otherwise `null`.
The worker uploads the snapshot to `<job_id>/database.db`, then uploads
the optional archive to `<job_id>/features.tar`. It writes one DynamoDB
summary for the requested reconstruction only after the uploads succeed. The
item includes each uploaded S3 key and object version ID. Set
`DYNAMODB_TABLE_NAME` to the Terraform output shown below.

To retry publication without running reconstruction, execute this command from
the project root with the job's UUID:

```bash
JOB_ID=123e4567-e89b-12d3-a456-426614174000
docker compose exec worker python -m app.queue.retry_publish "$JOB_ID"
```

The same `python -m app.queue.retry_publish "$JOB_ID"` command works in a configured
local worker environment. It reads the saved summary and artifacts under
`DATA_DIR`, uploads the files again, and replaces the DynamoDB item for that
job. The item records the new S3 object version IDs. After that write succeeds,
the retry command performs the same local cleanup as the worker.

### AWS infrastructure and worker permissions

The Terraform configuration in `infra/` manages the existing S3 bucket and an
on-demand DynamoDB jobs table. The table uses a string `job_id` partition key.
After applying the configuration, get its name with
`terraform -chdir=infra output -raw dynamodb_table_name`.

Grant the worker `s3:GetObject` and `s3:PutObject` on
`arn:aws:s3:::<bucket-name>/*`. For job records in DynamoDB, grant
`dynamodb:PutItem` on
`arn:aws:dynamodb:<region>:<account-id>:table/<table-name>`. Scope both
resources to the bucket and table managed by `infra/`. The bucket must have
versioning enabled so S3 returns object version IDs. The API's S3 upload
permission is separate from the worker permissions.

Finished models appear at [http://localhost:8000/reconstructions](http://localhost:8000/reconstructions). Sparse jobs produce `colmap/reconstructed.glb`; dense jobs also produce `colmap/fused.glb`.

### Local API

```bash
source venv/bin/activate
set -a
source .env
set +a
# Direct credentials need no named profile outside the Compose container.
unset AWS_PROFILE AWS_DEFAULT_PROFILE
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

All feature algorithms match each frame against the next 12 extracted frames in numerical filename order. SIFT uses an explicit COLMAP pair list; ORB and AKAZE use the same window with binary descriptors. Quadratic neighbors and loop-detection pairs are not added. Use a fresh job to evaluate this pairing policy: matching an existing database does not remove its previously stored pairs.

## Stack

- **Backend:** FastAPI, Uvicorn
- **Queue:** Redis, RQ
- **3D / vision:** COLMAP, OpenCV, Open3D, Trimesh, NumPy
- **Frontend:** Vanilla HTML / CSS / JS (uploader + reconstructions gallery, model-viewer)
- **Infra:** Docker, NVIDIA GPU
- **Tests:** Pytest (API, helpers, queue), Vitest (uploader and gallery)
