## Overview

App directory that contains the pipeline 3d reconstructions using Colmap. Redis queue server to handle the reconstructions asynchronous, and the API to upload video.

## Project structure

├── __init__.py
├── main.py
├── pipeline
├── __pycache__
├── queue
└── reconstruction.py

The reconstrucion should follow the microkernel architecture.

## Code style and stack

- You should use python 3.12 follow PEP 8. Every method should have their docstring following google's style.
- rq as queue server.
- colmap for 3d reconstruction pipeline, using both sparse and dense reconstructions.
- Use RQ's default worker heartbeat. Do not override ``--worker-ttl``.
- Set job timeout to 55 minutes (3300 seconds) on the queue and enqueued jobs.


## Bounderies

- always ask the user before installing a new package.

## Tests

- All methods should have their unit tests for pipeline, queue and API endpoints.