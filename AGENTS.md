## Overview
Monovideo that reconstructs 3d scene using Colmap.

## Project structure

The project should have the next structure:

├── app
├── data
├── docker-compose.yaml
├── Dockerfile
├── monovideo-env
├── node_modules
├── package.json
├── package-lock.json
├── README.md
├── requirements.txt
├── static
├── tests
└── vitest.config.js

## Code style and stack

You should use python 3.12 and follow the PEP 8. Every method should have their docstring following google's style.
For frontend you should use vanilla JS, split styles and scripts into different files.
Use Dockerfile and docker-compose to containarize the application.

## Queue worker

- Use RQ's default worker heartbeat. Do not override ``--worker-ttl``.
- Set job timeout to 55 minutes (3300 seconds) on the queue and enqueued jobs. RQ 2.x worker CLI has no ``--job-timeout``.

## Boundaries

- Every time you will use a new package for node or python that is not declared you should ask first.