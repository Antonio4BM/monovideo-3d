"""Configuration for the Redis job queue."""

import os

QUEUE_NAME = "monovideo-3d"
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
DEFAULT_STRIDES = int(os.getenv("STRIDES", "20"))
DATA_DIR = "data"


def redis_url() -> str:
    """Build the Redis connection URL from environment settings.

    Returns:
        str: Redis URL in the form ``redis://host:port/db``.
    """
    return f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
