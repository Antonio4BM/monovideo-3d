"""Redis and RQ connection helpers."""

from typing import Optional

from redis import Redis
from rq import Queue

from app.queue.settings import JOB_TIMEOUT, QUEUE_NAME, REDIS_DB, REDIS_HOST, REDIS_PORT


def get_redis_connection(
    host: Optional[str] = None,
    port: Optional[int] = None,
    db: Optional[int] = None,
) -> Redis:
    """Create a Redis client for the job queue broker.

    Args:
        host (str | None): Redis hostname; defaults to ``REDIS_HOST``.
        port (int | None): Redis port; defaults to ``REDIS_PORT``.
        db (int | None): Redis database index; defaults to ``REDIS_DB``.

    Returns:
        Redis: Connected Redis client instance.
    """
    return Redis(
        host=host or REDIS_HOST,
        port=port if port is not None else REDIS_PORT,
        db=db if db is not None else REDIS_DB,
    )


def get_queue(connection: Optional[Redis] = None) -> Queue:
    """Return the ``monovideo-3d`` RQ queue.

    Args:
        connection (Redis | None): Optional Redis client; created if omitted.

    Returns:
        Queue: RQ queue named ``monovideo-3d`` with a 55-minute default timeout.
    """
    conn = connection or get_redis_connection()
    return Queue(QUEUE_NAME, connection=conn, default_timeout=JOB_TIMEOUT)
