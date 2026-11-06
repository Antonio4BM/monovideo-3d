"""Redis and RQ connection helpers."""

from typing import Optional

from redis import Redis
from rq import Queue

from app.settings import get_settings


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
    settings = get_settings()
    return Redis(
        host=host or settings.redis_host,
        port=port if port is not None else settings.redis_port,
        db=db if db is not None else settings.redis_db,
    )


def get_queue(connection: Optional[Redis] = None) -> Queue:
    """Return the ``monovideo-3d`` RQ queue.

    Args:
        connection (Redis | None): Optional Redis client; created if omitted.

    Returns:
        Queue: RQ queue named ``monovideo-3d`` with a 55-minute default timeout.
    """
    conn = connection or get_redis_connection()
    settings = get_settings()
    return Queue(
        settings.queue_name, connection=conn, default_timeout=settings.job_timeout,
    )
