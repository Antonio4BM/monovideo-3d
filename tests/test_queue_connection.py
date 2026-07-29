"""Tests for Redis / RQ connection helpers."""

from unittest.mock import MagicMock, patch

from app.queue.connection import get_queue, get_redis_connection
from app.queue.settings import QUEUE_NAME


def test_get_redis_connection_uses_defaults():
    with patch("app.queue.connection.Redis") as redis_cls:
        redis_cls.return_value = MagicMock(name="redis")
        conn = get_redis_connection()

    redis_cls.assert_called_once_with(host="localhost", port=6379, db=0)
    assert conn is redis_cls.return_value


def test_get_redis_connection_overrides():
    with patch("app.queue.connection.Redis") as redis_cls:
        redis_cls.return_value = MagicMock(name="redis")
        get_redis_connection(host="redis", port=6380, db=1)

    redis_cls.assert_called_once_with(host="redis", port=6380, db=1)


def test_get_queue_uses_named_queue():
    fake_conn = MagicMock(name="redis")
    with patch("app.queue.connection.Queue") as queue_cls:
        queue_cls.return_value = MagicMock(name="queue")
        queue = get_queue(connection=fake_conn)

    queue_cls.assert_called_once_with(QUEUE_NAME, connection=fake_conn)
    assert queue is queue_cls.return_value


def test_get_queue_creates_connection_when_omitted():
    fake_conn = MagicMock(name="redis")
    with patch("app.queue.connection.get_redis_connection", return_value=fake_conn) as get_conn:
        with patch("app.queue.connection.Queue") as queue_cls:
            queue_cls.return_value = MagicMock(name="queue")
            get_queue()

    get_conn.assert_called_once_with()
    queue_cls.assert_called_once_with(QUEUE_NAME, connection=fake_conn)
