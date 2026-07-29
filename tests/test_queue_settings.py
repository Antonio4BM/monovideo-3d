"""Tests for Redis queue settings."""

import importlib

import app.queue.settings as settings


def test_queue_name_is_monovideo_3d():
    assert settings.QUEUE_NAME == "monovideo-3d"


def test_default_strides():
    assert settings.DEFAULT_STRIDES == 20


def test_redis_url_defaults(monkeypatch):
    monkeypatch.delenv("REDIS_HOST", raising=False)
    monkeypatch.delenv("REDIS_PORT", raising=False)
    monkeypatch.delenv("REDIS_DB", raising=False)
    importlib.reload(settings)

    assert settings.REDIS_HOST == "localhost"
    assert settings.REDIS_PORT == 6379
    assert settings.REDIS_DB == 0
    assert settings.redis_url() == "redis://localhost:6379/0"


def test_redis_url_from_environment(monkeypatch):
    monkeypatch.setenv("REDIS_HOST", "redis")
    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setenv("REDIS_DB", "2")
    importlib.reload(settings)

    assert settings.redis_url() == "redis://redis:6380/2"

    monkeypatch.delenv("REDIS_HOST", raising=False)
    monkeypatch.delenv("REDIS_PORT", raising=False)
    monkeypatch.delenv("REDIS_DB", raising=False)
    importlib.reload(settings)
