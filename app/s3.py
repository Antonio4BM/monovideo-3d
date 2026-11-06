"""Store uploaded videos in Amazon S3 using the configured AWS profile."""

import atexit
import os
import shutil
from pathlib import Path
from threading import Lock
from typing import Any

import boto3
from pydantic import SecretStr

from app.settings import Settings, get_settings

_client: Any = None
_client_lock = Lock()


def _reset_client_after_fork() -> None:
    """Discard inherited client and lock state in a forked worker process."""
    global _client, _client_lock
    _client = None
    _client_lock = Lock()


def close_s3_client() -> None:
    """Close this process's shared S3 client after operations have finished."""
    global _client
    with _client_lock:
        if _client is not None:
            _client.close()
            _client = None


atexit.register(close_s3_client)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_client_after_fork)


def _secret_value(secret: SecretStr) -> str | None:
    """Return a configured secret as plain text for the AWS SDK.

    Args:
        secret (SecretStr): Protected setting value.

    Returns:
        str | None: Plain secret value, or ``None`` when it is empty.
    """
    return secret.get_secret_value() or None


def _create_s3_client(settings: Settings) -> Any:
    """Create an S3 client from validated application settings.

    Args:
        settings (Settings): Application settings loaded from the environment.

    Returns:
        Any: Configured boto3 S3 client.

    Raises:
        ValueError: If the bucket or required credentials are missing.
    """
    bucket = settings.s3_bucket.strip()
    access_key = _secret_value(settings.aws_access_key_id)
    secret_key = _secret_value(settings.aws_secret_access_key)
    if not bucket:
        raise ValueError("S3_BUCKET must be configured")
    if not access_key or not secret_key:
        raise ValueError(
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be configured"
        )

    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=settings.aws_default_region,
        profile_name=settings.aws_profile,
    )
    return session.client("s3")


def _get_s3_client(settings: Settings) -> Any:
    """Return the shared client, initializing it once per process.

    Args:
        settings (Settings): Configuration used when creating the client.

    Returns:
        Any: S3 client shared by this process's upload and download threads.
    """
    global _client
    with _client_lock:
        if _client is None:
            _client = _create_s3_client(settings)
        return _client


def put_object(key: str, body: bytes) -> dict[str, Any]:
    """Upload video bytes to the configured S3 bucket.

    Credentials come from AWS environment variables supplied by Compose from
    ``.env``. Compose creates the selected profile's region configuration
    inside the container.

    Args:
        key (str): Object key, such as ``<job_id>/video.mp4``.
        body (bytes): Uploaded video contents.

    Returns:
        dict[str, Any]: S3 put_object response metadata.

    Raises:
        ValueError: If the S3 bucket or AWS credentials are not configured.
        botocore.exceptions.BotoCoreError: If AWS configuration is invalid.
        botocore.exceptions.ClientError: If S3 rejects the upload.
    """
    settings = get_settings()
    client = _get_s3_client(settings)
    return client.put_object(
        Bucket=settings.s3_bucket,
        Key=key,
        Body=body,
        ContentType="video/mp4",
    )


def get_object(key: str, destination: Path) -> None:
    """Download an S3 video to the worker's local reconstruction directory.

    Args:
        key (str): Object key, such as ``<job_id>/video.mp4``.
        destination (Path): Local path where the worker needs the video.

    Raises:
        ValueError: If the S3 bucket or AWS credentials are not configured.
        botocore.exceptions.BotoCoreError: If AWS configuration is invalid.
        botocore.exceptions.ClientError: If S3 rejects the download.
        OSError: If the local directory or file cannot be written.
    """
    settings = get_settings()
    client = _get_s3_client(settings)
    response = client.get_object(Bucket=settings.s3_bucket, Key=key)
    body = response["Body"]
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as video:
            shutil.copyfileobj(body, video)
    finally:
        body.close()
