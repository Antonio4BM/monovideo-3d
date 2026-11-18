"""Share one configured boto3 session within each process."""

import os
from threading import Lock

import boto3

from app.settings import Settings

_session: boto3.Session | None = None
_session_lock = Lock()


def _reset_session_after_fork() -> None:
    """Discard the parent's session and lock in an RQ child process."""
    global _session, _session_lock
    _session = None
    _session_lock = Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_session_after_fork)


def get_aws_session(settings: Settings) -> boto3.Session:
    """Return the process's configured session for S3 and DynamoDB.

    Args:
        settings (Settings): AWS credentials, profile, and region.

    Returns:
        boto3.Session: Shared session for this process.

    Raises:
        ValueError: If AWS access key or secret key is missing.
    """
    global _session
    with _session_lock:
        if _session is None:
            access_key = settings.aws_access_key_id.get_secret_value()
            secret_key = settings.aws_secret_access_key.get_secret_value()
            if not access_key or not secret_key:
                raise ValueError(
                    "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be configured"
                )
            _session = boto3.Session(
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=settings.aws_default_region,
                profile_name=settings.aws_profile or None,
            )
        return _session
