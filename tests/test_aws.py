"""Tests for the process-local AWS session shared by service clients."""

from unittest.mock import MagicMock

from app.aws import _reset_session_after_fork, get_aws_session
from app.s3 import _create_s3_client
from app.settings import Settings


def test_s3_and_dynamodb_use_one_session_per_process(monkeypatch):
    """Reuse the configured session when creating both AWS service handles."""
    monkeypatch.setattr("app.aws._session", None)
    session = MagicMock()
    create_session = MagicMock(return_value=session)
    monkeypatch.setattr("app.aws.boto3.Session", create_session)
    settings = Settings(
        s3_bucket="bucket",
        aws_access_key_id="access-key",
        aws_secret_access_key="secret-key",
        aws_default_region="us-east-1",
    )

    s3_client = _create_s3_client(settings)
    dynamodb = get_aws_session(settings).resource("dynamodb")

    assert s3_client is session.client.return_value
    assert dynamodb is session.resource.return_value
    create_session.assert_called_once_with(
        aws_access_key_id="access-key",
        aws_secret_access_key="secret-key",
        region_name="us-east-1",
        profile_name=None,
    )


def test_fork_discards_parent_session(monkeypatch):
    """Make an RQ child create its own boto3 session after a fork."""
    monkeypatch.setattr("app.aws._session", None)
    parent_session = MagicMock()
    child_session = MagicMock()
    create_session = MagicMock(side_effect=[parent_session, child_session])
    monkeypatch.setattr("app.aws.boto3.Session", create_session)
    settings = Settings(
        aws_access_key_id="access-key",
        aws_secret_access_key="secret-key",
    )

    assert get_aws_session(settings) is parent_session
    _reset_session_after_fork()
    assert get_aws_session(settings) is child_session
    assert create_session.call_count == 2
