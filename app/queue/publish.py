"""Publish prepared reconstruction artifacts and their DynamoDB summary."""

import json
import logging
from decimal import Decimal
from pathlib import Path

from app.aws import get_aws_session
from app.s3 import stream_file_to_s3
from app.settings import get_settings

logger = logging.getLogger(__name__)


def publish_artifacts(summary_path: Path) -> None:
    """Upload local artifacts in order, then store their versioned summary.

    Args:
        summary_path (Path): Local JSON summary written by preparation.

    Raises:
        ValueError: If the jobs table or an S3 version ID is missing.
        FileNotFoundError: If a required local artifact is absent.
        botocore.exceptions.ClientError: If S3 or DynamoDB rejects a write.
    """
    settings = get_settings()
    if not settings.dynamodb_table_name:
        raise ValueError("DYNAMODB_TABLE_NAME must be configured")

    with summary_path.open(encoding="utf-8") as summary_file:
        summary = json.load(summary_file, parse_float=Decimal)

    job_root = summary_path.parent.parent
    job_id = summary["job_id"]
    database_path = job_root / summary["database_snapshot"]
    if not database_path.is_file():
        raise FileNotFoundError(f"Database snapshot not found: {database_path}")
    archive_path = None
    if summary["feature_archive"] is not None:
        archive_path = job_root / summary["feature_archive"]
        if not archive_path.is_file():
            raise FileNotFoundError(f"Feature archive not found: {archive_path}")

    database_key = f"{job_id}/database.db"
    logger.info("Uploading database snapshot for job %s to %s", job_id, database_key)
    database_response = stream_file_to_s3(
        database_key, database_path, "application/vnd.sqlite3",
    )
    database_version = database_response.get("VersionId")
    if not database_version or database_version == "null":
        raise ValueError(f"S3 did not return a version ID for {database_key}")

    archive_key = None
    archive_version = None
    if archive_path is not None:
        archive_key = f"{job_id}/features.tar"
        logger.info("Uploading feature archive for job %s to %s", job_id, archive_key)
        archive_response = stream_file_to_s3(
            archive_key, archive_path, "application/x-tar",
        )
        archive_version = archive_response.get("VersionId")
        if not archive_version or archive_version == "null":
            raise ValueError(f"S3 did not return a version ID for {archive_key}")

    item = {
        key: value
        for key, value in summary.items()
        if key not in {"database_snapshot", "summary_path", "feature_archive"}
    }
    item.update({
        "database_s3_key": database_key,
        "database_s3_version_id": database_version,
        "feature_archive_s3_key": archive_key,
        "feature_archive_s3_version_id": archive_version,
    })
    dynamo_db = get_aws_session(settings).resource("dynamodb")
    try:
        dynamo_db.Table(settings.dynamodb_table_name).put_item(Item=item)
    finally:
        dynamo_db.meta.client.close()
    logger.info(
        "Finished uploading artifacts and saving summary for job %s",
        job_id,
    )
