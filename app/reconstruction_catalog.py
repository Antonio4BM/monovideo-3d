"""Read published reconstruction metadata and locate its local models."""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import HTTPException

from app.aws import get_aws_session
from app.reconstruction_helpers import (
    FUSED_GLB,
    RECONSTRUCTED_GLB,
    colmap_glb_path,
    list_finished_reconstructions,
    parse_job_id,
)
from app.settings import get_settings

logger = logging.getLogger(__name__)

METADATA_FIELDS = (
    "job_id",
    "feature_extractor_algorithm",
    "strides",
    "completed_at",
    "reconstruction_duration_seconds",
    "image_count",
    "total_keypoints",
    "total_matches",
    "total_descriptors",
    "median_triangulation_angle_degrees",
)


def read_published_jobs() -> list[dict]:
    """Read every page of the configured DynamoDB jobs table.

    Returns:
        list[dict]: Published job records, including legacy metadata.

    Raises:
        HTTPException: 503 when configuration or DynamoDB access fails.
    """
    resource = None
    try:
        settings = get_settings()
        if not settings.dynamodb_table_name:
            raise HTTPException(
                status_code=503,
                detail="DYNAMODB_TABLE_NAME must be configured to load jobs.",
            )
        resource = get_aws_session(settings).resource("dynamodb")
        table = resource.Table(settings.dynamodb_table_name)
        jobs = []
        scan_options = {}
        while True:
            page = table.scan(**scan_options)
            jobs.extend(page.get("Items", []))
            last_key = page.get("LastEvaluatedKey")
            if not last_key:
                return jobs
            scan_options = {"ExclusiveStartKey": last_key}
    except (BotoCoreError, ClientError, ValueError) as exc:
        logger.exception("Unable to read the reconstruction jobs table")
        raise HTTPException(
            status_code=503,
            detail=(
                "Unable to load reconstruction metadata from DynamoDB. "
                "Check AWS configuration and table read permissions."
            ),
        ) from exc
    finally:
        if resource is not None:
            resource.meta.client.close()


def _completion_timestamp(job: dict) -> float:
    """Resolve a completion time for chronological catalog sorting.

    Args:
        job (dict): Job with an optional ISO-8601 completion timestamp.

    Returns:
        float: Unix timestamp, or negative infinity for missing/invalid dates.
    """
    try:
        completed = datetime.fromisoformat(job.get("completed_at") or "")
        if completed.tzinfo is None:
            completed = completed.replace(tzinfo=timezone.utc)
        return completed.timestamp()
    except (TypeError, ValueError, OverflowError):
        return float("-inf")


def list_reconstruction_jobs(data_dir: Path) -> list[dict]:
    """Combine published metadata with availability of local GLB models.

    Args:
        data_dir (Path): Root directory containing local job folders.

    Returns:
        list[dict]: JSON-ready metadata and nullable model URLs, newest first.
    """
    jobs = []
    for item in read_published_jobs():
        job = {field: item.get(field) for field in METADATA_FIELDS}
        for field, value in job.items():
            if isinstance(value, Decimal):
                integral = int(value)
                job[field] = integral if value == integral else float(value)
        job.update(sparse_glb_url=None, dense_glb_url=None)
        try:
            job_id = parse_job_id(job["job_id"])
        except (AttributeError, TypeError, ValueError):
            job_id = None
        if job_id is not None:
            for field, filename, endpoint in (
                ("sparse_glb_url", RECONSTRUCTED_GLB, "glb"),
                ("dense_glb_url", FUSED_GLB, "fused"),
            ):
                if colmap_glb_path(data_dir, job_id, filename).is_file():
                    job[field] = f"/api/reconstructions/{job_id}/{endpoint}"
        jobs.append(job)
    jobs.sort(key=_completion_timestamp, reverse=True)
    return jobs


def get_reconstruction_catalog(data_dir: Path) -> dict[str, list[dict]]:
    """Build the metadata catalog while preserving the local model lists.

    Args:
        data_dir (Path): Root directory containing local job folders.

    Returns:
        dict[str, list[dict]]: Published jobs and legacy local GLB lists.
    """
    return {
        "jobs": list_reconstruction_jobs(data_dir),
        "reconstructions": list_finished_reconstructions(data_dir),
        "fused": list_finished_reconstructions(
            data_dir, filename=FUSED_GLB, url_name="fused",
        ),
    }
