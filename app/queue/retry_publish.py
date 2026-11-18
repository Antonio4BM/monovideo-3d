"""Worker command for republishing a saved reconstruction job."""

import argparse
import json
import uuid

from app.queue.artifacts import cleanup_published_artifacts
from app.queue.publish import publish_artifacts
from app.settings import get_settings


def main(argv: list[str] | None = None) -> None:
    """Republish a saved job without running reconstruction.

    Args:
        argv (list[str] | None): CLI arguments, or process arguments if omitted.

    Raises:
        FileNotFoundError: If the saved summary or an artifact is absent.
        botocore.exceptions.ClientError: If S3 or DynamoDB rejects a write.
    """
    parser = argparse.ArgumentParser(
        description="Republish saved reconstruction artifacts for a job",
    )
    parser.add_argument("job_id", help="UUID of the completed reconstruction")
    args = parser.parse_args(argv)
    try:
        job_id = str(uuid.UUID(args.job_id))
    except ValueError:
        parser.error("job_id must be a UUID")
    summary_path = (
        get_settings().data_directory / job_id / "artifacts" / "summary.json"
    )
    with summary_path.open(encoding="utf-8") as summary_file:
        summary = json.load(summary_file)
    if summary["job_id"] != job_id:
        raise ValueError("Saved summary job_id does not match the requested job")
    publish_artifacts(summary_path)
    cleanup_published_artifacts(
        summary_path.parent.parent, summary["feature_extractor_algorithm"],
    )


if __name__ == "__main__":
    main()
