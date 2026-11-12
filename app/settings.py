"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed configuration shared by the API and reconstruction worker."""

    s3_bucket: str = ""
    aws_profile: str = ""
    aws_access_key_id: SecretStr = SecretStr("")
    aws_secret_access_key: SecretStr = SecretStr("")
    aws_default_region: str = "us-east-1"
    data_directory: Path = Field(default=Path("data"), validation_alias="DATA_DIR")
    queue_name: str = Field(default="monovideo-3d", min_length=1)
    redis_host: str = Field(default="localhost", min_length=1)
    redis_port: int = Field(default=6379, ge=1, le=65535)
    redis_db: int = Field(default=0, ge=0)
    job_timeout: int = Field(default=3300, gt=0)

    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    @field_validator("s3_bucket", "aws_profile", "aws_default_region", mode="before")
    @classmethod
    def strip_text_settings(cls, value: object) -> object:
        """Remove surrounding whitespace from text configuration values.

        Args:
            value (object): Raw value supplied by the environment.

        Returns:
            object: Normalized string or the original non-string value.
        """
        return value.strip() if isinstance(value, str) else value


@lru_cache
def get_settings() -> Settings:
    """Load and cache application settings from the process environment.

    Returns:
        Settings: Validated application configuration.
    """
    return Settings()
