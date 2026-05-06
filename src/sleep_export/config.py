"""Application settings, populated from environment variables.

Environment variables are read once at process start. Override at runtime
by passing the relevant Settings argument explicitly (e.g. in tests).
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide configuration."""

    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False, extra="ignore")

    data_dir: Path = Field(default=Path("data"))
    """Directory for sleep.db + last_upload.zip. Mounted as a docker volume."""

    sleep_date_cutoff_hour: int = Field(default=15, ge=0, le=23)
    """Local hour-of-day boundary for assigning records to sleep_dates."""

    bind_host: str = Field(default="0.0.0.0")  # noqa: S104
    bind_port: int = Field(default=8000, ge=1, le=65535)
    no_cdn: bool = Field(default=False)
    """If true, expect Plotly + Tailwind to be vendored locally under static/.
    The default is to load them from CDNs; users on air-gapped networks can
    flip this and provide local copies."""

    @property
    def db_path(self) -> Path:
        """SQLite database file path inside the data directory."""
        return self.data_dir / "sleep.db"

    @property
    def upload_path(self) -> Path:
        """Path of the most recently uploaded zip (single-slot, overwritten)."""
        return self.data_dir / "last_upload.zip"


def get_settings() -> Settings:
    """Construct a fresh Settings instance from the current environment.

    Kept as a function (rather than a module-level singleton) so tests can
    monkeypatch `os.environ` and get a fresh view.
    """
    return Settings()
