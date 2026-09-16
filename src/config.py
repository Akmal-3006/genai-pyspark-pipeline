"""Shared configuration for the e-commerce data pipeline."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    """Filesystem settings used by both pipeline stages."""

    project_root: Path = PROJECT_ROOT

    @property
    def raw_data_dir(self) -> Path:
        """Return the directory containing generated input files."""
        return self.project_root / "data" / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """Return the directory containing Spark output tables."""
        return self.project_root / "data" / "processed"

    def ensure_directories(self) -> None:
        """Create pipeline data directories when they do not exist."""
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()