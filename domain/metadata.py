"""
Metadata-related domain models.

Immutable data structures representing file and release metadata.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass(frozen=True)
class FileMetadata:
    """Metadata for a single parsed file."""
    
    file_id: int
    release_year: str
    size_mb: float
    event_count: int
    processing_time_sec: float
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate file metadata."""
        if self.size_mb < 0:
            raise ValueError(f"size_mb must be non-negative, got {self.size_mb}")
        if self.event_count < 0:
            raise ValueError(f"event_count must be non-negative, got {self.event_count}")
        if self.processing_time_sec < 0:
            raise ValueError(f"processing_time_sec must be non-negative, got {self.processing_time_sec}")
        if not self.success and not self.error_message:
            raise ValueError("error_message must be provided when success=False")


@dataclass(frozen=True)
class ReleaseMetadata:
    """Metadata for a release year."""
    
    release_year: str
    file_ids: tuple[int, ...]  # Immutable tuple
    total_files: int
    fetched_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate release metadata."""
        if self.total_files != len(self.file_ids):
            raise ValueError(
                f"total_files ({self.total_files}) must match length of file_ids ({len(self.file_ids)})"
            )
        if self.total_files <= 0:
            raise ValueError(f"total_files must be positive, got {self.total_files}")
    
    @classmethod
    def from_file_list(cls, release_year: str, file_ids: list[int]) -> 'ReleaseMetadata':
        """Create ReleaseMetadata from a list of file IDs."""
        return cls(
            release_year=release_year,
            file_ids=tuple(file_ids),  # Convert to immutable tuple
            total_files=len(file_ids)
        )
