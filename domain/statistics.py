"""
Statistics-related domain models.

Immutable data structures for tracking parsing and processing statistics.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass(frozen=True)
class ChunkStatistics:
    """Statistics for a single chunk."""
    
    chunk_index: int
    size_bytes: int
    event_count: int
    file_count: int
    processing_time_sec: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate chunk statistics."""
        if self.chunk_index < 0:
            raise ValueError(f"chunk_index must be non-negative, got {self.chunk_index}")
        if self.size_bytes < 0:
            raise ValueError(f"size_bytes must be non-negative, got {self.size_bytes}")
        if self.event_count < 0:
            raise ValueError(f"event_count must be non-negative, got {self.event_count}")
        if self.file_count < 0:
            raise ValueError(f"file_count must be non-negative, got {self.file_count}")
    
    @property
    def size_mb(self) -> float:
        """Get size in megabytes."""
        return self.size_bytes / (1024 * 1024)


@dataclass(frozen=True)
class ParsingStatistics:
    """
    Comprehensive statistics for parsing operation.
    
    Immutable snapshot of parsing progress and results.
    """
    
    # Counts
    total_files: int
    successful_files: int
    failed_files: int
    total_events: int
    total_chunks: int
    
    # Sizes
    total_size_bytes: int
    max_chunk_size_bytes: int
    min_chunk_size_bytes: int
    
    # Memory
    max_memory_mb: float
    
    # Timing
    total_time_sec: float
    start_time: datetime
    end_time: datetime
    
    # Errors (immutable tuple of error messages)
    error_types: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    timeout_count: int = 0
    
    def __post_init__(self):
        """Validate parsing statistics."""
        if self.total_files < 0:
            raise ValueError(f"total_files must be non-negative, got {self.total_files}")
        if self.successful_files < 0:
            raise ValueError(f"successful_files must be non-negative, got {self.successful_files}")
        if self.failed_files < 0:
            raise ValueError(f"failed_files must be non-negative, got {self.failed_files}")
        if self.successful_files + self.failed_files != self.total_files:
            raise ValueError(
                f"successful_files ({self.successful_files}) + failed_files ({self.failed_files}) "
                f"must equal total_files ({self.total_files})"
            )
        if self.end_time < self.start_time:
            raise ValueError("end_time must be after start_time")
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.successful_files / self.total_files) * 100
    
    @property
    def average_events_per_file(self) -> float:
        """Calculate average events per successful file."""
        if self.successful_files == 0:
            return 0.0
        return self.total_events / self.successful_files
    
    @property
    def total_size_mb(self) -> float:
        """Get total size in megabytes."""
        return self.total_size_bytes / (1024 * 1024)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_files": self.total_files,
            "successful_files": self.successful_files,
            "failed_files": self.failed_files,
            "success_rate": f"{self.success_rate:.1f}%",
            "total_events": self.total_events,
            "total_chunks": self.total_chunks,
            "total_size_mb": f"{self.total_size_mb:.2f}",
            "max_memory_mb": f"{self.max_memory_mb:.1f}",
            "total_time_sec": f"{self.total_time_sec:.1f}",
            "average_events_per_file": f"{self.average_events_per_file:.0f}",
            "timeout_count": self.timeout_count,
            "error_types": {error_type: count for error_type, count in self.error_types},
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
        }
