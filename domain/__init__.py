"""
Domain models for ATLAS pipeline.

Pure data structures with validation, no business logic.
"""

from .events import EventChunk, EventBatch
from .metadata import FileMetadata, ReleaseMetadata
from .statistics import ParsingStatistics, ChunkStatistics
from .config import (
    PipelineConfig,
    ParsingConfig,
    TaskConfig,
    MassCalculationConfig,
    PostProcessingConfig,
    HistogramCreationConfig,
)

__all__ = [
    "EventChunk",
    "EventBatch",
    "FileMetadata",
    "ReleaseMetadata",
    "ParsingStatistics",
    "ChunkStatistics",
    "PipelineConfig",
    "ParsingConfig",
    "TaskConfig",
    "MassCalculationConfig",
    "PostProcessingConfig",
    "HistogramCreationConfig",
]
