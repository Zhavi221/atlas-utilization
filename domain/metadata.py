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
class MCDatasetMetadata:
    """
    Monte-Carlo dataset (DSID) metadata used to normalize simulated events.

    These fields are static per dataset (sample) and come from the ATLAS
    Open Data metadata (via atlasopenmagic ``get_metadata``). They feed the
    standard MC event-weight formula:

        w = (cross_section_pb * PB_TO_FB * kFactor * genFiltEff * L_fb) / sum_of_weights

    where ``L_fb`` is the target integrated luminosity (an analysis choice,
    not part of the dataset metadata).
    """

    dataset_number: int
    cross_section_pb: float           # generator cross section (sigma)
    sum_of_weights: float             # sum of per-event generator weights (normalization denominator)
    k_factor: float = 1.0             # higher-order correction; 1.0 when not provided
    gen_filt_eff: float = 1.0         # generator filter efficiency; 1.0 when not provided
    n_events: Optional[int] = None    # raw generated event count (cross-check only)
    physics_short: Optional[str] = None  # human-readable sample label
    generator: Optional[str] = None      # generator name (hints at weighted/negative events)

    def __post_init__(self):
        """Validate MC dataset metadata."""
        if self.cross_section_pb < 0:
            raise ValueError(f"cross_section_pb must be non-negative, got {self.cross_section_pb}")
        if self.sum_of_weights == 0:
            raise ValueError("sum_of_weights must be non-zero (cannot normalize a sample with zero effective size)")
        if self.k_factor < 0:
            raise ValueError(f"k_factor must be non-negative, got {self.k_factor}")
        if self.gen_filt_eff < 0:
            raise ValueError(f"gen_filt_eff must be non-negative, got {self.gen_filt_eff}")


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
