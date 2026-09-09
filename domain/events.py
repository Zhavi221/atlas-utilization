"""
Event-related domain models.

Immutable data structures representing parsed events and chunks.
"""

from dataclasses import dataclass, field
from typing import Optional
import awkward as ak
import numpy as np


def _empty_particle_collection(collection: ak.Array, event_count: int) -> ak.Array:
    """Build a typed jagged record collection containing no particles."""
    counts = np.zeros(event_count, dtype=np.int64)
    no_particles = ak.flatten(collection[:0])  # zero records, carrying ``collection``'s field types
    return ak.unflatten(no_particles, counts)


def _concatenate_events(arrays: list[ak.Array]) -> ak.Array:
    """
    Concatenate per-file event arrays, keeping collections only some files have.

    Concatenating the records directly builds a union type, which exposes only
    the fields common to *every* file - so one file without electrons would
    delete electrons from every event in the chunk. Concatenating one collection
    at a time keeps a plain record, with empty lists for the files that lack it.
    """
    example_of = {field: array[field] for array in arrays for field in array.fields}

    combined = {}
    for field, example in example_of.items():
        combined[field] = ak.concatenate([
            array[field] if field in array.fields else _empty_particle_collection(example, len(array))
            for array in arrays
        ])

    return ak.zip(combined, depth_limit=1)


@dataclass(frozen=True)
class EventBatch:
    """A batch of events from a single file."""
    
    events: ak.Array
    file_id: int
    release_year: str
    size_bytes: int
    event_count: int
    processing_time_sec: float
    
    def __post_init__(self):
        """Validate the event batch."""
        if self.event_count < 0:
            raise ValueError(f"event_count must be non-negative, got {self.event_count}")
        if self.size_bytes < 0:
            raise ValueError(f"size_bytes must be non-negative, got {self.size_bytes}")
        if self.processing_time_sec < 0:
            raise ValueError(f"processing_time_sec must be non-negative, got {self.processing_time_sec}")


@dataclass(frozen=True)
class EventChunk:
    """
    A chunk of accumulated events ready to be yielded.
    
    Represents multiple event batches accumulated until size threshold is reached.
    """
    
    events: ak.Array
    chunk_index: int
    release_year: str
    size_bytes: int
    event_count: int
    file_ids: tuple[int, ...]  # Use tuple for immutability
    
    def __post_init__(self):
        """Validate the event chunk."""
        if self.chunk_index < 0:
            raise ValueError(f"chunk_index must be non-negative, got {self.chunk_index}")
        if self.event_count < 0:
            raise ValueError(f"event_count must be non-negative, got {self.event_count}")
        if self.size_bytes < 0:
            raise ValueError(f"size_bytes must be non-negative, got {self.size_bytes}")
        if len(self.file_ids) == 0:
            raise ValueError("file_ids cannot be empty")
    
    @property
    def size_mb(self) -> float:
        """Get size in megabytes."""
        return self.size_bytes / (1024 * 1024)
    
    @classmethod
    def from_batches(
        cls,
        batches: list[EventBatch],
        chunk_index: int,
        release_year: str
    ) -> 'EventChunk':
        """
        Create an EventChunk from multiple EventBatches.
        
        Args:
            batches: List of event batches to combine
            chunk_index: Index of this chunk in the sequence
            release_year: Release year for this chunk
            
        Returns:
            EventChunk with concatenated events
        """
        if not batches:
            raise ValueError("Cannot create EventChunk from empty batch list")
        
        # Concatenate all events
        combined_events = _concatenate_events([batch.events for batch in batches])
        
        # Calculate totals
        total_size = sum(batch.size_bytes for batch in batches)
        total_events = sum(batch.event_count for batch in batches)
        file_ids = tuple(batch.file_id for batch in batches)
        
        return cls(
            events=combined_events,
            chunk_index=chunk_index,
            release_year=release_year,
            size_bytes=total_size,
            event_count=total_events,
            file_ids=file_ids
        )
