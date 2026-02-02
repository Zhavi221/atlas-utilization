"""
EventAccumulator service - Accumulates events into chunks.

Single responsibility: Manage event accumulation and chunking logic.
"""

import awkward as ak
from typing import Optional

from src.domain.events import EventChunk, EventBatch


class EventAccumulator:
    """
    Accumulates event batches into chunks based on size threshold.
    
    Stateful service that maintains current accumulation state.
    """
    
    def __init__(self, chunk_threshold_bytes: int):
        """
        Initialize accumulator.
        
        Args:
            chunk_threshold_bytes: Maximum bytes before yielding a chunk
        """
        if chunk_threshold_bytes <= 0:
            raise ValueError(f"chunk_threshold_bytes must be positive, got {chunk_threshold_bytes}")
        
        self._threshold_bytes = chunk_threshold_bytes
        self._current_batches: list[EventBatch] = []
        self._current_size_bytes = 0
        self._chunk_index = 0
        self._current_release_year: Optional[str] = None
    
    def add_batch(self, batch: EventBatch) -> Optional[EventChunk]:
        """
        Add an event batch to the accumulator.
        
        If adding this batch would exceed the threshold, the current
        accumulation is returned as a chunk and the new batch starts
        a fresh accumulation.
        
        Args:
            batch: EventBatch to add
            
        Returns:
            EventChunk if threshold exceeded, None otherwise
        """
        # Initialize release year if this is the first batch
        if self._current_release_year is None:
            self._current_release_year = batch.release_year
        
        # Check if adding this batch would exceed threshold
        would_exceed = (
            self._current_size_bytes + batch.size_bytes > self._threshold_bytes
            and len(self._current_batches) > 0  # Don't yield on first batch
        )
        
        chunk_to_return = None
        
        if would_exceed:
            # Create chunk from current accumulation
            chunk_to_return = self._create_chunk()
            # Reset for next accumulation
            self._reset_accumulation()
        
        # Add the new batch to current accumulation
        self._current_batches.append(batch)
        self._current_size_bytes += batch.size_bytes
        self._current_release_year = batch.release_year
        
        return chunk_to_return
    
    def flush(self) -> Optional[EventChunk]:
        """
        Force return any accumulated events as a chunk.
        
        Call this at the end of processing to get remaining events.
        
        Returns:
            EventChunk with accumulated events, or None if no events
        """
        if not self._current_batches:
            return None
        
        chunk = self._create_chunk()
        self._reset_accumulation()
        return chunk
    
    def _create_chunk(self) -> EventChunk:
        """Create a chunk from current batches."""
        if not self._current_batches:
            raise ValueError("Cannot create chunk with no batches")
        
        if self._current_release_year is None:
            raise ValueError("Release year not set")
        
        chunk = EventChunk.from_batches(
            batches=self._current_batches,
            chunk_index=self._chunk_index,
            release_year=self._current_release_year
        )
        
        self._chunk_index += 1
        return chunk
    
    def _reset_accumulation(self):
        """Reset accumulation state for next chunk."""
        self._current_batches = []
        self._current_size_bytes = 0
        self._current_release_year = None
    
    @property
    def current_size_bytes(self) -> int:
        """Get current accumulated size in bytes."""
        return self._current_size_bytes
    
    @property
    def current_size_mb(self) -> float:
        """Get current accumulated size in megabytes."""
        return self._current_size_bytes / (1024 * 1024)
    
    @property
    def batch_count(self) -> int:
        """Get number of batches currently accumulated."""
        return len(self._current_batches)
    
    @property
    def chunk_index(self) -> int:
        """Get current chunk index (number of chunks yielded so far)."""
        return self._chunk_index
