"""
EventAccumulator service - Accumulates events into chunks.

Single responsibility: Manage event accumulation and chunking logic.
"""

import logging
import awkward as ak
from typing import Optional

from domain.events import EventChunk, EventBatch
from services.calculations.weights_registry import extract_dsid_from_url


class EventAccumulator:
    """
    Accumulates event batches into chunks based on size threshold.
    
    Stateful service that maintains current accumulation state.
    """
    
    def __init__(self, chunk_threshold_bytes: int, split_by_dataset: bool = False):
        """
        Initialize accumulator.

        Args:
            chunk_threshold_bytes: Maximum bytes before yielding a chunk
            split_by_dataset: When True, also start a new chunk whenever the
                incoming batch belongs to a different MC dataset (DSID) than the
                current accumulation, guaranteeing one DSID per chunk. Required
                for correct per-file MC weighting on freshly-parsed data.
        """
        if chunk_threshold_bytes <= 0:
            raise ValueError(f"chunk_threshold_bytes must be positive, got {chunk_threshold_bytes}")

        self._threshold_bytes = chunk_threshold_bytes
        self._split_by_dataset = split_by_dataset
        self._current_batches: list[EventBatch] = []
        self._current_size_bytes = 0
        self._chunk_index = 0
        self._current_release_year: Optional[str] = None
        self._current_dsid: Optional[int] = None
    
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

        incoming_dsid = self._batch_dsid(batch)

        # Check if adding this batch would exceed threshold
        would_exceed = (
            self._current_size_bytes + batch.size_bytes > self._threshold_bytes
            and len(self._current_batches) > 0  # Don't yield on first batch
        )

        # When splitting by dataset, close the chunk at a DSID boundary so each
        # chunk contains exactly one dataset. Only applies when both the current
        # and incoming DSIDs are known (data files without a DSID chunk by size).
        crosses_dataset = (
            self._split_by_dataset
            and len(self._current_batches) > 0
            and self._current_dsid is not None
            and incoming_dsid is not None
            and incoming_dsid != self._current_dsid
        )

        chunk_to_return = None

        if would_exceed or crosses_dataset:
            # Create chunk from current accumulation
            chunk_to_return = self._create_chunk()
            # Reset for next accumulation
            self._reset_accumulation()

        # Add the new batch to current accumulation
        self._current_batches.append(batch)
        self._current_size_bytes += batch.size_bytes
        self._current_release_year = batch.release_year
        if self._current_dsid is None:
            self._current_dsid = incoming_dsid

        return chunk_to_return

    @staticmethod
    def _batch_dsid(batch: EventBatch) -> Optional[int]:
        """Extract the MC dataset number (DSID) from a batch's source URL."""
        source = getattr(batch, "source_url", None)
        return extract_dsid_from_url(source) if source else None
    
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
        
        self._warn_if_mixed_dsids()

        # Only label the chunk with a DSID when splitting by dataset guarantees
        # the chunk is single-DSID; otherwise a chunk may be mixed and a label
        # would be misleading.
        chunk_dsid = self._current_dsid if self._split_by_dataset else None

        chunk = EventChunk.from_batches(
            batches=self._current_batches,
            chunk_index=self._chunk_index,
            release_year=self._current_release_year,
            dsid=chunk_dsid
        )

        self._chunk_index += 1
        return chunk

    def _warn_if_mixed_dsids(self) -> None:
        """
        Warn when a chunk mixes events from more than one MC dataset (DSID).

        Chunks are accumulated by size alone, so a single chunk can span
        several source files. Per-file MC weighting assumes one DSID per
        chunk; if a chunk mixes DSIDs, a single weight cannot represent it
        correctly. A dataset legitimately split across many files resolves to
        one DSID and does NOT trigger this. Data files (no DSID) are ignored.
        """
        dsids = set()
        for batch in self._current_batches:
            source = getattr(batch, "source_url", None)
            dsid = extract_dsid_from_url(source) if source else None
            if dsid is not None:
                dsids.add(dsid)

        if len(dsids) > 1:
            logging.warning(
                "Chunk %d mixes %d MC datasets (DSIDs: %s); per-file MC weighting "
                "would be INCORRECT for this chunk. Accumulate one DSID per chunk "
                "or use per-event weighting before applying weights.",
                self._chunk_index, len(dsids), sorted(dsids),
            )
    
    def _reset_accumulation(self):
        """Reset accumulation state for next chunk."""
        self._current_batches = []
        self._current_size_bytes = 0
        self._current_release_year = None
        self._current_dsid = None
    
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
