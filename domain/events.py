"""
Event-related domain models.

Immutable data structures representing parsed events and chunks.
"""

from dataclasses import dataclass, field
from typing import Optional
import awkward as ak
import numpy as np

# Event-level Monte-Carlo information carried alongside the per-particle
# collections of an event array, as a record with one scalar per event.
# It is NOT a particle type: code that iterates particle fields must skip it,
# and every event filter must carry it through unchanged. Data files have no
# such field.
MC_EVENT_INFO_FIELD = "_mcEventInfo"
MC_EVENT_WEIGHT_FIELD = "_mcEventWeight"      # nominal per-event generator weight
MC_CHANNEL_NUMBER_FIELD = "_mcChannelNumber"  # dataset number (DSID) of the event's sample
MC_RUN_NUMBER_FIELD = "_mcRunNumber"          # MC run number, fixed per production campaign
NON_PARTICLE_FIELDS = frozenset({MC_EVENT_INFO_FIELD})

# Values assumed for events whose file lacks the MC info branches: an
# unweighted event (w_gen = 1) from an unknown dataset (DSID 0) and campaign (run 0).
_MC_EVENT_INFO_DEFAULTS = {MC_EVENT_WEIGHT_FIELD: 1.0, MC_CHANNEL_NUMBER_FIELD: 0, MC_RUN_NUMBER_FIELD: 0}


def particle_fields(events: ak.Array) -> list[str]:
    """Names of the particle collections in ``events`` (event-level fields excluded)."""
    return [f for f in events.fields if f not in NON_PARTICLE_FIELDS]


def dsids_in_events(events: ak.Array) -> np.ndarray:
    """Distinct MC dataset numbers (DSIDs) carried by ``events`` (empty for data / no MC info)."""
    if len(events) == 0 or MC_EVENT_INFO_FIELD not in events.fields:
        return np.array([], dtype=np.int64)
    info = events[MC_EVENT_INFO_FIELD]
    if MC_CHANNEL_NUMBER_FIELD not in info.fields:
        return np.array([], dtype=np.int64)
    channels = np.unique(ak.to_numpy(info[MC_CHANNEL_NUMBER_FIELD]))
    return channels[channels > 0]  # 0 marks events from files without the branch



def _empty_particle_collection(collection: ak.Array, event_count: int) -> ak.Array:
    """Build a typed jagged record collection containing no particles."""
    counts = np.zeros(event_count, dtype=np.int64)
    no_particles = ak.flatten(collection[:0])  # zero records, carrying ``collection``'s field types
    return ak.unflatten(no_particles, counts)


def _default_mc_event_info_column(name: str, example: ak.Array, event_count: int) -> ak.Array:
    """One MC info column holding its default value, typed like ``example[name]``."""
    return ak.values_astype(
        ak.Array(np.full(event_count, _MC_EVENT_INFO_DEFAULTS.get(name, 0))),
        example[name].layout.dtype,
    )


def _normalized_mc_event_info(info: Optional[ak.Array], example: ak.Array, event_count: int) -> ak.Array:
    """
    A per-event MC info record carrying every column of ``example``.

    Columns missing from ``info`` (a file without that branch, or no MC info at
    all) get their default value, so concatenation yields a plain record rather
    than a union type that would hide the columns only some files have.
    """
    return ak.zip({
        name: info[name] if info is not None and name in info.fields
        else _default_mc_event_info_column(name, example, event_count)
        for name in example.fields
    })


def _mc_event_info_example(arrays: list[ak.Array]) -> Optional[ak.Array]:
    """A record exposing the union of MC info columns seen across ``arrays``."""
    columns = {}
    for array in arrays:
        if MC_EVENT_INFO_FIELD in array.fields:
            info = array[MC_EVENT_INFO_FIELD]
            for name in info.fields:
                columns.setdefault(name, info[name][:0])
    return ak.zip(columns) if columns else None


def _concatenate_events(arrays: list[ak.Array]) -> ak.Array:
    """
    Concatenate per-file event arrays, keeping collections only some files have.

    Concatenating the records directly builds a union type, which exposes only
    the fields common to *every* file - so one file without electrons would
    delete electrons from every event in the chunk. Concatenating one collection
    at a time keeps a plain record, with empty lists for the files that lack it.
    """
    example_of = {field: array[field] for array in arrays for field in array.fields}
    mc_info_example = _mc_event_info_example(arrays)

    combined = {}
    for field, example in example_of.items():
        if field == MC_EVENT_INFO_FIELD:
            combined[field] = ak.concatenate([
                _normalized_mc_event_info(
                    array[field] if field in array.fields else None, mc_info_example, len(array)
                )
                for array in arrays
            ])
            continue
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
    file_url: str = ""
    
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
