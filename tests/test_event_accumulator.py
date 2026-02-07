"""
Tests for EventAccumulator service.

Tests that event accumulation and chunking works correctly.
"""

import pytest
import awkward as ak
from datetime import datetime

from services.parsing.event_accumulator import EventAccumulator
from domain.events import EventBatch, EventChunk


class TestEventAccumulator:
    """Tests for EventAccumulator service."""
    
    def test_create_accumulator_with_valid_threshold(self):
        """Test creating accumulator with valid threshold."""
        acc = EventAccumulator(chunk_threshold_bytes=1000)
        assert acc._threshold_bytes == 1000
        assert acc.current_size_bytes == 0
        assert acc.batch_count == 0
        assert acc.chunk_index == 0
    
    def test_create_accumulator_with_zero_threshold_fails(self):
        """Test that zero threshold raises ValueError."""
        with pytest.raises(ValueError, match="chunk_threshold_bytes must be positive"):
            EventAccumulator(chunk_threshold_bytes=0)
    
    def test_create_accumulator_with_negative_threshold_fails(self):
        """Test that negative threshold raises ValueError."""
        with pytest.raises(ValueError, match="chunk_threshold_bytes must be positive"):
            EventAccumulator(chunk_threshold_bytes=-100)
    
    def test_add_single_batch(self):
        """Test adding a single batch."""
        acc = EventAccumulator(chunk_threshold_bytes=10000)
        
        batch = EventBatch(
            events=ak.Array([{"pt": 1.0}]),
            file_id=1,
            release_year="2024r-pp",
            size_bytes=500,
            event_count=1,
            processing_time_sec=0.1
        )
        
        result = acc.add_batch(batch)
        
        # Should not yield chunk (below threshold)
        assert result is None
        assert acc.current_size_bytes == 500
        assert acc.batch_count == 1
    
    def test_add_batch_exceeds_threshold_yields_chunk(self):
        """Test that exceeding threshold yields a chunk."""
        acc = EventAccumulator(chunk_threshold_bytes=1000)
        
        # Add first batch (below threshold)
        batch1 = EventBatch(
            events=ak.Array([{"pt": 1.0}]),
            file_id=1,
            release_year="2024r-pp",
            size_bytes=600,
            event_count=1,
            processing_time_sec=0.1
        )
        result1 = acc.add_batch(batch1)
        assert result1 is None  # No chunk yet
        
        # Add second batch (would exceed threshold)
        batch2 = EventBatch(
            events=ak.Array([{"pt": 2.0}]),
            file_id=2,
            release_year="2024r-pp",
            size_bytes=600,
            event_count=1,
            processing_time_sec=0.1
        )
        result2 = acc.add_batch(batch2)
        
        # Should yield chunk
        assert result2 is not None
        assert isinstance(result2, EventChunk)
        assert result2.chunk_index == 0
        assert result2.event_count == 1  # Only batch1
        assert result2.size_bytes == 600
        assert result2.file_ids == (1,)
        
        # Accumulator should now have only batch2
        assert acc.current_size_bytes == 600
        assert acc.batch_count == 1
        assert acc.chunk_index == 1  # Incremented
    
    def test_flush_with_accumulated_batches(self):
        """Test flushing accumulated batches."""
        acc = EventAccumulator(chunk_threshold_bytes=10000)
        
        batch1 = EventBatch(
            events=ak.Array([{"pt": 1.0}]),
            file_id=1,
            release_year="2024r-pp",
            size_bytes=100,
            event_count=1,
            processing_time_sec=0.1
        )
        batch2 = EventBatch(
            events=ak.Array([{"pt": 2.0}]),
            file_id=2,
            release_year="2024r-pp",
            size_bytes=200,
            event_count=1,
            processing_time_sec=0.1
        )
        
        acc.add_batch(batch1)
        acc.add_batch(batch2)
        
        # Flush should return chunk with both batches
        chunk = acc.flush()
        
        assert chunk is not None
        assert chunk.event_count == 2
        assert chunk.size_bytes == 300
        assert chunk.file_ids == (1, 2)
        
        # Accumulator should be empty
        assert acc.current_size_bytes == 0
        assert acc.batch_count == 0
    
    def test_flush_with_empty_accumulator(self):
        """Test flushing empty accumulator."""
        acc = EventAccumulator(chunk_threshold_bytes=1000)
        
        result = acc.flush()
        
        assert result is None
    
    def test_size_mb_property(self):
        """Test size_mb property."""
        acc = EventAccumulator(chunk_threshold_bytes=10000)
        
        batch = EventBatch(
            events=ak.Array([{"pt": 1.0}]),
            file_id=1,
            release_year="2024r-pp",
            size_bytes=1024 * 1024,  # 1 MB
            event_count=1,
            processing_time_sec=0.1
        )
        
        acc.add_batch(batch)
        
        assert acc.current_size_mb == pytest.approx(1.0)
    
    def test_multiple_chunks(self):
        """Test yielding multiple chunks."""
        acc = EventAccumulator(chunk_threshold_bytes=500)
        
        chunks = []
        
        # Add 5 batches, each 400 bytes
        for i in range(5):
            batch = EventBatch(
                events=ak.Array([{"pt": float(i)}]),
                file_id=i,
                release_year="2024r-pp",
                size_bytes=400,
                event_count=1,
                processing_time_sec=0.1
            )
            chunk = acc.add_batch(batch)
            if chunk:
                chunks.append(chunk)
        
        # Should have yielded 4 chunks (batch 0+1, 1+2, 2+3, 3+4)
        # Actually: batch 0 accumulates, batch 1 triggers yield of 0, then 1 accumulates
        # So we get: chunk(0), chunk(1), chunk(2), chunk(3)
        assert len(chunks) == 4
        
        # Each chunk should have 1 batch
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.event_count == 1
        
        # Final batch should still be in accumulator
        assert acc.batch_count == 1
        
        # Flush to get last chunk
        final_chunk = acc.flush()
        assert final_chunk is not None
        assert final_chunk.chunk_index == 4
    
    def test_first_batch_never_yields_even_if_exceeds(self):
        """Test that first batch never yields even if it exceeds threshold."""
        acc = EventAccumulator(chunk_threshold_bytes=100)
        
        # Add a batch larger than threshold
        batch = EventBatch(
            events=ak.Array([{"pt": 1.0}]),
            file_id=1,
            release_year="2024r-pp",
            size_bytes=200,  # Larger than threshold
            event_count=1,
            processing_time_sec=0.1
        )
        
        result = acc.add_batch(batch)
        
        # Should not yield (first batch exception)
        assert result is None
        assert acc.current_size_bytes == 200
        assert acc.batch_count == 1
    
    def test_different_release_years_handled(self):
        """Test that batches from different release years are handled."""
        acc = EventAccumulator(chunk_threshold_bytes=10000)
        
        batch1 = EventBatch(
            events=ak.Array([{"pt": 1.0}]),
            file_id=1,
            release_year="2024r-pp",
            size_bytes=100,
            event_count=1,
            processing_time_sec=0.1
        )
        batch2 = EventBatch(
            events=ak.Array([{"pt": 2.0}]),
            file_id=2,
            release_year="2025r-pp",  # Different year
            size_bytes=100,
            event_count=1,
            processing_time_sec=0.1
        )
        
        acc.add_batch(batch1)
        acc.add_batch(batch2)
        
        chunk = acc.flush()
        
        # Chunk should use the last release year
        assert chunk.release_year == "2025r-pp"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
