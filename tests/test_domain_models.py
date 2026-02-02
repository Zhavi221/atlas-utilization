"""
Unit tests for domain models.

Tests that all domain models validate correctly and are immutable.
"""

import pytest
import awkward as ak
from datetime import datetime, timedelta

from src.domain import (
    EventChunk,
    EventBatch,
    FileMetadata,
    ReleaseMetadata,
    ParsingStatistics,
    ChunkStatistics,
    PipelineConfig,
    ParsingConfig,
    TaskConfig,
)


class TestEventBatch:
    """Tests for EventBatch domain model."""
    
    def test_create_valid_event_batch(self):
        """Test creating a valid EventBatch."""
        events = ak.Array([{"pt": 1.0}, {"pt": 2.0}])
        batch = EventBatch(
            events=events,
            file_id=12345,
            release_year="2024r-pp",
            size_bytes=1024,
            event_count=2,
            processing_time_sec=0.5
        )
        
        assert batch.file_id == 12345
        assert batch.event_count == 2
        assert batch.size_bytes == 1024
    
    def test_event_batch_negative_count_fails(self):
        """Test that negative event count raises ValueError."""
        events = ak.Array([{"pt": 1.0}])
        with pytest.raises(ValueError, match="event_count must be non-negative"):
            EventBatch(
                events=events,
                file_id=123,
                release_year="2024r-pp",
                size_bytes=100,
                event_count=-1,
                processing_time_sec=0.5
            )
    
    def test_event_batch_is_immutable(self):
        """Test that EventBatch is immutable."""
        events = ak.Array([{"pt": 1.0}])
        batch = EventBatch(
            events=events,
            file_id=123,
            release_year="2024r-pp",
            size_bytes=100,
            event_count=1,
            processing_time_sec=0.5
        )
        
        with pytest.raises(Exception):  # FrozenInstanceError
            batch.file_id = 999


class TestEventChunk:
    """Tests for EventChunk domain model."""
    
    def test_create_valid_event_chunk(self):
        """Test creating a valid EventChunk."""
        events = ak.Array([{"pt": 1.0}, {"pt": 2.0}, {"pt": 3.0}])
        chunk = EventChunk(
            events=events,
            chunk_index=0,
            release_year="2024r-pp",
            size_bytes=2048,
            event_count=3,
            file_ids=(1, 2)
        )
        
        assert chunk.chunk_index == 0
        assert chunk.event_count == 3
        assert len(chunk.file_ids) == 2
        assert chunk.size_mb == pytest.approx(2048 / (1024 * 1024))
    
    def test_event_chunk_empty_file_ids_fails(self):
        """Test that empty file_ids raises ValueError."""
        events = ak.Array([{"pt": 1.0}])
        with pytest.raises(ValueError, match="file_ids cannot be empty"):
            EventChunk(
                events=events,
                chunk_index=0,
                release_year="2024r-pp",
                size_bytes=100,
                event_count=1,
                file_ids=tuple()
            )
    
    def test_event_chunk_from_batches(self):
        """Test creating EventChunk from multiple EventBatches."""
        batch1 = EventBatch(
            events=ak.Array([{"pt": 1.0}]),
            file_id=1,
            release_year="2024r-pp",
            size_bytes=100,
            event_count=1,
            processing_time_sec=0.1
        )
        batch2 = EventBatch(
            events=ak.Array([{"pt": 2.0}, {"pt": 3.0}]),
            file_id=2,
            release_year="2024r-pp",
            size_bytes=200,
            event_count=2,
            processing_time_sec=0.2
        )
        
        chunk = EventChunk.from_batches([batch1, batch2], chunk_index=0, release_year="2024r-pp")
        
        assert chunk.event_count == 3
        assert chunk.size_bytes == 300
        assert chunk.file_ids == (1, 2)


class TestFileMetadata:
    """Tests for FileMetadata domain model."""
    
    def test_create_successful_file_metadata(self):
        """Test creating metadata for successful file."""
        meta = FileMetadata(
            file_id=123,
            release_year="2024r-pp",
            size_mb=10.5,
            event_count=1000,
            processing_time_sec=2.3,
            success=True
        )
        
        assert meta.success is True
        assert meta.error_message is None
    
    def test_create_failed_file_metadata(self):
        """Test creating metadata for failed file."""
        meta = FileMetadata(
            file_id=456,
            release_year="2024r-pp",
            size_mb=0.0,
            event_count=0,
            processing_time_sec=1.0,
            success=False,
            error_message="File not found"
        )
        
        assert meta.success is False
        assert meta.error_message == "File not found"
    
    def test_failed_file_without_error_message_fails(self):
        """Test that failed file without error message raises ValueError."""
        with pytest.raises(ValueError, match="error_message must be provided"):
            FileMetadata(
                file_id=456,
                release_year="2024r-pp",
                size_mb=0.0,
                event_count=0,
                processing_time_sec=1.0,
                success=False
            )


class TestReleaseMetadata:
    """Tests for ReleaseMetadata domain model."""
    
    def test_create_valid_release_metadata(self):
        """Test creating valid ReleaseMetadata."""
        meta = ReleaseMetadata(
            release_year="2024r-pp",
            file_ids=(1, 2, 3, 4, 5),
            total_files=5
        )
        
        assert meta.release_year == "2024r-pp"
        assert len(meta.file_ids) == 5
        assert meta.total_files == 5
    
    def test_release_metadata_from_list(self):
        """Test creating ReleaseMetadata from list."""
        file_list = [10, 20, 30]
        meta = ReleaseMetadata.from_file_list("2024r-pp", file_list)
        
        assert meta.total_files == 3
        assert meta.file_ids == (10, 20, 30)
    
    def test_mismatch_total_files_fails(self):
        """Test that mismatch between total_files and file_ids length fails."""
        with pytest.raises(ValueError, match="total_files.*must match"):
            ReleaseMetadata(
                release_year="2024r-pp",
                file_ids=(1, 2, 3),
                total_files=5  # Mismatch!
            )


class TestParsingStatistics:
    """Tests for ParsingStatistics domain model."""
    
    def test_create_valid_parsing_statistics(self):
        """Test creating valid ParsingStatistics."""
        start = datetime.now()
        end = start + timedelta(seconds=10)
        
        stats = ParsingStatistics(
            total_files=100,
            successful_files=95,
            failed_files=5,
            total_events=100000,
            total_chunks=10,
            total_size_bytes=1024 * 1024 * 100,  # 100 MB
            max_chunk_size_bytes=1024 * 1024 * 15,
            min_chunk_size_bytes=1024 * 1024 * 5,
            max_memory_mb=500.0,
            total_time_sec=10.0,
            start_time=start,
            end_time=end
        )
        
        assert stats.success_rate == 95.0
        assert stats.total_size_mb == pytest.approx(100.0)
        assert stats.average_events_per_file == pytest.approx(100000 / 95)
    
    def test_invalid_file_count_sum_fails(self):
        """Test that mismatched file counts fail validation."""
        start = datetime.now()
        end = start + timedelta(seconds=10)
        
        with pytest.raises(ValueError, match="must equal total_files"):
            ParsingStatistics(
                total_files=100,
                successful_files=50,
                failed_files=40,  # Should be 50
                total_events=100000,
                total_chunks=10,
                total_size_bytes=1024 * 1024 * 100,
                max_chunk_size_bytes=1024 * 1024 * 15,
                min_chunk_size_bytes=1024 * 1024 * 5,
                max_memory_mb=500.0,
                total_time_sec=10.0,
                start_time=start,
                end_time=end
            )
    
    def test_statistics_to_dict(self):
        """Test converting statistics to dictionary."""
        start = datetime.now()
        end = start + timedelta(seconds=10)
        
        stats = ParsingStatistics(
            total_files=100,
            successful_files=95,
            failed_files=5,
            total_events=100000,
            total_chunks=10,
            total_size_bytes=1024 * 1024 * 100,
            max_chunk_size_bytes=1024 * 1024 * 15,
            min_chunk_size_bytes=1024 * 1024 * 5,
            max_memory_mb=500.0,
            total_time_sec=10.0,
            start_time=start,
            end_time=end
        )
        
        stats_dict = stats.to_dict()
        assert stats_dict["total_files"] == 100
        assert stats_dict["success_rate"] == "95.0%"
        assert "start_time" in stats_dict


class TestTaskConfig:
    """Tests for TaskConfig domain model."""
    
    def test_no_tasks_enabled(self):
        """Test configuration with no tasks enabled."""
        config = TaskConfig()
        assert not config.any_enabled()
    
    def test_some_tasks_enabled(self):
        """Test configuration with some tasks enabled."""
        config = TaskConfig(do_parsing=True, do_mass_calculating=True)
        assert config.any_enabled()
        assert config.do_parsing
        assert config.do_mass_calculating
        assert not config.do_post_processing


class TestParsingConfig:
    """Tests for ParsingConfig domain model."""
    
    def test_create_valid_parsing_config(self):
        """Test creating valid ParsingConfig."""
        config = ParsingConfig(
            output_path="/tmp/output",
            file_urls_path="/tmp/urls.json",
            jobs_logs_path="/tmp/logs",
            release_years=("2024r-pp",),
            threads=4
        )
        
        assert config.threads == 4
        assert config.output_path == "/tmp/output"
    
    def test_invalid_threads_fails(self):
        """Test that invalid threads value fails."""
        with pytest.raises(ValueError, match="threads must be positive"):
            ParsingConfig(
                output_path="/tmp/output",
                file_urls_path="/tmp/urls.json",
                jobs_logs_path="/tmp/logs",
                threads=0
            )
    
    def test_empty_output_path_fails(self):
        """Test that empty output_path fails."""
        with pytest.raises(ValueError, match="output_path cannot be empty"):
            ParsingConfig(
                output_path="",
                file_urls_path="/tmp/urls.json",
                jobs_logs_path="/tmp/logs"
            )


class TestPipelineConfig:
    """Tests for PipelineConfig domain model."""
    
    def test_create_valid_pipeline_config(self):
        """Test creating valid PipelineConfig."""
        tasks = TaskConfig(do_parsing=True)
        parsing_config = ParsingConfig(
            output_path="/tmp/output",
            file_urls_path="/tmp/urls.json",
            jobs_logs_path="/tmp/logs"
        )
        
        config = PipelineConfig(
            tasks=tasks,
            parsing_config=parsing_config,
            run_name="test_run"
        )
        
        assert config.run_name == "test_run"
        assert config.tasks.do_parsing
    
    def test_no_tasks_enabled_fails(self):
        """Test that config with no tasks fails."""
        tasks = TaskConfig()  # All False
        
        with pytest.raises(ValueError, match="At least one task must be enabled"):
            PipelineConfig(tasks=tasks)
    
    def test_parsing_enabled_without_config_fails(self):
        """Test that parsing enabled without parsing_config fails."""
        tasks = TaskConfig(do_parsing=True)
        
        with pytest.raises(ValueError, match="parsing_config required"):
            PipelineConfig(tasks=tasks)
    
    def test_from_dict(self):
        """Test creating PipelineConfig from dictionary."""
        config_dict = {
            "tasks": {
                "do_parsing": True,
                "do_mass_calculating": False,
                "do_post_processing": False,
                "do_histogram_creation": False,
            },
            "parsing_task_config": {
                "output_path": "/tmp/output",
                "file_urls_path": "/tmp/urls.json",
                "jobs_logs_path": "/tmp/logs",
                "release_years": ["2024r-pp"],
                "threads": 4,
            },
            "run_metadata": {
                "run_name": "test_from_dict"
            }
        }
        
        config = PipelineConfig.from_dict(config_dict)
        
        assert config.run_name == "test_from_dict"
        assert config.tasks.do_parsing
        assert config.parsing_config.threads == 4
        assert config.parsing_config.release_years == ("2024r-pp",)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
