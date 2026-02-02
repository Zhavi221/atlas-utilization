"""
ParsingHandler - Handles parsing state.

Orchestrates file parsing using services.
"""

from datetime import datetime

from orchestration.context import PipelineContext
from orchestration.states import PipelineState
from .base import StateHandler
from services.parsing.file_parser import FileParser
from services.parsing.event_accumulator import EventAccumulator
from services.parsing.threaded_processor import ThreadedFileProcessor, ParsingStatisticsCollector
from domain.statistics import ParsingStatistics


class ParsingHandler(StateHandler):
    """
    Handler for PARSING state.
    
    Uses FileParser and ThreadedFileProcessor to parse files,
    EventAccumulator to create chunks, and saves results.
    """
    
    def __init__(
        self,
        file_parser: FileParser,
        threaded_processor: ThreadedFileProcessor,
        event_accumulator: EventAccumulator
    ):
        """
        Initialize handler.
        
        Args:
            file_parser: File parser service
            threaded_processor: Threaded processor service
            event_accumulator: Event accumulator service
        """
        super().__init__()
        self.file_parser = file_parser
        self.processor = threaded_processor
        self.accumulator = event_accumulator
    
    def handle(self, context: PipelineContext) -> tuple[PipelineContext, PipelineState]:
        """
        Parse files and determine next state.
        
        Args:
            context: Current pipeline context
            
        Returns:
            Tuple of (updated_context, next_state)
        """
        self._log_state_entry(context)
        
        parsing_config = context.config.parsing_config
        if not parsing_config or not context.metadata:
            self.logger.warning("No parsing config or metadata, skipping parsing")
            next_state = self._determine_next_state(context)
            return context, next_state
        
        start_time = datetime.now()
        stats_collector = ParsingStatisticsCollector()
        parsed_files = []
        
        # Parse each release year
        for release_year, file_urls in context.metadata.items():
            self.logger.info(
                f"Parsing {len(file_urls)} files for release year: {release_year}"
            )
            
            # Define callbacks
            def on_success(file_url: str, event_count: int, time_sec: float):
                stats_collector.record_success(file_url, event_count, 0, time_sec)
            
            def on_error(file_url: str, error: Exception):
                stats_collector.record_failure(file_url, error)
            
            # Process files
            for batch in self.processor.process_files(
                file_urls=file_urls,
                tree_names=list(parsing_config.possible_data_tree_names),
                release_year=release_year,
                batch_size=40_000,
                on_success=on_success,
                on_error=on_error
            ):
                # Accumulate batch into chunks
                chunk = self.accumulator.add_batch(batch)
                
                if chunk:
                    # Save chunk (placeholder - in real implementation, save to disk)
                    file_path = f"chunk_{chunk.chunk_index}_{release_year}.root"
                    parsed_files.append(file_path)
                    self.logger.info(
                        f"Saved chunk {chunk.chunk_index}: "
                        f"{chunk.event_count} events, {chunk.size_mb:.1f} MB"
                    )
        
        # Flush remaining events
        final_chunk = self.accumulator.flush()
        if final_chunk:
            file_path = f"chunk_final_{final_chunk.release_year}.root"
            parsed_files.append(file_path)
            self.logger.info(
                f"Saved final chunk: {final_chunk.event_count} events, {final_chunk.size_mb:.1f} MB"
            )
        
        # Create parsing statistics
        end_time = datetime.now()
        stats_summary = stats_collector.get_summary()
        
        parsing_stats = ParsingStatistics(
            total_files=stats_summary["total_files"],
            successful_files=stats_summary["successful_files"],
            failed_files=stats_summary["failed_files"],
            total_events=stats_summary["total_events"],
            total_chunks=len(parsed_files),
            total_size_bytes=int(stats_summary["total_size_mb"] * 1024 * 1024),
            max_chunk_size_bytes=parsing_config.chunk_yield_threshold_bytes,
            min_chunk_size_bytes=0,  # TODO: track min chunk size
            max_memory_mb=0.0,  # TODO: track memory
            total_time_sec=(end_time - start_time).total_seconds(),
            start_time=start_time,
            end_time=end_time
        )
        
        self.logger.info(
            f"Parsing complete: {parsing_stats.successful_files}/{parsing_stats.total_files} files, "
            f"{parsing_stats.total_events} events, {parsing_stats.success_rate:.1f}% success rate"
        )
        
        # Update context
        updated_context = context.with_parsed_files(parsed_files).with_parsing_stats(parsing_stats)
        
        # Determine next state
        next_state = self._determine_next_state(updated_context)
        
        self._log_state_exit(context, next_state)
        return updated_context, next_state
