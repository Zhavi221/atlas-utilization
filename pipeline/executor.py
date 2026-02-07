"""
PipelineExecutor - High-level pipeline orchestrator.

Wires together all services and executes the state machine.
"""

import logging
from typing import Optional

from domain.config import PipelineConfig
from orchestration import PipelineState, PipelineContext, StateMachine
from orchestration.handlers import FetchMetadataHandler, ParsingHandler
from services.metadata.fetcher import MetadataFetcher
from services.metadata.cache import MetadataCache
from services.parsing.file_parser import FileParser
from services.parsing.event_accumulator import EventAccumulator
from services.parsing.threaded_processor import ThreadedFileProcessor


class PipelineExecutor:
    """
    High-level pipeline executor.
    
    Responsible for:
    1. Creating all services with dependency injection
    2. Building the state machine with handlers
    3. Running the pipeline
    4. Returning results
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline executor.
        
        Args:
            config: Validated pipeline configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Build services and state machine
        self.state_machine = self._build_state_machine()
    
    def run(self) -> PipelineContext:
        """
        Execute the entire pipeline.
        
        Returns:
            Final pipeline context with results
        """
        self.logger.info("Initializing pipeline execution")
        
        # Create initial context
        initial_context = self._create_initial_context()
        
        # Run state machine
        final_context = self.state_machine.run(initial_context)
        
        # Log results
        self._log_results(final_context)
        
        return final_context
    
    def _create_initial_context(self) -> PipelineContext:
        """
        Create initial pipeline context.
        
        Determines starting state based on enabled tasks.
        
        Returns:
            Initial PipelineContext
        """
        tasks = self.config.tasks
        
        # Determine starting state
        if tasks.do_parsing:
            # Need metadata for parsing
            initial_state = PipelineState.FETCHING_METADATA
        elif tasks.do_mass_calculating:
            # Start with mass calculation (assumes files exist)
            initial_state = PipelineState.MASS_CALCULATION
        elif tasks.do_post_processing:
            initial_state = PipelineState.POST_PROCESSING
        elif tasks.do_histogram_creation:
            initial_state = PipelineState.HISTOGRAM_CREATION
        else:
            # No tasks enabled
            initial_state = PipelineState.IDLE
        
        self.logger.info(f"Starting state: {initial_state}")
        
        return PipelineContext(
            config=self.config,
            current_state=initial_state
        )
    
    def _build_state_machine(self) -> StateMachine:
        """
        Build state machine with all handlers and services.
        
        Uses dependency injection to wire everything together.
        
        Returns:
            Configured StateMachine
        """
        self.logger.info("Building state machine with services")
        
        # Create services
        services = self._create_services()
        
        # Create handlers
        handlers = self._create_handlers(services)
        
        # Create state machine
        return StateMachine(handlers)
    
    def _create_services(self) -> dict:
        """
        Create all services with dependency injection.
        
        Returns:
            Dict of service instances
        """
        services = {}
        
        # Metadata services
        if self.config.tasks.do_parsing and self.config.parsing_config:
            parsing_config = self.config.parsing_config
            
            services['metadata_fetcher'] = MetadataFetcher(
                timeout=parsing_config.fetching_metadata_timeout,
                show_progress=parsing_config.show_progress_bar
            )
            
            services['metadata_cache'] = MetadataCache(
                cache_path=parsing_config.file_urls_path,
                max_wait_time=300
            )
        
        # Parsing services
        if self.config.tasks.do_parsing and self.config.parsing_config:
            parsing_config = self.config.parsing_config
            
            services['file_parser'] = FileParser()
            
            services['event_accumulator'] = EventAccumulator(
                chunk_threshold_bytes=parsing_config.chunk_yield_threshold_bytes
            )
            
            services['threaded_processor'] = ThreadedFileProcessor(
                file_parser=services['file_parser'],
                max_threads=parsing_config.threads,
                show_progress=parsing_config.show_progress_bar
            )
        
        return services
    
    def _create_handlers(self, services: dict) -> dict:
        """
        Create state handlers with services.
        
        Args:
            services: Dict of service instances
            
        Returns:
            Dict mapping states to handlers
        """
        handlers = {}
        
        # Metadata fetching handler
        if 'metadata_fetcher' in services:
            handlers[PipelineState.FETCHING_METADATA] = FetchMetadataHandler(
                metadata_fetcher=services['metadata_fetcher'],
                metadata_cache=services['metadata_cache']
            )
        
        # Parsing handler
        if 'file_parser' in services:
            handlers[PipelineState.PARSING] = ParsingHandler(
                file_parser=services['file_parser'],
                threaded_processor=services['threaded_processor'],
                event_accumulator=services['event_accumulator']
            )
        
        # TODO: Add handlers for other states (MASS_CALCULATION, POST_PROCESSING, HISTOGRAM_CREATION)
        
        return handlers
    
    def _log_results(self, context: PipelineContext):
        """
        Log pipeline execution results.
        
        Args:
            context: Final pipeline context
        """
        self.logger.info("=" * 60)
        self.logger.info("Pipeline Execution Summary")
        self.logger.info("=" * 60)
        
        summary = context.get_summary()
        for key, value in summary.items():
            self.logger.info(f"{key:30s}: {value}")
        
        if context.parsing_stats:
            self.logger.info("\nParsing Statistics:")
            stats_dict = context.parsing_stats.to_dict()
            for key, value in stats_dict.items():
                self.logger.info(f"{key:30s}: {value}")
        
        self.logger.info("=" * 60)
