"""
FetchMetadataHandler - Handles metadata fetching state.

Fetches file URLs from ATLAS Open Data API.
"""

from orchestration.context import PipelineContext
from orchestration.states import PipelineState
from .base import StateHandler
from services.metadata.fetcher import MetadataFetcher
from services.metadata.cache import MetadataCache


class FetchMetadataHandler(StateHandler):
    """
    Handler for FETCHING_METADATA state.
    
    Fetches file metadata from ATLAS API and caches it.
    """
    
    def __init__(
        self,
        metadata_fetcher: MetadataFetcher,
        metadata_cache: MetadataCache
    ):
        """
        Initialize handler.
        
        Args:
            metadata_fetcher: Metadata fetcher service
            metadata_cache: Metadata cache service
        """
        super().__init__()
        self.fetcher = metadata_fetcher
        self.cache = metadata_cache
    
    def handle(self, context: PipelineContext) -> tuple[PipelineContext, PipelineState]:
        """
        Fetch metadata and determine next state.
        
        Args:
            context: Current pipeline context
            
        Returns:
            Tuple of (updated_context, next_state)
        """
        self._log_state_entry(context)
        
        parsing_config = context.config.parsing_config
        if not parsing_config:
            # Skip if no parsing config
            next_state = self._determine_next_state(context)
            self._log_state_exit(context, next_state)
            return context, next_state
        
        # Try to load from cache first
        metadata = self.cache.load()
        
        if metadata:
            self.logger.info(f"Loaded metadata from cache: {self.cache.cache_path}")
        else:
            self.logger.info("Cache miss, fetching metadata from API...")
            
            # Fetch fresh metadata
            metadata = self.fetcher.fetch(
                release_years=list(parsing_config.release_years) if parsing_config.release_years else None,
                record_ids=list(parsing_config.specific_record_ids) if parsing_config.specific_record_ids else None,
                separate_mc=parsing_config.parse_mc
            )
            
            # Save to cache
            try:
                self.cache.save(metadata)
            except TimeoutError as e:
                self.logger.warning(f"Could not save to cache: {e}")
        
        # Log summary
        total_files = sum(len(urls) for urls in metadata.values())
        self.logger.info(
            f"Fetched metadata: {len(metadata)} release year(s), {total_files} total files"
        )
        
        # Update context
        updated_context = context.with_metadata(metadata)
        
        # Determine next state
        next_state = self._determine_next_state(updated_context)
        
        self._log_state_exit(context, next_state)
        return updated_context, next_state
