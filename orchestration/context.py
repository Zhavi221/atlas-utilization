"""
Pipeline context.

Immutable context object passed between state handlers.
"""

from dataclasses import dataclass, field, replace
from typing import Optional, Any
from datetime import datetime

from domain.config import PipelineConfig
from domain.statistics import ParsingStatistics
from .states import PipelineState


@dataclass(frozen=True)
class PipelineContext:
    """
    Immutable context for pipeline execution.
    
    Contains all state needed for pipeline execution.
    Each state handler returns a new context with updated fields.
    """
    
    # Configuration
    config: PipelineConfig
    
    # Current state
    current_state: PipelineState
    
    # Execution metadata
    start_time: datetime = field(default_factory=datetime.now)
    
    # Data accumulated during pipeline
    metadata: Optional[dict[str, list[str]]] = None  # Release year -> file URLs
    parsed_files: list[str] = field(default_factory=list)
    im_files: list[str] = field(default_factory=list)
    processed_files: list[str] = field(default_factory=list)
    
    # Statistics
    parsing_stats: Optional[ParsingStatistics] = None
    
    # Error tracking
    error_message: Optional[str] = None
    error_details: Optional[dict] = None
    
    # Custom data (for extension)
    custom_data: dict[str, Any] = field(default_factory=dict)
    
    def with_state(self, new_state: PipelineState) -> 'PipelineContext':
        """
        Return new context with updated state.
        
        Args:
            new_state: New pipeline state
            
        Returns:
            New PipelineContext with updated state
        """
        return replace(self, current_state=new_state)
    
    def with_metadata(self, metadata: dict[str, list[str]]) -> 'PipelineContext':
        """
        Return new context with metadata.
        
        Args:
            metadata: Release year to file URLs mapping
            
        Returns:
            New PipelineContext with metadata
        """
        return replace(self, metadata=metadata)
    
    def with_parsed_files(self, files: list[str]) -> 'PipelineContext':
        """
        Return new context with parsed files.
        
        Args:
            files: List of parsed file paths
            
        Returns:
            New PipelineContext with parsed files
        """
        return replace(self, parsed_files=files)
    
    def with_im_files(self, files: list[str]) -> 'PipelineContext':
        """
        Return new context with IM files.
        
        Args:
            files: List of invariant mass file paths
            
        Returns:
            New PipelineContext with IM files
        """
        return replace(self, im_files=files)
    
    def with_processed_files(self, files: list[str]) -> 'PipelineContext':
        """
        Return new context with processed files.
        
        Args:
            files: List of processed file paths
            
        Returns:
            New PipelineContext with processed files
        """
        return replace(self, processed_files=files)
    
    def with_parsing_stats(self, stats: ParsingStatistics) -> 'PipelineContext':
        """
        Return new context with parsing statistics.
        
        Args:
            stats: Parsing statistics
            
        Returns:
            New PipelineContext with statistics
        """
        return replace(self, parsing_stats=stats)
    
    def with_error(self, message: str, details: Optional[dict] = None) -> 'PipelineContext':
        """
        Return new context with error information.
        
        Args:
            message: Error message
            details: Optional error details dict
            
        Returns:
            New PipelineContext with error information
        """
        return replace(
            self,
            current_state=PipelineState.FAILED,
            error_message=message,
            error_details=details or {}
        )
    
    def with_custom_data(self, key: str, value: Any) -> 'PipelineContext':
        """
        Return new context with custom data.
        
        Args:
            key: Data key
            value: Data value
            
        Returns:
            New PipelineContext with custom data
        """
        new_custom = self.custom_data.copy()
        new_custom[key] = value
        return replace(self, custom_data=new_custom)
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def is_terminal(self) -> bool:
        """Check if current state is terminal."""
        return self.current_state.is_terminal()
    
    @property
    def is_successful(self) -> bool:
        """Check if pipeline completed successfully."""
        return self.current_state == PipelineState.COMPLETED
    
    @property
    def has_error(self) -> bool:
        """Check if pipeline failed."""
        return self.current_state == PipelineState.FAILED
    
    def get_summary(self) -> dict:
        """
        Get summary of pipeline execution.
        
        Returns:
            Dict with execution summary
        """
        return {
            "state": str(self.current_state),
            "elapsed_time_sec": self.elapsed_time,
            "start_time": self.start_time.isoformat(),
            "parsed_files_count": len(self.parsed_files),
            "im_files_count": len(self.im_files),
            "processed_files_count": len(self.processed_files),
            "has_error": self.has_error,
            "error_message": self.error_message,
            "is_successful": self.is_successful,
        }
