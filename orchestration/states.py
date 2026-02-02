"""
Pipeline states.

Explicit state enumeration for the pipeline state machine.
"""

from enum import Enum, auto


class PipelineState(Enum):
    """
    All possible states in the pipeline execution.
    
    States represent discrete phases of pipeline execution with
    clear entry/exit conditions and transitions.
    """
    
    # Initial state
    IDLE = auto()
    
    # Metadata phase
    FETCHING_METADATA = auto()
    
    # Parsing phase
    PARSING = auto()
    
    # Mass calculation phase
    MASS_CALCULATION = auto()
    
    # Post-processing phase
    POST_PROCESSING = auto()
    
    # Histogram creation phase
    HISTOGRAM_CREATION = auto()
    
    # Terminal states
    COMPLETED = auto()
    FAILED = auto()
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self in (PipelineState.COMPLETED, PipelineState.FAILED)
    
    def __str__(self) -> str:
        """String representation of state."""
        return self.name


# Valid state transitions
VALID_TRANSITIONS = {
    PipelineState.IDLE: {
        PipelineState.FETCHING_METADATA,
        PipelineState.PARSING,
        PipelineState.MASS_CALCULATION,
        PipelineState.POST_PROCESSING,
        PipelineState.HISTOGRAM_CREATION,
        PipelineState.FAILED,
    },
    PipelineState.FETCHING_METADATA: {
        PipelineState.PARSING,
        PipelineState.MASS_CALCULATION,
        PipelineState.COMPLETED,
        PipelineState.FAILED,
    },
    PipelineState.PARSING: {
        PipelineState.MASS_CALCULATION,
        PipelineState.POST_PROCESSING,
        PipelineState.HISTOGRAM_CREATION,
        PipelineState.COMPLETED,
        PipelineState.FAILED,
    },
    PipelineState.MASS_CALCULATION: {
        PipelineState.POST_PROCESSING,
        PipelineState.HISTOGRAM_CREATION,
        PipelineState.COMPLETED,
        PipelineState.FAILED,
    },
    PipelineState.POST_PROCESSING: {
        PipelineState.HISTOGRAM_CREATION,
        PipelineState.COMPLETED,
        PipelineState.FAILED,
    },
    PipelineState.HISTOGRAM_CREATION: {
        PipelineState.COMPLETED,
        PipelineState.FAILED,
    },
    PipelineState.COMPLETED: set(),  # Terminal
    PipelineState.FAILED: set(),     # Terminal
}


def is_valid_transition(from_state: PipelineState, to_state: PipelineState) -> bool:
    """
    Check if a state transition is valid.
    
    Args:
        from_state: Current state
        to_state: Target state
        
    Returns:
        True if transition is valid
    """
    return to_state in VALID_TRANSITIONS.get(from_state, set())
