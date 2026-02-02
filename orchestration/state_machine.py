"""
State machine for pipeline execution.

Orchestrates state transitions and handler execution.
"""

import logging
from typing import Dict

from .context import PipelineContext
from .states import PipelineState, is_valid_transition
from .handlers.base import StateHandler


class StateMachine:
    """
    State machine for orchestrating pipeline execution.
    
    Manages state transitions and delegates work to state handlers.
    """
    
    def __init__(self, handlers: Dict[PipelineState, StateHandler]):
        """
        Initialize state machine.
        
        Args:
            handlers: Dict mapping states to their handlers
        """
        self.handlers = handlers
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Validate that all non-terminal states have handlers
        self._validate_handlers()
    
    def _validate_handlers(self):
        """Validate that all necessary handlers are provided."""
        required_states = {
            PipelineState.FETCHING_METADATA,
            PipelineState.PARSING,
            PipelineState.MASS_CALCULATION,
            PipelineState.POST_PROCESSING,
            PipelineState.HISTOGRAM_CREATION,
        }
        
        missing = required_states - set(self.handlers.keys())
        if missing:
            self.logger.warning(
                f"Missing handlers for states: {[str(s) for s in missing]}"
            )
    
    def run(self, initial_context: PipelineContext) -> PipelineContext:
        """
        Run the state machine until a terminal state is reached.
        
        Args:
            initial_context: Initial pipeline context
            
        Returns:
            Final pipeline context
        """
        context = initial_context
        iteration = 0
        max_iterations = 100  # Safety limit
        
        self.logger.info("=" * 60)
        self.logger.info("Starting pipeline execution")
        self.logger.info("=" * 60)
        
        while not context.is_terminal and iteration < max_iterations:
            iteration += 1
            
            try:
                context = self._execute_state(context)
            except Exception as e:
                self.logger.error(f"Error in state {context.current_state}: {e}", exc_info=True)
                context = context.with_error(
                    message=f"Error in {context.current_state}: {str(e)}",
                    details={"iteration": iteration, "state": str(context.current_state)}
                )
                break
        
        if iteration >= max_iterations:
            self.logger.error("State machine exceeded maximum iterations")
            context = context.with_error(
                message="Pipeline exceeded maximum iterations",
                details={"iterations": iteration}
            )
        
        self._log_final_state(context)
        return context
    
    def _execute_state(self, context: PipelineContext) -> PipelineContext:
        """
        Execute the current state's handler.
        
        Args:
            context: Current pipeline context
            
        Returns:
            Updated pipeline context
        """
        current_state = context.current_state
        
        self.logger.info(f"Current state: {current_state}")
        
        # Get handler for current state
        handler = self.handlers.get(current_state)
        
        if handler is None:
            self.logger.warning(f"No handler for state {current_state}, skipping to next")
            # Determine next state without executing handler
            from .handlers.base import StateHandler
            temp_handler = type('TempHandler', (StateHandler,), {
                'handle': lambda self, ctx: (ctx, self._determine_next_state(ctx))
            })()
            next_state = temp_handler._determine_next_state(context)
            return context.with_state(next_state)
        
        # Execute handler
        try:
            updated_context, next_state = handler.handle(context)
            
            # Validate transition
            if not is_valid_transition(current_state, next_state):
                self.logger.error(
                    f"Invalid transition: {current_state} → {next_state}"
                )
                return context.with_error(
                    message=f"Invalid state transition: {current_state} → {next_state}"
                )
            
            # Transition to next state
            self.logger.info(f"Transition: {current_state} → {next_state}")
            return updated_context.with_state(next_state)
            
        except Exception as e:
            self.logger.error(f"Handler error in {current_state}: {e}", exc_info=True)
            raise
    
    def _log_final_state(self, context: PipelineContext):
        """Log final pipeline state."""
        self.logger.info("=" * 60)
        
        if context.is_successful:
            self.logger.info("✓ Pipeline completed successfully")
        else:
            self.logger.error(f"✗ Pipeline failed: {context.error_message}")
        
        self.logger.info(f"Final state: {context.current_state}")
        self.logger.info(f"Elapsed time: {context.elapsed_time:.1f}s")
        
        summary = context.get_summary()
        for key, value in summary.items():
            self.logger.info(f"  {key}: {value}")
        
        self.logger.info("=" * 60)
