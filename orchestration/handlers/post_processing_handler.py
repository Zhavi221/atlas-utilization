"""
PostProcessingHandler - Handles post-processing state.

Delegates to the existing post_processing_pipeline module from atlas_utilization.
"""

import sys
from datetime import datetime
from pathlib import Path

from orchestration.context import PipelineContext
from orchestration.states import PipelineState
from .base import StateHandler


class PostProcessingHandler(StateHandler):
    """
    Handler for POST_PROCESSING state.

    Converts PipelineConfig into a plain dict and calls
    ``post_processing_pipeline.process_im_arrays`` from atlas_utilization.
    """

    def handle(self, context: PipelineContext) -> tuple[PipelineContext, PipelineState]:
        self._log_state_entry(context)

        pp = context.config.post_processing_config
        if pp is None:
            self.logger.warning("No post_processing_config – skipping")
            return context, self._determine_next_state(context)

        start = datetime.now()

        config_dict = {
            "input_dir": pp.input_dir,
            "output_dir": pp.output_dir,
            "peak_detection_bin_width_gev": pp.peak_detection_bin_width_gev,
        }

        # If the previous stage produced files, pass them explicitly
        file_list = None
        if context.im_files:
            file_list = [Path(f).name for f in context.im_files if f.endswith(".npy")]

        if '/srv01/agrp/netalev/atlas_utilization' not in sys.path:
            sys.path.insert(0, '/srv01/agrp/netalev/atlas_utilization')

        from src.pipelines.post_processing_pipeline import process_im_arrays  # type: ignore

        self.logger.info(
            f"Running post-processing: input={pp.input_dir}  output={pp.output_dir}"
        )

        processed_files = process_im_arrays(config_dict, file_list=file_list) or []

        elapsed = (datetime.now() - start).total_seconds()
        self.logger.info(
            f"Post-processing complete: {len(processed_files)} processed arrays "
            f"in {elapsed:.1f}s"
        )

        updated = context.with_processed_files(processed_files)
        next_state = self._determine_next_state(updated)
        self._log_state_exit(context, next_state)
        return updated, next_state
