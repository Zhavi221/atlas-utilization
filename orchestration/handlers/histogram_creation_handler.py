"""
HistogramCreationHandler - Handles histogram creation state.

Delegates to the existing histograms_pipeline module from atlas_utilization.
"""

import sys
from datetime import datetime
from pathlib import Path

from orchestration.context import PipelineContext
from orchestration.states import PipelineState
from .base import StateHandler


class HistogramCreationHandler(StateHandler):
    """
    Handler for HISTOGRAM_CREATION state.

    Converts PipelineConfig into a plain dict and calls
    ``histograms_pipeline.create_histograms`` from atlas_utilization.
    """

    def handle(self, context: PipelineContext) -> tuple[PipelineContext, PipelineState]:
        self._log_state_entry(context)

        hc = context.config.histogram_creation_config
        if hc is None:
            self.logger.warning("No histogram_creation_config – skipping")
            return context, self._determine_next_state(context)

        start = datetime.now()

        config_dict = {
            "input_dir": hc.input_dir,
            "output_dir": hc.output_dir,
            "bin_width_gev": hc.bin_width_gev,
            "single_output_file": hc.single_output_file,
            "output_filename": hc.output_filename,
            "exclude_outliers": hc.exclude_outliers,
            "use_bumpnet_naming": hc.use_bumpnet_naming,
        }

        # If the previous stage produced files, pass them explicitly
        file_list = None
        if context.processed_files:
            file_list = [Path(f).name for f in context.processed_files if f.endswith(".npy")]

        if '/srv01/agrp/netalev/atlas_utilization' not in sys.path:
            sys.path.insert(0, '/srv01/agrp/netalev/atlas_utilization')

        from src.pipelines.histograms_pipeline import create_histograms  # type: ignore

        self.logger.info(
            f"Running histogram creation: input={hc.input_dir}  output={hc.output_dir}"
        )

        create_histograms(config_dict, file_list=file_list)

        elapsed = (datetime.now() - start).total_seconds()
        self.logger.info(f"Histogram creation complete in {elapsed:.1f}s")

        next_state = self._determine_next_state(context)
        self._log_state_exit(context, next_state)
        return context, next_state
