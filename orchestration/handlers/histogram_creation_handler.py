"""
HistogramCreationHandler - Handles histogram creation state.

Delegates to the histograms_pipeline module.
"""

from datetime import datetime
from pathlib import Path

from orchestration.context import PipelineContext
from orchestration.states import PipelineState
from .base import StateHandler


class HistogramCreationHandler(StateHandler):
    """
    Handler for HISTOGRAM_CREATION state.

    Converts PipelineConfig into a plain dict and calls
    ``histograms_pipeline.create_histograms``.
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
            "apply_peak_removal_at_histogram_level": hc.apply_peak_removal_at_histogram_level,
            "batch_job_index": context.config.batch_job_index,
        }

        # Attach the MC weights registry when MC weighting is enabled so that
        # histograms are filled with per-source normalization weights.
        weights_registry = self._load_weights_registry(context)
        if weights_registry is not None:
            config_dict["weights_registry"] = weights_registry

        # If the previous stage produced files, pass them explicitly
        file_list = None
        if context.processed_files:
            file_list = [Path(f).name for f in context.processed_files if f.endswith(".npy") or f.endswith(".sqlite")]

        from services.pipelines.histograms_pipeline import create_histograms

        self.logger.info(
            f"Running histogram creation: input={hc.input_dir}  output={hc.output_dir}"
        )

        create_histograms(config_dict, file_list=file_list)

        elapsed = (datetime.now() - start).total_seconds()
        self.logger.info(f"Histogram creation complete in {elapsed:.1f}s")

        next_state = self._determine_next_state(context)
        self._log_state_exit(context, next_state)
        return context, next_state

    def _load_weights_registry(self, context: PipelineContext):
        """
        Load the MC weights registry when MC weighting is enabled.

        The registry is expected to be persisted (as weights_registry.json)
        next to the histogram input directory by an upstream step that knows
        each source file's DSID. Returns None when weighting is disabled or no
        registry is available (histograms then fill with unit weights).
        """
        mc_cfg = getattr(context.config, "mc_weighting_config", None)
        if mc_cfg is None or not mc_cfg.enabled:
            return None

        from pathlib import Path
        from services.calculations.weights_registry import WeightsRegistry

        hc = context.config.histogram_creation_config
        registry_path = str(Path(hc.input_dir) / "weights_registry.json")
        registry = WeightsRegistry.load(registry_path)
        if registry is None:
            self.logger.warning(
                f"MC weighting enabled but no weights registry found at {registry_path}; "
                "histograms will use unit weights"
            )
        return registry
