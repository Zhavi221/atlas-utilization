"""
HistogramCreationHandler - Handles histogram creation state.

Delegates to the histograms_pipeline module.
"""

import fcntl
import os
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

        # Global bin edges must exist before any histogram is filled, so the
        # scan runs first — otherwise concurrent batch jobs would each pick
        # their own local min/max and `hadd` would fail to merge them.
        self._ensure_global_ranges(hc)

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
            "total_batch_jobs": context.config.total_batch_jobs,
            "global_ranges_path": getattr(hc, 'global_ranges_path', None),
        }

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

    def _ensure_global_ranges(self, hc) -> None:
        """
        Compute global min/max per bumpnet signature across ALL processed
        SQLite shards and save to ``hc.global_ranges_path``, unless it has
        already been computed (e.g. by a concurrent histogram batch job).
        """
        global_ranges_path = getattr(hc, 'global_ranges_path', None)
        if not global_ranges_path:
            return
        if os.path.exists(global_ranges_path):
            self.logger.info(f"Using existing global ranges: {global_ranges_path}")
            return

        proc_dir = hc.input_dir
        if not os.path.isdir(proc_dir):
            self.logger.error(f"Cannot compute global ranges — input dir not found: {proc_dir}")
            raise RuntimeError(f"Cannot compute global ranges — input dir not found: {proc_dir}")

        sqlite_files = sorted(f for f in os.listdir(proc_dir) if f.endswith(".sqlite"))
        if not sqlite_files:
            self.logger.error(f"No processed SQLite files found in {proc_dir}")
            raise RuntimeError(f"No processed SQLite files found in {proc_dir}")

        ranges_dir = os.path.dirname(global_ranges_path)
        os.makedirs(ranges_dir, exist_ok=True)
        lock_path = global_ranges_path + ".lock"

        from services.pipelines.histograms_pipeline import compute_global_ranges, save_global_ranges

        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                if os.path.exists(global_ranges_path):
                    self.logger.info(
                        f"Global ranges already computed by a concurrent job: {global_ranges_path}"
                    )
                    return

                self.logger.info(
                    f"Scanning {len(sqlite_files)} SQLite files for global ranges..."
                )
                ranges = compute_global_ranges(
                    sqlite_files, proc_dir, exclude_outliers=hc.exclude_outliers
                )
                save_global_ranges(ranges, global_ranges_path)
                self.logger.info(
                    f"Saved global ranges for {len(ranges)} signatures to {global_ranges_path}"
                )
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
