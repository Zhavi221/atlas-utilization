#!/usr/bin/env python3
"""
Main entry point for refactored ATLAS pipeline.

Supports:
  - Single-job execution (default)
  - Batch job execution via --batch-job-index / --total-batch-jobs
  - Shared run directory via --run-dir  (for multi-job PBS arrays)
  - Merge-only mode via --merge-only --run-dir <path>
  - Post-run plot regeneration via --plots-only --run-dir <path>

Architecture (multi-job):
  Each batch job runs the FULL pipeline (parse → IM calc → post-proc → histograms)
  on its own slice of data.  Each batch writes a separate histogram file
  (batch_N.root) and a stats JSON (batch_N_stats.json).
  The merge job (--merge-only) combines histogram files via hadd, aggregates
  stats, and generates plots.
"""

import sys
import os
import logging
import argparse
import yaml
from pathlib import Path

from domain.config import PipelineConfig
from pipeline.executor import PipelineExecutor
from utils.paths import create_timestamped_run_dir, update_config_paths_with_run_dir


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ATLAS Pipeline - Refactored Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single job – all enabled stages (default)
  python main.py

  # Single job with custom config
  python main.py --config my_config.yaml

  # Batch array job – runs FULL pipeline on its file slice
  python main.py --batch-job-index 1 --total-batch-jobs 4 \\
      --run-dir /storage/agrp/netalev/data/run_20260217

  # Merge job – hadd histograms + aggregate stats + generate plots
  python main.py --merge-only --run-dir /storage/agrp/netalev/data/run_20260217

  # Re-generate plots from an existing run (no processing)
  python main.py --plots-only --run-dir /storage/agrp/netalev/data/run_20260217

  # Dry-run to validate config
  python main.py --dry-run
        """
    )

    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate configuration without running pipeline"
    )

    # --- Batch job arguments ---
    batch_group = parser.add_argument_group("Batch Job Options")
    batch_group.add_argument(
        "--batch-job-index", type=int, default=None,
        help="This job's index (1-based, matching PBS $PBS_ARRAY_INDEX)"
    )
    batch_group.add_argument(
        "--total-batch-jobs", type=int, default=None,
        help="Total number of batch jobs"
    )
    batch_group.add_argument(
        "--run-dir", type=str, default=None,
        help="Pre-created shared run directory (skips timestamped dir creation)"
    )

    # --- Post-run options ---
    post_group = parser.add_argument_group("Post-Run Options")
    post_group.add_argument(
        "--merge-only", action="store_true",
        help="Merge batch outputs: hadd histograms + aggregate stats + generate plots"
    )
    post_group.add_argument(
        "--plots-only", action="store_true",
        help="Only generate plots from existing data in --run-dir (no pipeline execution)"
    )

    args = parser.parse_args()

    # Validation
    if args.batch_job_index is not None and args.total_batch_jobs is None:
        parser.error("--total-batch-jobs is required when --batch-job-index is set")
    if args.total_batch_jobs is not None and args.batch_job_index is None:
        parser.error("--batch-job-index is required when --total-batch-jobs is set")
    if (args.plots_only or args.merge_only) and args.run_dir is None:
        parser.error("--run-dir is required when --plots-only or --merge-only is set")
    if args.plots_only and args.merge_only:
        parser.error("--plots-only and --merge-only are mutually exclusive")

    return args


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("ATLAS Pipeline - Refactored Architecture")
    logger.info("=" * 60)

    try:
        # ------------------------------------------------------------------
        # PLOTS-ONLY MODE: regenerate plots from existing run output
        # ------------------------------------------------------------------
        if args.plots_only:
            logger.info(f"Plots-only mode: reading data from {args.run_dir}")
            config_dict = load_config(args.config)
            config_dict = update_config_paths_with_run_dir(config_dict, args.run_dir)
            config = PipelineConfig.from_dict(config_dict)

            executor = PipelineExecutor(config)
            executor.generate_plots_from_output(args.run_dir)
            logger.info("✓ Plots generated successfully")
            return 0

        # ------------------------------------------------------------------
        # MERGE-ONLY MODE: hadd histograms + aggregate stats + plots
        # ------------------------------------------------------------------
        if args.merge_only:
            logger.info(f"Merge-only mode: merging outputs in {args.run_dir}")
            config_dict = load_config(args.config)
            config_dict = update_config_paths_with_run_dir(config_dict, args.run_dir)
            config = PipelineConfig.from_dict(config_dict)

            executor = PipelineExecutor(config)
            executor.merge_outputs(args.run_dir)
            logger.info("✓ Merge completed successfully")
            return 0

        # ------------------------------------------------------------------
        # NORMAL / BATCH PIPELINE MODE
        # ------------------------------------------------------------------
        logger.info(f"Loading configuration from: {args.config}")
        config_dict = load_config(args.config)

        # Inject batch job params from CLI into config (override YAML values)
        if args.batch_job_index is not None:
            config_dict.setdefault("run_metadata", {})
            config_dict["run_metadata"]["batch_job_index"] = args.batch_job_index
            config_dict["run_metadata"]["total_batch_jobs"] = args.total_batch_jobs

        # Determine run directory
        if args.run_dir:
            run_dir = args.run_dir
            os.makedirs(run_dir, exist_ok=True)
            logger.info(f"Using shared run directory: {run_dir}")
        else:
            run_metadata = config_dict.get('run_metadata', {})
            run_name = run_metadata.get('run_name', 'pipeline_run')
            base_output = run_metadata.get('base_output_dir', './output')
            run_dir = create_timestamped_run_dir(base_output, run_name)
            logger.info(f"Created timestamped run directory: {run_dir}")

        # Update config paths to use run directory
        config_dict = update_config_paths_with_run_dir(config_dict, run_dir)

        # Per-batch histogram filename so batch jobs don't collide
        if args.batch_job_index is not None:
            hist_cfg = config_dict.get("histogram_creation_task_config", {})
            hist_cfg["output_filename"] = f"batch_{args.batch_job_index}.root"

        # Create validated config
        config = PipelineConfig.from_dict(config_dict)
        logger.info("Configuration loaded and validated successfully")

        batch_info = ""
        if config.batch_job_index is not None:
            batch_info = f" (batch {config.batch_job_index}/{config.total_batch_jobs})"
        logger.info(f"Output directory: {run_dir}{batch_info}")

        if args.dry_run:
            logger.info("Dry run mode - configuration is valid, exiting")
            logger.info(f"Enabled tasks: {[k for k, v in vars(config.tasks).items() if v]}")
            logger.info(f"Run directory: {run_dir}")
            return 0

        # Create executor and run pipeline
        executor = PipelineExecutor(config)
        final_context = executor.run()

        # Save per-batch stats JSON for later aggregation
        if config.batch_job_index is not None:
            executor.save_batch_stats(run_dir, config.batch_job_index, final_context)

        # Generate plots only for single-job runs (batch plots are deferred to merge)
        if config.batch_job_index is None:
            executor.generate_plots_from_output(run_dir)

        if final_context.is_successful:
            logger.info(f"✓ Pipeline completed successfully{batch_info}")
            return 0
        else:
            logger.error(f"✗ Pipeline failed: {final_context.error_message}")
            return 1

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
