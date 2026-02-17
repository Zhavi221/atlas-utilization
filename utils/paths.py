"""
Path utilities for pipeline.

Handles timestamped directories and path management.
"""

import os
from pathlib import Path
from datetime import datetime


def create_timestamped_run_dir(base_output_dir: str, run_name: str = None) -> str:
    """
    Create a timestamped directory for the current pipeline run.

    Args:
        base_output_dir: Base output directory (e.g., "./output")
        run_name: Optional run name to include in directory

    Returns:
        Path to the timestamped run directory

    Example:
        create_timestamped_run_dir("./output", "test_run")
        -> "./output/test_run_20260216_211730"
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if run_name:
        dir_name = f"{run_name}_{timestamp}"
    else:
        dir_name = f"run_{timestamp}"

    run_dir = os.path.join(base_output_dir, dir_name)
    os.makedirs(run_dir, exist_ok=True)

    return run_dir


def update_config_paths_with_run_dir(config_dict: dict, run_dir: str) -> dict:
    """
    Inject all output paths into config to use the run directory.

    Paths are always set to the standard sub-directory layout, regardless of
    whether the keys already exist in the YAML.  This allows config files to
    omit per-task path fields entirely (they are derived from run_dir).

    Standard sub-directory layout under run_dir:
        parsed_data/        - parsed ROOT files
        im_arrays/          - invariant mass arrays
        im_arrays_processed/ - post-processed arrays
        histograms/         - final histograms
        plots/              - statistical plots
        logs/               - job logs
        metadata_cache.json - cached file URLs

    Args:
        config_dict: Configuration dictionary
        run_dir: Run directory path

    Returns:
        Updated configuration dictionary
    """
    updated_config = config_dict.copy()

    # Create all stage subdirectories upfront so the output structure is
    # always visible, even for stages that haven't run yet.
    STAGE_DIRS = ["parsed_data", "im_arrays", "im_arrays_processed",
                  "histograms", "plots", "logs"]
    for d in STAGE_DIRS:
        os.makedirs(os.path.join(run_dir, d), exist_ok=True)

    # Inject parsing paths
    if 'parsing_task_config' in updated_config:
        parsing_config = updated_config['parsing_task_config']
        parsing_config['output_path'] = os.path.join(run_dir, "parsed_data")
        parsing_config['file_urls_path'] = os.path.join(run_dir, "metadata_cache.json")
        parsing_config['jobs_logs_path'] = os.path.join(run_dir, "logs")

    # Inject mass calculation paths
    if 'mass_calculation_task_config' in updated_config:
        mass_config = updated_config['mass_calculation_task_config']
        mass_config['input_dir'] = os.path.join(run_dir, "parsed_data")
        mass_config['output_dir'] = os.path.join(run_dir, "im_arrays")

    # Inject post-processing paths
    if 'post_processing_task_config' in updated_config:
        post_config = updated_config['post_processing_task_config']
        post_config['input_dir'] = os.path.join(run_dir, "im_arrays")
        post_config['output_dir'] = os.path.join(run_dir, "im_arrays_processed")

    # Inject histogram creation paths
    if 'histogram_creation_task_config' in updated_config:
        hist_config = updated_config['histogram_creation_task_config']
        hist_config['input_dir'] = os.path.join(run_dir, "im_arrays_processed")
        hist_config['output_dir'] = os.path.join(run_dir, "histograms")

    return updated_config


def get_latest_run_dir(base_output_dir: str) -> str:
    """
    Get the most recent run directory.

    Args:
        base_output_dir: Base output directory

    Returns:
        Path to the latest run directory, or None if none exist
    """
    output_path = Path(base_output_dir)

    if not output_path.exists():
        return None

    # Find all timestamped directories
    run_dirs = [d for d in output_path.iterdir()
                if d.is_dir() and '_' in d.name]

    if not run_dirs:
        return None

    # Sort by modification time
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    return str(run_dirs[0])
