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


def _set_if_relative(config: dict, key: str, value: str):
    """Only overwrite a path if it's missing or relative (not an absolute override)."""
    if key not in config or not os.path.isabs(config[key]):
        config[key] = value


def update_config_paths_with_run_dir(config_dict: dict, run_dir: str) -> dict:
    """
    Inject default paths into config to use the run directory.

    Relative paths (the ``./output/...`` defaults from YAML) are replaced
    with the corresponding sub-directory under *run_dir*.  Absolute paths
    already set in the config are left untouched, so you can point any
    stage at an external directory by setting an absolute path in config.yaml.

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

    STAGE_DIRS = ["parsed_data", "im_arrays", "im_arrays_processed",
                  "histograms", "plots", "logs"]
    for d in STAGE_DIRS:
        os.makedirs(os.path.join(run_dir, d), exist_ok=True)

    if 'parsing_task_config' in updated_config:
        parsing_config = updated_config['parsing_task_config']
        _set_if_relative(parsing_config, 'output_path', os.path.join(run_dir, "parsed_data"))
        _set_if_relative(parsing_config, 'file_urls_path', os.path.join(run_dir, "metadata_cache.json"))
        _set_if_relative(parsing_config, 'jobs_logs_path', os.path.join(run_dir, "logs"))

    if 'mass_calculation_task_config' in updated_config:
        mass_config = updated_config['mass_calculation_task_config']
        _set_if_relative(mass_config, 'input_dir', os.path.join(run_dir, "parsed_data"))
        _set_if_relative(mass_config, 'output_dir', os.path.join(run_dir, "im_arrays"))

    if 'post_processing_task_config' in updated_config:
        post_config = updated_config['post_processing_task_config']
        _set_if_relative(post_config, 'input_dir', os.path.join(run_dir, "im_arrays"))
        _set_if_relative(post_config, 'output_dir', os.path.join(run_dir, "im_arrays_processed"))

    if 'histogram_creation_task_config' in updated_config:
        hist_config = updated_config['histogram_creation_task_config']
        _set_if_relative(hist_config, 'input_dir', os.path.join(run_dir, "im_arrays_processed"))
        _set_if_relative(hist_config, 'output_dir', os.path.join(run_dir, "histograms"))

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
