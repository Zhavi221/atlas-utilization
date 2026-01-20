import logging
import sys
import argparse
import yaml
import json
from datetime import datetime
import os

CONFIG_PATH = "configs/pipeline_config.yaml"

def main():
    """Main entry point for the pipeline."""
    logger = init_logging()
    args = parse_args()
    config = load_config(args)
    tasks = config["tasks"]

    # Initialize directories
    initialize_directories(config, logger)

    parsed_files = []
    im_files = []
    processed_im_files = []
    
    if tasks["do_parsing"]:
        parsed_files = parsing_task(
            config["parsing_task_config"], 
            config.get("testing_config", {}), 
            config.get("run_metadata", {}), logger)
    
    if tasks["do_mass_calculating"]:
        im_files = mass_calculation_task(
            config["mass_calculate_task_config"], 
            config.get("testing_config", {}), 
            logger, 
            parsed_files)
    
    if tasks["do_post_processing"]:
        processed_im_files = post_processing_task(
            config["post_processing_task_config"], 
            logger, 
            tasks["do_mass_calculating"],  #TODO pass tasks object instead of booleans 
            im_files
        )
    
    if tasks["do_histogram_creation"]:
        histogram_creation_task(
            config["histogram_creation_task_config"],
            logger,
            tasks["do_post_processing"],
            tasks["do_mass_calculating"],
            processed_im_files,
            im_files
        )


def log_task_boundary(logger, task_name: str):
    """Log task boundary for consistent formatting."""
    logger.info("=" * 60)
    logger.info(f"{task_name}")
    logger.info("=" * 60)


def normalize_list(value):
    """Normalize None to empty list."""
    return value if value is not None else []

def parse_args():
    """Parse command line arguments."""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", help="Config file", default=CONFIG_PATH)
    arg_parser.add_argument("--test_run_index", help="Config file", default=None)
    arg_parser.add_argument("--batch_job_index", default=None)
    arg_parser.add_argument("--total_batch_jobs", default=None)
    return arg_parser.parse_args()


def parsing_task(parsing_config, testing_config, run_metadata, logger):
    """Parse ROOT files and extract particle data."""
    log_task_boundary(logger, "Starting parsing task")
    
    pipeline_config = parsing_config["pipeline_config"]
    
    if pipeline_config["use_multiprocessing"]:
        logger.info("Using multiprocessing mode")
        from src.pipelines import multiprocessing_pipeline
        # Pass both configs - multiprocessing needs run_metadata from main config
        parsing_config_with_metadata = parsing_config.copy()
        parsing_config_with_metadata["run_metadata"] = run_metadata
        parsed_files = multiprocessing_pipeline.parse_with_per_chunk_subprocess(
            parsing_config_with_metadata, testing_config
        )
    else:
        logger.info("Using single-process mode")
        from src.pipelines import parsing_pipeline
        parsed_files = parsing_pipeline.parse(parsing_config, testing_config)
    
    parsed_files = normalize_list(parsed_files)
    logger.info(f"Parsed {len(parsed_files)} files")
    return parsed_files
         
            
def mass_calculation_task(mass_calculate_config, testing_config, logger, parsed_files):
    """Calculate invariant masses from parsed ROOT files."""
    log_task_boundary(logger, "Starting IM calculation task")
    
    from src.pipelines import im_pipeline
    im_files = im_pipeline.mass_calculate(mass_calculate_config, testing_config, file_list=parsed_files)
    im_files = normalize_list(im_files)
    
    logger.info(f"Created {len(im_files)} IM array files")
    return im_files


def post_processing_task(post_processing_config, logger, do_mass_calculating, im_files):
    """Post-process invariant mass arrays."""
    log_task_boundary(logger, "Starting post-processing task")
    
    from src.pipelines import post_processing_pipeline
    
    if do_mass_calculating and im_files:
        logger.info(f"Processing {len(im_files)} IM files from calculation")
        processed_files = post_processing_pipeline.process_im_arrays(
            post_processing_config, file_list=im_files
        )
    elif check_im_arrays_exist(post_processing_config["input_dir"]):
        logger.info("Scanning input directory for IM arrays")
        processed_files = post_processing_pipeline.process_im_arrays(post_processing_config)
    else:
        logger.warning("No IM arrays available. Skipping post-processing.")
        return []
    
    processed_files = normalize_list(processed_files)
    if processed_files:
        logger.info(f"Processed {len(processed_files)} IM array files")
    return processed_files


def histogram_creation_task(
    histogram_creation_config, 
    logger, 
    do_post_processing, 
    do_mass_calculating, 
    processed_im_files, 
    im_files
):
    """Create histograms from invariant mass arrays."""
    log_task_boundary(logger, "Starting histogram creation task")
    
    from src.pipelines import histograms_pipeline
    
    # Priority: processed > raw > directory scan
    if do_post_processing and processed_im_files:
        logger.info(f"Creating histograms from {len(processed_im_files)} processed files")
        histograms_pipeline.create_histograms(histogram_creation_config, file_list=processed_im_files)
    elif do_mass_calculating and im_files:
        logger.info(f"Creating histograms from {len(im_files)} raw IM files")
        histograms_pipeline.create_histograms(histogram_creation_config, file_list=im_files)
    elif check_im_arrays_exist(histogram_creation_config["input_dir"]):
        logger.info("Scanning input directory for IM arrays")
        histograms_pipeline.create_histograms(histogram_creation_config)
    else:
        logger.warning("No IM arrays found. Skipping histogram creation.")


def init_logging():
    """Initialize logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def check_im_arrays_exist(im_masses_dir):
    """Check if IM arrays exist in the given directory."""
    return os.path.exists(im_masses_dir) and bool(os.listdir(im_masses_dir))


def initialize_directories(config, logger):
    """
    Initialize only the output directories needed for enabled tasks.
    
    Only creates:
    - Output directories for enabled tasks (not input directories)
    - Skips archived paths (paths containing "archive")
    - Logging directories for parsing if enabled
    """
    tasks = config.get("tasks", {})
    directories_to_create = []
    
    # Logging directory (always create, regardless of which tasks are enabled)
    if "parsing_task_config" in config:
        parsing_config = config["parsing_task_config"]
        if "atlasparser_config" in parsing_config:
            atlasparser_config = parsing_config["atlasparser_config"]
            if "jobs_logs_path" in atlasparser_config:
                jobs_logs_path = atlasparser_config["jobs_logs_path"]
                # Skip archived paths
                if "archive" not in jobs_logs_path:
                    directories_to_create.append(jobs_logs_path)
    
    # Parsing directories (only if parsing is enabled)
    if tasks.get("do_parsing", False):
        if "parsing_task_config" in config:
            parsing_config = config["parsing_task_config"]
            
            # Output directory for parsed root files
            if "pipeline_config" in parsing_config:
                pipeline_config = parsing_config["pipeline_config"]
                if "output_path" in pipeline_config:
                    output_path = pipeline_config["output_path"]
                    # Skip archived paths
                    if "archive" not in output_path:
                        directories_to_create.append(output_path)
            
            # Logging directory
            if "atlasparser_config" in parsing_config:
                atlasparser_config = parsing_config["atlasparser_config"]
                if "logging_path" in atlasparser_config:
                    logging_path = atlasparser_config["logging_path"]
                    # Skip archived paths
                    if "archive" not in logging_path:
                        directories_to_create.append(logging_path)
    
    # Mass calculation directories (only if enabled)
    if tasks.get("do_mass_calculating", False):
        if "mass_calculate_task_config" in config:
            mass_config = config["mass_calculate_task_config"]
            # Only create output directory, input should already exist
            if "output_dir" in mass_config:
                output_dir = mass_config["output_dir"]
                if "archive" not in output_dir:
                    directories_to_create.append(output_dir)
    
    # Post-processing directories (only if enabled)
    if tasks.get("do_post_processing", False):
        if "post_processing_task_config" in config:
            post_config = config["post_processing_task_config"]
            # Only create output directory, input should already exist
            if "output_dir" in post_config:
                output_dir = post_config["output_dir"]
                if "archive" not in output_dir:
                    directories_to_create.append(output_dir)
    
    # Histogram creation directories (only if enabled)
    if tasks.get("do_histogram_creation", False):
        if "histogram_creation_task_config" in config:
            hist_config = config["histogram_creation_task_config"]
            # Only create output directory, input should already exist (or is archived)
            if "output_dir" in hist_config:
                output_dir = hist_config["output_dir"]
                if "archive" not in output_dir:
                    directories_to_create.append(output_dir)
    
    # Remove duplicates and None values
    directories_to_create = list(set(d for d in directories_to_create if d))
    
    # Create directories
    created_count = 0
    for directory in directories_to_create:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")
                created_count += 1
            except Exception as e:
                logger.warning(f"Could not create directory {directory}: {e}")
    
    if created_count > 0:
        logger.info(f"Initialized {created_count} directories")
    else:
        logger.debug("All directories already exist")


def load_config(args):
    """Load configuration from YAML file."""
    config_path = args.config if args.config else CONFIG_PATH
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    if config["testing_config"]["is_on"]:
        _load_testing_config(config, args)
    else:
        _load_production_config(config, args)
    
    return config


def _load_testing_config(config, args):
    """Load testing configuration."""
    test_run_index = int(args.test_run_index) if args.test_run_index else config["testing_config"]["test_run_index"]
    
    with open(config["testing_config"]["testing_jobs_path"]) as f:
        testing_runs = json.load(f)
        cur_run_config = testing_runs[test_run_index]
    
    if "pipeline_config" in cur_run_config:
        config["parsing_task_config"]["pipeline_config"].update(cur_run_config["pipeline_config"])
    
    if "atlasparser_config" in cur_run_config:
        config["parsing_task_config"]["atlasparser_config"].update(cur_run_config["atlasparser_config"])
    
    if "run_metadata" in cur_run_config:
        config["run_metadata"].update(cur_run_config["run_metadata"])


def _append_run_folder_to_path(path, run_folder):
    """Append run folder to a data path, inserting it after the base directory.
    
    For logs, the run folder contains a "logs" subdirectory.
    
    Example:
        Input:  "/storage/agrp/netalev/data/root_files/"
        Output: "/storage/agrp/netalev/data/{run_folder}/root_files/"
        
        Input:  "/storage/agrp/netalev/data/logs/"
        Output: "/storage/agrp/netalev/data/{run_folder}/logs/"
    """
    if not path or not run_folder:
        return path
    
    # Check if path is absolute and contains "data"
    has_trailing_slash = path.endswith('/')
    is_absolute = path.startswith('/')
    
    # Skip relative paths that don't contain data directory
    if not is_absolute and "data" not in path:
        return path
    
    # Normalize path (remove trailing slash temporarily)
    path_normalized = path.rstrip('/')
    
    # Split path into components
    parts = [p for p in path_normalized.split('/') if p]  # Remove empty strings
    
    # Find where "data" appears in the path
    target_index = None
    for i, part in enumerate(parts):
        if part == "data":
            target_index = i
            break
    
    if target_index is not None:
        # Insert run_folder after "data"
        new_parts = parts[:target_index + 1] + [run_folder] + parts[target_index + 1:]
        result = '/' + '/'.join(new_parts)
        return result + '/' if has_trailing_slash else result
    
    # Fallback: if absolute path but no data found, append at end
    if is_absolute:
        return os.path.join(path_normalized, run_folder) + ('/' if has_trailing_slash else '')
    
    # Return original if we can't process it
    return path


def _load_production_config(config, args):
    """Load production configuration and append run folder to data paths.
    
    Sets batch_job_index and total_batch_jobs dynamically from command-line args
    for all pipeline stages that support batching.
    """
    run_time = datetime.now().strftime("%d_%m_%Y_%H:%M")
    
    if args.batch_job_index is not None:
        batch_job_index = int(args.batch_job_index)
        total_batch_jobs = int(args.total_batch_jobs) if args.total_batch_jobs else None
        run_name = (
            config["run_metadata"]["run_name"] 
            or f"job_idx{batch_job_index}_{run_time}"
        )
    else:
        batch_job_index = None
        total_batch_jobs = None
        run_name = config["run_metadata"]["run_name"] or f"run_{run_time}"
    
    # Create run folder name: {run_name}_{date}
    run_folder = f"{run_name}_{run_time}"
    
    # Set batch job parameters in run_metadata (used by parsing pipeline)
    config["run_metadata"] = {
        "batch_job_index": batch_job_index,
        "run_name": run_name,
        "total_batch_jobs": total_batch_jobs
    }
    
    # Set batch job parameters for all stages that support batching
    if batch_job_index is not None:
        if "mass_calculate_task_config" in config:
            config["mass_calculate_task_config"]["batch_job_index"] = batch_job_index
            config["mass_calculate_task_config"]["total_batch_jobs"] = total_batch_jobs
        
        if "post_processing_task_config" in config:
            config["post_processing_task_config"]["batch_job_index"] = batch_job_index
            config["post_processing_task_config"]["total_batch_jobs"] = total_batch_jobs
        
        if "histogram_creation_task_config" in config:
            config["histogram_creation_task_config"]["batch_job_index"] = batch_job_index
            config["histogram_creation_task_config"]["total_batch_jobs"] = total_batch_jobs
    
    # Append run folder to all data paths
    if "parsing_task_config" in config:
        if "pipeline_config" in config["parsing_task_config"]:
            pipeline_config = config["parsing_task_config"]["pipeline_config"]
            if "output_path" in pipeline_config:
                pipeline_config["output_path"] = _append_run_folder_to_path(
                    pipeline_config["output_path"], run_folder
                )
            if "file_urls_path" in pipeline_config:
                pipeline_config["file_urls_path"] = _append_run_folder_to_path(
                    pipeline_config["file_urls_path"], run_folder
                )
        
        if "atlasparser_config" in config["parsing_task_config"]:
            atlasparser_config = config["parsing_task_config"]["atlasparser_config"]
            if "jobs_logs_path" in atlasparser_config:
                # Append run folder to logs path as well
                atlasparser_config["jobs_logs_path"] = _append_run_folder_to_path(
                    atlasparser_config["jobs_logs_path"], run_folder
                )
    
    # Update mass calculation paths
    if "mass_calculate_task_config" in config:
        mass_config = config["mass_calculate_task_config"]
        if "input_dir" in mass_config:
            mass_config["input_dir"] = _append_run_folder_to_path(
                mass_config["input_dir"], run_folder
            )
        if "output_dir" in mass_config:
            mass_config["output_dir"] = _append_run_folder_to_path(
                mass_config["output_dir"], run_folder
            )
    
    # Update post-processing paths (skip archived paths)
    if "post_processing_task_config" in config:
        post_config = config["post_processing_task_config"]
        if "input_dir" in post_config:
            # Only append run folder if not an archived path
            if "archive" not in post_config["input_dir"]:
                post_config["input_dir"] = _append_run_folder_to_path(
                    post_config["input_dir"], run_folder
                )
        if "output_dir" in post_config:
            # Only append run folder if not an archived path
            if "archive" not in post_config["output_dir"]:
                post_config["output_dir"] = _append_run_folder_to_path(
                    post_config["output_dir"], run_folder
                )
    
    # Update histogram creation paths (skip archived paths)
    if "histogram_creation_task_config" in config:
        hist_config = config["histogram_creation_task_config"]
        if "input_dir" in hist_config:
            # Only append run folder if not an archived path
            if "archive" not in hist_config["input_dir"]:
                hist_config["input_dir"] = _append_run_folder_to_path(
                    hist_config["input_dir"], run_folder
                )
        if "output_dir" in hist_config:
            # Only append run folder if not an archived path
            if "archive" not in hist_config["output_dir"]:
                hist_config["output_dir"] = _append_run_folder_to_path(
                    hist_config["output_dir"], run_folder
                )


if __name__ == "__main__":
    main()