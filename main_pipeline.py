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
        parsed_files = parsing_task(config["parsing_config"], logger)
    
    if tasks["do_mass_calculating"]:
        im_files = mass_calculation_task(config["mass_calculate"], logger, parsed_files)
    
    if tasks["do_post_processing"]:
        processed_im_files = post_processing_task(
            config["post_processing"], 
            logger, 
            tasks["do_mass_calculating"], 
            im_files
        )
    
    if tasks["do_histogram_creation"]:
        histogram_creation_task(
            config["histogram_creation"],
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


def parsing_task(parsing_config, logger):
    """Parse ROOT files and extract particle data."""
    log_task_boundary(logger, "Starting parsing task")
    
    pipeline_config = parsing_config["pipeline_config"]
    
    if pipeline_config["parse_in_multiprocessing"]:
        logger.info("Using multiprocessing mode")
        from src.pipelines import multiprocessing_pipeline
        parsed_files = multiprocessing_pipeline.parse_with_per_chunk_subprocess(parsing_config)
    else:
        logger.info("Using single-process mode")
        from src.pipelines import parsing_pipeline
        parsed_files = parsing_pipeline.parse(parsing_config)
    
    parsed_files = normalize_list(parsed_files)
    logger.info(f"Parsed {len(parsed_files)} files")
    return parsed_files
         
            
def mass_calculation_task(mass_calculate_config, logger, parsed_files):
    """Calculate invariant masses from parsed ROOT files."""
    log_task_boundary(logger, "Starting IM calculation task")
    
    from src.pipelines import im_pipeline
    im_files = im_pipeline.mass_calculate(mass_calculate_config, file_list=parsed_files)
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
    Initialize all data and log directories if they don't exist.
    
    Creates directories for:
    - Parsing output and logs
    - Invariant mass calculation input/output
    - Post-processing input/output
    - Histogram creation input/output
    """
    directories_to_create = []
    
    # Parsing directories
    if "parsing_config" in config:
        parsing_config = config["parsing_config"]
        
        if "pipeline_config" in parsing_config:
            pipeline_config = parsing_config["pipeline_config"]
            if "output_path" in pipeline_config:
                directories_to_create.append(pipeline_config["output_path"])
        
        if "atlasparser_config" in parsing_config:
            atlasparser_config = parsing_config["atlasparser_config"]
            if "logging_path" in atlasparser_config:
                directories_to_create.append(atlasparser_config["logging_path"])
    
    # Mass calculation directories
    if "mass_calculate" in config:
        mass_config = config["mass_calculate"]
        if "input_dir" in mass_config:
            directories_to_create.append(mass_config["input_dir"])
        if "output_dir" in mass_config:
            directories_to_create.append(mass_config["output_dir"])
    
    # Post-processing directories
    if "post_processing" in config:
        post_config = config["post_processing"]
        if "input_dir" in post_config:
            directories_to_create.append(post_config["input_dir"])
        if "output_dir" in post_config:
            directories_to_create.append(post_config["output_dir"])
    
    # Histogram creation directories
    if "histogram_creation" in config:
        hist_config = config["histogram_creation"]
        if "input_dir" in hist_config:
            directories_to_create.append(hist_config["input_dir"])
        if "output_dir" in hist_config:
            directories_to_create.append(hist_config["output_dir"])
    
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
        config["parsing_config"]["pipeline_config"].update(cur_run_config["pipeline_config"])
    
    if "atlasparser_config" in cur_run_config:
        config["parsing_config"]["atlasparser_config"].update(cur_run_config["atlasparser_config"])
    
    config["parsing_config"]["run_metadata"] = cur_run_config["run_metadata"]


def _load_production_config(config, args):
    """Load production configuration."""
    run_time = datetime.now().strftime("%d_%m_%Y_%H:%M")
    
    if args.batch_job_index is not None:
        batch_job_index = int(args.batch_job_index)
        total_batch_jobs = int(args.total_batch_jobs) if args.total_batch_jobs else None
        run_name = (
            config["parsing_config"]["run_metadata"]["run_name"] 
            or f"job_idx{batch_job_index}_{run_time}"
        )
        
        if "mass_calculate" in config:
            config["mass_calculate"]["batch_job_index"] = batch_job_index
            config["mass_calculate"]["total_batch_jobs"] = total_batch_jobs
    else:
        batch_job_index = None
        total_batch_jobs = None
        run_name = config["parsing_config"]["run_metadata"]["run_name"] or f"run_{run_time}"
    
    config["parsing_config"]["run_metadata"] = {
        "batch_job_index": batch_job_index,
        "run_name": run_name,
        "total_batch_jobs": total_batch_jobs
    }


if __name__ == "__main__":
    main()