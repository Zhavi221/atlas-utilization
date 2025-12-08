import logging
import sys
import argparse
import yaml
import json
from datetime import datetime


CONFIG_PATH = "configs/pipeline_config.yaml"

def parse_args():
    """Parse command line arguments."""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", help="Config file", default=CONFIG_PATH)
    arg_parser.add_argument("--test_run_index", help="Config file", default=None)
    arg_parser.add_argument("--batch_job_index", default=None)
    arg_parser.add_argument("--total_batch_jobs", default=None)
    return arg_parser.parse_args()

def main():
    """Main entry point for the pipeline."""
    logger = init_logging()
    args = parse_args()  # Get args to access config path
    config = load_config(args)
    tasks = config["tasks"]

    saved_files = []  # Track files saved by parsing for IM calculation
    
    if tasks["do_parsing"]:
        logger.info("Starting parsing task")
        parsing_config = config["parsing_config"]
        pipeline_config = parsing_config["pipeline_config"]
            
        if pipeline_config["parse_in_multiprocessing"]:
            logger.info("Parsing with multiprocessing")
            from src.pipelines import multiprocessing_pipeline
            saved_files = multiprocessing_pipeline.parse_with_per_chunk_subprocess(parsing_config)   
        else:
            logger.info("Parsing without a subprocess")
            from src.pipelines import parsing_pipeline
            saved_files = parsing_pipeline.parse(parsing_config)
        
        if saved_files is None:
            saved_files = []
        logger.info(f"Parsing completed. {len(saved_files)} files saved for IM calculation.")
            
    if tasks["do_mass_calculating"]:
        if tasks["do_parsing"]:
            # Sequential processing: IM calculation on files parsed in this job
            if saved_files:
                logger.info(f"Starting IM calculation on {len(saved_files)} files from parsing")
                from src.pipelines import im_pipeline
                im_pipeline.mass_calculate(config["mass_calculate"], file_list=saved_files)
            else:
                logger.warning("No files were successfully parsed. Skipping IM calculation.")
        else:
            # Parsing is disabled, run IM calculation directly (scans directory)
            logger.info("Starting IM calculation task (scanning input directory)")
            from src.pipelines import im_pipeline
            im_pipeline.mass_calculate(config["mass_calculate"])

def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    main_logger = logging.getLogger(__name__)

    return main_logger

def load_config(args):
    """Load configuration from YAML file."""
    config_path = args.config if args.config else CONFIG_PATH
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    if config["testing_config"]["is_on"]:
        if args.test_run_index:
            test_run_index = int(args.test_run_index)
        else:
            test_run_index = config["testing_config"]["test_run_index"]

        with open(config["testing_config"]["testing_jobs_path"]) as f:
            testing_runs = json.load(f)
            cur_run_config = testing_runs[test_run_index]

            # Update pipeline_config
        if "pipeline_config" in cur_run_config:
            for key, value in cur_run_config["pipeline_config"].items():
                config["parsing_config"]["pipeline_config"][key] = value

        # Update atlasparser_config
        if "atlasparser_config" in cur_run_config:
            for key, value in cur_run_config["atlasparser_config"].items():
                config["parsing_config"]["atlasparser_config"][key] = value

        config["parsing_config"]["run_metadata"] = cur_run_config["run_metadata"]
    else:
        ts = datetime.now()
        run_time = ts.strftime("%d_%m_%Y_%H:%M")
        if args.batch_job_index is not None:
            batch_job_index = int(args.batch_job_index) 
            total_batch_jobs = int(args.total_batch_jobs) if args.total_batch_jobs else None

            run_name = f"job_idx{batch_job_index}_{run_time}" if config["parsing_config"]["run_metadata"]["run_name"] is None else config["parsing_config"]["run_metadata"]["run_name"]
            
            # Also set batch job info for IM pipeline if it exists
            # For IM pipeline, batching is based on file+combination pairs
            if "mass_calculate" in config:
                config["mass_calculate"]["batch_job_index"] = batch_job_index
                config["mass_calculate"]["total_batch_jobs"] = total_batch_jobs

        else:
            batch_job_index = None
            run_name = config["parsing_config"]["run_metadata"]["run_name"] if config["parsing_config"]["run_metadata"]["run_name"] is None else f"run_{run_time}"
            total_batch_jobs = None
        
        config["parsing_config"]["run_metadata"] = {
            "batch_job_index": batch_job_index,
            "run_name": run_name,
            "total_batch_jobs": total_batch_jobs
        }
                
    return config



if __name__ == "__main__":
    main()