import logging
import sys
import argparse
import yaml
import json

CONFIG_PATH = "configs/pipeline_config.yaml"

def main():
    main_logger = init_logging()
    config = load_config(CONFIG_PATH)
    tasks = config["tasks"]
    testing_config = config["testing_config"]

    if tasks["do_parsing"]:
        main_logger.info("Starting parsing task")
        parsing_config = config["parsing_config"]
        pipeline_config = parsing_config["pipeline_config"]

        if pipeline_config["parse_in_multiprocessing"]:
            main_logger.info("Parsing with multiprocessing")
            from src.pipelines import multiprocessing_pipeline
            multiprocessing_pipeline.parse_with_per_chunk_subprocess(parsing_config)
        else:
            main_logger.info("Parsing without a subprocess")
            from src.pipelines import parsing_pipeline
            parsing_pipeline.parse(parsing_config)
    
    if tasks["do_mass_calculating"]:
        main_logger.info("Starting calculations task")
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

def load_config(config_path):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", help="Config file", default=config_path)
    arg_parser.add_argument("--test_run_index", help="Config file")
    args = arg_parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    if config["testing_config"]["status"]:
        test_run_index = int(args.test_run_index)
        with open("testing_runs.json") as f:
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
        
    return config



if __name__ == "__main__":
    main()