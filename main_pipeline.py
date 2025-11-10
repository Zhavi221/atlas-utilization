import logging
import sys
import argparse
import yaml

CONFIG_PATH = "configs/pipeline_config.yaml"

def main():
    main_logger = init_logging()
    config = load_config(CONFIG_PATH)
    tasks = config["tasks"]
        
    if tasks["do_parsing"]:
        main_logger.info("Starting parsing task")
        parsing_config = config["parsing"]

        if parsing_config.get("parse_in_subprocess", True):
            main_logger.info("Parsing with a subprocess")
            from src.pipelines import subprocessing_parsing
            subprocessing_parsing.parse_with_per_chunk_subprocess(parsing_config)
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
    args = arg_parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    return config



if __name__ == "__main__":
    main()