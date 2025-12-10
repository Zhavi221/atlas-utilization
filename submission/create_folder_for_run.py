import argparse
import yaml
from datetime import datetime
#CHECK test this script with submitting real jobs
CONFIG_PATH = "configs/pipeline_config.yaml"

# DATE_FOLDER=\$(date +%Y%m%d_%H%M%S)
# OUTPUT_DATA_PATH="/storage/agrp/netalev/data/root_files/\${DATE_FOLDER}"
# OUTPUT_IM_PATH="/storage/agrp/netalev/data/inv_masses/\${DATE_FOLDER}"
# mkdir -p "\${OUTPUT_PATH}"
# mkdir -p "\${OUTPUT_IM_PATH}"

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--config", help="Config file", default=CONFIG_PATH)
arg_parser.add_argument("--test_run_index", help="Config file", default=None)
arg_parser.add_argument("--batch_job_index", default=None)
arg_parser.add_argument("--total_batch_jobs", default=None)
args = arg_parser.parse_args()

with open(CONFIG_PATH, "w") as f:
    config = yaml.safe_load(f)
    ts = datetime.now()
    run_time = ts.strftime("%d_%m_%Y_%H:%M")

    config["parsing_config"]["pipeline_config"]["output_path"] += run_time
    config["parsing_config"]["atlasparser_config"]["logging_path"] += run_time
    config["mass_calculate"]["input_dir"] += run_time
    config["mass_calculate"]["output_dir"] += run_time
    yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
