import os
import math
import argparse
from src.parse_atlas import parser
import atlasopenmagic as atom
import yaml
import itertools

CONFIG_PATH = "configs/pipeline_config.yaml"
TIME_FOR_FILE_SEC = 5
WALLTIME_PER_JOB_SEC = 24 * 3600  # 24 hours

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--config", help="Config file", default=CONFIG_PATH)
args = arg_parser.parse_args()
with open(args.config) as f:
    config = yaml.safe_load(f)

release_year_file_ids: dict = parser.ATLAS_Parser.fetch_record_ids_for_release_years(
    release_years=config["parsing_config"]["atlasparser_config"]["release_years"],
    timeout=60
)
all_file_ids = set(itertools.chain.from_iterable(release_year_file_ids.values()))
print(f"Total unique files to process: {len(all_file_ids)}")
num_files = len(all_file_ids)

files_per_job_sec = WALLTIME_PER_JOB_SEC / TIME_FOR_FILE_SEC
print(f"Files per job: {files_per_job_sec}")
num_jobs = math.ceil(files_per_job_sec/num_files)
print(num_jobs)
