import os
import math
import argparse
from src.parse_atlas import parser
import atlasopenmagic as atom
import yaml
import itertools

CONFIG_PATH = "configs/pipeline_config.yaml"
TIME_FOR_FILE_SEC = 5 * 60
WALLTIME_PER_JOB_SEC = 24 * 3600  # 24 hours

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--config", help="Config file", default=CONFIG_PATH)
arg_parser.add_argument("--time_for_file_sec", type=int, default=TIME_FOR_FILE_SEC)
arg_parser.add_argument("--walltime_per_job_sec", type=int, default=WALLTIME_PER_JOB_SEC)

args = arg_parser.parse_args()
with open(args.config) as f:
    config = yaml.safe_load(f)

release_year_file_ids: dict = parser.AtlasOpenParser.fetch_record_ids_for_release_years(
    release_years=config["parsing_config"]["atlasparser_config"]["release_years"],
    timeout=60
)

all_file_ids = set(itertools.chain.from_iterable(release_year_file_ids.values()))
print(f"Total unique files to process: {len(all_file_ids)}")
num_files = len(all_file_ids)

files_per_job = args.walltime_per_job_sec / args.time_for_file_sec
print(f"Files per job: {files_per_job}")
num_jobs = math.ceil(num_files/files_per_job)
print(num_jobs)
