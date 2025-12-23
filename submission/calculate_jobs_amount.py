import os
import math
import argparse
import atlasopenmagic as atom
import yaml
import itertools
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.parse_atlas import parser
import io
import contextlib

CONFIG_PATH = "configs/pipeline_config.yaml"
TIME_FOR_FILE_SEC = 20
WALLTIME_PER_JOB_SEC = 24 * 3600  # 24 hours

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--config", help="Config file", default=CONFIG_PATH)
arg_parser.add_argument("--time_for_file_sec", type=int, default=TIME_FOR_FILE_SEC)
arg_parser.add_argument("--walltime_per_job_sec", type=int, default=WALLTIME_PER_JOB_SEC)

args = arg_parser.parse_args()
with open(args.config) as f:
    config = yaml.safe_load(f)

_buf = io.StringIO()
with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf):
    temp_parser = parser.AtlasOpenParser(
        chunk_yield_threshold_bytes=0, max_threads=0, logging_path=None,
        release_years=config["parsing_config"]["atlasparser_config"]["release_years"])
    release_year_file_ids: dict = temp_parser.fetch_record_ids(
        timeout=160, seperate_mc=True
    )
    if not config["parsing_config"]["pipeline_config"]["parse_mc"]:
            release_year_file_ids = {k: v for k, v in release_year_file_ids.items() if "mc" not in k}

    if config["parsing_config"]["pipeline_config"]["limit_files_per_year"]:
        parser.AtlasOpenParser.limit_files_per_year(release_year_file_ids, 
        config["parsing_config"]["pipeline_config"]["limit_files_per_year"])

all_file_ids = set(itertools.chain.from_iterable(release_year_file_ids.values()))
num_files = len(all_file_ids)

files_per_job = args.walltime_per_job_sec / args.time_for_file_sec
num_jobs = math.ceil(num_files/files_per_job)
print(num_jobs)