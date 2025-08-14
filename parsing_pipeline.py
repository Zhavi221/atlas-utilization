import logging
import sys
from src.parse_atlas import parser, combinatorics, consts, schemas
import matplotlib.pyplot as plt # plotting
import awkward as ak
import tqdm
import argparse
import yaml
import uproot
import awkward as ak

release_years = ["2016", "2020", "2024", "2025"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--config", help="Config file", default="configs/pipeline_config.yaml")
args = arg_parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

parsing_config = config["parsing"]

def flatten_for_root(awk_arr):
    """Flatten a top-level awkward Array into ROOT-friendly dict."""
    root_ready = {}

    for obj_name in awk_arr.fields:
        obj = awk_arr[obj_name]

        try:
            # If this succeeds, obj is a record array (possibly jagged)
            for field in obj.fields:
                root_ready[f"{obj_name}_{field}"] = obj[field]
        except AttributeError:
            # Not a record, already primitive or jagged primitive
            root_ready[obj_name] = obj

    return root_ready


def run():
    atlasparser = parser.ATLAS_Parser()
    release_files_uris = atlasparser.fetch_records_ids(
        release_year=parsing_config["release_year"]
    )

    categories = combinatorics.make_objects_categories(
        schemas.PARTICLE_LIST, min_n=4, max_n=4
    )
    main_category = categories[0]
    
    for events_chunk in atlasparser.parse_files_v2(
        files_ids=release_files_uris, 
        limit=parsing_config["file_limit"],
        max_workers=parsing_config["max_workers"],
        max_chunk_events=parsing_config["max_chunk_events"]
    ):
        combo_events = atlasparser.filter_events_by_combination(
            events_chunk, main_category
        )   

        root_ready = flatten_for_root(combo_events)

        with uproot.recreate("test/filtered.root") as f:
            f["tree"] = root_ready

        # TODO: IMPLEMENT SAVING TO NETWORK DRIVE

run()