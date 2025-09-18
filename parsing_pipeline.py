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
import os

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



def run():
    atlasparser = parser.ATLAS_Parser(
        max_chunk_size_bytes=parsing_config["max_chunk_size_bytes"],
        max_threads=parsing_config["max_threads"],
        max_processes=parsing_config["max_processes"]
        )

    release_files_uris = atlasparser.fetch_records_ids(
        release_year=parsing_config["release_year"]
    )

    categories = combinatorics.make_objects_categories(
        schemas.PARTICLE_LIST, min_n=4, max_n=4
    )
    main_category = categories[0]
    
    for events_chunk in atlasparser.parse_files(
        files_ids=release_files_uris, 
        limit=parsing_config["file_limit"]
    ):
        
        logging.info("Cutting events")
        cut_events = parser.ATLAS_Parser.filter_events_by_kinematics(
            events_chunk, parsing_config["kinematic_cuts"]
        )
        
        logging.info("Filtering events")
        filtered_events = parser.ATLAS_Parser.filter_events_by_counts(
            # cut_events, parsing_config["particle_counts"]
            cut_events, {'nElectrons_eta': 2, 'nJets_eta': 4}, use_range=False
        )    

        logging.info("Flattening root")
        root_ready = atlasparser.flatten_for_root(filtered_events)

        logging.info("Saving events")
        atlasparser.save_events_as_root(root_ready, parsing_config["output_path"])        
        

if __name__ == "__main__":
    run()