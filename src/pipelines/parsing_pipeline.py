import logging
import sys
from src.parse_atlas import parser, consts, schemas
from src.calculations import combinatorics, physics_calcs
import matplotlib.pyplot as plt # plotting
import awkward as ak
import tqdm
import argparse
import yaml
import uproot
import awkward as ak
import os

def parse(config):
    logger = init_logging()

    atlasparser = parser.ATLAS_Parser(
        max_chunk_size_bytes=config["max_chunk_size_bytes"],
        max_threads=config["max_threads"],
        max_processes=config["max_processes"]
        )

    release_files_uris = atlasparser.fetch_records_ids(
        release_year=config["release_year"]
    )
    
    for events_chunk in atlasparser.parse_files(
        files_ids=release_files_uris, 
        limit=config["file_limit"]
    ):
        
        logger.info("Cutting events")
        cut_events = parser.ATLAS_Parser.filter_events_by_kinematics(
            events_chunk, config["kinematic_cuts"]
        )
        
        logger.info("Filtering events")
        
        filtered_events = physics_calcs.filter_events_by_combination(
            cut_events, config["particle_counts"], use_count_range=False
        ) 

        logger.info("Flattening root")
        root_ready = atlasparser.flatten_for_root(filtered_events)

        logger.info("Saving events")
        atlasparser.save_events_as_root(root_ready, config["output_path"])        
        

def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logger = logging.getLogger(__name__)

    return logger