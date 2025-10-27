import logging
import sys
from src.parse_atlas import parser, consts, schemas
from src.calculations import combinatorics, physics_calcs
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt # plotting
import awkward as ak
import tqdm
import argparse
import yaml
import uproot
import awkward as ak
import os
import gc
import random

def parse(config):
    logger = init_logging()

    atlasparser = parser.ATLAS_Parser(
        max_process_memory_mb=config["max_process_memory_mb"],
        max_chunk_size_bytes=config["max_chunk_size_bytes"],
        max_threads=config["max_threads"],
        logging_path=config["logging_path"],
        initialize_statistics=True
        )

    release_files_uris = atlasparser.fetch_records_ids(
        release_year=config["release_year"]
    )

    if config.get("random_files", True):
        random.shuffle(release_files_uris)

    for events_chunk in atlasparser.parse_files(
        files_ids=release_files_uris, 
        limit=config["file_limit"]
    ):
        
        logger.info("Cutting events")
        cut_events = physics_calcs.filter_events_by_kinematics(
            events_chunk, config["kinematic_cuts"]
        )
        #del events_chunk  

        logger.info("Filtering events")
        filtered_events = physics_calcs.filter_events_by_particle_counts(
            events=cut_events, 
            particle_counts=config["particle_counts"], 
            is_particle_counts_range=True
        ) 
        #del cut_events

        logger.info("Flattening root")
        root_ready = atlasparser.flatten_for_root(filtered_events)
        #del filtered_events

        logger.info("Saving events")
        atlasparser.save_events_as_root(root_ready, config["output_path"])        
        #del root_ready

        #gc.collect()  

def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logger = logging.getLogger(__name__)

    return logger

