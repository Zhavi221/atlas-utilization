import logging
import sys
from src.parse_atlas import parser, schemas
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
    pipeline_config = config["pipeline_config"]
    atlasparser_config = config["atlasparser_config"]

    atlasparser = parser.AtlasOpenParser(
        max_environment_memory_mb=atlasparser_config["max_environment_memory_mb"],
        chunk_yield_threshold_bytes=atlasparser_config["chunk_yield_threshold_bytes"],
        max_threads=atlasparser_config["max_threads"],
        logging_path=atlasparser_config["logging_path"],
        release_years=atlasparser_config["release_years"],
        create_dirs=atlasparser_config["create_dirs"],
        possible_tree_names=atlasparser_config["possible_tree_names"]
        )

    release_years_file_ids = atlasparser.fetch_record_ids(pipeline_config["fetching_metadata_timeout"])

    if pipeline_config["limit_files_per_year"]:
        parser.AtlasOpenParser.limit_files_per_year(release_years_file_ids, pipeline_config["limit_files_per_year"])
    
    if pipeline_config["random_files"]:
        random.shuffle(release_years_file_ids)

    for events_chunk in atlasparser.parse_files(
        release_years_file_ids=release_years_file_ids,
        save_statistics=True
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
        atlasparser.save_events_as_root(root_ready, pipeline_config["output_path"])        
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

