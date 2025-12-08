"""
Parsing Pipeline - Single Process Mode

Processes ATLAS Open Data files sequentially in a single process.
For multiprocessing mode, see multiprocessing_pipeline.py
"""
import logging
import sys
from src.parse_atlas import parser
from src.calculations import physics_calcs

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
        possible_tree_names=atlasparser_config["possible_tree_names"],
        temp_directory=atlasparser_config.get("temp_directory"),
        show_progress_bar=atlasparser_config.get("show_progress_bar", True),
        max_file_retries=pipeline_config["count_retries_failed_files"]
        )

    release_years_file_ids = atlasparser.fetch_record_ids(pipeline_config["fetching_metadata_timeout"])

    if pipeline_config["limit_files_per_year"]:
        parser.AtlasOpenParser.limit_files_per_year(release_years_file_ids, pipeline_config["limit_files_per_year"])
    

    saved_files = []  # Track successfully saved ROOT files for IM pipeline
    import os
    
    for events_chunk in atlasparser.parse_files(
        release_years_file_ids=release_years_file_ids,
        save_statistics=True
    ):
        logger.info("Cutting events")
        cut_events = physics_calcs.filter_events_by_kinematics(
            events_chunk, config["kinematic_cuts"]
        )
        del events_chunk  

        logger.info("Filtering events")
        filtered_events = physics_calcs.filter_events_by_particle_counts(
            events=cut_events, 
            particle_counts=config["particle_counts"], 
            is_particle_counts_range=True
        ) 
        del cut_events

        logger.info("Flattening root")
        root_ready = atlasparser.flatten_for_root(filtered_events)
        del filtered_events

        logger.info("Saving events")
        output_path = atlasparser.save_events_as_root(root_ready, pipeline_config["output_path"])
        if output_path:
            saved_filename = os.path.basename(output_path)
            saved_files.append(saved_filename)
            logger.info(f"Saved file: {saved_filename}")
        del root_ready
    
    logger.info(f"Parsing completed. Successfully saved {len(saved_files)} ROOT files for IM calculation.")
    return saved_files  

def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logger = logging.getLogger(__name__)

    return logger

