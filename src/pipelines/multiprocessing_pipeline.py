import logging
import sys
import time
import multiprocessing as mp
from multiprocessing import Process, Queue
from src.parse_atlas import parser, schemas
from src.calculations import combinatorics, physics_calcs, math_utils, consts
import awkward as ak
from tqdm import tqdm
import yaml
import os
import gc
import random
import json
from datetime import datetime

def worker_parse_and_process_one_chunk(config, worker_num, release_years_file_ids):
    """
    Runs in a separate process.
    Parses files, processes (filter/flatten), and saves ONE chunk.
    Sends only status updates back via queue (no data).
    Exits after chunk is saved.
    """
    logger = init_logging(worker_num)
    pipeline_config = config["pipeline_config"]
    parser_config = config["parser_config"]
    
    try:
        atlasparser = parser.ATLAS_Parser(
            max_environment_memory_mb=parser_config["max_environment_memory_mb"],
            chunk_yield_threshold_bytes=parser_config["chunk_yield_threshold_bytes"],
            max_threads=parser_config["max_threads"],
            logging_path=parser_config["logging_path"],
            possible_tree_names=parser_config["possible_tree_names"]
        )
        
        # Parse until we get ONE chunk
        for events_chunk in atlasparser.parse_files(
            release_years_file_ids=release_years_file_ids,
            # save_statistics=True #FOR TESTING - statistics, weird behavior consider removing
        ):
            files_parsed = atlasparser.cur_file_ids.copy()
            chunk_size_before = events_chunk.layout.nbytes / (1024**2)
            num_events = len(events_chunk)
            
            logger.info(f"Got chunk: {num_events} events, {chunk_size_before:.1f} MB")
            
            # PROCESS IN SUBPROCESS
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
            
            logger.info("Flattening for ROOT")
            root_ready = atlasparser.flatten_for_root(filtered_events)
            del filtered_events
            
            logger.info("Saving events to disk")
            atlasparser.cur_file_ids = files_parsed
            atlasparser.save_events_as_root(root_ready, pipeline_config["output_path"])

            stats = atlasparser.get_statistics(
                total_files=len(files_parsed),
                successful_count=len(files_parsed),
            )
            
            logger.info("Subprocess exiting")
            return {
                "status": "chunk_complete",
                "files_parsed": files_parsed,
                "num_events": num_events,
                "chunk_size_mb": chunk_size_before,
                "stats": stats  # Return statistics
            }
    except Exception as e:
        logger.error(f"Subprocess error: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }

def parse_with_per_chunk_subprocess(config):
    """
    Main process orchestrator.
    - Gets file list
    - Spawns subprocess for each chunk
    - Subprocess does ALL processing (parse/filter/save)
    - Subprocess exits (memory freed to OS)
    - Main just waits for status and updates tracking
    """
    logger = init_logging()
    pipeline_config = config["pipeline_config"]
    parser_config = config["parser_config"]

    # Get file list upfront (in main process)
    temp_parser = parser.ATLAS_Parser(
        max_environment_memory_mb=parser_config["max_environment_memory_mb"],
        chunk_yield_threshold_bytes=parser_config["chunk_yield_threshold_bytes"],
        max_threads=parser_config["max_threads"],
        logging_path=parser_config["logging_path"],
        release_years=parser_config["release_years"],
        create_dirs=parser_config["create_dirs"],
        possible_tree_names=parser_config["possible_tree_names"]
    )
    
    release_years_file_ids = temp_parser.fetch_record_ids(timeout=pipeline_config["fetching_metadata_timeout"])
    
    if pipeline_config["random_files"]:
        random.shuffle(release_years_file_ids)

    if pipeline_config["limit_files_per_year"]:
        parser.ATLAS_Parser.limit_files_per_year(release_years_file_ids, pipeline_config["limit_files_per_year"])
    
    chunk_count = 0
    total_events = 0
    
    all_stats = []
    max_parallel_workers = pipeline_config["max_parallel_workers"]
    for release_year, file_ids in release_years_file_ids.items():
        logger.info(f"Found {len(file_ids)} files to process")
        
        # Progress bar
        pbar = tqdm(total=len(file_ids), desc="Parsing files", unit="file", dynamic_ncols=True, mininterval=3)
        
        # Loop: spawn subprocess for each chunk
        files_remaining = file_ids
    
        while files_remaining:
            logger.info(f"Files remaining: {len(files_remaining)}, spawning {max_parallel_workers} workers...")
            
            # Split remaining files among workers
            num_workers = min(max_parallel_workers, len(files_remaining))
            if len(files_remaining) < (max_parallel_workers * 10):  # Less than 10 files per worker
                num_workers = max(1, len(files_remaining) // 20)

            file_chunks = chunk_list(files_remaining, num_workers)
            
            # Prepare arguments for each worker (no queue needed)
            worker_args = [
                (config, worker_num, {release_year: chunk}) 
                for worker_num, chunk in enumerate(file_chunks)
            ]
            
            # Launch workers in parallel using Pool
            with mp.Pool(processes=num_workers) as pool:
                # Use starmap to launch all workers and wait for results
                results = pool.starmap(worker_parse_and_process_one_chunk, worker_args)
                
                # Process results from all workers
                for i, result in enumerate(results):
                    if result["status"] == "chunk_complete":
                        all_stats.append(result["stats"])
                        files_parsed = result["files_parsed"]
                        
                        # Update tracking
                        for f in files_parsed:
                            if f in files_remaining:
                                files_remaining.remove(f)
                        
                        chunk_count += 1
                        total_events += result["num_events"]
                        pbar.update(len(files_parsed))
                        
                        logger.info(f"Worker {i+1} completed: {len(files_parsed)} files, {result['num_events']} events")
                    
                    elif result["status"] == "error":
                        logger.error(f"Worker {i+1} failed: {result['message']}")
                    
                    elif result["status"] == "no_chunks":
                        logger.info(f"Worker {i+1}: No chunks to process")
            
            logger.info("All workers completed, sleeping for memory reclaim...")
            #TEMP time.sleep(2) 
        
        # Handle crashed files
        if not files_remaining:
            crashed_files_path = parser_config["logging_path"] + "crashed_files.json"
            if os.path.exists(crashed_files_path):
                with open(crashed_files_path, "r") as f:
                    data = json.load(f)
                    files_remaining = data.get("failed_files", [])
                    if files_remaining:
                        logger.info(f"Retrying {len(files_remaining)} crashed files...")
        if all_stats:
            aggregate_statistics(all_stats, parser_config["logging_path"], pipeline_config, parser_config)    

        pbar.close()
    
    logger.info(
        f"Parsing complete! Processed {chunk_count} chunks, "
        f"{total_events:,} total events"
    )

def chunk_list(lst, n):
    """Split list into n roughly equal chunks"""
    k, m = divmod(len(lst), n)
    return [lst[i*k+min(i,m):(i+1)*k+min(i+1,m)] for i in range(n)]

def aggregate_statistics(stats_list, output_path, pipeline_config, parser_config):
    """Combine statistics from all worker chunks into consolidated metrics"""
    if not stats_list:
        return
    
    total_chunks = len(stats_list)
    
    # Aggregate parsing session metrics
    chunk_times = [s["parsing_session"]["total_time_seconds"] for s in stats_list]
    all_files = sum(s["parsing_session"]["total_files"] for s in stats_list)
    successful_files = sum(s["parsing_session"]["successful_files"] for s in stats_list)
    failed_files = sum(s["parsing_session"]["failed_files"] for s in stats_list)
    
    # Aggregate performance metrics
    all_chunk_sizes = [s["performance"]["avg_chunk_size_mb"] for s in stats_list]
    total_events = sum(s["performance"]["total_events_processed"] for s in stats_list)
    total_data_mb = sum(s["performance"]["total_data_processed_mb"] for s in stats_list)
    all_memory_captures = [s["performance"]["max_memory_captured_mb"] for s in stats_list]

    # Collect all chunk details
    all_chunk_details = []
    for i, s in enumerate(stats_list):
        for chunk in s["chunk_details"]:
            chunk["chunk_id"] = i  # Renumber sequentially
            all_chunk_details.append(chunk)
    
    # Aggregate error information
    all_error_types = {}
    all_failed_files = []
    total_timeouts = sum(s["errors"]["timeout_count"] for s in stats_list)
    
    for s in stats_list:
        for error_type, count in s["errors"].get("error_types", {}).items():
            all_error_types[error_type] = all_error_types.get(error_type, 0) + count
        all_failed_files.extend(s["errors"].get("failed_file_list", []))
    
    # Build consolidated statistics
    
    #TREND STATISTICS
    #TODO correctly attach a data point to a time point, sort and then output the two arrays
    timestamps = [datetime.fromisoformat(s["parsing_session"]["timestamp"]).second for s in stats_list]
    max_mem_captures = [s["performance"]["max_memory_captured_mb"] for s in stats_list]
    chunks_sizes = [s["performance"]["avg_chunk_size_mb"] for s in stats_list]
    
    zipped_data = list(zip(timestamps, max_mem_captures, chunks_sizes))
    zipped_data.sort(key=lambda tuple: tuple[0])
    timestamps, max_mem_captures, chunks_sizes = zip(*zipped_data)
    
    logging.info(f"CHUNK TIMESTAMPS: {timestamps}")
    mem_first_deriv_avg, mem_second_deriv_avg = math_utils.calc_n_derivs_avg(
        timestamps, 
        max_mem_captures,
        n_derivs=2)
    chunk_first_deriv_avg, chunk_second_deriv_avg = math_utils.calc_n_derivs_avg(
        timestamps, 
        chunks_sizes,
        n_derivs=2)
    
    #TIME STATISTICS
    start_timestamp = datetime.fromisoformat(stats_list[0]["parsing_session"]["timestamp"])
    last_timestamp = datetime.fromisoformat(stats_list[-1]["parsing_session"]["timestamp"])
    avg_chunk_parsing_mins = (sum(chunk_times) / total_chunks) / 60
    mb_per_second = total_data_mb / sum(chunk_times) if sum(chunk_times) > 0 else 0
    #CONFIG VARIABLES
    config_yield_threshold_mb = parser_config["chunk_yield_threshold_bytes"] / 1024**2

    consolidated = {
        "parsing_session": {
            "start_timestamp": str(start_timestamp),  
            "last_timestamp": str(last_timestamp),  
            "last_to_first_timestamp_diff_minutes": (last_timestamp - start_timestamp).seconds / 60,
            "total_processing_time_minutes": sum(chunk_times) / 60,
            "avg_chunk_parsing_mins": avg_chunk_parsing_mins,
            "min_chunk_parsing_mins": min(chunk_times) / 60,
            "max_chunk_parsing_mins": max(chunk_times) / 60,
            "total_files": all_files,
            "successful_files": successful_files,
            "failed_files": failed_files,
            "success_rate": (successful_files / all_files * 100) if all_files > 0 else 0
            #FEATURE add a measurment for time between first worker to finish and it's current iteration parsing 
        },
        "performance": {
            "total_data_processed_mb": total_data_mb,
            "total_events_processed": total_events,
            "chunks_created": total_chunks,
            "avg_chunk_size_mb": total_data_mb / total_chunks,
            "min_chunk_size_mb": min(all_chunk_sizes),
            "max_chunk_size_mb": max(all_chunk_sizes),
            "events_per_second": total_events / sum(chunk_times) if sum(chunk_times) > 0 else 0,
            "mb_per_second": mb_per_second,
            "max_process_memory_captured": max(all_memory_captures),
            
        },
        "advanced_stats" :{
            "mem_first_deriv_avg":mem_first_deriv_avg,
            "mem_second_deriv_avg":mem_second_deriv_avg,
            "chunk_first_deriv_avg":chunk_first_deriv_avg,
            "chunk_second_deriv_avg":chunk_second_deriv_avg
        },
        "for_all_data":{
            "days_to_parse":
                (consts.OPEN_DATA_SIZE_MB/mb_per_second)/
                (60*60*24),
        },
        "errors": {
            "timeout_count": total_timeouts,
            "error_types": all_error_types,
            "failed_file_list": list(set(all_failed_files))  # Deduplicate
        },
        "raw_data":{
            "timestamps":timestamps,
            "avg_chunks_sizes":chunks_sizes,
            "max_memory_captures":max_mem_captures
        },
        "chunk_details": all_chunk_details
    }
    
    # Save consolidated statistics
    output_file = os.path.join(output_path, "consolidated_statistics.json")
    with open(output_file, "w") as f:
        json.dump(consolidated, f, indent=2)
    
    return consolidated

def init_logging(worker_num=None):
    format = "%(asctime)s - %(name)s - %(levelname)s - "
    if worker_num:
        format += f"Worker {worker_num} - "
    
    format += "%(message)s"

    logging.basicConfig(
        level=logging.INFO,
        format=format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )

    return logging.getLogger(__name__)