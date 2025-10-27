import logging
import sys
import time
import multiprocessing as mp
from multiprocessing import Process, Queue
from src.parse_atlas import parser, consts, schemas
from src.calculations import combinatorics, physics_calcs
import awkward as ak
from tqdm import tqdm
import yaml
import os
import gc
import random
import json

def subprocess_parse_and_process_one_chunk(config, files_to_parse, status_queue):
    """
    Runs in a separate process.
    Parses files, processes (filter/flatten), and saves ONE chunk.
    Sends only status updates back via queue (no data).
    Exits after chunk is saved.
    """
    logger = init_logging()
    
    try:
        atlasparser = parser.ATLAS_Parser(
            max_process_memory_mb=config["max_process_memory_mb"],
            max_chunk_size_bytes=config["max_chunk_size_bytes"],
            max_threads=config["max_threads"],
            logging_path=config["logging_path"]
        )
        
        # Parse until we get ONE chunk
        chunk_received = False
        for events_chunk in atlasparser.parse_files(
            files_ids=files_to_parse,
            limit=config.get("file_limit", 0),
            tracking_enabled=True, #FOR TESTING
            save_statistics=True #FOR TESTING
        ):

            files_parsed = atlasparser.cur_files_ids.copy()
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
            atlasparser.cur_files_ids = files_parsed
            atlasparser.save_events_as_root(root_ready, config["output_path"])
            
            # Send status back to main
            status_queue.put({
                "status": "chunk_complete",
                "files_parsed": files_parsed,
                "num_events": num_events,
                "chunk_size_mb": chunk_size_before
            })
            
            chunk_received = True
            
            
            # Exit after processing ONE chunk
            break
        
        # If loop completed without yielding (no chunks)
        if not chunk_received:
            status_queue.put({
                "status": "no_chunks",
                "files_parsed": []
            })
        
        logger.info("Subprocess exiting")
    
    except Exception as e:
        logger.error(f"Subprocess error: {e}", exc_info=True)
        status_queue.put({
            "status": "error",
            "message": str(e)
        })


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
    
    # Get file list upfront (in main process)
    temp_parser = parser.ATLAS_Parser(
        max_process_memory_mb=config["max_process_memory_mb"],
        max_chunk_size_bytes=config["max_chunk_size_bytes"],
        max_threads=config["max_threads"],
        logging_path=config["logging_path"]
    )
    
    all_files = temp_parser.fetch_records_ids(
        release_year=config["release_year"]
    )
    
    if config.get("random_files", True):
        random.shuffle(all_files)

    if config.get("file_limit"):
        all_files = all_files[:config["file_limit"]]
    
    logger.info(f"Found {len(all_files)} files to process")
    
    files_remaining = list(all_files)
    chunk_count = 0
    total_events = 0
    
    # Progress bar
    pbar = tqdm(total=len(all_files), desc="Parsing files", unit="file", dynamic_ncols=True, mininterval=3)
    
    # Loop: spawn subprocess for each chunk
    while files_remaining:
        
        logger.info(f"Files remaining: {len(files_remaining)}, spawning subprocess for next chunk...")
        
        # Create queue for status updates only
        status_queue = Queue()
        
        # Spawn subprocess
        worker_process = Process(
            target=subprocess_parse_and_process_one_chunk,
            args=(config, files_remaining, status_queue),
            daemon=False
        )
        
        worker_process.start()
        
        # Wait for subprocess to complete
        try:
            #FOR TESTING - timeout
            status = status_queue.get(timeout=None) 
            
            if status["status"] == "chunk_complete":
                files_parsed = status["files_parsed"]
                num_events = status["num_events"]
                chunk_size_mb = status["chunk_size_mb"]
                
                logger.info(
                    f"Chunk {chunk_count + 1} complete: "
                    f"{len(files_parsed)} files, {num_events} events, {chunk_size_mb:.1f} MB"
                )
                
                # Update tracking
                for f in files_parsed:
                    if f in files_remaining:
                        files_remaining.remove(f)
                
                chunk_count += 1
                total_events += num_events
                pbar.update(len(files_parsed))
                
                if not files_remaining:
                    crashed_files_path = config["logging_path"] + "crashed_files.json"
                    if os.path.exists(crashed_files_path):
                        with open(crashed_files_path, "r") as f:
                            data = json.load(f)
                            files_remaining = data.get("failed_files", [])

            elif status["status"] == "no_chunks":
                logger.info("No more chunks to process")
                break
            
            elif status["status"] == "error":
                logger.error(f"Subprocess error: {status['message']}")
                raise RuntimeError(f"Subprocess failed: {status['message']}")
        
        except mp.TimeoutError:
            logger.error("Subprocess timeout")
            raise
        
        finally:
            # Ensure subprocess is fully terminated
            if worker_process.is_alive():
                logger.warning("Subprocess still alive, terminating...")
                worker_process.terminate()
                worker_process.join(timeout=10)
                
                if worker_process.is_alive():
                    logger.warning("Subprocess did not terminate, killing...")
                    worker_process.kill()
            
            worker_process.join()
            logger.info("Subprocess terminated, memory freed to OS")
            logger.info("Sleeping 10 seconds to allow OS to reclaim memory...")
            
            #FOR TESTING - sleep 
            time.sleep(10)
            logger.info("10 seconds passed")
    
    pbar.close()
    logger.info(
        f"Parsing complete! Processed {chunk_count} chunks, "
        f"{total_events:,} total events"
    )



def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    return logger