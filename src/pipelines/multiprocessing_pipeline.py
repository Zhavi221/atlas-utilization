import logging
import sys
import time
import multiprocessing as mp
from multiprocessing import Process, Queue
from src.parse_atlas import parser, schemas
from src.utils.calculations import combinatorics, physics_calcs, math_utils
from . import consts
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
    atlasparser_config = config["atlasparser_config"]
    
    try:
        # Filter out files that have exceeded retries before processing
        # Load retry counts from crashed_files.json if it exists
        crashed_files_name = "crashed_files.json"
        if pipeline_config.get("run_metadata", {}).get("batch_job_index") is not None:
            batch_job_index = pipeline_config["run_metadata"]["batch_job_index"]
            crashed_files_name = f"crashed_files_{batch_job_index}.json"
        crashed_files_path = atlasparser_config["logging_path"] + crashed_files_name
        
        worker_retry_counts = {}
        if os.path.exists(crashed_files_path):
            try:
                with open(crashed_files_path, 'r') as f:
                    worker_data = json.load(f)
                    worker_retry_counts = worker_data.get("retry_counts", {})
            except (json.JSONDecodeError, IOError):
                pass
        
        # Filter release_years_file_ids to exclude files that exceeded retries
        filtered_release_years_file_ids = {}
        skipped_files = []  # Track files that exceeded max retries
        max_retries = pipeline_config["count_retries_failed_files"]
        for release_year, file_ids in release_years_file_ids.items():
            filtered_file_ids = []
            for file_path in file_ids:
                retry_count = worker_retry_counts.get(file_path, 0)
                if retry_count < max_retries:
                    filtered_file_ids.append(file_path)
                else:
                    skipped_files.append(file_path)
                    logger.warning(f"Worker {worker_num}: Skipping {file_path} - exceeded max retries ({retry_count} >= {max_retries})")
            
            if filtered_file_ids:
                filtered_release_years_file_ids[release_year] = filtered_file_ids
            else:
                logger.warning(f"Worker {worker_num}: All files for {release_year} exceeded max retries, skipping")
        
        if not filtered_release_years_file_ids:
            logger.warning(f"Worker {worker_num}: No files to process after filtering (all exceeded max retries)")
            return {
                "status": "skipped",
                "files_parsed": [],
                "skipped_files": skipped_files,  # Return list of skipped files so main process can remove them
                "saved_filename": None,  # No file saved
                "num_events": 0,
                "worker_num": worker_num
            }
        
        specific_record_ids = atlasparser_config.get("specific_record_ids", None)
        logger.info(f"Worker {worker_num}: Config specific_record_ids value: {specific_record_ids} (type: {type(specific_record_ids)})")
        atlasparser = parser.AtlasOpenParser(
            max_environment_memory_mb=atlasparser_config["max_environment_memory_mb"],
            chunk_yield_threshold_bytes=atlasparser_config["chunk_yield_threshold_bytes"],
            max_threads=atlasparser_config["max_threads"],
            logging_path=atlasparser_config["logging_path"],
            possible_tree_names=atlasparser_config["possible_tree_names"],
            wrapping_logger=logger,
            show_progress_bar=atlasparser_config.get("show_progress_bar", True),
            max_file_retries=pipeline_config["count_retries_failed_files"],
            specific_record_ids=specific_record_ids
        )
        
        # Parse until we get ONE chunk (using filtered file list)
        chunk_received = False
        for events_chunk in atlasparser.parse_files(
            release_years_file_ids=filtered_release_years_file_ids
        ):
            chunk_received = True
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

            # Measure filtered logical size before freeing
            filtered_size_mb = filtered_events.layout.nbytes / (1024**2)
            del filtered_events
            
            logger.info("Saving events to disk")
            output_path = atlasparser.save_events_as_root(root_ready, pipeline_config["output_path"])
            
            # Extract just the filename (not full path) for IM pipeline
            saved_filename = os.path.basename(output_path) if output_path else None
            
            # Measure on-disk size of the produced ROOT file
            disk_size_mb = os.path.getsize(output_path) / (1024**2) if os.path.exists(output_path) else 0.0

            stats = atlasparser.get_statistics(
                total_files=len(files_parsed)
            )
            # Attach filtered and disk size measurements to stats
            stats.setdefault("performance", {})
            stats["performance"]["total_filtered_data_mb"] = filtered_size_mb
            stats["performance"]["total_disk_data_mb"] = disk_size_mb
            
            logger.info("Subprocess exiting")
            return {
                "status": "chunk_complete",
                "files_parsed": files_parsed,
                "saved_filename": saved_filename,  # Return saved ROOT filename for IM pipeline
                "num_events": num_events,
                "chunk_size_mb": chunk_size_before,
                "stats": stats  # Return statistics
            }
        
        # If no chunks were yielded (all files failed/timeout), return appropriate status
        if not chunk_received:
            # Get list of files that were attempted
            attempted_files = []
            for release_year, file_ids in filtered_release_years_file_ids.items():
                attempted_files.extend(file_ids)
            
            logger.warning(f"Worker {worker_num}: No chunks yielded from parse_files() - all {len(attempted_files)} files may have failed or timed out")
            return {
                "status": "no_chunks",
                "files_parsed": attempted_files,  # Return attempted files so they can be marked as failed
                "saved_filename": None,
                "num_events": 0,
                "worker_num": worker_num,
                "message": f"No chunks produced - all {len(attempted_files)} files failed or timed out"
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
    atlasparser_config = config["atlasparser_config"]
    run_metadata = config["run_metadata"]

    # Get file list upfront (in main process)
    specific_record_ids = atlasparser_config.get("specific_record_ids", None)
    logger.info(f"Config specific_record_ids value: {specific_record_ids} (type: {type(specific_record_ids)})")
    temp_parser = parser.AtlasOpenParser(
        max_environment_memory_mb=atlasparser_config["max_environment_memory_mb"],
        chunk_yield_threshold_bytes=atlasparser_config["chunk_yield_threshold_bytes"],
        max_threads=atlasparser_config["max_threads"],
        logging_path=atlasparser_config["logging_path"],
        release_years=atlasparser_config["release_years"],
        create_dirs=atlasparser_config["create_dirs"],
        possible_tree_names=atlasparser_config["possible_tree_names"],
        show_progress_bar=atlasparser_config.get("show_progress_bar", True),
        max_file_retries=pipeline_config["count_retries_failed_files"],
        specific_record_ids=specific_record_ids
    )
    
    release_years_file_ids = {}
    file_urls_path = pipeline_config["file_urls_path"]
    lock_file_path = file_urls_path + ".lock"
    
    # If specific_record_ids is set, skip cache and always fetch fresh
    # (since cached file might have been created without specific_record_ids)
    skip_cache = specific_record_ids and (isinstance(specific_record_ids, list) and len(specific_record_ids) > 0)
    if skip_cache:
        logger.info(f"specific_record_ids is set, skipping cache and fetching fresh file URLs")
        whole_file_ids = None
    elif os.path.exists(file_urls_path):
        # First, try to read existing file (for all jobs, including job 1)
        try:
            with open(file_urls_path, "r") as f:
                whole_file_ids = json.load(f)
            logger.info(f"Loaded file URLs from existing file: {file_urls_path}")
            
            # Extract batch for this job (or use all if no batching)
            if run_metadata.get("batch_job_index") is not None:
                release_years_file_ids = get_batch_by_index(
                    whole_file_ids, 
                    run_metadata["batch_job_index"], 
                    run_metadata["total_batch_jobs"]
                )
                logger.info(f"Extracted batch {run_metadata['batch_job_index']}/{run_metadata['total_batch_jobs']} from existing file")
            else:
                # No batching, use all files
                release_years_file_ids = whole_file_ids
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error reading existing file URLs: {e}. Will fetch new ones.")
            whole_file_ids = None
    else:
        whole_file_ids = None
    
    # Only fetch if file doesn't exist or was corrupted
    if whole_file_ids is None:
        # Use file locking to ensure only one process creates the file
        max_wait_time = 300  # Wait up to 5 minutes for another job to create the file
        wait_interval = 2  # Check every 2 seconds
        elapsed_time = 0
        lock_acquired = False
        
        while elapsed_time < max_wait_time:
            try:
                # Try to create lock file exclusively (atomic operation)
                lock_fd = os.open(lock_file_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                # We got the lock! Now fetch and write
                lock_acquired = True
                try:
                    logger.info("Acquired lock - fetching file URLs...")
                    if skip_cache:
                        logger.info(f"Fetching with specific_record_ids: {specific_record_ids}")
                    whole_file_ids = temp_parser.fetch_record_ids(
                        timeout=pipeline_config["fetching_metadata_timeout"], 
                        seperate_mc=True
                    )
                    if skip_cache:
                        logger.info(f"Fetched {len(whole_file_ids)} release year(s) with specific_record_ids")
                    if not pipeline_config["parse_mc"]:
                        whole_file_ids = {
                            k: v for k, v in whole_file_ids.items() 
                            if "mc" not in k
                        }
                    
                    parser.AtlasOpenParser.limit_files_per_year(
                        whole_file_ids, 
                        pipeline_config["limit_files_per_year"]
                    )
                    
                    # Write atomically using temp file + rename
                    temp_file_path = file_urls_path + ".tmp"
                    with open(temp_file_path, "w") as f:
                        json.dump(whole_file_ids, f, indent=2)
                    os.rename(temp_file_path, file_urls_path)
                    logger.info(f"Successfully created file URLs file: {file_urls_path}")
                    break  # Success, exit loop
                except Exception as e:
                    logger.error(f"Error while fetching/writing file URLs: {e}")
                    # Clean up temp file if it exists
                    temp_file_path = file_urls_path + ".tmp"
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                    raise
                finally:
                    os.close(lock_fd)
                    if os.path.exists(lock_file_path):
                        os.unlink(lock_file_path)  # Remove lock file
            except FileExistsError:
                # Lock file exists - another job is creating the file
                logger.info(f"Lock file exists, waiting for another job to create file URLs... (waited {elapsed_time}s)")
                time.sleep(wait_interval)
                elapsed_time += wait_interval
                
                # Check if file was created while we waited
                if os.path.exists(file_urls_path):
                    try:
                        with open(file_urls_path, "r") as f:
                            whole_file_ids = json.load(f)
                        logger.info("File was created by another job, loaded successfully")
                        break
                    except (json.JSONDecodeError, IOError):
                        # File might be incomplete, continue waiting
                        pass
            except Exception as e:
                logger.error(f"Unexpected error while acquiring lock: {e}")
                if lock_acquired and os.path.exists(lock_file_path):
                    try:
                        os.unlink(lock_file_path)
                    except:
                        pass
                raise
        
        # If we still don't have the file, it's an error
        if whole_file_ids is None:
            logger.error(f"Timeout waiting for file URLs to be created. File may not exist at {file_urls_path}")
            return
        
        # Extract batch for this job (or use all if no batching)
        if run_metadata.get("batch_job_index") is not None:
            release_years_file_ids = get_batch_by_index(
                whole_file_ids,
                run_metadata["batch_job_index"],
                run_metadata["total_batch_jobs"]
            )
            logger.info(f"Extracted batch {run_metadata['batch_job_index']}/{run_metadata['total_batch_jobs']} after fetching")
        else:
            # No batching, use all files
            release_years_file_ids = whole_file_ids
    
    crashed_files_name = "crashed_files.json"
    if run_metadata.get("batch_job_index",None) is not None:
        crashed_files_name = f"crashed_files_{run_metadata['batch_job_index']}.json"
    else:
        crashed_files_name = "crashed_files.json"

    crashed_files_path = atlasparser_config["logging_path"] + crashed_files_name
    
    if pipeline_config["limit_files_per_year"]:
        parser.AtlasOpenParser.limit_files_per_year(release_years_file_ids, pipeline_config["limit_files_per_year"])
    
    count_retries_failed_files = pipeline_config["count_retries_failed_files"]
    chunk_count = 0
    total_events = 0
    
    all_stats = []
    all_saved_files = []  # Track all successfully saved ROOT files for IM pipeline
    max_parallel_workers = pipeline_config["max_parallel_workers"]
    show_progress_bar = atlasparser_config.get("show_progress_bar", True)
    
    for release_year, file_ids in release_years_file_ids.items():
        logger.info(f"Found {len(file_ids)} files to process")
        cur_retries = 0

        # Progress bar (conditional based on config)
        if show_progress_bar:
            pbar = tqdm(total=len(file_ids), desc="Parsing files", unit="file", dynamic_ncols=True, mininterval=3)
        else:
            # No-op progress bar when disabled
            class NoOpProgressBar:
                def __init__(self, *args, **kwargs):
                    pass
                def update(self, n=1):
                    pass
                def set_postfix_str(self, s):
                    pass
                def close(self):
                    pass
            pbar = NoOpProgressBar()
        
        # Filter initial file list to exclude files that have already exceeded retries
        # Load retry counts from crashed_files.json if it exists
        crashed_files_path = atlasparser_config["logging_path"] + crashed_files_name
        initial_retry_counts = {}
        if os.path.exists(crashed_files_path):
            try:
                with open(crashed_files_path, 'r') as f:
                    initial_data = json.load(f)
                    initial_retry_counts = initial_data.get("retry_counts", {})
            except (json.JSONDecodeError, IOError):
                pass
        
        # Filter out files that have exceeded retries from initial list
        files_remaining = []
        skipped_initial = []
        for file_path in file_ids:
            retry_count = initial_retry_counts.get(file_path, 0)
            if retry_count < count_retries_failed_files:
                files_remaining.append(file_path)
            else:
                skipped_initial.append(file_path)
                logger.warning(f"Skipping {file_path} from initial list - already exceeded max retries ({retry_count} >= {count_retries_failed_files})")
        
        if skipped_initial:
            logger.info(f"Filtered out {len(skipped_initial)} files from initial list that exceeded max retries")
        
        if not files_remaining:
            logger.warning(f"All {len(file_ids)} files for {release_year} have exceeded max retries, skipping this release year")
            pbar.close()
            continue
        MIN_FILES_PER_WORKER = 10
        FALLBACK_FILES_PER_WORKER = 20
        # Track retry counts across retry iterations
        retry_counts_global = {}
        # Initialize with initial retry counts
        retry_counts_global.update(initial_retry_counts)
        # Track files that have permanently exceeded retries to prevent reprocessing
        permanently_skipped_files = set()
        # Initialize with files that already exceeded from initial check
        for file_path in skipped_initial:
            permanently_skipped_files.add(file_path)
        # Initialize with files that already exceeded from initial check
        for file_path in skipped_initial:
            permanently_skipped_files.add(file_path)

        while cur_retries <= count_retries_failed_files: # CHECK retry files
            while files_remaining:
                # Double-check: filter out any files that have exceeded retries before processing
                # This prevents race conditions where files exceed limit between loading and processing
                # Update retry_counts_global from crashed_files.json
                if os.path.exists(crashed_files_path):
                    try:
                        with open(crashed_files_path, 'r') as f:
                            check_data = json.load(f)
                            retry_counts_global.update(check_data.get("retry_counts", {}))
                    except (json.JSONDecodeError, IOError):
                        pass
                
                # Filter files_remaining to exclude those that exceeded retries
                files_to_process = []
                files_to_remove = []
                for file_path in files_remaining:
                    # Skip if already permanently marked as exceeded
                    if file_path in permanently_skipped_files:
                        files_to_remove.append(file_path)
                        continue
                    
                    retry_count = retry_counts_global.get(file_path, 0)
                    if retry_count < count_retries_failed_files:  # Use the same limit
                        files_to_process.append(file_path)
                    else:
                        logger.warning(f"Filtering out {file_path} - exceeded max retries ({retry_count} >= {count_retries_failed_files}) before processing")
                        files_to_remove.append(file_path)
                        permanently_skipped_files.add(file_path)  # Mark as permanently skipped
                
                # Remove exceeded files from files_remaining
                if files_to_remove:
                    files_remaining = [f for f in files_remaining if f not in files_to_remove]
                
                if not files_to_process:
                    logger.info("No files to process after filtering (all exceeded max retries)")
                    if not files_remaining:
                        break
                    # If files_remaining still has items but all exceeded retries, clear it
                    files_remaining = []
                    continue
                
                logger.info(f"Files remaining: {len(files_to_process)}, spawning {max_parallel_workers} workers...")
                                
                num_workers = min(max_parallel_workers, len(files_to_process))
                if len(files_to_process) < (max_parallel_workers * MIN_FILES_PER_WORKER):
                    # Not enough files for full parallelism, reduce workers
                    num_workers = max(1, len(files_to_process) // FALLBACK_FILES_PER_WORKER)
                
                # Final filter: remove any files that have exceeded retries from chunks
                # This is a last check before sending to workers
                final_files_to_process = []
                for file_path in files_to_process:
                    if file_path in permanently_skipped_files:
                        continue
                    # Double-check retry count one more time
                    retry_count = retry_counts_global.get(file_path, 0)
                    if retry_count >= count_retries_failed_files:
                        logger.warning(f"Final filter: Removing {file_path} - exceeded max retries ({retry_count} >= {count_retries_failed_files})")
                        permanently_skipped_files.add(file_path)
                        continue
                    final_files_to_process.append(file_path)
                
                if not final_files_to_process:
                    logger.info("No files to process after final filtering (all exceeded max retries)")
                    files_remaining = [f for f in files_remaining if f not in files_to_process]
                    continue
                
                file_chunks = chunk_list(final_files_to_process, num_workers)
                
                # Prepare arguments for each worker (no queue needed)
                worker_args = [
                    (config, worker_num, {release_year: chunk}) 
                    for worker_num, chunk in enumerate(file_chunks)
                ]
                
                # Launch workers in parallel using Pool
                with mp.Pool(processes=num_workers) as pool:
                    # Launch all workers asynchronously
                    async_results = []
                    for args in worker_args:
                        async_result = pool.apply_async(
                            worker_parse_and_process_one_chunk, 
                            args=args
                        )
                        async_results.append(async_result)
                    
                    # Process results as they become available (not in order)
                    results = [None] * len(async_results)  # Pre-allocate to maintain order
                    completed_count = 0
                    
                    while completed_count < len(async_results):
                        for i, async_result in enumerate(async_results):
                            if results[i] is None and async_result.ready():
                                try:
                                    result = async_result.get(timeout=0.1)
                                    results[i] = result
                                    completed_count += 1
                                    
                                    # Handle invalid result (None or missing status)
                                    if result is None or not isinstance(result, dict) or "status" not in result:
                                        logger.error(f"Worker {i+1} returned invalid result: {result}")
                                        # Mark as error to prevent infinite loop
                                        result = {
                                            "status": "error",
                                            "message": f"Invalid result returned: {result}",
                                            "files_parsed": []
                                        }
                                    
                                    if result["status"] == "chunk_complete":
                                        all_stats.append(result["stats"])
                                        files_parsed = result["files_parsed"]
                                        
                                        # Collect saved filename for IM pipeline
                                        if result.get("saved_filename"):
                                            all_saved_files.append(result["saved_filename"])
                                            logger.info(f"Worker {i+1} saved file: {result['saved_filename']}")
                                        
                                        # Update tracking - use set for O(1) removal
                                        files_parsed_set = set(files_parsed)
                                        files_remaining_set = set(files_remaining)
                                        files_remaining_set -= files_parsed_set
                                        files_remaining = list(files_remaining_set)
                                        
                                        chunk_count += 1
                                        total_events += result["num_events"]
                                        pbar.update(len(files_parsed))
                                        
                                        logger.info(f"Worker {i+1} completed: {len(files_parsed)} files, {result['num_events']} events")
                                    
                                    elif result["status"] == "error":
                                        logger.error(f"Worker {i+1} failed: {result['message']}")
                                        # Mark files as failed so they can be retried
                                        if "files_parsed" in result:
                                            files_parsed = result["files_parsed"]
                                            files_parsed_set = set(files_parsed)
                                            files_remaining_set = set(files_remaining)
                                            # Don't remove from files_remaining - they should be retried
                                    
                                    elif result["status"] == "no_chunks":
                                        logger.warning(f"Worker {i+1}: No chunks produced - {result.get('message', 'unknown reason')}")
                                        # Remove attempted files from files_remaining since they failed
                                        # These will be handled by the retry mechanism via crashed_files.json
                                        if "files_parsed" in result:
                                            attempted_files = result["files_parsed"]
                                            files_parsed_set = set(attempted_files)
                                            files_remaining_set = set(files_remaining)
                                            files_remaining_set -= files_parsed_set
                                            files_remaining = list(files_remaining_set)
                                            logger.info(f"Removed {len(attempted_files)} failed files from files_remaining (will be retried via crashed_files.json)")
                                    
                                    elif result["status"] == "skipped":
                                        logger.info(f"Worker {i+1}: Skipped (all files exceeded max retries)")
                                        # Remove skipped files from files_remaining since they exceeded max retries
                                        if "skipped_files" in result and result["skipped_files"]:
                                            skipped_files_set = set(result["skipped_files"])
                                            files_remaining_set = set(files_remaining)
                                            files_remaining_set -= skipped_files_set
                                            files_remaining = list(files_remaining_set)
                                            logger.info(f"Removed {len(result['skipped_files'])} files from files_remaining (exceeded max retries)")
                                        else:
                                            # Fallback: if skipped_files not provided, remove all files assigned to this worker
                                            # This handles edge cases where the worker structure might differ
                                            if i < len(file_chunks):
                                                assigned_chunk = file_chunks[i]
                                                assigned_files_set = set(assigned_chunk)
                                                files_remaining_set = set(files_remaining)
                                                files_remaining_set -= assigned_files_set
                                                files_remaining = list(files_remaining_set)
                                                logger.info(f"Removed {len(assigned_chunk)} assigned files from files_remaining (all exceeded max retries, fallback removal)")
                                except Exception as e:
                                    logger.error(f"Error getting result from worker {i+1}: {e}")
                                    results[i] = {"status": "error", "message": str(e)}
                                    completed_count += 1
                        
                        # Small sleep to avoid busy-waiting
                        if completed_count < len(async_results):
                            time.sleep(0.01)
                    
                    # Get timestamp when all workers are done
                    all_workers_end_timestamp = datetime.now()
                    logger.info(f"All workers completed at: {all_workers_end_timestamp}")
                
            if not os.path.exists(crashed_files_path):
                logger.info("No crashed files to retry.")
                break
            
            # Validate and load crashed files
            try:
                with open(crashed_files_path, "r") as f:
                    data = json.load(f)
                
                # Get failed files and filter out those that exceeded max retries
                all_failed_files = data.get("failed_files", [])
                retry_counts = data.get("retry_counts", {})
                # Update global retry counts for use in inner loop
                retry_counts_global.update(retry_counts)
                max_file_retries = count_retries_failed_files  # Use same limit as global retries
                
                # Filter out files that have exceeded max retries
                # Note: retry_count is the number of times we've already retried
                # So if retry_count >= max_file_retries, we've exceeded the limit
                files_remaining = []
                files_exceeded_retries = []
                for file_path in all_failed_files:
                    # Skip if already permanently marked
                    if file_path in permanently_skipped_files:
                        files_exceeded_retries.append(file_path)
                        continue
                    
                    retry_count = retry_counts.get(file_path, 0)
                    if retry_count < max_file_retries:  # Use < instead of <= to prevent retrying when at limit
                        files_remaining.append(file_path)
                    else:
                        files_exceeded_retries.append(file_path)
                        permanently_skipped_files.add(file_path)  # Mark as permanently skipped
                        logger.warning(f"Skipping {file_path} - exceeded max retries ({retry_count} >= {max_file_retries})")
                
                if not isinstance(all_failed_files, list):
                    logger.warning(f"Invalid format in {crashed_files_path}, expected list")
                    files_remaining = []
                
                # Clean up: remove files that exceeded retries from the JSON file IMMEDIATELY
                # This prevents them from being processed by workers
                if files_exceeded_retries:
                    for file_path in files_exceeded_retries:
                        if file_path in data["failed_files"]:
                            data["failed_files"].remove(file_path)
                        # Also remove from retry_counts to prevent future issues
                        if "retry_counts" in data and file_path in data["retry_counts"]:
                            del data["retry_counts"][file_path]
                        # Mark as permanently skipped
                        permanently_skipped_files.add(file_path)
                    # Update the file to remove exceeded retry files BEFORE processing
                    with open(crashed_files_path, 'w') as f:
                        json.dump(data, f, indent=2)
                    logger.info(f"Cleaned up {len(files_exceeded_retries)} files that exceeded max retries from crashed_files.json (removed from both failed_files and retry_counts)")
                
                # Only proceed if there are files remaining that haven't exceeded retries
                if not files_remaining:
                    logger.info("No files to retry (all exceeded max retries or list is empty).")
                    # Remove the crashed files list since we're done with retries
                    if os.path.exists(crashed_files_path):
                        os.remove(crashed_files_path)
                    break
                
                # Update files_remaining to only include files that haven't exceeded retries
                # This ensures workers don't process files that exceeded the limit
                logger.info(f"Retrying {len(files_remaining)} crashed files (out of {len(all_failed_files)} total failed, {len(files_exceeded_retries)} exceeded limit)...")
                    
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error reading crashed files: {e}. Skipping retry.")
                break
            
            # Allow for some sleep to let remote dataserver rest
            logger.info(f"Retrying... after {cur_retries} times")
            cur_retries += 1
            time.sleep(10)
        
        # Close progress bar when done with this release year
        if show_progress_bar:
            pbar.close()

    if all_stats:
        aggregate_statistics(all_stats, atlasparser_config["logging_path"], pipeline_config, atlasparser_config, run_metadata)    
    else:
        logging.info(f"No stats to aggregate.")

    logger.info(
        f"Parsing complete! Processed {chunk_count} chunks, "
        f"{total_events:,} total events"
    )
    
    # Return list of successfully saved files for IM pipeline
    logger.info(f"Successfully saved {len(all_saved_files)} ROOT files for IM calculation.")
    return all_saved_files

def get_batch_by_index(whole_file_ids, batch_index, total_batch_jobs):
    """Extracts a batch of file IDs using a moving window across all years"""
    batch_index = int(batch_index)
    total_batch_jobs = int(total_batch_jobs)
    
    # Flatten all file IDs into a single array with year tracking
    all_files = []
    for year, file_ids in whole_file_ids.items():
        for file_id in file_ids:
            all_files.append((year, file_id))
    
    # Calculate batch boundaries
    total_files = len(all_files)
    files_per_batch = total_files // total_batch_jobs
    start_idx = (batch_index - 1) * files_per_batch
    
    if batch_index == total_batch_jobs:
        end_idx = total_files
    else:
        end_idx = start_idx + files_per_batch
    
    # Extract the batch slice
    batch_slice = all_files[start_idx:end_idx]
    
    # Reconstruct dictionary format
    batch_file_ids = {}
    for year, file_id in batch_slice:
        if year not in batch_file_ids:
            batch_file_ids[year] = []
        batch_file_ids[year].append(file_id)
    
    return batch_file_ids

def chunk_list(lst, n):
    """Split list into n roughly equal chunks"""
    k, m = divmod(len(lst), n)
    return [lst[i*k+min(i,m):(i+1)*k+min(i+1,m)] for i in range(n)]

def aggregate_statistics(stats_list, output_path, pipeline_config, atlasparser_config, run_metadata):
    """Combine statistics from all worker chunks into consolidated metrics"""   
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
    total_filtered_data_mb = sum(s["performance"].get("total_filtered_data_mb", 0.0) for s in stats_list)
    total_disk_data_mb = sum(s["performance"].get("total_disk_data_mb", 0.0) for s in stats_list)
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
    timestamps = [datetime.fromisoformat(s["parsing_session"]["timestamp"]).second for s in stats_list]
    max_mem_captures = [s["performance"]["max_memory_captured_mb"] for s in stats_list]
    chunks_sizes = [s["performance"]["avg_chunk_size_mb"] for s in stats_list]
    
    zipped_data = list(zip(timestamps, max_mem_captures, chunks_sizes))
    zipped_data.sort(key=lambda tuple: tuple[0])
    timestamps, max_mem_captures, chunks_sizes = zip(*zipped_data)
    
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
    config_yield_threshold_mb = atlasparser_config["chunk_yield_threshold_bytes"] / 1024**2

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
        },
        "performance": {
            "total_data_processed_mb": total_data_mb,
            "total_filtered_data_mb": total_filtered_data_mb,
            "total_disk_data_mb": total_disk_data_mb,
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
        "config_variables":{
            "config_yield_threshold_mb": config_yield_threshold_mb,
            "max_parallel_workers": pipeline_config["max_parallel_workers"],
            "limit_files_per_year": pipeline_config["limit_files_per_year"],
            "max_threads": atlasparser_config["max_threads"],
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
    output_file = os.path.join(output_path, run_metadata["run_name"]) + ".json"
    with open(output_file, "w") as f:
        json.dump(consolidated, f, indent=2)
        logging.info(f"Succesfully aggregated and saved all statistics under the name {output_file}")
    
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