import awkward as ak
import numpy as np
import vector
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import itertools

import uproot
import requests
import atlasopenmagic as atom

import csv
import random
from tqdm import tqdm
import os
import psutil
import traceback
import logging
import sys
import io
from contextlib import redirect_stdout

import threading
import datetime
import time
import json
import gc
import tracemalloc
import hashlib

from . import schemas
from src.utils import memory_utils

class AtlasOpenParser():
    '''
    AtlasOpenParser class is responsible for parsing ATLAS Open Data files and handling event selection, filtering, and output.
    Attributes:
        categories (list): List of object categories for combinatorics.
        files_ids (list): List of file URIs to be parsed.
        file_parsed_count (int): Counter for parsed files.
        cur_chunk (int): Current chunk index.
        events (ak.Array): Accumulated events.
        total_size_kb (int): Total size of accumulated events in kilobytes.
        max_chunk_size_bytes (int): Maximum chunk size in bytes before yielding.
    Methods:
        generate_histograms():
            Generates histograms for object combinations (not yet implemented).
        calculate_mass_for_combination(events: ak.Array) -> ak.Array:
            Computes invariant mass per event from combined objects.
        save_events_as_root(events, output_dir):
            Saves filtered events to ROOT files in the specified directory.
        flatten_for_root(awk_arr):
            Flattens awkward arrays for ROOT compatibility.
        fetch_records_ids(release_year):
            Fetches file URIs for a given release year.
        parse_files(files_ids: list = None, limit: int = 0, max_threads: int = 3):
            Parses files in parallel and yields chunks of events.
            Checks if the accumulated events exceed the chunk size.
        fetch_mc_files_ids(release_year, is_random=False, all=False):
            Fetches Monte Carlo records IDs for a given release year.
        log_cur_size():
            Logs the current size of accumulated events.
        _prepare_obj_branch_name(obj_name):
            Prepares object name and zip function for parsing.
        _normalize_fields(ak_array):
            Normalizes fields in the awkward array to match schema.
    '''
    def __init__(self, 
                chunk_yield_threshold_bytes, 
                max_threads, 
                logging_path, 
                possible_tree_names=["CollectionTree"],
                create_dirs=False, 
                max_environment_memory_mb=None, 
                release_years=[],
                show_available_releases=False,
                wrapping_logger=None):
        self.files_ids = None
        self.file_parsed_count = 0
        self.cur_chunk = 0
        self.cur_file_ids = []
        
        self.events = None
        self.total_size_kb = 0
        
        #SET UP RELEASES VARIABLES
        self._fetch_available_releases(show_available_releases)
        self._setup_release_years(release_years)
        
        self._init_flags(
            create_dirs,
            possible_tree_names,
            chunk_yield_threshold_bytes,
            max_environment_memory_mb,
            logging_path
            )
        
        self._init_logger(wrapping_logger)
        # New statistics tracking
        self.crash_lock = threading.Lock()
        self.failed_files = []
        self.parsing_start_time = time.time()
        self.chunk_stats = []
        self.error_types = {}
        self.timeout_count = 0
        self.max_file_size_mb = 0
        self.min_file_size_mb = float('inf')
        self.total_events_processed = 0
        self.max_memory_captured = 0
        self.total_chunks = 0
        # ================================================================

        #PARALLELISM CONFIG
        self.max_threads = max_threads

    def _init_flags(self, 
                          create_dirs, 
                          possible_tree_names, 
                          chunk_yield_threshold_bytes, 
                          max_environment_memory_mb,
                          logging_path):
        if possible_tree_names:
            self.possible_tree_names = possible_tree_names
            logging.info(
                f"Possible tree names set to: {self.possible_tree_names}.")
        
        if chunk_yield_threshold_bytes:
            self.chunk_yield_threshold_bytes = chunk_yield_threshold_bytes
            logging.info(
                f"Chunk yield threshold set to: {self.chunk_yield_threshold_bytes//1024**2} MB.")
        
        if max_environment_memory_mb:
            self.max_environment_memory_mb = max_environment_memory_mb
            logging.info(
                f"Max environment memory set to: {self.max_environment_memory_mb} MB.")

        if logging_path:
            self.crash_log = logging_path + "atlas_crashes.log"
            self.stats_log = logging_path + "atlas_stats.json"
            self.crashed_files = logging_path + "crashed_files.json"
        
            logging.info(
                f"Logging paths set to: {self.crash_log}, {self.stats_log}, {self.crashed_files}.")
            
        if create_dirs:
            self._initialize_statistics()
            logging.info(
                f"Recreated directories...")

    def _init_logger(self, wrapping_logger=None):
        if wrapping_logger is not None and isinstance(wrapping_logger, logging.Logger):
            root = logging.getLogger()
            root.handlers = list(wrapping_logger.handlers)
            root.setLevel(wrapping_logger.level)
            
    #FETCHING FILE IDS
    def _fetch_available_releases(self, show_available_releases):
        if show_available_releases:
            self.available_releases = atom.available_releases() 
        else:
            with redirect_stdout(io.StringIO()):
                self.available_releases = atom.available_releases()
                
    def _setup_release_years(self, release_years):
        invalid_releases = [year for year in release_years if year not in self.available_releases]
        if invalid_releases:
            formatted_releases = list(self.available_releases.keys())
            raise ValueError(f"Release years {invalid_releases} are not recognized. Available releases: {formatted_releases}")
        
        self.input_release_years = release_years

    def fetch_record_ids(self, timeout=60):
        '''
            Fetches the real records IDs for a given release year.
            Returns a list of file URIs.
        '''
        release_years = []
        if self.input_release_years:
            release_years = self.input_release_years
        else:
            release_years = self.available_releases
        
        release_files_uris: dict = ATLAS_Parser.fetch_record_ids_for_release_years(
            release_years,
            timeout=timeout)

        return release_files_uris
    
    @staticmethod
    def fetch_record_ids_for_release_years(release_years, timeout=60) -> dict:
        release_years_file_ids = {}
        with ThreadPoolExecutor(max_workers=1) as executor:
            for year in release_years:
                if year not in release_years_file_ids.keys():
                    release_years_file_ids[year] = []
                try:
                    future = executor.submit(atom.set_release, year)
                    future.result(timeout=timeout)
                    
                    datasets = atom.available_datasets()

                    for dataset_id in datasets:
                        future = executor.submit(atom.get_urls, dataset_id) #FEATURE PARSING handle 'noskim' absence
                        urls = future.result(timeout=timeout)
                        if urls:
                            release_years_file_ids[year].extend(urls)
                except TimeoutError:
                    logging.warning(f"Timeout while fetching metadata for release year {year}")

                except Exception as e:
                    logging.warning(f"Could not fetch metadata for release year {year}: {e}")
        
        return release_years_file_ids
    
    def fetch_mc_files_ids(self, release_year, is_random=False, all=False):
        '''
            Fetches the Monte Carlo records IDs for a given release year.
            Returns a list of file URIs.
        '''
        release = consts.RELEASES_YEARS[release_year]
        metadata_url = consts.LIBRARY_RELEASES_METADATA[release]

        _metadata = {}

        response = requests.get(metadata_url)

        response.raise_for_status()
        lines = response.text.splitlines()

        reader = csv.DictReader(lines)
        for row in reader:
            dataset_number = row['dataset_number'].strip()
            _metadata[dataset_number] = row

        all_mc_ids = list(_metadata.keys())

        if is_random:
            random_mc_id = random.choice(all_mc_ids)
            all_metadata = atom.get_metadata(random_mc_id)
            print(all_metadata['process'], all_metadata['short_name'])
            return random_mc_id

        return all_mc_ids

    #PARSING METHODS
    def parse_files(self,
                       release_years_file_ids: dict = None,
                       save_statistics: bool = False):
        '''
            Parses the input files by their IDs, otherwise uses the member release_years_file_ids.
            Yields chunks of events as awkward arrays each size from the input limit.
        '''
        if release_years_file_ids is None:
            raise ValueError("No release_years_file_ids provided.")
        
        self.successful_count = 0
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            for release_year, file_ids in release_years_file_ids.items():
                logging.info(f"Starting to parse year {release_year}'s {len(file_ids)} files with {self.max_threads} threads.")
                with tqdm(total=len(file_ids), desc="Parsing files", unit="file", dynamic_ncols=True, mininterval=1) as pbar:
                    futures = {
                        executor.submit(
                            AtlasOpenParser.parse_file, 
                            file_index,
                            self.possible_tree_names
                        ): file_index
                        for file_index in file_ids
                    }
                
                    for future in as_completed(futures):
                        file_index = futures[future]
                        self._process_completed_future(
                            future, file_index, release_year
                        )
                        
                        if self._chunk_exceed_threshold():
                            self._save_chunk_metadata()
                            chunk_to_yield = self._prepare_chunk_for_yield()
                        
                            yield chunk_to_yield
                            del chunk_to_yield
                            gc.collect() 
                            memory_utils.print_gc_stats() #FOR TESTING
                            memory_utils.print_top_memory_variables(n=10) #FOR TESTING
                        #CHECK remove this commented?
                        '''
                        try:
                            cur_file_data = future.result(timeout=10)
                            
                            if cur_file_data is not None:
                                successful_count += 1

                                self._save_parsed_file_metadata(cur_file_data, release_year, file_index)
                                cur_file_data = AtlasOpenParser._normalize_fields(cur_file_data)
                                self._concatenate_events(cur_file_data)
                                
                                tqdm.write(f"{memory_utils.get_process_mem_usage_mb():.1f} MB used after parsing {self.file_parsed_count} files.")
                                if self._chunk_exceed_threshold():
                                    self._save_chunk_metadata()
                                    
                                    # Store chunk reference
                                    chunk_to_yield = self.events
                                    
                                    # CRITICAL: Clear reference BEFORE yielding
                                    self.events = None
                                    self.file_parsed_count = 0
                                    self.total_chunks += 1
                                    
                                    mem_before_yield = memory_utils.get_process_mem_usage_mb()
                                    self.max_memory_captured = max(self.max_memory_captured, mem_before_yield)
                                    
                                    yield chunk_to_yield
                                    
                                    # CRITICAL: delete local reference after yield
                                    del chunk_to_yield
                                    
                                    # Force aggressive cleanup after yield
                                    gc.collect()
                                    memory_utils.print_gc_stats()
                                    memory_utils.print_top_memory_variables(n=10)
                                    
                                    mem_after_yield = memory_utils.get_process_mem_usage_mb()
                                    mem_freed = mem_before_yield - mem_after_yield
                                    

                                    tqdm.write(
                                        f"🧹 Memory before yield: {mem_before_yield:.1f} MB "
                                        f"🧹 Memory after yield: {mem_after_yield:.1f} MB "
                                        f"(freed: {mem_freed:.1f} MB)"
                                    )

                        except Exception as e:
                            file_processing_time = time.time() - file_start_time
                            self._log_crash(file_index, e, file_processing_time)
                            tqdm.write(f"⚠️ Error: {file_index} - {type(e).__name__}")

                        '''
                        status = self._get_parsing_status_for_pbar()
                        pbar.set_postfix_str(status)
                        pbar.update(1)
                
                gc.collect()
                        
            #TEMP for when the subprocess eliminates the generator and never returns so yielding has to be last
            if self.events is not None:
                self.total_chunks += 1
                self._save_chunk_metadata()
                
            if save_statistics:
                logging.info(f"Saving statistics to {self.stats_log}")
                total_files = [file_id for file_id in file_ids for file_ids in release_years_file_ids.values()]
                stats = self.get_statistics(len(total_files))
                
                with open(self.stats_log, 'a') as f:
                    json.dump(stats, f, indent=2)

                self.print_statistics_summary(stats)

            if self.events is not None:
                self.max_memory_captured = memory_utils.get_process_mem_usage_mb()
                yield self.events
                self.events = None
    
    def _process_completed_future(self, future, file_index, release_year):
        """Process a single completed future. Returns updated success count."""
        file_start_time = time.time()
        try:
            cur_file_data = future.result(timeout=10)
            if cur_file_data is not None:
                self.successful_count += 1
                self._save_parsed_file_metadata(cur_file_data, release_year, file_index)
                cur_file_data = AtlasOpenParser._normalize_fields(cur_file_data)
                self._concatenate_events(cur_file_data)
                
                tqdm.write(f"{memory_utils.get_process_mem_usage_mb():.1f} MB used...")
                
        except Exception as e:
            file_processing_time = time.time() - file_start_time
            self._log_crash(file_index, e, file_processing_time)
            tqdm.write(f"⚠️ Error: {file_index} - {type(e).__name__}") 
    
    def _prepare_chunk_for_yield(self):    
        chunk_to_yield = self.events
        self.events = None
        self.file_parsed_count = 0
        self.total_chunks += 1
        
        mem_before_yield = memory_utils.get_process_mem_usage_mb()
        self.max_memory_captured = max(self.max_memory_captured, mem_before_yield)
        
        return chunk_to_yield

    @staticmethod
    def parse_file(file_index, tree_names=["CollectionTree"], batch_size=40_000) -> ak.Array:
        """
        Parse an ATLAS DAOD file in batches if necessary.
        Accumulates raw arrays and zips into vector objects only once per object.
        """
        with uproot.open(file_index) as root_file:
            root_file_keys = root_file.keys()
            tree_name = AtlasOpenParser.get_data_tree_name_for_root_file(root_file_keys, tree_names)
            tree = root_file[tree_name]
            all_tree_branches = set(tree.keys())
            n_entries = tree.num_entries
            is_file_big = n_entries > batch_size 

            obj_branches_and_quantities: dict[str, dict[str, str]] = AtlasOpenParser.extract_branches_by_obj_in_schema(
                all_tree_branches, schema=schemas.INVARIANT_MASS_SCHEMA)

            if not obj_branches_and_quantities.values():
                return None
            
            all_branches = set(itertools.chain.from_iterable(obj_branches_and_quantities.values()))

            # 2. Initialize storage for raw batches
            obj_events_by_quantities = {obj_name: [] for obj_name in obj_branches_and_quantities.keys()}

            # 3. Define entry ranges
            entry_ranges = []
            if is_file_big:
                entry_ranges = [
                    (start, min(start + batch_size, n_entries)) 
                    for start in range(0, n_entries, batch_size)
                ]
            else: #If file not big entry_ranges is the entire file
                entry_ranges = [(0, n_entries)]

            # 4. Read batches
            for entry_start, entry_stop in entry_ranges:
                batch_data = tree.arrays(all_branches, entry_start=entry_start, entry_stop=entry_stop)
                
                for obj_name, branch_mapping in obj_branches_and_quantities.items():
                    subset = batch_data[list(branch_mapping.keys())]  # Get branches as list
                    if len(subset) > 0:
                        obj_events_by_quantities[obj_name].append(subset)

            for obj_name, chunks in obj_events_by_quantities.items():
                concatenated = ak.concatenate(chunks)
                
                obj_events_by_quantities[obj_name] = ak.zip({
                    quantity: concatenated[full_branch]
                    for full_branch, quantity in obj_branches_and_quantities[obj_name].items()
                })

            return ak.zip(obj_events_by_quantities, depth_limit=1)

    def _get_parsing_status_for_pbar(self):
        success_rate = (
                        (self.successful_count / (self.successful_count + len(self.failed_files)) * 100)
                        if (self.successful_count + len(self.failed_files)) > 0 else 0
                    )
        status = (
                f"✅ {self.successful_count} | "
                f"❌ {len(self.failed_files)} | "
                f"✨ {success_rate:.1f}% | "
                f"💾 {self.total_size_kb / (1024 * 1024):.1f} MB | "
                f"🎯 {self.total_events_processed:,} events"
            )
        
        return status

    def _concatenate_events(self, cur_file_data):
        chunk_size_kb = cur_file_data.layout.nbytes
        self.total_size_kb += chunk_size_kb
        
        mem_before = memory_utils.get_process_mem_usage_mb()

        if self.events is None:
            self.events = cur_file_data
        else:
            old_events = self.events
            self.events = ak.concatenate([old_events, cur_file_data], axis=0)
        
        mem_after = memory_utils.get_process_mem_usage_mb()
        mem_delta = mem_after - mem_before
        
        # Log if memory grew unexpectedly
        if mem_delta > chunk_size_kb / (1024 * 1024) * 3:  # More than 3x file size
            tqdm.write(f"⚠️  High memory growth: {mem_delta:.1f} MB for {chunk_size_kb/(1024*1024):.1f} MB file")
        
        self.file_parsed_count += 1
    
    #SAVING FILES
    def save_events_as_root(self, events, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        file_name = f"{self.cur_release_year}_{list_to_filename_hash(self.cur_file_ids)}"

        output_path = os.path.join(output_dir, f"{file_name}.root")
        with uproot.recreate(output_path) as file:
            file["CollectionTree"] = events
            file["metadata"] = {
                'file_ids': ak.Array([','.join(self.cur_file_ids)]),
                'n_files': ak.Array([len(self.cur_file_ids)])
            }

        self.cur_file_ids.clear()
    
    def flatten_for_root(self, awk_arr):
        """
        Flatten a top-level awkward Array into a ROOT-friendly dict
        compatible with _parse_file()'s expected field structure.
        Specifically, each particle object is stored under
        "<cur_obj_name>AuxDyn.<field>" branches.
        """
        root_ready = {}

        for obj_name in awk_arr.fields:
            obj = awk_arr[obj_name]

            # e.g., obj_name = "Jets"  -> cur_obj_name = "AnalysisJets"
            # obj_branches_and_quantities: dict[str, dict[str, str]] = AtlasOpenParser.extract_branches_by_obj_in_schema(
            #     all_tree_branches, schema=schemas.INVARIANT_MASS_SCHEMA)
            
            cur_obj_branch_name = AtlasOpenParser._prepare_obj_branch_name(
                obj_name, schema=schemas.INVARIANT_MASS_SCHEMA) #TODO replace with extract_branches_by_obj_in_schema?

            if cur_obj_branch_name is None:
                cur_obj_branch_name = obj_name

            try:
                # If obj is a record array, iterate over its fields
                for field in obj.fields:
                    branch = obj[field]
                    # ROOT doesn't like None — fill with 0.0
                    filled_branch = ak.fill_none(branch, 0.0)

                    # IMPORTANT: match the structure used by _parse_file
                    # e.g., "AnalysisJetsAuxDyn.pt"
                    branch_name = f"{cur_obj_branch_name}AuxDyn.{field}"
                    root_ready[branch_name] = filled_branch

            except AttributeError:
                # Not a record array — save as a top-level branch
                logging.info(f"Warning: {obj_name} is not a record array, saving as-is.")
                root_ready[obj_name] = ak.fill_none(obj, 0.0)

            except Exception as e:
                logging.info(f"Error processing {obj_name}: {e}")
                continue
        
        
        return root_ready

    def _chunk_exceed_threshold(self):
        """Check if we should yield based on ACTUAL memory pressure"""
        if self.events is None:
            return False
        
        logical_size = self.events.layout.nbytes
        actual_memory = memory_utils.get_process_mem_usage_mb() 
        
        if hasattr(self, 'max_environment_memory_mb'):
            if (actual_memory + 1000) > self.max_environment_memory_mb:
                tqdm.write(f"High memory usage: {actual_memory:.1f} MB (limit: {self.max_environment_memory_mb} MB)")
                return True

        return logical_size >= self.chunk_yield_threshold_bytes

    #LOGGING METHODS
    def _log_crash(self, file_index, exception, processing_time=None):
        """Enhanced thread-safe crash logging with statistics"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_type = type(exception).__name__
        
        with self.crash_lock:
            self.failed_files.append(file_index)
            
            # Track error types for statistics
            if error_type not in self.error_types:
                self.error_types[error_type] = 0
            self.error_types[error_type] += 1
            
            # Track timeouts separately
            if isinstance(exception, TimeoutError):
                self.timeout_count += 1
            
            with open(self.crash_log, 'a') as f:
                f.write(f"\n💥 CRASH at {timestamp}\n")
                f.write(f"File: {file_index}\n")
                f.write(f"Error Type: {error_type}\n")
                f.write(f"Error: {exception}\n")
                if processing_time:
                    f.write(f"Processing Time: {processing_time:.2f}s\n")
                f.write(f"Traceback:\n{traceback.format_exc()}\n")
                f.write("-" * 60 + "\n")

            if os.path.exists(self.crashed_files):
                with open(self.crashed_files, 'r') as f:
                    data = json.load(f)
            else:
                data = {"failed_files": []}

            data["failed_files"].append(file_index)

            with open(self.crashed_files, 'w+') as f:
                json.dump(data, f, indent=2)

    def _save_parsed_file_metadata(self, cur_file_data, release_year, file_index):                    
        file_size_mb = cur_file_data.layout.nbytes / (1024 * 1024)
        self.max_file_size_mb = max(self.max_file_size_mb, file_size_mb)
        self.min_file_size_mb = min(self.min_file_size_mb, file_size_mb)

        events_in_file = len(cur_file_data)
        self.total_events_processed += events_in_file
        tqdm.write(
            f"✅ File processed: {file_size_mb:.2f} MB logical size, " 
            f"{events_in_file:,} events."
        )
        
        self.cur_file_ids.append(file_index)
        self.cur_release_year = release_year
    
    def _save_chunk_metadata(self):
        chunk_info = {
            "chunk_id": self.cur_chunk,
            "events": len(self.events),
            "size_mb": self.events.layout.nbytes / (1024 * 1024),
            "files_included": self.file_parsed_count
        }
        self.chunk_stats.append(chunk_info)
        
        self.cur_chunk += 1
    
    def get_statistics(self, total_files):
        """Save comprehensive parsing statistics to JSON"""
        end_time = time.time()
        total_time = end_time - self.parsing_start_time if self.parsing_start_time else 0
        avg_chunk_time = total_time / self.total_chunks

        stats = {
            "parsing_session": {
                "timestamp": datetime.datetime.now().isoformat(),
                "total_time_seconds": total_time,
                "avg_chunk_time": avg_chunk_time,
                "total_files": total_files,
                "successful_files": self.successful_count,
                "failed_files": len(self.failed_files),
                "success_rate": (self.successful_count / total_files * 100) if total_files > 0 else 0
            },
            "performance": {
                "max_file_size_mb": self.max_file_size_mb if self.max_file_size_mb > 0 else 0,
                "min_file_size_mb": self.min_file_size_mb if self.min_file_size_mb != float('inf') else 0,
                "total_data_processed_mb": self.total_size_kb / (1024 * 1024),
                "total_events_processed": self.total_events_processed,
                "chunks_created": self.cur_chunk,
                "avg_chunk_size_mb": (self.total_size_kb / (1024 * 1024)) / max(self.cur_chunk, 1),
                "max_memory_captured_mb": self.max_memory_captured
            },
            "errors": {
                "timeout_count": self.timeout_count,
                "error_types": self.error_types,
                "failed_file_list": self.failed_files[:10]  # Only store first 10 for brevity
            },
            "chunk_details": self.chunk_stats
        }
        
        return stats

    def print_statistics_summary(self, stats):
        """Print a nice summary of parsing statistics"""
        tqdm.write(f"\n" + "="*60)
        tqdm.write(f"📊 ATLAS PARSING STATISTICS SUMMARY")
        tqdm.write(f"="*60)
        tqdm.write(f"⏱️  Total Time: {stats['parsing_session']['total_time_seconds']:.1f}s")
        tqdm.write(f"📁 Files: ✅{stats['parsing_session']['successful_files']} / ❌{stats['parsing_session']['failed_files']} / 📊{stats['parsing_session']['total_files']}")
        tqdm.write(f"✨ Success Rate: {stats['parsing_session']['success_rate']:.1f}%")
        tqdm.write(f"💾 Data Processed: {stats['performance']['total_data_processed_mb']:.1f} MB")
        tqdm.write(f"🎯 Events Processed: {stats['performance']['total_events_processed']:,}")
        tqdm.write(f"📦 Chunks Created: {stats['performance']['chunks_created']}")
        tqdm.write(f"📊 Max memory captured: {stats['performance']['max_memory_captured_mb']:.1f} MB")
        
        if stats['errors']['error_types']:
            tqdm.write(f"⚠️  Error Breakdown:")
            for error_type, count in stats['errors']['error_types'].items():
                tqdm.write(f"   • {error_type}: {count}")
        
        tqdm.write(f"📋 Full stats saved to: {self.stats_log}")
        if stats['parsing_session']['failed_files'] > 0:
            tqdm.write(f"💥 Crash log saved to: {self.crash_log}")
        tqdm.write(f"="*60)
   
    def _initialize_statistics(self):
        if os.path.exists(self.crash_log):
            os.remove(self.crash_log) 
        os.makedirs(os.path.dirname(self.crash_log), exist_ok=True)

        if os.path.exists(self.stats_log):
            os.remove(self.stats_log) 
        os.makedirs(os.path.dirname(self.stats_log), exist_ok=True)

        if os.path.exists(self.crashed_files):
            os.remove(self.crashed_files)
        os.makedirs(os.path.dirname(self.crashed_files), exist_ok=True)

    #STATIC METHODS
    @staticmethod
    def get_data_tree_name_for_root_file(root_file_keys, possible_tree_names):
        if not possible_tree_names:
            logging.warning("No possible_tree_names provided, defaulting to 'CollectionTree'.")
            return "CollectionTree"
        
        available_trees = [key[:-2] for key in root_file_keys] # Remove trailing ';1'
        for tree_name in possible_tree_names:
            if tree_name in available_trees:
                return tree_name
        
        return "CollectionTree" 
    
    @staticmethod
    def extract_branches_by_obj_in_schema(tree_branches, schema: dict):
        """Returns {obj_name: {full_branch: quantity, ...}}"""
        obj_branches = {}
        
        for obj_name, fields in schema.items():
            cur_obj_branch_name = AtlasOpenParser._prepare_obj_branch_name(obj_name, schema)
            obj_available_physical_quantities = [
                f for f in fields if f"{cur_obj_branch_name}.{f}" in tree_branches
            ]

            if AtlasOpenParser._can_calculate_inv_mass(obj_available_physical_quantities):
                obj_branches[obj_name] = {
                    f"{cur_obj_branch_name}.{quantity}": quantity
                    for quantity in obj_available_physical_quantities
                }

        return obj_branches

    @staticmethod
    def _can_calculate_inv_mass(available_fields, ref_system={'phi', 'eta', 'pt'}):
        """
        Check if we have the minimum fields needed to calculate invariant mass.
        """
        available_fields = set(available_fields)
        return ref_system.issubset(available_fields)
    
    @staticmethod
    def _prepare_obj_branch_name(obj_name, schema: dict):
        if obj_name in schema.keys():
            return "Analysis" + obj_name + "AuxDyn"
        else:
            return obj_name

    @staticmethod
    def _normalize_fields(ak_array):
        for field in schemas.INVARIANT_MASS_SCHEMA.keys():
            if field not in ak_array.fields:
                # Set to empty list per event (not None)
                ak_array = ak.with_field(
                    ak_array,
                    ak.Array([[]] * len(ak_array)),
                    field
                )
        return ak_array
    
    @staticmethod
    def limit_files_per_year(years_record_ids, limit_files_per_year):
        '''
            Limits the number of files per year to the specified limit.
        '''
        if limit_files_per_year is None:
            logging.warning("No limit_files_per_year provided.") 
            return
        
        for release_year, files_ids in years_record_ids.items():
            cur_files = files_ids[:limit_files_per_year]
            if len(cur_files) > 0:
                years_record_ids[release_year] = files_ids[:limit_files_per_year]

        logging.info(
            f"Limited files to given limit: {limit_files_per_year}.")
        

def list_to_filename_hash(strings):
    combined = "|".join(strings)  # delimiter avoids ambiguity
    digest = hashlib.sha1(combined.encode('utf-8')).hexdigest()
    return digest[:16]  # shorten to 16 chars if desired
    
 