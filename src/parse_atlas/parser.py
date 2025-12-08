import awkward as ak
import numpy as np
import vector
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import itertools

import uproot
import tarfile
import gzip
import tempfile
import shutil
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
from pathlib import Path

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
                wrapping_logger=None,
                temp_directory=None,
                show_progress_bar=True,
                max_file_retries=3):
        self.files_ids = None
        self.file_parsed_count = 0
        self.cur_chunk = 0
        self.cur_file_ids = []
        
        self.events = None
        self.total_size_kb = 0
        
        #SET UP TEMP DIRECTORY
        if temp_directory is None:
            self.temp_directory = tempfile.gettempdir()
        else:
            self.temp_directory = temp_directory
            # Create temp directory if it doesn't exist
            os.makedirs(self.temp_directory, exist_ok=True)
        
        #SET UP RELEASES VARIABLES
        self._fetch_available_releases(show_available_releases)
        self._setup_release_years(release_years)
        
        #SET UP PROGRESS BAR CONFIGURATION
        self.show_progress_bar = show_progress_bar
        
        #SET UP RETRY CONFIGURATION
        self.max_file_retries = max_file_retries
        
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
    
    @staticmethod
    def _get_noop_progress_bar():
        """Returns a no-op progress bar context manager for when progress bar is disabled."""
        class NoOpProgressBar:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def update(self, n=1):
                pass
            def set_postfix_str(self, s):
                pass
        return NoOpProgressBar()

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

    def fetch_record_ids(self, timeout=60, seperate_mc=False) -> dict:
        '''
            Fetches the real records IDs for a given release year.
            Returns a list of file URIs.
        '''
        release_years = []
        if self.input_release_years:
            release_years = self.input_release_years
        else:
            release_years = self.available_releases
        
        release_files_uris: dict = AtlasOpenParser.fetch_record_ids_for_release_years(
            release_years,
            timeout=timeout)

        #CHECK seperate mc feature
        if seperate_mc:
            release_files_uris = AtlasOpenParser._seperate_mc_files(release_files_uris)
        
        return release_files_uris
    
    @staticmethod
    def _seperate_mc_files(release_files_uris):
        """
        Separate MC files from data files by creating a view with _mc suffix.
        
        Note: The _mc suffix is only for file organization. Schema lookups
        automatically normalize release years (strip _mc) so "2024r-pp_mc" uses
        the same schema as "2024r-pp".
        
        Args:
            release_files_uris: Dictionary mapping release years to file URIs
            
        Returns:
            Dictionary with MC files under "{year}_mc" keys and data files under "{year}" keys
        """
        seperated_release_files_uris = {}
        for year, file_uris in release_files_uris.items():
            mc_file_uris = []
            data_file_uris = []
            for uri in file_uris:
                if "mc" in uri.lower():
                    mc_file_uris.append(uri)
                else:
                    data_file_uris.append(uri)    

            if mc_file_uris:
                mc_year = f"{year}_mc"
                seperated_release_files_uris[mc_year] = mc_file_uris
            if data_file_uris:
                seperated_release_files_uris[year] = data_file_uris
        
        return seperated_release_files_uris

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
            
            NOTE: This method is currently unused and may need refactoring.
            It relies on external constants that are not defined in this codebase.
        '''
        # TODO: This method references undefined constants (RELEASES_YEARS, LIBRARY_RELEASES_METADATA)
        # If this functionality is needed, these constants should be defined or
        # the method should be refactored to use the atlasopenmagic library directly.
        raise NotImplementedError(
            "fetch_mc_files_ids is not fully implemented. "
            "Use fetch_record_ids with appropriate release years instead."
        )

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
                self.cur_release_year = release_year
                logging.info(f"Starting to parse year {release_year}'s {len(file_ids)} files with {self.max_threads} threads.")
                
                # Use progress bar or no-op context manager based on configuration
                if self.show_progress_bar:
                    progress_bar = tqdm(total=len(file_ids), desc="Parsing files", unit="file", dynamic_ncols=True, mininterval=1)
                else:
                    # Create a no-op context manager for when progress bar is disabled
                    # Define it once outside the loop for efficiency
                    progress_bar = self._get_noop_progress_bar()
                
                with progress_bar as pbar:
                    futures = {
                        executor.submit(
                            AtlasOpenParser.parse_root_file, 
                            file_index,
                            self.possible_tree_names,
                            release_year,
                            40_000,  # batch_size
                            self.temp_directory  # temp_directory
                        ): file_index
                        for file_index in file_ids
                    }
                
                    for future in as_completed(futures):
                        file_index = futures[future]
                        self._process_completed_future(
                            future, file_index, release_year
                        )
                        
                        if self._chunk_exceed_threshold():
                            self.total_chunks += 1
                            self._save_chunk_metadata()
                            chunk_to_yield = self._prepare_chunk_for_yield()
                        
                            yield chunk_to_yield
                            del chunk_to_yield
                            gc.collect() 
                            memory_utils.print_gc_stats() #FOR TESTING
                            memory_utils.print_top_memory_variables(n=10) #FOR TESTING
                        
                        if self.show_progress_bar:
                            status = self._get_parsing_status_for_pbar()
                            pbar.set_postfix_str(status)
                        pbar.update(1)
                
                gc.collect()

                if self.events is None:
                    continue
                        
                self.total_chunks += 1
                self._save_chunk_metadata()
                if save_statistics:
                    logging.info(f"Saving statistics to {self.stats_log}")
                    total_files = [file_id for file_id in file_ids for file_ids in release_years_file_ids.values()]
                    stats = self.get_statistics(len(total_files))
                    
                    with open(self.stats_log, 'a') as f:
                        json.dump(stats, f, indent=2)

                    self.print_statistics_summary(stats)

                self.max_memory_captured = memory_utils.get_process_mem_usage_mb()
                yield self.events
                self.events = None
    
    def _process_completed_future(self, future, file_index, release_year):
        """Process a single completed future. Returns updated success count."""
        file_start_time = time.time()
        # Use configurable timeout, default to 10 seconds
        file_timeout = getattr(self, 'file_processing_timeout', 10)
        
        try:
            cur_file_data = future.result(timeout=file_timeout)
            if cur_file_data is not None:
                self.successful_count += 1
                file_size_mb, events_in_file = self._save_parsed_file_metadata(
                    cur_file_data, release_year, file_index)
                
                if self.show_progress_bar:
                    tqdm.write(
                        f"✅ File processed: {file_size_mb:.2f} MB logical size, " 
                        f"{events_in_file:,} events."
                    )
                cur_file_data = AtlasOpenParser._normalize_fields(cur_file_data, release_year)
                self._concatenate_events(cur_file_data)

        except TimeoutError as e:
            file_processing_time = time.time() - file_start_time
            self._log_crash(file_index, e, file_processing_time)
            if self.show_progress_bar:
                tqdm.write(f"⚠️ Timeout: {file_index} - processing took >{file_timeout}s")
        except (ValueError, KeyError, AttributeError) as e:
            file_processing_time = time.time() - file_start_time
            self._log_crash(file_index, e, file_processing_time)
            if self.show_progress_bar:
                tqdm.write(f"⚠️ Data error: {file_index} - {type(e).__name__}: {str(e)[:100]}")
        except Exception as e:
            file_processing_time = time.time() - file_start_time
            self._log_crash(file_index, e, file_processing_time)
            if self.show_progress_bar:
                tqdm.write(f"⚠️ Unexpected error: {file_index} - {type(e).__name__}") 
    
    def _prepare_chunk_for_yield(self):    
        chunk_to_yield = self.events
        self.events = None
        self.file_parsed_count = 0
        self.total_chunks += 1
        
        mem_before_yield = memory_utils.get_process_mem_usage_mb()
        self.max_memory_captured = max(self.max_memory_captured, mem_before_yield)
        
        return chunk_to_yield
    
    @staticmethod
    def extract_release_year_from_filename(filename: str) -> str:
        """
        Extract the release year from a ROOT filename.
        
        Filenames typically follow the pattern: {release_year}_{hash}.root
        or {release_year}_mc_{hash}.root
        
        Examples:
            - "2024r-pp_mc_b1c18ef224d66cb7.root" -> "2024r-pp"
            - "2024r-pp_f2fbf9c5651948e9.root" -> "2024r-pp"
            - "2025r-evgen-13tev_abc123.root" -> "2025r-evgen-13tev"
        
        Args:
            filename: ROOT filename (with or without path, with or without .root extension)
            
        Returns:
            Release year identifier (e.g., "2024r-pp") or "2024r-pp" as default
        """
        # Get just the filename without path
        base_name = os.path.basename(filename)
        # Remove .root extension if present
        base_name = base_name.replace(".root", "")
        
        # Extract release year (everything before first underscore)
        # Handle both patterns: {release_year}_{hash} and {release_year}_mc_{hash}
        if "_" in base_name:
            # Split by underscore and take the first part
            release_year = base_name.split("_")[0]
            
            # Validate against available releases
            available_releases = schemas.get_available_releases()
            # Normalize the release year (remove _mc suffix if present in the extracted part)
            normalized_year = schemas.normalize_release_year(release_year)
            
            if normalized_year in available_releases:
                return normalized_year
            else:
                logging.warning(f"Extracted release year '{release_year}' (normalized: '{normalized_year}') "
                              f"not found in available releases. Available: {available_releases}. "
                              f"Using default '2024r-pp'")
        else:
            logging.warning(f"Could not extract release year from filename '{filename}': "
                          f"no underscore found. Using default '2024r-pp'")
        
        return "2024r-pp"  # Default fallback
    
    #CHECK check performance of this function, even for big local files its slow
    @staticmethod
    def parse_root_file(file_index, tree_names=["CollectionTree"], release_year="2024r-pp", batch_size=40_000, temp_directory=None) -> ak.Array:
        """
        Parse an ATLAS DAOD file in batches if necessary.
        Accumulates raw arrays and zips into vector objects only once per object.
        
        Automatically normalizes release years with _mc suffix for schema lookups.
        Handles compressed files (.tar.gz, .gz) by extracting to temp directory.
        
        Args:
            file_index: Path or URI to ROOT file (can be .tar.gz or .gz)
            tree_names: List of possible tree names to search for
            release_year: Release year identifier (e.g., "2024r-pp" or "2024r-pp_mc")
            batch_size: Number of entries to process per batch
            temp_directory: Directory for extracting compressed files (None = system temp)
        """
        # Check if file is compressed
        is_tar_gz = '.tar.gz' in file_index
        is_gzipped = '.gz' in file_index and not is_tar_gz
        
        if is_tar_gz:
            # Handle tar.gz files
            return AtlasOpenParser._parse_tar_gz_file(file_index, tree_names, release_year, batch_size, temp_directory)
        elif is_gzipped:
            # Handle .gz files
            return AtlasOpenParser._parse_gzipped_file(file_index, tree_names, release_year, batch_size, temp_directory)
        
        # Regular ROOT file - parse directly
        with uproot.open(file_index) as root_file:
            root_file_keys = root_file.keys()
            tree_name = AtlasOpenParser.get_data_tree_name_for_root_file(root_file_keys, tree_names)
            tree = root_file[tree_name]
            all_tree_branches = set(tree.keys())
            n_entries = tree.num_entries
            is_file_big = n_entries > batch_size if batch_size is not None else False
            
            obj_branches_and_quantities: dict[str, dict[str, str]] = AtlasOpenParser.extract_branches_by_obj_in_schema(
                all_tree_branches, release_year=release_year)
            
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
    
    @staticmethod
    def _parse_tar_gz_file(file_index, tree_names, release_year, batch_size, temp_directory):
        """
        Extract tar.gz file and parse ROOT file inside.
        
        Args:
            file_index: URI or path to .tar.gz file
            tree_names: List of possible tree names
            release_year: Release year identifier
            batch_size: Number of entries per batch
            temp_directory: Directory for extraction (None = system temp)
            
        Returns:
            Parsed awkward array or None if extraction fails
        """
        if temp_directory is None:
            temp_directory = tempfile.gettempdir()
        
        # Create unique temp directory for this extraction
        extract_dir = tempfile.mkdtemp(dir=temp_directory, prefix='atlas_tar_')
        
        try:
            # Download tar.gz if remote
            if file_index.startswith('root://') or file_index.startswith('http'):
                tar_path = os.path.join(extract_dir, 'file.tar.gz')
                logging.info(f"Downloading {file_index} to {tar_path}")
                
                if file_index.startswith('http'):
                    response = requests.get(file_index, stream=True, timeout=300)
                    response.raise_for_status()
                    with open(tar_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    # For root:// protocol, try to download using xrdcp or fallback
                    # Try xrdcp first (common on CERN infrastructure)
                    import subprocess
                    try:
                        result = subprocess.run(
                            ['xrdcp', file_index, tar_path],
                            capture_output=True,
                            timeout=600,
                            check=True
                        )
                    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
                        # Fallback: try using uproot's underlying mechanism
                        # Note: uproot may not handle tar.gz directly, so we skip
                        logging.warning(
                            f"Cannot download root:// tar.gz file {file_index}. "
                            f"xrdcp not available or failed. Error: {e}"
                        )
                        return None
            else:
                # Local file
                tar_path = file_index
            
            # Extract tar.gz
            logging.info(f"Extracting {tar_path}")
            with tarfile.open(tar_path, 'r:gz') as tar:
                # Find ROOT file(s) in archive
                logging.info(f"TAR FILE MEMBERS: {tar.getmembers()}")
                root_files = [m for m in tar.getmembers() if m.name.endswith('.root') and m.isfile()]
                if not root_files:
                    logging.warning(f"No ROOT file found in {file_index}")
                    return None
                
                # Extract first ROOT file
                root_file_member = root_files[0]
                tar.extract(root_file_member, extract_dir)
                extracted_root = os.path.join(extract_dir, root_file_member.name)
                
                logging.info(f"Extracted ROOT file: {extracted_root}")
                
                # Parse extracted ROOT file
                result = AtlasOpenParser.parse_root_file(
                    extracted_root, tree_names, release_year, batch_size, temp_directory
                )
                
                return result
                
        except Exception as e:
            logging.error(f"Error extracting/parsing tar.gz file {file_index}: {e}")
            return None
        finally:
            # Clean up temp directory
            try:
                shutil.rmtree(extract_dir)
                logging.debug(f"Cleaned up temp directory: {extract_dir}")
            except Exception as e:
                logging.warning(f"Failed to clean up temp directory {extract_dir}: {e}")
    
    @staticmethod
    def _parse_gzipped_file(file_index, tree_names, release_year, batch_size, temp_directory):
        """
        Decompress .gz file and parse ROOT file.
        
        Args:
            file_index: URI or path to .gz file
            tree_names: List of possible tree names
            release_year: Release year identifier
            batch_size: Number of entries per batch
            temp_directory: Directory for extraction (None = system temp)
            
        Returns:
            Parsed awkward array or None if decompression fails
        """
        if temp_directory is None:
            temp_directory = tempfile.gettempdir()
        
        # Create unique temp file for decompressed ROOT file
        temp_file = tempfile.NamedTemporaryFile(
            dir=temp_directory, 
            suffix='.root', 
            delete=False,
            prefix='atlas_gz_'
        )
        temp_path = temp_file.name
        temp_file.close()
        
        try:
            # Download and decompress if remote
            if file_index.startswith('root://') or file_index.startswith('http'):
                logging.info(f"Downloading and decompressing {file_index}")
                
                if file_index.startswith('http'):
                    response = requests.get(file_index, stream=True, timeout=300)
                    response.raise_for_status()
                    with gzip.GzipFile(fileobj=response.raw) as gz_file:
                        with open(temp_path, 'wb') as f:
                            for chunk in iter(lambda: gz_file.read(8192), b''):
                                f.write(chunk)
                else:
                    # For root:// protocol, try xrdcp
                    import subprocess
                    try:
                        # Download first to temp location
                        temp_gz = os.path.join(os.path.dirname(temp_path), 'temp_file.gz')
                        result = subprocess.run(
                            ['xrdcp', file_index, temp_gz],
                            capture_output=True,
                            timeout=600,
                            check=True
                        )
                        # Decompress
                        with gzip.open(temp_gz, 'rb') as gz_file:
                            with open(temp_path, 'wb') as f:
                                f.write(gz_file.read())
                        # Clean up temp gz
                        os.unlink(temp_gz)
                    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
                        logging.warning(
                            f"Cannot download root:// .gz file {file_index}. "
                            f"xrdcp not available or failed. Error: {e}"
                        )
                        return None
            else:
                # Local file - decompress directly
                logging.info(f"Decompressing {file_index}")
                with gzip.open(file_index, 'rb') as gz_file:
                    with open(temp_path, 'wb') as f:
                        f.write(gz_file.read())
            
            logging.info(f"Decompressed to: {temp_path}")
            
            # Parse decompressed ROOT file
            result = AtlasOpenParser.parse_root_file(
                temp_path, tree_names, release_year, batch_size, temp_directory
            )
            
            return result
            
        except Exception as e:
            logging.error(f"Error decompressing/parsing .gz file {file_index}: {e}")
            return None
        finally:
            # Clean up temp file
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    logging.debug(f"Cleaned up temp file: {temp_path}")
            except Exception as e:
                logging.warning(f"Failed to clean up temp file {temp_path}: {e}")

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
            if self.show_progress_bar:
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
        return output_path
    
    #FEATURE make static
    def flatten_for_root(self, awk_arr, release_year=None):
        """
        Flatten a top-level awkward Array into a ROOT-friendly dict
        compatible with _parse_root_file()'s expected field structure.
        Specifically, each particle object is stored under
        "<cur_obj_name>AuxDyn.<field>" branches (dotted) or "<obj>_<field>" (flat).
        
        Automatically normalizes release years with _mc suffix for schema lookups.
        
        Args:
            awk_arr: Awkward array to flatten
            release_year: Release year identifier (defaults to self.cur_release_year if available)
                         Can include _mc suffix (e.g., "2024r-pp_mc") - will be normalized
        """
        if release_year is None:
            release_year = getattr(self, 'cur_release_year', '2024r-pp')
        
        root_ready = {}

        for obj_name in awk_arr.fields:
            obj = awk_arr[obj_name]

            cur_obj_branch_name = AtlasOpenParser._prepare_obj_branch_name(
                obj_name, release_year=release_year) 

            if cur_obj_branch_name is None:
                cur_obj_branch_name = obj_name

            try:
                # If obj is a record array, iterate over its fields
                for field in obj.fields:
                    branch = obj[field]
                    # ROOT doesn't like None — fill with 0.0
                    filled_branch = ak.fill_none(branch, 0.0)

                    # IMPORTANT: match the structure used by _parse_root_file
                    # Determine naming pattern for this release
                    try:
                        schema_config = schemas.get_schema_for_release(release_year)
                        naming_pattern = schema_config.get("naming_pattern", "dotted")
                    except KeyError:
                        naming_pattern = "dotted"  # Default to dotted
                    
                    if naming_pattern == "flat":
                        # Flat naming: object_field (e.g., lep_pt)
                        branch_name = f"{cur_obj_branch_name}_{field}"
                    else:
                        # Dotted naming: object.field (e.g., AnalysisJetsAuxDyn.pt)
                        branch_name = f"{cur_obj_branch_name}.{field}"
                    
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
                if self.show_progress_bar:
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
            
            if "ValueError" in str(exception):
                return
            
            if os.path.exists(self.crashed_files):
                with open(self.crashed_files, 'r') as f:
                    data = json.load(f)
            else:
                data = {"failed_files": [], "retry_counts": {}}
            
            # Initialize retry_counts if not present (for backward compatibility)
            if "retry_counts" not in data:
                data["retry_counts"] = {}
            
            # Initialize failed_files if not present
            if "failed_files" not in data:
                data["failed_files"] = []
            
            # Check retry count BEFORE incrementing to prevent exceeding limit
            max_retries = self.max_file_retries
            if file_index not in data["retry_counts"]:
                data["retry_counts"][file_index] = 0
            
            current_retry_count = data["retry_counts"][file_index]
            
            # If already exceeded, don't increment or add to failed_files
            if current_retry_count >= max_retries:
                # File has already exceeded max retries - remove from failed_files and skip
                if file_index in data["failed_files"]:
                    data["failed_files"].remove(file_index)
                logging.warning(f"File {file_index} has already exceeded max retries ({current_retry_count} >= {max_retries}), skipping")
                # Still write the updated data to remove it from failed_files
                with open(self.crashed_files, 'w+') as f:
                    json.dump(data, f, indent=2)
                return  # Don't add to failed_files or increment count
            
            # Increment retry count (we know it's still within limit)
            data["retry_counts"][file_index] += 1
            
            # Add to failed_files if not already there (for retry)
            if file_index not in data["failed_files"]:
                data["failed_files"].append(file_index)

            with open(self.crashed_files, 'w+') as f:
                json.dump(data, f, indent=2)

    def _save_parsed_file_metadata(self, cur_file_data, release_year, file_index):                    
        file_size_mb = cur_file_data.layout.nbytes / (1024 * 1024)
        self.max_file_size_mb = max(self.max_file_size_mb, file_size_mb)
        self.min_file_size_mb = min(self.min_file_size_mb, file_size_mb)

        events_in_file = len(cur_file_data)
        self.total_events_processed += events_in_file
        
        self.cur_file_ids.append(file_index)

        return file_size_mb, events_in_file
    
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
            return "CollectionTree"
        
        available_trees = [key[:-2] for key in root_file_keys] # Remove trailing ';1'
        for tree_name in possible_tree_names:
            if tree_name in available_trees:
                return tree_name
        
        return "CollectionTree" 
    
    @staticmethod
    def extract_branches_by_obj_in_schema(tree_branches, release_year="2024r-pp"):
        """
        Extract branches by object based on release-specific schema.
        Returns {obj_name: {full_branch: quantity, ...}}
        
        Automatically normalizes release years with _mc suffix for schema lookups.
        
        Args:
            tree_branches: Set of all branch names in the ROOT tree
            release_year: Release year identifier (e.g., "2024r-pp" or "2024r-pp_mc")
        """
        try:
            schema_config = schemas.get_schema_for_release(release_year)
        except KeyError:
            # Fallback: try to auto-detect branch names
            logging.warning(f"Release year '{release_year}' not found in schemas. Attempting auto-detection.")
            return AtlasOpenParser._auto_detect_branches(tree_branches)
        
        obj_branches = {}
        objects = schema_config["objects"]
        naming_pattern = schema_config.get("naming_pattern", "dotted")

        for obj_name, fields in objects.items():
            obj_available_physical_quantities = []
            
            if naming_pattern == "flat":
                # Flat naming: object_field (e.g., lep_pt, jet_pt)
                cur_obj_branch_base = AtlasOpenParser._prepare_obj_branch_name(obj_name, release_year=release_year)
                obj_available_physical_quantities = [
                    f for f in fields if f"{cur_obj_branch_base}_{f}" in tree_branches
                ]
                
                if AtlasOpenParser._can_calculate_inv_mass(obj_available_physical_quantities):
                    obj_branches[obj_name] = {
                        f"{cur_obj_branch_base}_{quantity}": quantity
                        for quantity in obj_available_physical_quantities
                    }
            else:
                # Dotted naming: object.field (e.g., AnalysisElectronsAuxDyn.pt)
                cur_obj_branch_name = AtlasOpenParser._prepare_obj_branch_name(obj_name, release_year=release_year)
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
    def _auto_detect_branches(tree_branches):
        """
        Auto-detect branch names when release schema is not available.
        Attempts to find branches matching common patterns.
        
        Args:
            tree_branches: Set of all branch names in the ROOT tree
            
        Returns:
            Dictionary mapping object names to branch mappings
        """
        obj_branches = {}
        tree_branches_str = " ".join(tree_branches)
        
        # Common object names and their possible branch patterns
        object_patterns = {
            "Electrons": ["Electron", "electron", "el"],
            "Muons": ["Muon", "muon", "mu"],
            "Jets": ["Jet", "jet"],
            "Photons": ["Photon", "photon", "gamma"]
        }
        
        required_fields = ["pt", "eta", "phi"]
        
        for obj_name, patterns in object_patterns.items():
            # Try to find branches matching each pattern
            for pattern in patterns:
                # Look for branches containing the pattern
                matching_branches = [b for b in tree_branches if pattern.lower() in b.lower()]
                
                if matching_branches:
                    # Try to find the base branch name (e.g., "AnalysisElectronsAuxDyn")
                    # by looking for branches with common suffixes
                    base_branch = None
                    for branch in matching_branches:
                        # Remove field suffix to get base branch
                        parts = branch.split(".")
                        if len(parts) == 2:
                            potential_base = parts[0]
                            # Check if this base has the required fields
                            has_required = all(
                                f"{potential_base}.{field}" in tree_branches 
                                for field in required_fields
                            )
                            if has_required:
                                base_branch = potential_base
                                break
                    
                    if base_branch:
                        # Extract available fields
                        available_fields = []
                        for field in required_fields + ["mass"]:
                            if f"{base_branch}.{field}" in tree_branches:
                                available_fields.append(field)
                        
                        if AtlasOpenParser._can_calculate_inv_mass(available_fields):
                            obj_branches[obj_name] = {
                                f"{base_branch}.{field}": field
                                for field in available_fields
                            }
                        break
        
        return obj_branches
    
    @staticmethod
    def _prepare_obj_branch_name(obj_name, release_year="2024r-pp", field=None):
        """
        Prepare object branch name using release-specific template.
        
        Automatically normalizes release years with _mc suffix (e.g., "2024r-pp_mc" -> "2024r-pp").
        The _mc suffix is only for file organization; schema lookups use the base release year.
        
        Args:
            obj_name: Canonical object name (e.g., "Electrons", "Muons")
            release_year: Release year identifier (e.g., "2024r-pp" or "2024r-pp_mc")
            field: Optional field name (required for flat naming pattern)
            
        Returns:
            Branch name:
            - For dotted naming: "AnalysisElectronsAuxDyn" (base name without field)
            - For flat naming: "lep_pt" (full name with field if provided, else just "lep")
        """
        try:
            schema = schemas.get_schema_for_release(release_year)
            naming_pattern = schema.get("naming_pattern", "dotted")
            object_mappings = schema.get("object_mappings", {})
            
            # Get the branch object name (may differ from canonical name)
            branch_obj_name = object_mappings.get(obj_name, obj_name)
            
            if naming_pattern == "flat":
                if field:
                    return f"{branch_obj_name}_{field}"
                else:
                    return branch_obj_name  # Return base name for flat pattern
            else:
                # Dotted naming
                prefix = schema["branch_prefix"]
                suffix = schema["branch_suffix"]
                return f"{prefix}{branch_obj_name}{suffix}"
        except KeyError:
            # Fallback: use default pattern if release not found
            logging.warning(f"Release year '{release_year}' not found. Using default branch naming.")
            return "Analysis" + obj_name + "AuxDyn"

    @staticmethod
    def _normalize_fields(ak_array, release_year="2024r-pp"):
        """
        Normalize fields in awkward array to match schema.
        Ensures all expected objects are present (even if empty).
        
        Automatically normalizes release years with _mc suffix for schema lookups.
        
        Args:
            ak_array: Awkward array to normalize
            release_year: Release year identifier (e.g., "2024r-pp" or "2024r-pp_mc")
        """
        try:
            schema_config = schemas.get_schema_for_release(release_year)
            expected_objects = schema_config["objects"].keys()
        except KeyError:
            # Fallback to legacy schema
            logging.warning(f"Release year '{release_year}' not found. Using legacy schema.")
            expected_objects = schemas.INVARIANT_MASS_SCHEMA.keys()
        
        for field in expected_objects:
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
    
 