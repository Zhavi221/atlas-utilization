import awkward as ak
import numpy as np
import vector
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import uproot
import requests
import atlasopenmagic as atom

import csv
import random
from tqdm import tqdm
import os
import traceback
import logging

import threading
import datetime
import time
import json

from . import schemas, consts

class ATLAS_Parser():
    '''
    ATLAS_Parser class is responsible for parsing ATLAS Open Data files and handling event selection, filtering, and output.
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
        filter_events_by_kinematics(events, kinematic_cuts):
            Filters events based on kinematic cuts.
        filter_events_by_counts(events, particle_counts):
            Filters events based on particle counts.
        _prepare_obj_name(obj_name):
            Prepares object name and zip function for parsing.
        _normalize_fields(ak_array):
            Normalizes fields in the awkward array to match schema.
    '''
    def __init__(self, max_chunk_size_bytes, max_threads, max_processes):
        self.files_ids = None
        self.file_parsed_count = 0
        self.cur_chunk = 0
        self.cur_chunk_hash = None
        
        self.events = None
        self.total_size_kb = 0

        self.max_chunk_size_bytes = max_chunk_size_bytes
        
        # ===== Enhanced: Comprehensive crash and statistics tracking =====
        self.crash_log = "atlas_crashes.log"
        self.stats_log = "atlas_stats.json"
        self.crash_lock = threading.Lock()
        self.failed_files = []
        
        # New statistics tracking
        self.parsing_start_time = None
        self.chunk_stats = []
        self.error_types = {}
        self.timeout_count = 0
        self.max_file_size_mb = 0
        self.min_file_size_mb = float('inf')
        self.total_events_processed = 0
        # ================================================================

        #PARALLELISM CONFIG
        self.max_threads = max_threads
        self.max_processes = max_processes

    #SAVING FILES
    def save_events_as_root(self, events, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        #TODO: for each chunk, create a has representing it's underlying files
        output_path = os.path.join(output_dir, f"chunk_{self.cur_chunk}.root")
        with uproot.recreate(output_path) as file:
            file["CollectionTree"] = events

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
            cur_obj_name = ATLAS_Parser._prepare_obj_name(obj_name)

            if cur_obj_name is None:
                cur_obj_name = obj_name

            try:
                # If obj is a record array, iterate over its fields
                for field in obj.fields:
                    branch = obj[field]
                    # ROOT doesn't like None — fill with 0.0
                    filled_branch = ak.fill_none(branch, 0.0)

                    # IMPORTANT: match the structure used by _parse_file
                    # e.g., "AnalysisJetsAuxDyn.pt"
                    branch_name = f"{cur_obj_name}AuxDyn.{field}"
                    root_ready[branch_name] = filled_branch

            except AttributeError:
                # Not a record array — save as a top-level branch
                logging.info(f"Warning: {obj_name} is not a record array, saving as-is.")
                root_ready[obj_name] = ak.fill_none(obj, 0.0)

            except Exception as e:
                logging.info(f"Error processing {obj_name}: {e}")
                continue

        return root_ready
    
    #PARSING METHODS
    def fetch_records_ids(self, release_year):
        '''
            Fetches the real records IDs for a given release year.
            Returns a list of file URIs.
        '''
        atom.set_release(consts.RELEASES_YEARS.get(release_year))

        datasets_ids = atom.available_data()
        release_files_uris = []

        for dataset_id in datasets_ids:
            release_files_uris.extend(atom.get_urls_data(dataset_id))

        self.files_ids = release_files_uris
        return release_files_uris

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

    def _save_statistics(self, total_files, successful_count):
        """Save comprehensive parsing statistics to JSON"""
        end_time = time.time()
        total_time = end_time - self.parsing_start_time if self.parsing_start_time else 0
        
        stats = {
            "parsing_session": {
                "timestamp": datetime.datetime.now().isoformat(),
                "total_time_seconds": total_time,
                "total_files": total_files,
                "successful_files": successful_count,
                "failed_files": len(self.failed_files),
                "success_rate": (successful_count / total_files * 100) if total_files > 0 else 0
            },
            "performance": {
                "max_file_size_mb": self.max_file_size_mb if self.max_file_size_mb > 0 else 0,
                "min_file_size_mb": self.min_file_size_mb if self.min_file_size_mb != float('inf') else 0,
                "total_data_processed_mb": self.total_size_kb / (1024 * 1024),
                "total_events_processed": self.total_events_processed,
                "chunks_created": self.cur_chunk,
                "avg_chunk_size_mb": (self.total_size_kb / (1024 * 1024)) / max(self.cur_chunk, 1)
            },
            "errors": {
                "timeout_count": self.timeout_count,
                "error_types": self.error_types,
                "failed_file_list": self.failed_files[:10]  # Only store first 10 for brevity
            },
            "chunk_details": self.chunk_stats
        }
        
        with open(self.stats_log, 'w') as f:
            json.dump(stats, f, indent=2)
        
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
        
        if stats['errors']['error_types']:
            tqdm.write(f"⚠️  Error Breakdown:")
            for error_type, count in stats['errors']['error_types'].items():
                tqdm.write(f"   • {error_type}: {count}")
        
        tqdm.write(f"📋 Full stats saved to: {self.stats_log}")
        if stats['parsing_session']['failed_files'] > 0:
            tqdm.write(f"💥 Crash log saved to: {self.crash_log}")
        tqdm.write(f"="*60)
   
    def _initialize_statistics(self):
        self.parsing_start_time = time.time()
        if os.path.exists(self.crash_log):
            os.remove(self.crash_log) 
        if os.path.exists(self.stats_log):
            os.remove(self.stats_log) 

    def parse_files(self,
                       files_ids: list = None,
                       limit: int = 0):
        '''
            Parses the input files by their IDs, otherwise uses the member files_ids.
            Yields chunks of events as awkward arrays each size from the input limit.
        '''
        if files_ids is None:
            if self.files_ids is None:
                raise ValueError("No files_ids provided and self.files_ids is None.")
            files_ids = self.files_ids

        if limit:
            files_ids = files_ids[:limit]

        successful_count = 0
        self._initialize_statistics()
        
        tqdm.write(
            f"Starting to parse {len(files_ids)} files with {self.max_threads} threads.")
        
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = {
                executor.submit(ATLAS_Parser.parse_file, file_index): file_index
                for file_index in files_ids
            }
        
            with tqdm(total=len(files_ids), desc="Parsing files", unit="file", dynamic_ncols=True) as pbar:

                for future in as_completed(futures):
                    file_index = futures[future]
                    file_start_time = time.time()

                    try:
                        cur_file_data = future.result(timeout=10)
                        
                        if cur_file_data is not None:
                            successful_count += 1
                            self._log_file_metadata(cur_file_data)
                            cur_file_data = ATLAS_Parser._normalize_fields(cur_file_data)
                            self._concatenate_events(cur_file_data)
                            
                            if self._chunk_size_enough():
                                self._log_chunk()
                                yield self.events
                                self.events = None

                    except TimeoutError:
                        file_processing_time = time.time() - file_start_time
                        self._log_crash(file_index, TimeoutError(f"Parsing took longer than 10s"), file_processing_time)
                        tqdm.write(f"⏱️ Timeout: {file_index} ({file_processing_time:.1f}s)")
                        
                    except Exception as e:
                        file_processing_time = time.time() - file_start_time
                        self._log_crash(file_index, e, file_processing_time)
                        tqdm.write(f"⚠️ Error: {file_index} - {type(e).__name__}")

                    status = self._get_parsing_status(successful_count)
                    pbar.set_postfix_str(status)
                    pbar.update(1)

        stats = self._save_statistics(len(files_ids), successful_count)
        self.print_statistics_summary(stats) 
        
        if self.events is not None:
            self._log_chunk()
            yield self.events
            self.events = None

    def _get_parsing_status(self, successful_count):
        success_rate = (
                        (successful_count / (successful_count + len(self.failed_files)) * 100)
                        if (successful_count + len(self.failed_files)) > 0 else 0
                    )
        status = (
                f"✅ {successful_count} | "
                f"❌ {len(self.failed_files)} | "
                f"✨ {success_rate:.1f}% | "
                f"💾 {self.total_size_kb / (1024 * 1024):.1f} MB | "
                f"🎯 {self.total_events_processed:,} events"
            )
        
        return status

    def _log_file_metadata(self, cur_file_data):
                            
        file_size_mb = cur_file_data.layout.nbytes / (1024 * 1024)
        self.max_file_size_mb = max(self.max_file_size_mb, file_size_mb)
        self.min_file_size_mb = min(self.min_file_size_mb, file_size_mb)
                            
        events_in_file = len(cur_file_data)
        self.total_events_processed += events_in_file
                            
        tqdm.write(f"✅ File processed: {file_size_mb:.2f} MB, {events_in_file:,} events")
    
    def _log_chunk(self):
        chunk_info = {
            "chunk_id": self.cur_chunk,
            "events": len(self.events),
            "size_mb": self.events.layout.nbytes / (1024 * 1024),
            "files_included": self.file_parsed_count
        }
        self.chunk_stats.append(chunk_info)
        
        self.cur_chunk += 1
        
        tqdm.write(
                f"Yielding chunk #{self.cur_chunk} with {len(self.events):,} events "
                f"from {self.file_parsed_count} files "
                f"({self.events.layout.nbytes / (1024 * 1024):.2f} MB). "
                f"Total: {self.total_size_kb  / (1024 * 1024):.2f} MB accumulated.")

    @staticmethod
    #TODO new parse file method
    def parse_file(file_index, tree_name="CollectionTree", batch_size=40_000) -> ak.Array:
        """
        Parse an ATLAS DAOD file in batches if necessary.
        Accumulates raw arrays and zips into vector objects only once per object.
        """
        with uproot.open({file_index: tree_name}) as tree:
            all_keys = set(tree.keys())
            n_entries = tree.num_entries
            is_big = n_entries > batch_size  # flag large files

            # 1. Precompute all fields to read
            field_map = ATLAS_Parser._extract_branches_by_schema(all_keys, schema=schemas.INVARIANT_MASS_SCHEMA)

            if not field_map.values():
                return None
            
            all_fields = sorted({x for v in field_map.values() for x in v})

            # 2. Initialize storage for raw batches
            all_events = {obj_name: [] for obj_name in field_map}

            # 3. Define entry ranges
            entry_ranges = []
            parsing_label = ""
            if is_big:
                parsing_label = "Parsing file as batches"
                entry_ranges = [
                    (start, min(start + batch_size, n_entries)) 
                    for start in range(0, n_entries, batch_size)
                ]
            else:
                parsing_label = "Parsing file"
                entry_ranges = [(0, n_entries)]

            # 4. Read batches
            for entry_start, entry_stop in tqdm(entry_ranges, desc=parsing_label, unit="batch"):
                batch_data = tree.arrays(all_fields, entry_start=entry_start, entry_stop=entry_stop)

                # Append raw arrays only
                for obj_name, fields in field_map.items():
                    subset = batch_data[fields]
                    all_events[obj_name].append(subset)

            # 5. Concatenate batches and zip once per object
            for obj_name, chunks in all_events.items():
                concatenated = ak.concatenate(chunks)
                field_names = [f.split('.')[-1] for f in field_map[obj_name]]
                all_events[obj_name] = vector.zip({name: concatenated[full] 
                                                for name, full in zip(field_names, field_map[obj_name])})

            #TODO what is this
            return ak.zip({k: v for k, v in all_events.items() if v is not None}, depth_limit=1)

    @staticmethod
    #TODO go over
    def _extract_branches_by_schema(all_keys, schema: dict):
        field_map = {}
        all_fields = []
        for obj_name, fields in schema.items():
            cur_obj_name = ATLAS_Parser._prepare_obj_name(obj_name)
            physical_quantities = [
                    f for f in fields if f"{cur_obj_name}AuxDyn.{f}" in all_keys
                ]

            if ATLAS_Parser.can_calculate_inv_mass(physical_quantities):
                full_branches = [f"{cur_obj_name}AuxDyn.{f}" for f in physical_quantities]
                field_map[obj_name] = full_branches

        return field_map

    @staticmethod
    def parse_filea(file_index, tree_name="CollectionTree") -> ak.Array:
        '''
            Parses a single file by a generic schema.
        '''
        with uproot.open({file_index: tree_name}, 
                vector_read=False) as tree:
            events = {}
            objs_filtered = []

            tree_keys = tree.keys()
            # Wrap the schema iteration with tqdm
            for obj_name, fields in tqdm(
                schemas.INVARIANT_MASS_SCHEMA.items(),
                desc=f"Parsing file",
                unit="obj"
            ):
                cur_obj_name = ATLAS_Parser._prepare_obj_name(obj_name)

                filtered_fields = list(filter(
                    lambda field: f"{cur_obj_name}AuxDyn.{field}" in tree_keys,
                    fields))

                #TODO make this check robust and less fatal
                if not ATLAS_Parser.can_calculate_inv_mass(filtered_fields):
                    continue
                # if len(filtered_fields) < len(fields):
                #     continue

                objs_filtered.append(cur_obj_name)
                tree_as_rows = tree.arrays(
                    filtered_fields,
                    aliases={field: f"{cur_obj_name}AuxDyn.{field}" for field in filtered_fields}
                )

                sep_to_arrays = ak.unzip(tree_as_rows)
                field_names = tree_as_rows.fields

                tree_as_rows = vector.zip(
                    dict(zip(field_names, sep_to_arrays))
                )

                events[obj_name] = tree_as_rows

            if events:
                return ak.zip(events, depth_limit=1)
            else:
                return None
    
    def _concatenate_events(self, cur_file_data):
        if self.events is None:
            self.events = cur_file_data
        else:
            self.events = ak.concatenate(
                [self.events, cur_file_data], axis=0)
        
        self.file_parsed_count += 1
        # chunk_size_mb = sys.getsizeof(cur_file_data) / (1024 * 1024)
        # chunk_size_mb = ak.nbytes(cur_file_data)
        chunk_size_kb = cur_file_data.layout.nbytes
        self.total_size_kb += chunk_size_kb
    
    def _chunk_size_enough(self):
        return self.events.layout.nbytes >= self.max_chunk_size_bytes

    #TESTING METHODS
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

    #STATIC METHODS
    #TODO: check this
    @staticmethod
    def can_calculate_inv_mass(filtered_fields):
        required_fields = {'pt', 'eta', 'phi', 'm'} #NAME ACCORDING TO QUANTITIES PRESENT
        required_fields2 = {'rho', 'eta', 'phi', 'tau'} #NAME ACCORDING TO QUANTITIES PRESENT
        
        temp = set(filtered_fields)

        if 'm' not in temp:
            temp.update('m')  #what is this

        return required_fields == temp or required_fields2 == temp
    
    @staticmethod
    def filter_events_by_kinematics(events, kinematic_cuts):
        filtered_events = {}

        for obj in events.fields:
            particles = events[obj]

            # Skip empty arrays
            if len(particles) == 0:
                continue

            # Start with all True mask
            mask = ak.ones_like(particles, dtype=bool)

            # rho cut
            if "rho" in kinematic_cuts and hasattr(particles, "rho"):
                rho_vals = ak.values_astype(particles.rho, float)
                mask = ak.mask(mask, rho_vals >= kinematic_cuts["rho"]["min"])

            # eta cut
            if "eta" in kinematic_cuts and hasattr(particles, "eta"):
                eta_vals = ak.values_astype(particles.eta, float)
                mask = ak.mask(mask, (eta_vals >= kinematic_cuts["eta"]["min"]) &
                                        (eta_vals <= kinematic_cuts["eta"]["max"]))

            # phi cut
            if "phi" in kinematic_cuts and hasattr(particles, "phi"):
                phi_vals = ak.values_astype(particles.phi, float)
                mask = ak.mask(mask, (phi_vals >= kinematic_cuts["phi"]["min"]) &
                                        (phi_vals <= kinematic_cuts["phi"]["max"]))

            # tau cut
            if "tau" in kinematic_cuts and hasattr(particles, "tau"):
                tau_vals = ak.values_astype(particles.tau, float)
                mask = ak.mask(mask, tau_vals >= kinematic_cuts["tau"]["min"])

            # Apply mask to particles
            filtered_events[obj] = ak.mask(particles, mask)

        return ak.Array(filtered_events)
    
    @staticmethod
    def _prepare_obj_name(obj_name):
        if obj_name in schemas.PARTICLE_LIST:
            return "Analysis" + obj_name
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
