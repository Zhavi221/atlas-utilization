import csv
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import uproot
import numpy as np
import awkward as ak
import vector
import requests
import atlasopenmagic as atom
import os

from . import combinatorics, schemas, consts

class ATLAS_Parser():
    '''
        ATLAS_Parser class is responsible for parsing ATLAS Open Data files.
        Available methods:
        - fetch_records_ids: Fetches the real records IDs for a given release year.
        - fetch_mc_files_ids: Fetches the MC records IDs for a given release year.
        - parse_files: Parses the files with the given IDs and yields chunks of events.
        - _parse_file: Parses a single file by a generic schema.
        - calculate_mass_for_combination: Computes invariant mass per event from the combined objects.
        - filter_events_by_combination: Filters the events by the given combination dictionary.
        - save_events: Saves the events to a file.
        - load_events_from_file: Loads the events from a file.
    '''
    def __init__(self, max_chunk_size_bytes):
        self.categories = combinatorics.make_objects_categories(
            schemas.PARTICLE_LIST, min_n=2, max_n=4)
        self.files_ids = None
        self.file_parsed_count = 0
        self.cur_chunk = 0
        
        self.events = None
        self.total_size_kb = 0

        self.max_chunk_size_bytes = max_chunk_size_bytes

    def fetch_records_ids(self, release_year):
        '''
            Fetches the real records IDs for a given release year.
            Returns a list of file URIs.
        '''
        atom.set_release(consts.RELEASES_YEARS.get(release_year))

        datasets_ids = atom.available_data()
        release_files_uris = []

        for file_id in datasets_ids:
            release_files_uris.extend(atom.get_urls_data(file_id))
        
        self.files_ids = release_files_uris
        return release_files_uris

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

    def parse_files(self,
                       files_ids: list = None,
                       limit: int = 0,
                       max_workers: int = 3):
        '''
            Parses the input files by their IDs, otherwise uses the member files_ids.
            Yields chunks of events as awkward arrays each 5GB.
        '''
        if files_ids is None:
            if self.files_ids is None:
                raise ValueError("No files_ids provided and self.files_ids is None.")
            files_ids = self.files_ids

        if limit:
            files_ids = files_ids[:limit]

        tqdm.write(
            f"Starting to parse {len(files_ids)} files with {max_workers} threads.")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._parse_file, file_index): file_index
                for file_index in files_ids
            }
            with tqdm(total=len(files_ids), desc="Parsing files", unit="file", dynamic_ncols=True) as pbar:
                for future in as_completed(futures):
                    file_index = futures[future]
                    cur_file_data = future.result()
                    cur_file_data = ATLAS_Parser.normalize_fields(
                                cur_file_data)
                    
                    self._concatenate_events(cur_file_data)
                    
                    tqdm.write(
                        f"Current file size:{cur_file_data.layout.nbytes / (1024 * 1024):.2f} MB.")
                    if self._should_yield_chunk():
                        self.log_cur_size()
                        self.cur_chunk += 1
                        yield self.events
                        self.events = None
                        
                    pbar.set_postfix_str(f"{self.total_size_kb  / (1024 * 1024):.2f} MB | {file_index}")
                    pbar.update(1)

                    tqdm.write(
                        f"Parsed file with objects: {list(cur_file_data.fields)}")
            
        if self.events is not None:
            self.log_cur_size()
            self.cur_chunk += 1
            yield self.events
            self.events = None

    def _parse_file(self, file_index):
        '''
            Parses a single file by a generic schema.
        '''
        with uproot.open({file_index: "CollectionTree"}, vector_read=False) as tree:
            events = {}
            objs_filtered = []

            # Wrap the schema iteration with tqdm
            for obj_name, fields in tqdm(
                schemas.INVARIANT_MASS_SCHEMA.items(),
                desc=f"Parsing file",
                unit="obj"
            ):
                cur_obj_name, zip_function = ATLAS_Parser._prepare_obj_name(
                    obj_name)

                filtered_fields = list(filter(
                    lambda field: f"{cur_obj_name}AuxDyn.{field}" in tree.keys(),
                    fields))

                if len(filtered_fields) < len(fields):
                    continue

                objs_filtered.append(cur_obj_name)
                tree_as_rows = tree.arrays(
                    filtered_fields,
                    aliases={field: f"{cur_obj_name}AuxDyn.{field}" for field in filtered_fields}
                )

                sep_to_arrays = ak.unzip(tree_as_rows)
                field_names = tree_as_rows.fields

                tree_as_rows = zip_function(
                    dict(zip(field_names, sep_to_arrays))
                )

                events[obj_name] = tree_as_rows

            return ak.zip(events, depth_limit=1)
    
    def _load_and_parse_in_parallel(self, files_ids, max_workers):            
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._parse_file, file_index): file_index
                for file_index in files_ids
            }

            return futures
    
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
    
    def _should_yield_chunk(self):
        return self.events.layout.nbytes >= self.max_chunk_size_bytes

    def log_cur_size(self):
        tqdm.write(
                f"Yielding final chunk with {len(self.events)} events "
                f"after a total of {self.file_parsed_count} files "
                f"at {self.events.layout.nbytes / (1024 * 1024):.2f} MB. "
                f"{self.total_size_kb  / (1024 * 1024):.2f} MB - accumulated size.")

    #NOT YET IMPLEMENTED
    def generate_histograms(self):
        categories = combinatorics.make_objects_categories(
            schemas.PARTICLE_LIST, min_n=2, max_n=4)
        for category in categories:
            combination = combinatorics.make_objects_combinations_for_category(
                category, min_k=2, max_k=4)
            # events = self.filter_events_by_combination(combination)

    #NOT YET IMPLEMENTED
    def calculate_mass_for_combination(self, events: ak.Array) -> ak.Array:
        """
        Compute invariant mass per event from the combined objects.
        Compatible with jagged structure (variable number of particles per event).
        """
        if len(events) == 0:
            return ak.Array([])

        vectors_per_event = []

        for obj in events.fields:
            arr = events[obj]
            if len(arr) == 0:
                continue
 
            # Assign mass
            if "m" in arr.fields:
                mass = arr.m
            elif obj in consts.KNOWN_MASSES:
                mass = ak.ones_like(arr.pt) * consts.KNOWN_MASSES[obj]
            else:
                raise ValueError(f"Cannot compute mass for '{obj}': missing 'm' and no known constant.")

            vec = ak.zip({
                "pt": arr.pt,
                "eta": arr.eta,
                "phi": arr.phi,
                "mass": mass
            }, with_name="Momentum4D")

            vectors_per_event.append(vec)

        if not vectors_per_event:
            return ak.Array([])

        # Concatenate all objects into one jagged array per event
        all_vectors = ak.concatenate(vectors_per_event, axis=1)  # concat along the per-event axis

        # Sum per event
        total_vec = ak.sum(all_vectors, axis=1)

        return total_vec.mass

    def filter_events_by_kinematics(self, events, kinematic_cuts):
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

    def filter_events_by_counts(self, events, particle_counts):
        '''
            Filters the events by the given combination dictionary.
        '''
        for obj, range_dict in particle_counts.items():
            obj_array = events[obj]
            if ak.all(ak.is_none(obj_array)):
                continue

            obj_count = ak.num(obj_array)
            # Use bitwise & instead of logical and for array operations
            mask = (obj_count >= range_dict['min']) & (obj_count <= range_dict['max'])
            events = events[mask]

        return ak.to_packed(events)

    
    def save_events(self, events, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"filtered_{self.cur_chunk}.root")
        with uproot.recreate(output_path) as f:
            f["tree"] = events

    def flatten_for_root(self, awk_arr):
        """Flatten a top-level awkward Array into ROOT-friendly dict."""
        root_ready = {}

        for obj_name in awk_arr.fields:
            obj = awk_arr[obj_name]

            try:
                # If this succeeds, obj is a record array (possibly jagged)
                for field in obj.fields:
                    root_ready[f"{obj_name}_{field}"] = obj[field]
            except AttributeError:
                # Not a record, already primitive or jagged primitive
                root_ready[obj_name] = obj

        return root_ready

    def _load_events_from_file(self, file_path):
        with uproot.open(file_path) as f:
            return f["tree"]
        
    @staticmethod
    def _prepare_obj_name(obj_name):
        final_name = None
        if obj_name in schemas.PARTICLE_LIST:
            final_name = "Analysis" + obj_name
            zip_function = vector.zip
        else:
            zip_function = ak.zip

        return final_name, zip_function

    @staticmethod
    def normalize_fields(ak_array):
        for field in schemas.INVARIANT_MASS_SCHEMA.keys():
            if field not in ak_array.fields:
                # Set to empty list per event (not None)
                ak_array = ak.with_field(
                    ak_array,
                    ak.Array([[]] * len(ak_array)),
                    field
                )
        return ak_array
