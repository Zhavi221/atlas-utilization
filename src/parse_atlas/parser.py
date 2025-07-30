import csv
import sys
import logging
import pickle
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import uproot
import numpy as np
import awkward as ak
import vector
import requests
import atlasopenmagic as atom

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
    def __init__(self):
        self.categories = combinatorics.make_objects_categories(
            schemas.PARTICLE_LIST, min_n=2, max_n=4)

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
        
        tqdm.write("Total amount of files found: %d",
                     len(release_files_uris))

        self.files_ids = release_files_uris
        return release_files_uris

    def fetch_mc_files_ids(self, release_year, is_random=False, all=False):
        '''
            Fetches the Monte Carlo records IDs for a given release year.
            Returns a list of file URIs.
        '''
        release = consts.RELEASES_YEARS[release_year]
        metadata_url = consts.LIBRARY_RELEASES_METADATA[release]

        # TODO: IMPLEMENT ALL VARIABLE
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

    def parse_files(self, files_ids: list = None, limit: int = 0, max_workers: int = 3):
        '''
            Parses the files with the given IDs and yields chunks of events.
            If limit is specified, only that many files will be parsed.
        '''

        if files_ids is None:
            if self.files_ids is None:
                raise ValueError("No files_ids provided and self.files_ids is None.")
            files_ids = self.files_ids

        if limit:
            files_ids = files_ids[:limit]

        file_parsed_count = 0
        total_size_mb = 0
        events = None

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

                    if events is None:
                        events = cur_file_data
                    else:
                        events = ak.concatenate(
                            [events, cur_file_data], axis=0)

                    file_parsed_count += 1
                    # chunk_size_mb = sys.getsizeof(cur_file_data) / (1024 * 1024)
                    chunk_size_mb = sys.getsizeof(cur_file_data)
                    total_size_mb += chunk_size_mb

                    pbar.set_postfix_str(
                        f"{total_size_mb:.2f} KB | {file_index}")
                    pbar.update(1)

                    tqdm.write(
                        f"Parsed file with objects: {list(cur_file_data.fields)}")

                    # TODO: WHY EVENTS DONT GROW IN SIZE?
                    if sys.getsizeof(events) >= 50:
                    # if any(ak.num(events[field]) > 0 for field in events.fields):
                    # if True:
                        tqdm.write(
                            f"Yielding chunk with {len(events)} events after {file_parsed_count} files.")
                        yield events
                        events = None

        if len(events):
            tqdm.write(
                f"Yielding final chunk with total {file_parsed_count} files.")
            yield events

    def _parse_file(self, file_index):
        '''
            Parses a single file by a generic schema.
        '''
        with uproot.open({file_index: "CollectionTree"}, vector_read=False) as tree:
            events = {}
            objs_filtered = []

            for obj_name, fields in schemas.INVARIANT_MASS_SCHEMA.items():
                cur_obj_name, zip_function = ATLAS_Parser._prepare_obj_name(
                    obj_name)
                
                filtered_fields = list(filter(
                    lambda field: f"{cur_obj_name}AuxDyn.{field}" in tree.keys(),
                    fields))
                # print(cur_obj_name)
                # print(tree.keys(filter_name= f"{cur_obj_name}AuxDyn.*"))
                # print(filtered_fields, fields)

                if len(filtered_fields) < len(fields):
                    continue

                objs_filtered.append(cur_obj_name)
                tree_as_rows = tree.arrays(
                    filtered_fields,
                    aliases={
                        field: f"{cur_obj_name}AuxDyn.{field}" for field in filtered_fields}
                )
                sep_to_arrays = ak.unzip(tree_as_rows)
                field_names = tree_as_rows.fields

                tree_as_rows = zip_function(
                    dict(zip(field_names, sep_to_arrays)))

                events[obj_name] = tree_as_rows

            return ak.zip(events, depth_limit=1)

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

    def filter_events_by_combination(self, events_file, combination_dict):
        '''
            Filters the events by the given combination dictionary.
        '''
        # TODO: CODE REVIEW

        for obj, count in combination_dict.items():
            obj_array = events_file[obj]
            if ak.all(ak.is_none(obj_array)):
                continue

            obj_count = ak.num(obj_array)
            events_file = events_file[
                obj_count == count]

        return ak.to_packed(events_file)

    def save_events(self, dumped_object, file_path):
        file_path = consts.LOCAL_DATA_PATH + file_path

        with open(file_path, 'wb') as file:
            pickle.dump(dumped_object, file)
        print(f"List saved successfully to {file_path}")

    def load_events_from_file(self, file_path):
        file_path = consts.LOCAL_DATA_PATH + file_path

        with open(file_path, 'rb') as file:
            ak_zip_list = pickle.load(file)
        print(f"List loaded successfully from {file_path}")
        self.loaded_events = ak_zip_list

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
