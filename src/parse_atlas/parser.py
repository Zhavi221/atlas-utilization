from . import consts

import atlasopenmagic as atom

import uproot
import awkward as ak
import vector
import logging
import pickle
import random
import requests
import csv
import sys

from . import combinatorics, schemas

logging.basicConfig(level=logging.INFO)

class ATLAS_Parser():
    def __init__(self, domain=consts.CERN_OPENDATA_URI):
        self.domain = domain
        self.files_uris = []
        self.events = []
        self.file_parsed_count = 0
        

        self.categories = combinatorics.make_objects_categories(consts.PARTICLE_LIST, min_n=2, max_n=4)
        
    def fetch_real_records_ids(self, release_year):
        atom.set_release(consts.RELEASES_YEARS.get(release_year))

        datasets_ids = atom.available_data()             
        release_files_uris = []
        
        for id in datasets_ids:
            release_files_uris.extend(atom.get_urls_data(id))

        logging.info("Total amount of files found: %d", len(release_files_uris))
        # self.files_uris = release_files_uris
        return release_files_uris

    def fetch_mc_files_ids(self, release_year, is_random=False, all=False):
        release = consts.RELEASES_YEARS[release_year]   
        metadata_url = consts.LIBRARY_RELEASES_METADATA[release]

        #TODO: IMPLEMENT ALL VARIABLE
        _metadata = {}
        
        response = requests.get(metadata_url)

        response.raise_for_status()
        lines = response.text.splitlines()

        reader = csv.DictReader(lines)
        for row in reader:
            dataset_number = row['dataset_number'].strip()
            _metadata[dataset_number] = row

            # We can use the physics short name to get the metadata as well
            # physics_short = row['physics_short'].strip()
            # _metadata[physics_short] = row
        
        all_mc_ids =  list(_metadata.keys())
        
        if is_random:
            random_mc_id = random.choice(all_mc_ids)
            all_metadata = atom.get_metadata(random_mc_id)
            print(all_metadata['process'], all_metadata['short_name'])
            return random_mc_id
        
        return all_mc_ids

    def parsing_pipeline(self, limit: int=0, files_ids: list=[]) -> None:
        '''
            parses all files that were fetched
        '''
        events = None
        self.file_parsed_count = 0

        if not files_ids:
            files_ids = self.files_indexes[:limit] #slicing by zero is negligble
        
        logging.info(f"Processing {len(files_ids)} files")
        for file_index in files_ids:
            logging.info(f"Processing file - {file_index}")
            
            cur_file_data = self._parse_file(schemas.GENERIC_SCHEMA, file_index)
            
            if sys.getsizeof(events) >= 1e+9:
                logging.info(f"Dumping events to memory, size: {sys.getsizeof(events)} bytes")
                self.save_events(events, f"events_{self.file_parsed_count}.pkl")
                events = None

            if events is None:
                events = cur_file_data
            else:
                events = ak.concatenate([events, cur_file_data], axis=0)
            
            self.file_parsed_count += 1
            logging.info(f"Finished, file number {self.file_parsed_count}")

    def _parse_file(self, schema, file_index):   
        '''
            parses a single file by a generic schema
        '''
        #TODO: change this function so it parses a file with a generic schema 
        #                   (a schema that will fit all use cases) 
        
        with uproot.open({file_index: "CollectionTree"}) as tree:
            events = {}
            
            for obj_name, fields in schema.items():
                cur_obj_name, zip_function = ATLAS_Parser._prepare_obj_name(obj_name)
                filtered_fields = list(filter(
                    lambda field: f"{cur_obj_name}AuxDyn.{field}" in tree.keys(),
                    fields))
                
                if not any(filtered_fields):
                    continue
                
                logging.info(f"Parsing {cur_obj_name} with fields: {filtered_fields}")

                tree_as_rows = tree.arrays(
                    filtered_fields,
                    aliases={field: f"{cur_obj_name}AuxDyn.{field}" for field in filtered_fields}
                )

                sep_to_arrays = ak.unzip(tree_as_rows)
                field_names = tree_as_rows.fields

                tree_as_rows = zip_function(dict(zip(field_names, sep_to_arrays)))

                events[obj_name] = tree_as_rows

            return ak.zip(events, depth_limit=1)

    def generate_histograms(self):
        categories = combinatorics.make_objects_categories(consts.PARTICLE_LIST, min_n=2, max_n=4)
        for category in categories:
            combination = combinatorics.make_objects_combinations_for_category(
                category, min_k=2, max_k=4)
            events = self.filter_events_by_combination(combination)

    def filter_events_by_combination(self, events_file, combination_dict):
        #TODO: implement filter_events_by_category

        '''LEARN FROM THIS CODE'''
        # events["Electrons"] = selected_electrons(events.Electrons)
        # events["Muons"] = selected_muons(events.Muons)
        # events["Jets"] = selected_jets(events.Jets)
        # events["Jets"] = events.Jets[no_overlap(events.Jets, events.Electrons)]
        # events["Jets", "is_bjet"] = events.Jets.btag_prob > 0.85
        
        for obj, count in combination_dict.items():
            events_file = events_file[
                ak.num(events_file[obj]) == count]
        # events_file = events_file[
        #     (ak.num(events_file.Jets) == combination_dict.get("Jets", 0)) # at least N jets
        #     (ak.num(events_file.Jets) >= 4) # at least 4 jets
        #     & ((ak.num(events_file.Electrons) + ak.num(events_file.Muons)) == 1) # exactly one lepton
        #     & (ak.num(events_file.Jets[events_file.Jets.is_bjet]) >= 2) # at least two btagged jets with prob > 0.85
        # ]
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
        self.events = ak_zip_list

    def testing_load_file_as_object(self, file_index):
        with uproot.open({file_index: "CollectionTree"}) as tree:
            return tree

    @staticmethod
    def _prepare_obj_name(obj_name):
        final_name = None
        if obj_name in ["Electrons", "Muons", "Jets"]:
            final_name = "Analysis" + obj_name
            zip_function = vector.zip
        else:
            zip_function = ak.zip
        
        
        return final_name, zip_function
