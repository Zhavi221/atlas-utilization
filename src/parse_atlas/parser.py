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

from . import combinatorics

logging.basicConfig(level=logging.INFO)

class ATLAS_Parser():
    def __init__(self, domain=consts.CERN_OPENDATA_URI, release_year=''):
        self.domain = domain
        self.files_uris = []
        self.events = []
        self.file_parsed_count = 0

        if release_year:
            self.release = consts.RELEASES_YEARS.get(release_year)

        self.categories = combinatorics.make_objects_categories(consts.PARTICLE_LIST, min_n=2, max_n=4)
        
    def fetch_real_records_ids(self):
        datasets_ids = []
        if self.release:
            atom.set_release(self.release)
        else:
            for release in consts.RELEASES_YEARS.values():
                atom.set_release(self.release)  
                datasets_ids.append(atom.available_data())
            
        release_files_uris = []

        for id in datasets_ids:
            release_files_uris.extend(atom.get_urls_data(id))

        logging.info("Total amount of files found: %d", len(release_files_uris))
        self.files_uris = release_files_uris

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

    #DEPRECATED
    '''
    def get_records_file_index(self, recids=[], file_idx=[]):
        file_indices = self._retrieve_file_indices(recids, file_idx)
        print("Successfuly retrieved all indexes.")

        total_files = 0
        all_files_indexes = []

        for file_indice in file_indices:
            files = file_indice["files"]
            
            for file in files:
                uri = file["uri"]
                parent_file = file["key"]
                
                if not file_idx or any([file in parent_file for file in file_idx]):
                    all_files_indexes.append(uri)
                    total_files += 1

        print("Total amount of files found - ", total_files)
        self.files_indexes = all_files_indexes
    
    
    #MAKE THIS A FUNCTION TO RETRIEVE FROM A SINGLE RECORD
    def _retrieve_file_indices(self, recids: list=[], specific_file_index: list=[]) -> list:
        indices = []
        for recid in recids:
            cern_client.verify_recid(self.domain, recid)
            
            metadata_from_recid = cern_client.get_record_as_json(self.domain, recid)

            file_indices = metadata_from_recid["metadata"]["_file_indices"]

            indices.extend(file_indices)
        
        return indices        
    '''

    def parse_all_files(self, schema: dict, limit: int=0, files_ids: list=[]) -> None:
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
            
            cur_file_data = self._parse_file(schema, file_index)
            if events is None:
                events = cur_file_data
            else:
                events = ak.concatenate([events, cur_file_data], axis=0)
            # events.append(cur_file_data)
            
            self.file_parsed_count += 1
            logging.info(f"Finished, file number {self.file_parsed_count}")

        # self.events = ak.concatenate(events, axis=0)
        self.events = events

    def load_single_file(self, schema, file_index):
        with uproot.open({file_index: "CollectionTree"}) as tree:
            return tree

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
    
    def filter_events_by_categories(self):
        #TODO: implement filter_events_by_category

        '''LEARN FROM THIS CODE'''
        # events = copy.copy(events) # shallow copy
        # events["Jets", "btag_prob"] = events.BTagging_AntiKt4EMPFlow.DL1dv01_pb
        # events["Electrons"] = selected_electrons(events.Electrons)
        # events["Muons"] = selected_muons(events.Muons)
        # events["Jets"] = selected_jets(events.Jets)
        # events["Jets"] = events.Jets[no_overlap(events.Jets, events.Electrons)]
        # events["Jets", "is_bjet"] = events.Jets.btag_prob > 0.85
        # events = events[
        #     (ak.num(events.Jets) >= 4) # at least 4 jets
        #     & ((ak.num(events.Electrons) + ak.num(events.Muons)) == 1) # exactly one lepton
        #     & (ak.num(events.Jets[events.Jets.is_bjet]) >= 2) # at least two btagged jets with prob > 0.85
        # ]
        # return ak.to_packed(events)

    def save_events(self, file_path):
        file_path = consts.LOCAL_DATA_PATH + file_path

        with open(file_path, 'wb') as file:
            pickle.dump(self.events, file)
        print(f"List saved successfully to {file_path}")

    def load_events_from_file(self, file_path):
        file_path = consts.LOCAL_DATA_PATH + file_path

        with open(file_path, 'rb') as file:
            ak_zip_list = pickle.load(file)
        print(f"List loaded successfully from {file_path}")
        self.events = ak_zip_list

    @staticmethod
    def _prepare_obj_name(obj_name):
        final_name = None
        if obj_name in ["Electrons", "Muons", "Jets"]:
            final_name = "Analysis" + obj_name
            zip_function = vector.zip
        else:
            zip_function = ak.zip
        
        
        return final_name, zip_function
