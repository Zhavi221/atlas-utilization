from . import consts

from cernopendata_client import searcher as cern_client
import atlasopenmagic as atom

import uproot
import awkward as ak
import vector
import logging
import pickle
import random

logging.basicConfig(level=logging.INFO)

class ATLAS_Parser():
    def __init__(self, release_year='', domain=consts.CERN_OPENDATA_URI, rand_release=False):
        self.domain = domain
        self.files_uris = []
        self.events = []
        self.file_parsed_count = 0

        if rand_release:
            release = random.choice(consts.AVILABLE_RELEASES)
        else:
            release = consts.AVILABLE_RELEASES.get(release_year)
        
        atom.set_release(release)
        
        logging.info("Initialize atom with release: %s", release)

    def get_real_record_uris(self, recids=[], file_idx=[]):
        real_datasets_ids = atom.available_data()
        all_files_uris = []

        for id in real_datasets_ids:
            all_files_uris.extend(atom.get_urls_data(id))

        logging.info("Total amount of files found: %d", len(all_files_uris))
        self.files_uris = all_files_uris

    def get_mc_files_uris(self, random=False):
        #TODO: implement get_mc_uris
        pass
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
        events = None
        self.file_parsed_count = 0

        if not files_ids:
            files_ids = self.files_indexes[:limit]

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

            
            self.file_parsed_count += 1
            logging.info(f"Finished, file number {self.file_parsed_count}")

        # self.events = ak.concatenate(events, axis=0)
        self.events = events

    def _parse_file(self, schema, file_index):
        with uproot.open({file_index: "CollectionTree"}) as tree:
            events = {}

            for container_name, fields in schema.items():
                cur_container_name, zip_function = ATLAS_Parser._prepare_container_name(container_name)

                tree_as_rows = tree.arrays(
                    fields,
                    aliases={var: f"{cur_container_name}AuxDyn.{var}" for var in fields}
                )
                sep_to_arrays = ak.unzip(tree_as_rows)
                field_names = tree_as_rows.fields

                tree_as_rows = zip_function(dict(zip(field_names, sep_to_arrays)))

                events[container_name] = tree_as_rows
                events[container_name] = tree_as_rows

            return ak.zip(events, depth_limit=1)
    
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
    def _prepare_container_name(container_name):
        final_name = container_name
        if final_name in ["Electrons", "Muons", "Jets"]:
            final_name = "Analysis" + final_name
            zip_function = vector.zip
        else:
            zip_function = ak.zip
        
        return final_name, zip_function
