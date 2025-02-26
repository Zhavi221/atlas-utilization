from . import consts

from cernopendata_client import searcher as cern_client

import uproot
import awkward as ak
import vector
import logging
import pickle
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

class ATLAS_Parser():
    def __init__(self, server=consts.CERN_OPENDATA_URI):
        self.server = server
        self.files_indexes = []
        self.events = []
        self.file_parsed_count = 0

    def get_records_file_index(self, recids=[], file_idx=[]):
        file_indexes = self._retrieve_file_indexes(recids, file_idx)
        logging.info("Successfuly retrieved all indexes.")

        total_files = 0
        all_files_indexes = []

        for file_index in file_indexes:
            files = file_index["files"]
            total_files += len(files)
            
            for file in files:
                uri = file["uri"]
                parent_file = file["key"]
                
                if not file_idx or any([file in parent_file for file in file_idx]):
                    all_files_indexes.append(uri)
                    total_files += 1


                parent_file = file["key"]
                
                if not file_idx or any([file in parent_file for file in file_idx]):
                    all_files_indexes.append(uri)
                    total_files += 1



        logging.info("Total amount of files found - ", total_files)
        self.files_indexes = all_files_indexes

    #MAKE THIS A FUNCTION TO RETRIEVE FROM A SINGLE RECORD
    def _retrieve_file_indexes(self, recids: list=[], specific_file_index: list=[]) -> list:
        indexes = []
        for recid in recids:
            cern_client.verify_recid(self.server, recid)
            
            metadata_from_recid = cern_client.get_record_as_json(self.server, recid)

            file_indexes = metadata_from_recid["metadata"]["_file_indices"]

            indexes.extend(file_indexes)
        
        return indexes

    def parse_indexed_files(self, schema: dict, limit=None) -> None:
        events = None
        
        if limit == None:
            limit = len(self.files_indexes)

        for file_index in tqdm(self.files_indexes[:limit]):
            logging.info(f"Processing file number {self.file_parsed_count} - {file_index}")
            
            cur_file_data = self._parse_file(schema, file_index)
            if events is None:
                events = cur_file_data
            else:
                events = ak.concatenate([events, cur_file_data], axis=0)
            
            self.file_parsed_count += 1

        self.events = events

    def _parse_file(self, schema: dict, file_index: str) -> ak.Array:
        with uproot.open({file_index: "CollectionTree"}) as tree:
            events = {}

            for obj_name, fields in schema.items():
                tree_as_rows = self._extract_obj(tree, obj_name, fields)
                events[obj_name] = tree_as_rows

            return ak.zip(events, depth_limit=1)

    def _extract_obj(self, tree: uproot.ReadOnlyDirectory, obj_name: str, fields: list):
        adjusted_obj_name: str = ATLAS_Parser._adjust_obj_name(obj_name)

        obj_tree_data = self._extract_obj_fields(tree, fields, adjusted_obj_name)
        return obj_tree_data

    def _extract_obj_fields(self, tree: uproot.ReadOnlyDirectory, fields: list, obj_name: str):
        obj_fields_arrays = tree.arrays(
            fields,
            aliases=ATLAS_Parser._format_field_names(fields, obj_name)
        )
        sep_to_arrays = ak.unzip(obj_fields_arrays)
        field_names = obj_fields_arrays.fields

        tree_as_rows: ak.zip = ak.zip(dict(zip(field_names, sep_to_arrays))) 
        return tree_as_rows

    @staticmethod
    def _format_field_names(self, fields, modified_container):
        return {var: f"{modified_container}AuxDyn.{var}" for var in fields}

    @staticmethod
    def _adjust_obj_name(container_name):
        final_name = container_name
        if final_name in ["Electrons", "Muons", "Jets"]:
            #ADJUSTS THE CONTAINER NAME TO SUIT THE TREE 
            final_name = "Analysis" + final_name
        return final_name

    def save_events(self, file_path):
        file_path = consts.LOCAL_DATA_PATH + file_path

        with open(file_path, 'wb') as file:
            pickle.dump(self.events, file)
        logging.info(f"List saved successfully to {file_path}")

    def load_events_from_file(self, file_path):
        file_path = consts.LOCAL_DATA_PATH + file_path

        with open(file_path, 'rb') as file:
            ak_zip_list = pickle.load(file)
        logging.info(f"List loaded successfully from {file_path}")
        self.events = ak_zip_list
