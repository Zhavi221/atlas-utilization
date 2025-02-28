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
    def __init__(self, server: str=consts.CERN_OPENDATA_URI) -> None:
        self.server = server
        self.files_indexes = []
        self.events = []
        self.file_parsed_count = 0

    def get_records_file_index(self, recids=[], file_idx=[]) -> None:
        file_indexes: list = self._retrieve_file_indexes(recids, file_idx)
        logging.info("Successfuly retrieved all indexes.")

        total_files = 0
        all_files_indexes = []

        for file_index in file_indexes:
            files = file_index["files"]
            total_files += len(files)
            
            for file in files:
                uri = file["uri"]                
                all_files_indexes.append(uri)
                total_files += 1


        logging.info("Total amount of files found - ", total_files)
        self.files_indexes = all_files_indexes

    def _retrieve_file_indexes(self, recids: list=[], specific_file_index: list=[]) -> list:
        indexes = []
        for recid in recids:
            cern_client.verify_recid(self.server, recid)
            
            metadata_from_recid = cern_client.get_record_as_json(self.server, recid)

            file_indexes = metadata_from_recid["metadata"]["_file_indices"]

            if specific_file_index:
                file_indexes = [
                    file_index for file_index in file_indexes 
                    if file_index["key"] in specific_file_index
                ]

            indexes.extend(file_indexes)
        
        return indexes

    def parse_indexed_files(self, schema: dict, limit=None) -> None:
        events: ak.Array = None
        
        if limit is  None:
            limit = len(self.files_indexes)

        for file_index in tqdm(self.files_indexes[:limit]):
            logging.info(f"Processing file number {self.file_parsed_count} - {file_index}")
            
            cur_file_data: ak.Array = self._parse_file(schema, file_index)
            if events is None:
                events: ak.Array = cur_file_data
            else:
                events: ak.Array = ak.concatenate([events, cur_file_data], axis=0)
            
            self.file_parsed_count += 1

        self.events = events

    def _parse_file(self, schema: dict, file_index: str) -> ak.Array:
        with uproot.open({file_index: "CollectionTree"}) as tree:
            events = {}

            for obj_name, fields in schema.items():
                tree_as_rows: ak.Array = self._extract_obj(tree, obj_name, fields)
                events[obj_name] = tree_as_rows

            return ak.zip(events, depth_limit=1)

    def _extract_obj(self, tree: uproot.ReadOnlyDirectory, obj_name: str, fields: list) -> ak.Array:
        adjusted_obj_name: str = ATLAS_Parser._adjust_obj_name(obj_name)

        obj_tree_data: ak.Array = self._extract_obj_fields(tree, fields, adjusted_obj_name)
        return obj_tree_data

    def _extract_obj_fields(self, tree: uproot.ReadOnlyDirectory, fields: list, obj_name: str) -> ak.Array:
        """
        Extracts specified fields given by a list for a certain particle (obj_name) from a given root tree file
        And returns them as an awkward array.
        Args:
            tree (uproot.ReadOnlyDirectory): The directory object from which to extract fields.
            fields (list): A list of field names to extract from the directory.
            obj_name (str): The name of the object (particle type) to extract fields from.
        Returns:
            ak.Array: An awkward array containing the extracted fields, zipped together as rows.
        """
        
        obj_fields_arrays: ak.Array = tree.arrays(
            fields,
            aliases=ATLAS_Parser._format_field_names(fields, obj_name)
        )
        sep_to_arrays: tuple = ak.unzip(obj_fields_arrays)
        field_names: list = obj_fields_arrays.fields

        tree_as_rows: ak.Array = ak.zip(dict(zip(field_names, sep_to_arrays))) #TODO: Check if this is the correct way to zip the arrays
        return tree_as_rows

    @staticmethod
    def _format_field_names(self, fields, modified_container):
        return {var: f"{modified_container}AuxDyn.{var}" for var in fields}

    @staticmethod
    def _adjust_obj_name(container_name: str) -> str:
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
