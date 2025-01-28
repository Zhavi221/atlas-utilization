from . import consts

from cernopendata_client import searcher as cern_client

import uproot
import awkward as ak
import vector
import logging

logging.basicConfig(level=logging.INFO)

class ATLAS_Parser():
    def __init__(self, server=consts.CERN_OPENDATA_URI):
        self.server = server
        self.file_indexes = []
        self.events = []

    def get_data_index(self, recids=[]):
        file_indices = self._retrieve_file_indices(recids)
        print("Successfuly retrieved all indexes.")

        total_files = 0
        all_files_uris = []

        for file_indice in file_indices:
            files = file_indice["files"]
            total_files += len(files)
            
            for file in files:
                uri = file["uri"]
                all_files_uris.append(uri)

        print("Total amount of files found - ", total_files)
        self.file_indexes = all_files_uris

    def _retrieve_file_indices(self, recids=[]):
        indices = []
        for recid in recids:
            cern_client.verify_recid(self.server, recid)
            
            metadata_from_recid = cern_client.get_record_as_json(self.server, recid)

            file_indices = metadata_from_recid["metadata"]["_file_indices"]
                
            indices.extend(file_indices)
        
        return indices

    def parse_all_files(self, schema, limit=0):
        events = []
        for file_index in self.file_indexes[:limit]:
            logging.info(f"Processing file - {file_index}")
            
            cur_file_data = self._parse_file(schema, file_index)
            events.append(cur_file_data)

            logging.info("Finished")
        
        self.events = ak.concatenate(events, axis=0)

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

                events[cur_container_name] = tree_as_rows

            return ak.zip(events, depth_limit=1)
    
    @staticmethod
    def _prepare_container_name(container_name):
        final_name = container_name
        if final_name in ["Electrons", "Muons", "Jets"]:
            final_name = "Analysis" + final_name
            zip_function = vector.zip
        else:
            zip_function = ak.zip
        
        return final_name, zip_function
