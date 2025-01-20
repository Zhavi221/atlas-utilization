from . import consts

from cernopendata_client import searcher as cern_client

import uproot
import pandas as pd

class ATLAS_Parser():
    def __init__(self, server=consts.CERN_OPENDATA_URI):
        self.server = server
        self.file_uris = []

    def get_data(self, recids=[]):
        file_indices = self._retrieve_file_indices(recids)
        print('Successfuly retrieved all indexes.')

        total_files = 0
        all_files_uris = []

        for file_indice in file_indices:
            files = file_indice['files']
            total_files += len(files)
            
            for file in files:
                uri = file['uri']
                all_files_uris.append(uri)

        print('Total amount of files found - ', total_files)
        self.file_uris = all_files_uris

    def _retrieve_file_indices(self, recids=[]):

        indices = []
        for recid in recids:
            cern_client.verify_recid(self.server, recid)
            
            metadata_from_recid = cern_client.get_record_as_json(self.server, recid)

            file_indices = metadata_from_recid['metadata']['_file_indices']
                
            indices.extend(file_indices)
        
        return indices

    def parse_data(self, schema):
        for file_uri in self.file_uris:# Open the ROOT file and access the "CollectionTree"
            with uproot.open(file_uri) as file:
                tree = file['CollectionTree']
                
                events = {}
                for objname, fields in schema.items():
                    base = objname
                    if objname in ['Electrons', 'Muons', 'Jets']:
                        base = 'Analysis' + objname
                    
                    # Construct the full branch names
                    full_fields = [f'{base}AuxDyn.{field}' for field in fields]
                    
                    # Read the branches into a pandas DataFrame
                    df = tree.arrays(full_fields, library='pd')
                    
                    # Rename columns to remove the prefix
                    df.columns = fields
                    
                    events[objname] = df
                
                return events