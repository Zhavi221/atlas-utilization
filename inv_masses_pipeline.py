import logging
import sys
from src.parse_atlas import parser, combinatorics, consts, schemas
import matplotlib.pyplot as plt # plotting
import awkward as ak
import tqdm
import argparse
import yaml
import uproot
import awkward as ak
import os


from src.parse_atlas import parser, schemas, combinatorics
import argparse, yaml, uproot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--config", help="Config file", default="configs/pipeline_config.yaml")
args = arg_parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

parsing_config = config["parsing"]

def run():
    atlasparser = parser.ATLAS_Parser(
        max_chunk_size_bytes=parsing_config["max_chunk_size_bytes"],
        max_threads=parsing_config["max_threads"],
        max_processes=parsing_config["max_processes"]
        )
   
    # TODO: get all the combinations for the each category
    all_combinations = combinatorics.get_all_combinations(parsing_config["objects_to_calculate"])

    #TODO: write a for loop over all the root files in output_dir
    with uproot.open("output_dir/filtered_1_test.root") as f:
        atlasparser.events = f["tree"]

    e = atlasparser.events
    arrays = e.arrays(library="ak")
    print(arrays.fields)
    print(type(arrays))
    
    #TODO: filter each file by all the combinations 
    # AND THEN calculate the invariant mass for each combination
    # AND SAVE the results as a new numpy array OR directly train BumpNet? (requiers post processing) 
    twojets_twoelectrons = atlasparser.filter_events_by_counts(
        arrays, {'nElectrons_eta': 2, 'nJets_eta': 4}, use_range=False)
    
    inv_mass = atlasparser.calculate_mass_for_combination(twojets_twoelectrons) 
    #######

if __name__ == "__main__":
    run()