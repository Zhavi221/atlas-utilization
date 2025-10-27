import logging
import sys
import matplotlib.pyplot as plt # plotting
import awkward as ak
import tqdm
import argparse
import yaml
import uproot
import awkward as ak
import os
import vector

from src.calculations import combinatorics, physics_calcs
from src.parse_atlas import parser

import argparse, yaml, uproot
import numpy as np

branches = set()
def mass_calculate(config):
    logger = init_logging()
    
    if len(os.listdir(config["input_dir"])) == 0: 
        logger.warning(f"Input directory '{config['input_dir']}' is empty.")
        return
    
    os.makedirs(config["output_dir"], exist_ok=True)
    
    all_combinations = combinatorics.get_all_combinations(config["objects_to_calculate"])

    for filename in os.listdir(config["input_dir"]):
        if filename.endswith(".root"):
            logger.info(f"Processing file: {filename}")
            file_path = os.path.join(config["input_dir"], filename)
            
            particle_arrays: ak.Array = parser.ATLAS_Parser.parse_file(file_path)

            #FOR TESTING - limit
            for combination in all_combinations[:30]:  # Limit to first 5 combinations for testing
                logger.info(f"Processing combination: {combination}")
                filtered_events: ak.Array = physics_calcs.filter_events_by_particle_counts(
                    events=particle_arrays, 
                    particle_counts=combination, 
                    is_particle_counts_range=False
                )    
                
                if len(filtered_events) == 0:
                    continue
                
                inv_mass: list = physics_calcs.calc_inv_mass(filtered_events) 
                
                if not ak.any(inv_mass):
                    continue

                combination_name = prepare_combination_name(combination)
                output_path = os.path.join(
                    config["output_dir"], 
                    f"{filename}_{combination_name}_inv_mass.npy" 
                    )
                
                np.save(output_path, ak.to_numpy(inv_mass))

def prepare_combination_name(combination: dict) -> str:
    combination_name = ''
    for object, amount in combination.items():
        combination_name += str(amount)
        combination_name += object[0].lower()

    return combination_name

def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logger = logging.getLogger(__name__)
    return logger



#LEARN FROM THIS - OLD PIPELINE IS IT RELEVANT?
# def run():
#     atlasparser = parser.ATLAS_Parser()
#     release_files_uris = atlasparser.fetch_records_ids(release_year='2024')

#     categories = combinatorics.make_objects_categories(schemas.PARTICLE_LIST, min_n=2, max_n=4)

#     # for events_chunk in atlasparser.parse_files(files_ids=release_files_uris, limit=30):
#     for events_chunk in atlasparser.parse_files(files_ids=
#                                                 random.sample(release_files_uris, k=1)):
#         for category in categories:
#             # logging.info(f"Processing category: {category}")
#             combination_dict_gen = combinatorics.make_objects_combinations_for_category(
#                     category, min_k=2, max_k=4)
#             combination_dict = next(combination_dict_gen)
#             #IF CAN FILTER ACCORDING TO ITERATION'S COMBINATION
#             if not all(obj in events_chunk.fields for obj in combination_dict.keys()):
#                 logging.info ('Not all of the combination objects are present in the events chunk. ')
#                 continue    

#             combo_events = atlasparser.filter_events_by_combination(
#                 events_chunk, combination_dict)

#             combination_events_mass = atlasparser.calculate_mass_for_combination(combo_events)
            
#             #COMBO_EVENTS IS THE EVENTS FILTERED FOR EACH COMBINTATION
#             #NEXT STEP, MAKE A MASS HIST OUT OF IT
#             plt.hist(ak.flatten(combination_events_mass / consts.GeV, axis=None), bins=100)
#             plt.xlabel("Reconstructed Top Quark Mass (GeV)")
#             plt.ylabel("Number of Events")
#             plt.title("Distribution of Reconstructed Top Quark Mass")
#             plt.axvline(172.76, color='r', linestyle='dashed', linewidth=2, label='Expected Top Quark Mass')
#             plt.legend()
#             plt.show()
#             0/0