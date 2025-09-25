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

from src.calculations import combinatorics, physics_calcs

import argparse, yaml, uproot
import numpy as np

def mass_calculate(config):
    logger = init_logging()
    
    # categories = combinatorics.make_objects_categories(
    #     schemas.PARTICLE_LIST, min_n=4, max_n=4
    # )
    # main_category = categories[0]

    os.makedirs(config["input_dir"], exist_ok=True)
    all_combinations = combinatorics.get_all_combinations(config["objects_to_calculate"])

    for combination in all_combinations:    
        logger.info(f"Processing combination: {combination}")
        for filename in os.listdir(config["input_dir"]):
            if filename.endswith(".root"):
                logger.info(f"Processing file: {filename}")
                file_path = os.path.join(config["input_dir"], filename)
                
                #TODO: MAKE SURE THIS IS GOOD
                with uproot.open(file_path) as f:
                    e = f["tree"]
                arrays = e.arrays(library="ak")
                
                #TODO: filter each file by all the combinations 
                # AND THEN calculate the invariant mass for each combination
                # AND SAVE the results as a new numpy array OR directly train BumpNet? (requiers post processing) 
                
                filtered_events = physics_calcs.filter_events_by_combination(
                    arrays, combination, use_count_range=False
                )    

                inv_mass = physics_calcs.calc_events_mass(filtered_events) 

                combination_name = prepare_combination_name(combination)
                output_path = os.path.join(
                    config["output_dir"], 
                    f"{filename}_{combination_name}_inv_mass.npy" #TODO: MAKE THIS MORE ROBUST (e.g. 2e2j instead of 2electrons_2jets
                    )
                np.save(output_path, ak.to_numpy(inv_mass))
    #######

def prepare_combination_name(combination: dict) -> str:
    combination_name = ''
    for object, amount in combination:
        combination_name += str(amount)
        combination_name += object

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