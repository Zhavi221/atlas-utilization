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
import psutil

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
    
    #TODO check if combinations method adhere to new logic
    all_combinations: list = combinatorics.get_all_combinations(
        config["objects_to_calculate"],
        min_particles=config["min_particles"],
        max_particles=config["max_particles"],
        min_count=config["min_count"],
        max_count=config["max_count"],
        limit=30) #FOR TESTING - limit

    for filename in os.listdir(config["input_dir"]):
        if filename.endswith(".root"):
            logger.info(f"Processing file: {filename}")
            file_path = os.path.join(config["input_dir"], filename)
            
            particle_arrays: ak.Array = parser.ATLAS_Parser.parse_file(file_path)
            fs_im_mapping = {}
            for cur_fs, fs_events in physics_calcs.group_by_final_state(particle_arrays):
                if cur_fs not in fs_im_mapping:
                    fs_im_mapping[cur_fs] = {}
                for combination in all_combinations:
                    if not physics_calcs.is_finalstate_contain_combination(cur_fs, combination):
                        continue

                    logger.info(f"Processing combination: {combination}")
                    filtered_events: ak.Array = physics_calcs.filter_events_by_particle_counts(
                        events=fs_events, 
                        particle_counts=combination
                    )    

                    sliced_events_by_pt: ak.Array = physics_calcs.slice_events_by_field(
                        events=filtered_events, 
                        particle_counts=combination,
                        slice_by_field="pt"
                    )

                    if len(sliced_events_by_pt) == 0:
                        continue
                    
                    inv_mass: list = physics_calcs.calc_inv_mass(sliced_events_by_pt) 
                    
                    if not ak.any(inv_mass):
                        continue
                    
                    #TODO add year, final state and combination to filename
                    combination_name = prepare_combination_name(filename, cur_fs, combination)
                    
                    if combination_name not in fs_im_mapping[cur_fs]:
                        fs_im_mapping[cur_fs][combination_name] = inv_mass
                    else:
                        combination_im = fs_im_mapping[cur_fs][combination_name]
                        fs_im_mapping[cur_fs][combination_name] = combination_im.extend(inv_mass)
                    
                    
                    #TODO check for size of each "final state list", or just output all of it once chunk size enough
                    if chunk_size_enough(config, fs_im_mapping):
                        #FOR TESTING commented saving files mechanism
                        
                        # for fs, combinations in fs_im_mapping.items():
                        #     for combination_im in combinations:
                                
                        output_path = os.path.join(
                            config["output_dir"], 
                            f"{combination_name}.npy" 
                            )
                        
                        np.save(output_path, ak.to_numpy(fs_im_mapping[cur_fs][combination_name]))
                        break

def get_actual_memory_mb():
    """Get actual process memory usage"""
    process = psutil.Process(os.getpid())
    process_rss_bytes = process.memory_info().rss
    return process_rss_bytes / (1024**2)

def chunk_size_enough(config, fs_im_mapping):
    """Check if we should yield based on ACTUAL memory pressure"""
    logical_size = sys.getsizeof(fs_im_mapping)
    actual_memory = get_actual_memory_mb() 
    
    if (actual_memory + 1000) < 8192:
        return True

    return logical_size >= 50000000

def prepare_combination_name(filename, final_state, combination: dict) -> str:
    combination_name = ''
    combination_name += f"{filename}_FS_{final_state}_IM_"

    for object, amount in combination.items():
        combination_name += f"{amount}{object[0].lower()}_"

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