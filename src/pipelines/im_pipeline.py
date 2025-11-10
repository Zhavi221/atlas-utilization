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
                    
                    cur_im: list = physics_calcs.calc_inv_mass(sliced_events_by_pt) 
                    
                    if not ak.any(cur_im):
                        continue
                    
                    cur_combination_name = prepare_im_combination_name(filename, cur_fs, combination)
                    
                    if cur_combination_name not in fs_im_mapping[cur_fs]:
                        fs_im_mapping[cur_fs][cur_combination_name] = cur_im
                    else:
                        combination_im = fs_im_mapping[cur_fs][cur_combination_name]
                        fs_im_mapping[cur_fs][cur_combination_name] = combination_im.extend(cur_im)
                    
                    
                    if fs_dict_exceedng_threshold(fs_im_mapping, config["fs_mapping_threshold_bytes"]):
                        logger.info(f"FS mapping exceeding threshold. Saving big IM arrays.")
                        
                        combinations_to_save: list = get_combinations_exceeding_mem(fs_im_mapping, config["fs_mapping_threshold_bytes"])
                        for combination in combinations_to_save:  
                            combination_tuple = list(combination.items())[0]
                            im_combination_name = combination_tuple[0]                     
                            im_arr = combination_tuple[1]       

                            logger.info(f"Saving {im_combination_name}")   

                            output_path = os.path.join(
                                config["output_dir"], 
                                f"{im_combination_name}.npy" 
                                )
                            
                            np.save(output_path, ak.to_numpy(im_arr))

def get_process_memory_mb():
    """Get actual process memory usage"""
    process = psutil.Process(os.getpid())
    process_rss_bytes = process.memory_info().rss
    return process_rss_bytes / (1024**2)

def fs_dict_exceedng_threshold(fs_im_mapping, threshold):
    """Check if we should yield based on ACTUAL memory pressure"""
    process_memory = get_process_memory_mb() 
    
    # if (process_memory + 1000) < 8192: #FEATURE this ideally considers process memory limitations
    #     return False

    return sys.getsizeof(fs_im_mapping) >= threshold

def get_combinations_exceeding_mem(fs_im_mapping, threshold):
    combinations_exceeding = []
    mem_per_combination = threshold / len(fs_im_mapping.values())

    for fs, im_combinations in fs_im_mapping.items():
        for combination_name, im_arr in im_combinations.items():
            
            if sys.getsizeof(im_arr) > mem_per_combination:
                combinations_exceeding.append({combination_name: im_arr})
    
    return combinations_exceeding

def get_combination_dict_repr(combination, combination_name=""):
    for object, amount in combination.items():
        combination_name += f"{amount}{object[0].lower()}_"
    
    return combination_name

def prepare_im_combination_name(filename, final_state, combination: dict) -> str:
    combination_name = ''
    combination_name += f"{filename}_FS_{final_state}_IM_"
    combination_name = get_combination_dict_repr(combination, combination_name)

    return combination_name

def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logger = logging.getLogger(__name__)
    return logger