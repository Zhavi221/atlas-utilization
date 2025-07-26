import logging
import sys
from src.parse_atlas import parser, combinatorics, consts, schemas
import matplotlib.pyplot as plt # plotting
import awkward as ak
import tqdm
import random

release_years = ['2016', '2020', '2024', '2025']

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def run():
    atlasparser = parser.ATLAS_Parser()
    release_files_uris = atlasparser.fetch_records_ids(release_year='2024')

    categories = combinatorics.make_objects_categories(schemas.PARTICLE_LIST, min_n=2, max_n=4)

    # for events_chunk in atlasparser.parse_files(files_ids=release_files_uris, limit=30):
    for events_chunk in atlasparser.parse_files(files_ids=
                                                random.sample(release_files_uris, k=1)):
        for category in categories:
            # logging.info(f"Processing category: {category}")
            combination_dict_gen = combinatorics.make_objects_combinations_for_category(
                    category, min_k=2, max_k=4)
            combination_dict = next(combination_dict_gen)
            #IF CAN FILTER ACCORDING TO ITERATION'S COMBINATION
            if not all(obj in events_chunk.fields for obj in combination_dict.keys()):
                logging.info ('Not all of the combination objects are present in the events chunk. ')
                continue    

            combo_events = atlasparser.filter_events_by_combination(
                events_chunk, combination_dict)

            combination_events_mass = atlasparser.calculate_mass_for_combination(combo_events)
            
            #COMBO_EVENTS IS THE EVENTS FILTERED FOR EACH COMBINTATION
            #NEXT STEP, MAKE A MASS HIST OUT OF IT
            plt.hist(ak.flatten(combination_events_mass / consts.GeV, axis=None), bins=100)
            plt.xlabel("Reconstructed Top Quark Mass (GeV)")
            plt.ylabel("Number of Events")
            plt.title("Distribution of Reconstructed Top Quark Mass")
            plt.axvline(172.76, color='r', linestyle='dashed', linewidth=2, label='Expected Top Quark Mass')
            plt.legend()
            plt.show()
            0/0
            
run()