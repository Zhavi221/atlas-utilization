import logging
import sys
from src.parse_atlas import parser, combinatorics, consts, schemas

release_years = ['2016', '2020', '2024', '2025']

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def run():
    atlasparser = parser.ATLAS_Parser()
    release_files_uris = atlasparser.fetch_real_records_ids(release_year='2024')
    # atlasparser.parsing_pipeline(file_uris=release_files_uris)

    categories = combinatorics.make_objects_categories(consts.PARTICLE_LIST, min_n=2, max_n=4)

    for events_chunk in atlasparser.parse_files(files_ids=release_files_uris, limit=30):
        for category in categories:
            logging.info(f"Processing category: {category}")
            combinations = combinatorics.make_objects_combinations_for_category(
                    category, min_k=2, max_k=4)
            
            for combination in combinations:
                combination_dict = {
                    obj[:-1]: int(obj[-1]) for obj in combination}

                #IF CAN FILTER ACCORDING TO ITERATION'S COMBINATION
                if all(obj in events_chunk.fields for obj in combination_dict.keys()):
                    combo_events = atlasparser.filter_events_by_combination(events_chunk, combination_dict)
                    
                    #COMBO_EVENTS IS THE EVENTS FILTERED FOR EACH COMBINTATION
                    #NEXT STEP, MAKE A MASS HIST OUT OF IT

run()