from src.parse_atlas import parser, combinatorics, consts, schemas

release_years = ['2016', '2020', '2024', '2025']
def run():
    atlasparser = parser.ATLAS_Parser()
    release_files_uris = atlasparser.fetch_real_records_ids(release_year='2024')
    # atlasparser.parsing_pipeline(file_uris=release_files_uris)

    categories = combinatorics.make_objects_categories(consts.PARTICLE_LIST, min_n=2, max_n=4)
    for uri in release_files_uris:
        root_file = atlasparser._parse_file(schema=schemas.GENERIC_SCHEMA, file_index=uri)
        
        for category in categories:
            combinations = combinatorics.make_objects_combinations_for_category(
                    category, min_k=2, max_k=4)
           
            for combination in combinations:
                combination_dict = {
                    obj[:-1]: int(obj[-1]) for obj in combination}

                if all(obj in root_file.fields for obj in combination_dict.keys()):
                    combo_events = atlasparser.filter_events_by_combination(root_file, combination_dict)
                    
                    print(len(combo_events)/len(root_file),
                          len(combo_events), len(root_file))
                    #COMBO_EVENTS IS THE EVENTS FILTERED FOR EACH COMBINTATION
                    #NEXT STEP, MAKE A MASS HIST OUT OF IT
