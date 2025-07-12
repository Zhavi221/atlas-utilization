from src import parse_atlas

release_years = ['2016', '2020', '2024', '2025']
parser = parse_atlas.ATLAS_Parser(releases_year=release_years[3])
parser.fetch_real_records_ids()
parser.parse_all_files()
