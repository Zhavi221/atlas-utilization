#CONSTS
GeV = 1000.
KNOWN_MASSES = {
    "Muons": 0.105,
    "Photons": 0.0,
    "Electrons": 0.000511,
    "Jets": 0.0
}

#LOCAL FILES
LOCAL_DATA_PATH = 'data/'
ATLAS_ELECTROWEAK_BOSON_DATA = 'mc_electroweak_boson.pkl'
ATLAS_MC_TOP_NOMINAL_DATA = 'mc_top_nominal.pkl'
LIBRARY_RELEASES_METADATA = {
    '2016e-8tev': 'https://opendata.atlas.cern/files/metadata_8tev.csv',
    '2020e-13tev': 'https://opendata.atlas.cern/files/metadata_2020e_13tev.csv',
    '2024r-pp': 'https://opendata.atlas.cern/files/metadata.csv',
    '2025e-13tev-beta': 'https://opendata.atlas.cern/files/metadata.csv'
}

#RECORDS
ATLAS_13TEV_2015_RECORD = 80000
ATLAS_13TEV_2016_RECORD = 80001
ATLAS_13TEV_RECIDS = [ATLAS_13TEV_2015_RECORD, ATLAS_13TEV_2016_RECORD]

ATLAS_MC_TOP_NOMINAL_2024 = 80017
ATLAS_ELECTROWEAK_BOSON_2024 = 80010
ATLAS_MC_2024_IDS = [ATLAS_MC_TOP_NOMINAL_2024, ATLAS_ELECTROWEAK_BOSON_2024]

#SPECIFIC FILE LISTS
ZMUMU = "mc20_13TeV_MC_Sh_2211_Zmumu_maxHTpTV2_m10_40_pT5_CFilterBVeto_file_index.json"

