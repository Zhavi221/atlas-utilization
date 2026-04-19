"""
Centralized constants for particle physics calculations.
"""
KNOWN_MASSES = {
    "Muons": 0.105,
    "Photons": 0.0,
    "Electrons": 0.000511,
    "Jets": 0.0,
    "Taus": 1.77686,
}

# ATLAS PHYSLITE / AnalysisElectronsAuxDyn — relative isolation uses cone energy / pT
ELECTRON_REL_ISOLATION_FIELD = "ptvarcone30_Nonprompt_All_MaxWeightTTVALooseCone_pt1000"

LETTER_PARTICLE_MAPPING = {
    "e": "Electrons",
    "j": "Jets",
    "g": "Photons",
    "m": "Muons",
    "t": "Taus",
    "b": "BJets",
    "l": "LJets",
}
