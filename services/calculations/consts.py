"""
Centralized constants for particle physics calculations.
"""

# Known masses in MeV (converted to GeV later by _convert_array_to_gev). Values taken from PDG.
# TODO Should Jets/BJets be zero?
# TODO Tau candidates are stored as zero-mass in PHYSLITE; handle it.
KNOWN_MASSES = {
    "Muons": 105.6583755,
    "Photons": 0.0,
    "Electrons": 0.51099895069,
    "Jets": 0.0,
    "BJets": 0.0,
    "Taus": 1776.93,
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
}
