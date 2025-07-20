GENERIC_SCHEMA = {
    "Electrons": [
        "pt", "eta", "phi", "m", "charge",
        "isTight", "isLoose", "d0", "z0"
    ],
    "Muons": [
        "pt", "eta", "phi", "m", "charge",
        "isTight", "isLoose", "d0", "z0"
    ],
    "Jets": [
        "pt", "eta", "phi", "m", "charge",
        "DL1dv01_pb", "DL1dv01_pc", "DL1dv01_pu"
    ],
    "Photons": [
        "pt", "eta", "phi", "m", "isConverted", "isTight", "isLoose"
    ],
    "MET": [
        "et", "phi"
    ]
    # Add others if needed: Taus, TrackJets, etc.
    # Category definitions and event labels stay exactly as you had them:
    # e.g., "2ex_0mx_0gx_0jx"
}

# E_POSI_MUON_SCHEMA = {
#     "Electrons": [
#         # "pt", "eta", "phi", "charge"
#         "pt", "eta", "phi", "charge"
#     ],
#     "Muons": [
#         "pt", "eta", "phi", "charge"
#     ]
# }