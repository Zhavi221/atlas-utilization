"""
Centralized constants for particle physics calculations.
All particle-related constants should be defined here to avoid duplication.
"""
KNOWN_MASSES = {
    "Muons": 0.105,
    "Photons": 0.0,
    "Electrons": 0.000511,
    "Jets": 0.0
}

LETTER_PARTICLE_MAPPING = {
    "e": "Electrons",
    "j": "Jets",
    "p": "Photons",  # Fixed typo: was "Photos"
    "m": "Muons"
}
