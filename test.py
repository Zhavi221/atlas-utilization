#!/usr/bin/env python3
"""Minimal script to open and read ROOT file from URL"""

import uproot
from src.parse_atlas import parser
import random

atlasparser = parser.ATLAS_Parser(1,000,000)
link1 = random.choice(atlasparser.fetch_records_ids('2024'))
print(f"Opening ROOT file from: {link1}")

# Open and inspect
with uproot.open(link1) as f:
    print(f"Keys: {list(f.keys())}")