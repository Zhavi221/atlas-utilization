#!/usr/bin/env python3
"""Minimal script to open and read ROOT file from URL"""

import uproot
from src.parse_atlas import parser
import random

atlasparser = parser.ATLAS_Parser(1,000,000)
ids = atlasparser.fetch_records_ids('2024')
first_link = ids[0]
with uproot.open(first_link) as f:
    print(f"Opening first ROOT file from: {first_link}")
    print(f"Keys: {list(f.keys())}")

random_link = random.choice(ids)
with uproot.open(random_link) as f:
    print(f"Opening random ROOT file from: {random_link}")
    print(f"Keys: {list(f.keys())}")
    