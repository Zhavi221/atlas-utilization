"""
Build the MC weights registry (weights_registry.json) for the histogram stage.

This is the *producer* for MC weighting: it runs where each source file's DSID
is known, fetches per-dataset metadata, computes the per-file normalization
weight, and writes weights_registry.json next to the histogram input data. The
histogram stage then reads that file and fills histograms with the weights.

Assembling source -> DSID:
  1. Auto: scan --input-dir for files whose name contains a canonical ATLAS
     DSID (e.g. mc20_13TeV.410470.<name>...). Their DSID is extracted directly.
  2. Explicit: --dsid-map <json> provides {source_prefix: dsid_or_url} for
     sources whose filename does NOT carry a DSID (e.g. hash-named files like
     2024r-pp_<hash>.root). URL values are resolved to a DSID automatically.
  The explicit map wins on conflicts.

Usage:
    python -m utils.build_weights_registry \
        --config config.yaml \
        --input-dir /path/to/im_arrays_processed \
        --dsid-map source_dsids.json

Note: requires the pipeline runtime deps (atlasopenmagic, etc.), so run it in
the analysis environment, not a bare checkout.
"""

import argparse
import json
import logging
import os
import sys

import yaml

# Allow running as a script (python utils/build_weights_registry.py ...)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domain.config import MCWeightingConfig
from services.metadata.fetcher import MetadataFetcher
from services.calculations.weights_registry import (
    build_and_save_registry,
    extract_dsid_from_url,
    source_prefix_from_signature,
)

_SOURCE_FILE_SUFFIXES = (".root", ".npy", ".sqlite")


def _load_weighting_config(config_path: str) -> MCWeightingConfig:
    """Load just the mc_weighting_config block from the pipeline YAML."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f) or {}
    mc_dict = config_dict.get("mc_weighting_config") or {}
    return MCWeightingConfig(
        enabled=mc_dict.get("enabled", False),
        target_luminosity_fb=mc_dict.get("target_luminosity_fb", 1.0),
        luminosity_by_campaign=mc_dict.get("luminosity_by_campaign"),
        require_metadata=mc_dict.get("require_metadata", False),
    )


def _scan_source_prefixes(input_dir: str) -> set:
    """Collect distinct source-file prefixes from files in input_dir."""
    prefixes = set()
    if not os.path.isdir(input_dir):
        return prefixes
    for name in os.listdir(input_dir):
        if name.endswith(_SOURCE_FILE_SUFFIXES):
            prefixes.add(source_prefix_from_signature(name))
    return prefixes


def _resolve_dsid(value):
    """A map value may be a DSID (int/str digits) or a URL/name to parse."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if value.strip().isdigit():
            return int(value.strip())
        return extract_dsid_from_url(value)
    return None


def build_source_to_dsid(input_dir: str, dsid_map_path: str = None) -> dict:
    """
    Assemble {source_prefix: dsid} from auto-extraction plus an explicit map.

    Sources that resolve to no DSID are omitted (they will get the default
    weight at histogram time).
    """
    source_to_dsid = {}

    # 1. Auto-extract from filenames that carry a canonical DSID.
    for prefix in _scan_source_prefixes(input_dir):
        dsid = extract_dsid_from_url(prefix)
        if dsid is not None:
            source_to_dsid[prefix] = dsid

    # 2. Explicit map (wins on conflict); values may be DSIDs or URLs.
    if dsid_map_path:
        with open(dsid_map_path, "r") as f:
            explicit = json.load(f)
        for source, value in explicit.items():
            prefix = source_prefix_from_signature(source)
            dsid = _resolve_dsid(value)
            if dsid is None:
                logging.warning("Could not resolve DSID for source '%s' (value=%r)", source, value)
                continue
            source_to_dsid[prefix] = dsid

    return source_to_dsid


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build MC weights_registry.json")
    parser.add_argument("--config", default="config.yaml", help="Pipeline YAML config")
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory with parsed/IM files; registry is written here",
    )
    parser.add_argument(
        "--dsid-map", default=None,
        help="Optional JSON {source_prefix: dsid_or_url} for files without a DSID in the name",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path (default: <input-dir>/weights_registry.json)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    weighting_config = _load_weighting_config(args.config)
    if not weighting_config.enabled:
        logging.warning(
            "mc_weighting_config.enabled is false; building the registry anyway "
            "(the histogram stage will only use it once enabled)."
        )

    source_to_dsid = build_source_to_dsid(args.input_dir, args.dsid_map)
    if not source_to_dsid:
        logging.error(
            "No source->DSID mappings resolved. For hash-named files, supply --dsid-map."
        )
        return 1

    output_path = args.output or os.path.join(args.input_dir, "weights_registry.json")
    fetcher = MetadataFetcher()
    build_and_save_registry(source_to_dsid, fetcher, weighting_config, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
