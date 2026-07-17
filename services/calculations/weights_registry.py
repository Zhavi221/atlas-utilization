"""
MC weight registry: maps a parsed source file to its per-dataset weight.

For the per-file (leading-order) weighting scheme, every event from a given
simulated dataset receives the same normalization weight:

    w = (cross_section_pb * 1000 * kFactor * genFiltEff * L) / sumOfWeights

so the weight only needs to be known per *source file* (one DSID per file),
not per event. This module builds, persists, and resolves that
``source_prefix -> weight`` mapping.

Why a registry keyed by source prefix:
    Invariant-mass signatures keep the originating filename as a prefix
    (``<prefix>_FS_<fs>_IM_<im>[_main|_outliers]``). That prefix survives
    combinatorics, storage, and post-processing, so it is the one piece of
    dataset identity still available at histogram-fill time. We look the
    weight up by that prefix and fill histograms with it.

DSID recovery:
    The DSID lives in the ATLAS Open Data file URL, not in the (hashed)
    parsed filename. The mapping ``source_prefix -> DSID`` must therefore be
    captured at parse/fetch time (when the URL is known) via
    :func:`extract_dsid_from_url`, then combined with fetched dataset metadata
    to produce weights.
"""

import json
import logging
import os
import re
from typing import Optional

# Match the 6-digit ATLAS DSID only in its canonical position, e.g.
# "mc20_13TeV.410470.PhPy8EG_..." -> "410470". Deliberately strict: a wrong
# DSID would silently produce a wrong weight, so anything that isn't clearly a
# DSID (rucio container numbers, file-sequence indices like _000001) must
# return None and fall back to the default weight instead.
_DSID_PATTERNS = (
    re.compile(r"[Tt]e[Vv]\.(\d{6})\."),      # ...TeV.410470.  (canonical)
    re.compile(r"\.(\d{6})\.[A-Za-z]"),        # .410470.PhPy8... (DSID before physics_short)
)

# Source prefix = everything before the "_FS_" marker in a signature/filename.
_SOURCE_PREFIX_RE = re.compile(r"^(.*?)_FS_")


def extract_dsid_from_url(url_or_name: str) -> Optional[int]:
    """
    Extract the dataset number (DSID) from an ATLAS Open Data URL or filename.

    Returns the DSID as an int, or None if no DSID-like token is found.
    """
    if not url_or_name:
        return None
    for pattern in _DSID_PATTERNS:
        match = pattern.search(url_or_name)
        if match:
            return int(match.group(1))
    return None


def source_prefix_from_signature(signature: str) -> str:
    """
    Extract the source-file prefix from an IM signature or .npy filename.

    ``2024r-pp_abc123_FS_1e_1m_2j_IM_1e_1m_2j_main`` -> ``2024r-pp_abc123``.
    Falls back to the input (minus a trailing .npy) if no ``_FS_`` marker.
    """
    name = signature[:-4] if signature.endswith(".npy") else signature
    match = _SOURCE_PREFIX_RE.match(name)
    if match:
        return match.group(1)
    return name


class WeightsRegistry:
    """
    Resolves a MC normalization weight for a given IM signature / source file.

    Unknown source prefixes resolve to ``default_weight`` (1.0) so that
    unweighted data flows through unchanged; each unknown prefix is warned
    about once.
    """

    def __init__(self, weights_by_prefix: dict, default_weight: float = 1.0):
        # Normalize keys to plain source prefixes so lookups are robust whether
        # callers pass a full signature or a bare prefix.
        self._weights = {
            source_prefix_from_signature(k): float(v)
            for k, v in weights_by_prefix.items()
        }
        self.default_weight = float(default_weight)
        self._warned_prefixes = set()

    def __len__(self) -> int:
        return len(self._weights)

    def weight_for(self, signature_or_prefix: str) -> float:
        """Resolve the weight for a signature, filename, or bare source prefix."""
        prefix = source_prefix_from_signature(signature_or_prefix)
        if prefix in self._weights:
            return self._weights[prefix]
        if prefix not in self._warned_prefixes:
            logging.warning(
                "No MC weight for source '%s'; using default weight %s",
                prefix, self.default_weight,
            )
            self._warned_prefixes.add(prefix)
        return self.default_weight

    def save(self, path: str) -> None:
        """Persist the registry to JSON."""
        with open(path, "w") as f:
            json.dump(
                {"default_weight": self.default_weight, "weights": self._weights},
                f, indent=2,
            )

    @classmethod
    def load(cls, path: str) -> Optional["WeightsRegistry"]:
        """Load a registry from JSON, or None if the file is missing/invalid."""
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return cls(data.get("weights", {}), data.get("default_weight", 1.0))
        except (json.JSONDecodeError, IOError) as e:
            logging.warning("Failed to load weights registry from %s: %s", path, e)
            return None

    @classmethod
    def build_from_metadata(
        cls,
        source_to_dsid: dict,
        metadata_by_dsid: dict,
        weighting_config,
        source_to_campaign: Optional[dict] = None,
        default_weight: float = 1.0,
    ) -> "WeightsRegistry":
        """
        Build a registry from source->DSID mapping and fetched dataset metadata.

        Args:
            source_to_dsid: {source_prefix: dsid} captured at parse/fetch time.
            metadata_by_dsid: {dsid: MCDatasetMetadata} from the fetcher.
            weighting_config: MCWeightingConfig (provides target luminosity,
                including per-campaign values via get_luminosity()).
            source_to_campaign: Optional {source_prefix: campaign} for
                campaign-specific luminosity.
            default_weight: Weight for sources without resolvable metadata.

        Returns:
            A WeightsRegistry. Sources whose DSID has no metadata are omitted
            (they will resolve to default_weight at lookup time).
        """
        from services.calculations.mc_weights import compute_normalization

        source_to_campaign = source_to_campaign or {}
        weights = {}
        for source_prefix, dsid in source_to_dsid.items():
            metadata = metadata_by_dsid.get(dsid)
            if metadata is None:
                logging.warning(
                    "No metadata for DSID %s (source '%s'); will use default weight",
                    dsid, source_prefix,
                )
                continue
            campaign = source_to_campaign.get(source_prefix)
            luminosity = weighting_config.get_luminosity(campaign)
            weights[source_prefix] = compute_normalization(metadata, luminosity)

        return cls(weights, default_weight=default_weight)


def build_and_save_registry(
    source_to_dsid: dict,
    fetcher,
    weighting_config,
    output_path: str,
    source_to_campaign: Optional[dict] = None,
) -> "WeightsRegistry":
    """
    Fetch metadata for the given sources' DSIDs, build a weights registry, and
    persist it to ``output_path``.

    This is the producer counterpart to the histogram-stage consumer: it runs
    where DSIDs are known and writes the ``weights_registry.json`` that the
    histogram stage later reads.

    Args:
        source_to_dsid: {source_prefix: dsid} for every source to weight.
        fetcher: A MetadataFetcher (needs fetch_mc_metadata_for_datasets).
        weighting_config: MCWeightingConfig (luminosity + require_metadata).
        output_path: Where to write weights_registry.json.
        source_to_campaign: Optional {source_prefix: campaign} for per-campaign
            luminosity.

    Returns:
        The saved WeightsRegistry.
    """
    dsids = sorted(set(source_to_dsid.values()))
    logging.info("Fetching MC metadata for %d unique DSID(s)", len(dsids))
    metadata_by_dsid = fetcher.fetch_mc_metadata_for_datasets(
        dsids, require_metadata=weighting_config.require_metadata
    )
    registry = WeightsRegistry.build_from_metadata(
        source_to_dsid, metadata_by_dsid, weighting_config, source_to_campaign
    )
    registry.save(output_path)
    logging.info(
        "Wrote weights registry with %d weighted source(s) to %s",
        len(registry), output_path,
    )
    return registry
