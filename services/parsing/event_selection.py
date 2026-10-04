"""
Apply parsing-stage event filters from YAML (particle count ranges + kinematic cuts).

Maps YAML keys (e.g. ``electrons``) to awkward record fields (``Electrons``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import awkward as ak
import logging
import numpy as np

from services.calculations import physics_calcs
from services.parsing import schemas

YAML_PARTICLE_KEYS: Dict[str, str] = {
    "electrons": "Electrons",
    "muons": "Muons",
    "jets": "Jets",
    "bjets": "BJets",
    "photons": "Photons",
    "taus": "Taus",
}

LIGHT_JET_FIELD = "Jets"
MAX_NON_JET_OBJECTS = 4


def canonical_particle_field_name(key: str) -> str:
    return YAML_PARTICLE_KEYS.get(key.lower(), key)


def normalize_yaml_kinematic_cuts(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Turn ``pt_min`` / ``eta_max`` / ``rel_isolation_max`` into internal cut dict."""
    out: Dict[str, Any] = {}
    if "pt" in raw and isinstance(raw["pt"], dict):
        out["pt"] = dict(raw["pt"])
    elif "pt_min" in raw:
        out["pt"] = {"min": float(raw["pt_min"])}

    if "eta" in raw and isinstance(raw["eta"], dict):
        out["eta"] = dict(raw["eta"])
    elif "eta_max" in raw:
        em = float(raw["eta_max"])
        out["eta"] = {"min": -em, "max": em}

    if "phi" in raw and isinstance(raw["phi"], dict):
        out["phi"] = dict(raw["phi"])
    elif "phi_min" in raw or "phi_max" in raw:
        out["phi"] = {
            "min": float(raw.get("phi_min", -np.pi)),
            "max": float(raw.get("phi_max", np.pi)),
        }

    if "rel_isolation_max" in raw:
        out["rel_isolation_max"] = float(raw["rel_isolation_max"])

    return out


def apply_parsing_event_selection(
    events: ak.Array,
    particle_counts: Optional[Dict[str, Any]] = None,
    kinematic_cuts: Optional[Dict[str, Any]] = None,
    allowed_objects: Optional[tuple[str, ...]] = None,
) -> ak.Array:
    """
    Retain configured objects, apply per-object kinematic cuts, then apply
    event-level count ranges.  Events may contain unconfigured objects; those
    collections are discarded and do not participate in event selection.

    At most four retained non-light-jet objects are allowed per event.  Light
    jets are deliberately excluded from this total and have no upper bound.
    """
    if allowed_objects is not None:
        events = retain_objects_for_storage(events, allowed_objects)

    if kinematic_cuts:
        by_obj: Dict[str, Dict[str, Any]] = {}
        for key, val in kinematic_cuts.items():
            if not isinstance(val, dict):
                continue
            cname = canonical_particle_field_name(key)
            by_obj[cname] = normalize_yaml_kinematic_cuts(val)
        events = physics_calcs.filter_events_by_kinematics(events, by_obj)

    if particle_counts:
        mapped: Dict[str, Any] = {}
        for key, val in particle_counts.items():
            cname = canonical_particle_field_name(key)
            if cname == LIGHT_JET_FIELD and isinstance(val, dict):
                # Additional light jets must not reject an otherwise valid
                # final state.  Keep an optional lower bound, but remove any
                # configured upper bound.
                mapped[cname] = {**val, "max": float("inf")}
            else:
                mapped[cname] = val

    else:
        mapped = {}

    if mapped:
        events = physics_calcs.filter_events_by_particle_counts(
            events,
            mapped,
            is_exact_count=False,
            is_particle_counts_range=True,
        )

    return _filter_events_by_non_jet_object_total(events)


def _filter_events_by_non_jet_object_total(events: ak.Array) -> ak.Array:
    """Reject events with more than four retained non-light-jet objects."""
    non_jet_fields = [field for field in events.fields if field != LIGHT_JET_FIELD]
    if not non_jet_fields:
        return events

    total = ak.zeros_like(ak.num(events[non_jet_fields[0]]), dtype=np.int64)
    for field in non_jet_fields:
        total = total + ak.num(events[field])
    return events[total <= MAX_NON_JET_OBJECTS]


def retain_objects_for_storage(
    events: ak.Array, objects_to_store: tuple[str, ...]
) -> ak.Array:
    """Drop non-persisted physics collections before event selection."""
    return ak.zip(
        {
            field: events[field]
            for field in events.fields
            if field in objects_to_store
        },
        depth_limit=1,
    )


def apply_trigger_selection(
    events: ak.Array,
    release_year: str = "2024r-pp",
    file_path: str = "",
) -> ak.Array:
    """
    Keep only events where at least one lepton fired a single-lepton trigger.

    The ``_triggerMatch`` field (if present) is a record of per-event booleans,
    one per trigger chain.  An event passes if ANY electron chain OR ANY muon
    chain is True.

    Returns the filtered events array (``_triggerMatch`` field is dropped
    from the output to avoid downstream issues with non-particle fields).
    """
    logger = logging.getLogger(__name__)

    if "_triggerMatch" not in events.fields:
        logger.warning(
            "Skipping file %s: no trigger-match branches on file (%d events dropped)",
            file_path, len(events),
        )
        return events[:0]

    trig = events["_triggerMatch"]
    chain_defs = schemas.SINGLE_LEPTON_TRIGGER_CHAINS

    if "_runNumber" in events.fields:
        # MC: each event's trigger year is set by its random run number
        rrn = events["_runNumber"]
        trigger_years = [
            (year, (rrn >= lo) & (rrn <= hi))
            for year, (lo, hi) in schemas.YEAR_RUN_RANGES.items()
        ]
    else:
        # Data: the file's year applies to every event
        trigger_years = [
            (year, True) for year in schemas.get_trigger_years(release_year, file_path)
        ]

    # Build per-event booleans: did any electron / muon chain of the event's year fire?
    electron_pass = ak.zeros_like(ak.Array([False] * len(events)))
    muon_pass = ak.zeros_like(ak.Array([False] * len(events)))
    for year, in_year in trigger_years:
        if year not in chain_defs:
            raise ValueError(
                f"No trigger chains defined for year '{year}'. "
                f"Supported: {sorted(chain_defs.keys())}"
            )
        year_chains = chain_defs[year]
        for chain in year_chains.get("Electrons", []):
            full_branch = chain + schemas.TRIGGER_BRANCH_SUFFIX
            if full_branch in trig.fields:
                electron_pass = electron_pass | (in_year & trig[full_branch])
        for chain in year_chains.get("Muons", []):
            full_branch = chain + schemas.TRIGGER_BRANCH_SUFFIX
            if full_branch in trig.fields:
                muon_pass = muon_pass | (in_year & trig[full_branch])

    # Event passes if any lepton trigger fired
    event_mask = electron_pass | muon_pass

    # Log trigger efficiency
    n_total = len(events)
    n_pass = int(ak.sum(event_mask))
    logger.info(
        "Trigger selection: %d / %d events pass (%.1f%%), "
        "electron-only: %d, muon-only: %d",
        n_pass, n_total, 100 * n_pass / n_total if n_total else 0,
        int(ak.sum(electron_pass & ~muon_pass)),
        int(ak.sum(muon_pass & ~electron_pass)),
    )

    filtered = events[event_mask]

    # Drop _triggerMatch from the output — it's event-level metadata that
    # would cause axis errors in downstream particle-level operations
    particle_fields = {f: filtered[f] for f in filtered.fields if f not in ("_triggerMatch", "_runNumber")}
    return ak.zip(particle_fields, depth_limit=1)
