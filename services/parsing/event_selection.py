"""
Apply parsing-stage event filters from YAML (particle count ranges + kinematic cuts).

Maps YAML keys (e.g. ``electrons``) to awkward record fields (``Electrons``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import awkward as ak
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
) -> ak.Array:
    """
    Kinematic cuts are applied per particle type first, then event-level count ranges.
    """
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
            mapped[cname] = val
        events = physics_calcs.filter_events_by_particle_counts(
            events,
            mapped,
            is_exact_count=False,
            is_particle_counts_range=True,
        )

    return events


def apply_trigger_selection(
    events: ak.Array,
    release_year: str = "2024r-pp",
) -> ak.Array:
    """
    Keep only events where at least one lepton fired a single-lepton trigger.

    The ``_triggerMatch`` field (if present) is a record of per-event booleans,
    one per trigger chain.  An event passes if ANY electron chain OR ANY muon
    chain is True.

    For MC (``release_year`` ending in ``_mc``), all Run-2 year chains are
    OR'd together since the MC covers the full period.

    Returns the filtered events array (``_triggerMatch`` field is dropped
    from the output to avoid downstream issues with non-particle fields).
    """
    if "_triggerMatch" not in events.fields:
        return events

    trig = events["_triggerMatch"]
    chain_defs = schemas.SINGLE_LEPTON_TRIGGER_CHAINS

    # Collect all electron and muon chains across years (OR within flavour)
    e_chains = set()
    mu_chains = set()
    for year_chains in chain_defs.values():
        e_chains.update(year_chains.get("Electrons", []))
        mu_chains.update(year_chains.get("Muons", []))

    # Build per-event boolean: did any electron chain fire?
    electron_pass = ak.zeros_like(ak.Array([False] * len(events)))
    for chain in e_chains:
        full_branch = chain + schemas.TRIGGER_BRANCH_SUFFIX
        if full_branch in trig.fields:
            electron_pass = electron_pass | trig[full_branch]

    # Did any muon chain fire?
    muon_pass = ak.zeros_like(ak.Array([False] * len(events)))
    for chain in mu_chains:
        full_branch = chain + schemas.TRIGGER_BRANCH_SUFFIX
        if full_branch in trig.fields:
            muon_pass = muon_pass | trig[full_branch]

    # Event passes if any lepton trigger fired
    event_mask = electron_pass | muon_pass

    # Log trigger efficiency
    n_total = len(events)
    n_pass = int(ak.sum(event_mask))
    import logging
    logger = logging.getLogger(__name__)
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
    particle_fields = {f: filtered[f] for f in filtered.fields if f != "_triggerMatch"}
    return ak.zip(particle_fields, depth_limit=1)
