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

    return events


def filter_events_by_non_jet_object_total(events: ak.Array) -> ak.Array:
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



DEFAULT_OVERLAP_REMOVAL_CUTS: Dict[str, Any] = {
    # Steps 1-2: e-mu (proxy for shared-track; PHYSLITE lacks track/calo info)
    "e_mu_dr": 0.1,
    # Step 3: e-jet
    "e_jet_dr": 0.2,
    # Step 5: mu-jet
    "mu_jet_dr": 0.2,
    "mu_jet_use_track_condition": True,
    "mu_jet_track_field": "NumTrkPt500",
    "mu_jet_track_min": 3,
    "mu_jet_pt_ratio_max": 0.7,
    # Steps 4,6: lepton vs jet (fixed cone, replaces sliding cone)
    #lepton-jet sliding cone (pT must be in GeV for this formula;
    # this pipeline stores pT in MeV, hence the /1000 conversion below)
    "lepton_jet_dr_fixed": 0.4,
    "lepton_jet_dr_pt_offset": 0.04,
    "lepton_jet_dr_pt_coeff_gev": 10.0,
    "lepton_near_jet_dr": 0.4,
    "photon_lepton_dr": 0.4,
    "photon_jet_dr": 0.4,
    "tau_electron_dr": 0.1,
}


def _pairs(a: ak.Array, b: ak.Array) -> tuple[ak.Array, ak.Array]:
    """Every (a_i, b_j) pair per event, each side shaped [event][a_i][b_j]."""
    return ak.unzip(ak.cartesian([a, b], axis=1, nested=True))


def _delta_r(a_side: ak.Array, b_side: ak.Array) -> ak.Array:
    """Element-wise ΔR between two equally shaped arrays of eta/phi records."""
    deta = a_side.eta - b_side.eta
    dphi = (a_side.phi - b_side.phi + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt(deta ** 2 + dphi ** 2)


def _overlap_mask(a: ak.Array, b: ak.Array, threshold: float) -> ak.Array:
    """Per-object [event][a_i] bool: True if any b_j within *threshold*."""
    a_side, b_side = _pairs(a, b)
    return ak.any(_delta_r(a_side, b_side) < threshold, axis=2)


def _overlap_mask_with_veto(
    jets: ak.Array, muons: ak.Array, dr_threshold: float,
    track_field: str, track_min: int, pt_ratio_max: float,
    use_track: bool, logger,
) -> ak.Array:
    """mu-jet overlap mask with optional track/pT-ratio veto."""
    has_track = use_track and track_field in jets.fields
    if use_track and not has_track:
        logger.warning(
            "Overlap removal: Jets field '%s' not available; "
            "mu-jet step falls back to ΔR-only (no track/pT-ratio veto).",
            track_field,
        )
    jet_side, muon_side = _pairs(jets, muons)
    close = _delta_r(jet_side, muon_side) < dr_threshold
    if has_track:
        ratio = muon_side.pt / jet_side.pt
        looks_real = (jet_side[track_field] >= track_min) | (ratio <= pt_ratio_max)
        close = close & ~looks_real
    return ak.any(close, axis=2)

#for now we are not using this but keep it
def _sliding_cone_mask(leptons: ak.Array, jets: ak.Array, cfg: dict) -> ak.Array:
    """Lepton-jet overlap with pT-dependent sliding cone."""
    lepton_side, jet_side = _pairs(leptons, jets)
    pt_gev = lepton_side.pt / 1000.0
    cone = np.minimum(
        cfg["lepton_jet_dr_fixed"],
        cfg["lepton_jet_dr_pt_offset"] + cfg["lepton_jet_dr_pt_coeff_gev"] / pt_gev,
    )
    return ak.any(_delta_r(lepton_side, jet_side) < cone, axis=2)


def _has_particles(collection: Optional[ak.Array]) -> bool:
    return (collection is not None
            and len(collection.fields) > 0
            and int(ak.sum(ak.num(collection))) > 0)

def apply_overlap_removal(
    events: ak.Array,
    cuts: Optional[Dict[str, Any]] = None,
) -> ak.Array:
    """
    ATLAS-style overlap removal, following Table 4.1.
    Assumes ``pt`` is in MeV, consistent with the rest of this pipeline.
    """
    logger = logging.getLogger(__name__)

    cfg = {**DEFAULT_OVERLAP_REMOVAL_CUTS, **(cuts or {})}

    electrons = events["Electrons"] if "Electrons" in events.fields else None
    muons = events["Muons"] if "Muons" in events.fields else None
    jets = events["Jets"] if "Jets" in events.fields else None
    photons = events["Photons"] if "Photons" in events.fields else None
    taus = events["Taus"] if "Taus" in events.fields else None

    # 0. e-mu: drop muon (proxy for shared-track, Table 4.1 steps 1-2)
    if _has_particles(electrons) and _has_particles(muons):
        muons = muons[~_overlap_mask(muons, electrons, cfg["e_mu_dr"])]

    # 1. e-jet: drop the jet (step 3)
    if _has_particles(electrons) and _has_particles(jets):
        jets = jets[~_overlap_mask(jets, electrons, cfg["e_jet_dr"])]

    # 2. electron vs jet: drop the electron (step 4, fixed dR<0.4)
    if _has_particles(electrons) and _has_particles(jets):
        electrons = electrons[~_overlap_mask(electrons, jets, cfg["lepton_near_jet_dr"])]

    # 3. mu-jet: drop the jet (step 5)
    if _has_particles(muons) and _has_particles(jets):
        jets = jets[~_overlap_mask_with_veto(
            jets, muons, cfg["mu_jet_dr"],
            track_field=cfg["mu_jet_track_field"],
            track_min=cfg["mu_jet_track_min"],
            pt_ratio_max=cfg["mu_jet_pt_ratio_max"],
            use_track=cfg["mu_jet_use_track_condition"],
            logger=logger,
        )]

    # 4. muon vs jet: drop the muon (step 6, fixed dR<0.4)
    if _has_particles(muons) and _has_particles(jets):
        muons = muons[~_overlap_mask(muons, jets, cfg["lepton_near_jet_dr"])]

    # 5. photon vs electron and muon: drop the photon (step 7)
    if _has_particles(photons) and _has_particles(electrons):
        photons = photons[~_overlap_mask(photons, electrons, cfg["photon_lepton_dr"])]
    if _has_particles(photons) and _has_particles(muons):
        photons = photons[~_overlap_mask(photons, muons, cfg["photon_lepton_dr"])]

    # 6. jet vs photon: drop the jet (step 8)
    if _has_particles(photons) and _has_particles(jets):
        jets = jets[~_overlap_mask(jets, photons, cfg["photon_jet_dr"])]

    # 7. tau-electron: drop the tau
    if _has_particles(taus) and _has_particles(electrons):
        taus = taus[~_overlap_mask(taus, electrons, cfg["tau_electron_dr"])]

    result = events
    if electrons is not None:
        result = ak.with_field(result, electrons, "Electrons")
    if muons is not None:
        result = ak.with_field(result, muons, "Muons")
    if jets is not None:
        if "NumTrkPt500" in jets.fields:
            jets = ak.zip({f: jets[f] for f in jets.fields if f != "NumTrkPt500"})
        result = ak.with_field(result, jets, "Jets")
    if photons is not None:
        result = ak.with_field(result, photons, "Photons")
    if taus is not None:
        result = ak.with_field(result, taus, "Taus")
    return result

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
