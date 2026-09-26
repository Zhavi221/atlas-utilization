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


DEFAULT_OVERLAP_REMOVAL_CUTS: Dict[str, Any] = {
    # Step 1: e-jet
    "e_jet_dr": 0.2,
    # Step 2: mu-jet
    "mu_jet_dr": 0.2,
    "mu_jet_use_track_condition": True,
    "mu_jet_track_field": "NumTrkPt500",
    "mu_jet_track_min": 3,        # jet kept as "real" (not removed) if n_track >= this...
    "mu_jet_pt_ratio_max": 0.7,   # ...or pT_mu / pT_jet <= this
    # Step 3: lepton-jet sliding cone (pT must be in GeV for this formula;
    # this pipeline stores pT in MeV, hence the /1000 conversion below)
    "lepton_jet_dr_fixed": 0.4,
    "lepton_jet_dr_pt_offset": 0.04,
    "lepton_jet_dr_pt_coeff_gev": 10.0,
    # Step 4/5/6
    "photon_jet_dr": 0.2,
    "photon_electron_dr": 0.1,
    "tau_electron_dr": 0.1,
}


def _delta_r(eta1: ak.Array, phi1: ak.Array, eta2: ak.Array, phi2: ak.Array) -> ak.Array:
    dphi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt((eta1 - eta2) ** 2 + dphi ** 2)


def _pair_dr(a: ak.Array, b: ak.Array):
    """Cross ``a[event][i]`` with ``b[event][j]``; returns (a, b, dr) each shaped [event][i][j]."""
    pairs = ak.cartesian({"a": a, "b": b}, axis=1, nested=True)
    dr = _delta_r(pairs.a.eta, pairs.a.phi, pairs.b.eta, pairs.b.phi)
    return pairs.a, pairs.b, dr


def _has_particles(collection: Optional[ak.Array]) -> bool:
    return collection is not None and len(collection.fields) > 0


def apply_overlap_removal(
    events: ak.Array,
    cuts: Optional[Dict[str, Any]] = None,
) -> ak.Array:
    """
    ATLAS-style overlap removal, following Table 2 of arXiv:1606.03903.

    Applied to the baseline objects surviving the parsing-stage kinematic
    cuts, in this order (each step uses the objects surviving the previous
    one):

      1. e-jet   (ΔR<0.2): drop the jet.
      2. mu-jet  (ΔR<0.2): drop the jet, unless (optionally) it looks like a
         real jet: n_track >= ``mu_jet_track_min`` or
         pT_mu/pT_jet <= ``mu_jet_pt_ratio_max``.
      3. lepton-jet (sliding ΔR < min(0.4, 0.04 + 10 GeV/pT_lepton)): drop
         the *lepton* (electron or muon), using the jets surviving 1-2.
      4. photon-jet (ΔR<0.2): drop the jet.
      5. photon-electron (ΔR<0.1): drop the photon.
      6. tau-electron (ΔR<0.1): drop the tau.

    ``Jets`` in this pipeline already excludes b-tagged jets whenever jet
    tagging is enabled (they are split into ``BJets`` at parse time -- see
    ``FileParser._calculate_btagging_and_split``), so steps 1-2 naturally
    implement the paper's "jet not b-tagged" condition without needing a
    b-tag flag here; ``BJets`` itself is left untouched.

    The e-mu step (ΔR<0.01, calo-tagged muon only) is intentionally skipped:
    this pipeline does not currently flag calo-tagged muons, the effect is
    rare, and it has no bearing on electron-jet overlap.

    Missing collections (e.g. no Photons/Taus on a given file) are skipped.
    Assumes ``pt`` is in MeV, consistent with the rest of this pipeline.
    """
    logger = logging.getLogger(__name__)

    if "Jets" not in events.fields:
        return events

    cfg = {**DEFAULT_OVERLAP_REMOVAL_CUTS, **(cuts or {})}

    electrons = events["Electrons"] if "Electrons" in events.fields else None
    muons = events["Muons"] if "Muons" in events.fields else None
    jets = events["Jets"]
    photons = events["Photons"] if "Photons" in events.fields else None
    taus = events["Taus"] if "Taus" in events.fields else None

    # 1. e-jet: drop the jet
    if _has_particles(electrons) and _has_particles(jets):
        _, _, dr = _pair_dr(jets, electrons)
        jets = jets[~ak.any(dr < cfg["e_jet_dr"], axis=2)]

    # 2. mu-jet: drop the jet, unless it looks like a genuine jet
    if _has_particles(muons) and _has_particles(jets):
        j, m, dr = _pair_dr(jets, muons)
        overlap = dr < cfg["mu_jet_dr"]

        if cfg["mu_jet_use_track_condition"]:
            track_field = cfg["mu_jet_track_field"]
            if track_field in jets.fields:
                n_track = getattr(j, track_field)
                pt_ratio = m.pt / j.pt
                looks_real = (n_track >= cfg["mu_jet_track_min"]) | (
                    pt_ratio <= cfg["mu_jet_pt_ratio_max"]
                )
                overlap = overlap & ~looks_real
            else:
                logger.warning(
                    "Overlap removal: Jets field '%s' not available; "
                    "mu-jet step falls back to ΔR-only (no track/pT-ratio veto).",
                    track_field,
                )

        jets = jets[~ak.any(overlap, axis=2)]

    # 3. lepton-jet: sliding cone, drop the lepton
    def _drop_leptons_near_jets(leptons: Optional[ak.Array]) -> Optional[ak.Array]:
        if not _has_particles(leptons) or not _has_particles(jets):
            return leptons
        lep, _, dr = _pair_dr(leptons, jets)
        pt_gev = lep.pt / 1000.0
        cone = np.minimum(
            cfg["lepton_jet_dr_fixed"],
            cfg["lepton_jet_dr_pt_offset"] + cfg["lepton_jet_dr_pt_coeff_gev"] / pt_gev,
        )
        return leptons[~ak.any(dr < cone, axis=2)]

    electrons = _drop_leptons_near_jets(electrons)
    muons = _drop_leptons_near_jets(muons)

    # 4. photon-jet: drop the jet
    if _has_particles(photons) and _has_particles(jets):
        _, _, dr = _pair_dr(jets, photons)
        jets = jets[~ak.any(dr < cfg["photon_jet_dr"], axis=2)]

    # 5. photon-electron: drop the photon
    if _has_particles(photons) and _has_particles(electrons):
        _, _, dr = _pair_dr(photons, electrons)
        photons = photons[~ak.any(dr < cfg["photon_electron_dr"], axis=2)]

    # 6. tau-electron: drop the tau
    if _has_particles(taus) and _has_particles(electrons):
        _, _, dr = _pair_dr(taus, electrons)
        taus = taus[~ak.any(dr < cfg["tau_electron_dr"], axis=2)]

    updated = {field: events[field] for field in events.fields}
    if electrons is not None:
        updated["Electrons"] = electrons
    if muons is not None:
        updated["Muons"] = muons
    updated["Jets"] = jets
    if photons is not None:
        updated["Photons"] = photons
    if taus is not None:
        updated["Taus"] = taus

    return ak.zip(updated, depth_limit=1)


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
