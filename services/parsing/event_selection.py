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
    # Steps unchanged from before
    "photon_jet_dr": 0.2,
    "photon_electron_dr": 0.1,
    "tau_electron_dr": 0.1,
}


def _delta_r_np(eta1, phi1, eta2, phi2):
    """NumPy ΔR between [n_a] and [n_b] arrays, returns [n_a, n_b]."""
    deta = eta1[:, None] - eta2[None, :]
    dphi = (phi1[:, None] - phi2[None, :] + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt(deta ** 2 + dphi ** 2)


def _overlap_mask(a: ak.Array, b: ak.Array, threshold: float) -> ak.Array:
    """Per-object [event][a_i] bool: True if any b_j within *threshold*."""
    a_counts = ak.to_numpy(ak.num(a))
    b_counts = ak.to_numpy(ak.num(b))
    a_eta = np.asarray(ak.flatten(a.eta, axis=None))
    a_phi = np.asarray(ak.flatten(a.phi, axis=None))
    b_eta = np.asarray(ak.flatten(b.eta, axis=None))
    b_phi = np.asarray(ak.flatten(b.phi, axis=None))
    result = []
    ao, bo = 0, 0
    for na, nb in zip(a_counts, b_counts):
        if na == 0 or nb == 0:
            result.append([False] * na)
        else:
            dr = _delta_r_np(a_eta[ao:ao+na], a_phi[ao:ao+na],
                             b_eta[bo:bo+nb], b_phi[bo:bo+nb])
            result.append((dr < threshold).any(axis=1).tolist())
        ao += na
        bo += nb
    return ak.Array(result)


def _overlap_mask_with_veto(
    jets: ak.Array, muons: ak.Array, dr_threshold: float,
    track_field: str, track_min: int, pt_ratio_max: float,
    use_track: bool, logger,
) -> ak.Array:
    """mu-jet overlap mask with optional track/pT-ratio veto."""
    j_counts = ak.to_numpy(ak.num(jets))
    m_counts = ak.to_numpy(ak.num(muons))
    j_eta = np.asarray(ak.flatten(jets.eta, axis=None))
    j_phi = np.asarray(ak.flatten(jets.phi, axis=None))
    j_pt  = np.asarray(ak.flatten(jets.pt, axis=None))
    m_eta = np.asarray(ak.flatten(muons.eta, axis=None))
    m_phi = np.asarray(ak.flatten(muons.phi, axis=None))
    m_pt  = np.asarray(ak.flatten(muons.pt, axis=None))
    has_track = use_track and track_field in jets.fields
    if use_track and not has_track:
        logger.warning(
            "Overlap removal: Jets field '%s' not available; "
            "mu-jet step falls back to ΔR-only (no track/pT-ratio veto).",
            track_field,
        )
    j_ntrk = np.asarray(ak.flatten(jets[track_field], axis=None)) if has_track else None
    result = []
    jo, mo = 0, 0
    for nj, nm in zip(j_counts, m_counts):
        if nj == 0 or nm == 0:
            result.append([False] * nj)
        else:
            dr = _delta_r_np(j_eta[jo:jo+nj], j_phi[jo:jo+nj],
                             m_eta[mo:mo+nm], m_phi[mo:mo+nm])
            close = dr < dr_threshold
            if has_track:
                nt = j_ntrk[jo:jo+nj]
                ratio = m_pt[mo:mo+nm][None, :] / j_pt[jo:jo+nj][:, None]
                looks_real = (nt[:, None] >= track_min) | (ratio <= pt_ratio_max)
                close = close & ~looks_real
            result.append(close.any(axis=1).tolist())
        jo += nj
        mo += nm
    return ak.Array(result)

#for now we are not using this but keep it
def _sliding_cone_mask(leptons: ak.Array, jets: ak.Array, cfg: dict) -> ak.Array:
    """Lepton-jet overlap with pT-dependent sliding cone."""
    l_counts = ak.to_numpy(ak.num(leptons))
    j_counts = ak.to_numpy(ak.num(jets))
    l_eta = np.asarray(ak.flatten(leptons.eta, axis=None))
    l_phi = np.asarray(ak.flatten(leptons.phi, axis=None))
    l_pt  = np.asarray(ak.flatten(leptons.pt, axis=None))
    j_eta = np.asarray(ak.flatten(jets.eta, axis=None))
    j_phi = np.asarray(ak.flatten(jets.phi, axis=None))
    result = []
    lo, jo = 0, 0
    for nl, nj in zip(l_counts, j_counts):
        if nl == 0 or nj == 0:
            result.append([False] * nl)
        else:
            dr = _delta_r_np(l_eta[lo:lo+nl], l_phi[lo:lo+nl],
                             j_eta[jo:jo+nj], j_phi[jo:jo+nj])
            pt_gev = l_pt[lo:lo+nl] / 1000.0
            cone = np.minimum(
                cfg["lepton_jet_dr_fixed"],
                cfg["lepton_jet_dr_pt_offset"] + cfg["lepton_jet_dr_pt_coeff_gev"] / pt_gev,
            )
            result.append((dr < cone[:, None]).any(axis=1).tolist())
        lo += nl
        jo += nj
    return ak.Array(result)


def _has_particles(collection: Optional[ak.Array]) -> bool:
    return (collection is not None
            and len(collection.fields) > 0
            and int(ak.sum(ak.num(collection))) > 0)

def apply_overlap_removal(
    events: ak.Array,
    cuts: Optional[Dict[str, Any]] = None,
) -> ak.Array:
    """
    ATLAS-style overlap removal, following Table 2 of arXiv:1606.03903.
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

    # 5. photon-jet: drop the jet (unchanged)
    if _has_particles(photons) and _has_particles(jets):
        jets = jets[~_overlap_mask(jets, photons, cfg["photon_jet_dr"])]

    # 6. photon-electron: drop the photon (unchanged)
    if _has_particles(photons) and _has_particles(electrons):
        photons = photons[~_overlap_mask(photons, electrons, cfg["photon_electron_dr"])]

    # 7. tau-electron: drop the tau (unchanged)
    if _has_particles(taus) and _has_particles(electrons):
        taus = taus[~_overlap_mask(taus, electrons, cfg["tau_electron_dr"])]

    result = events
    if electrons is not None:
        result = ak.with_field(result, electrons, "Electrons")
    if muons is not None:
        result = ak.with_field(result, muons, "Muons")
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
