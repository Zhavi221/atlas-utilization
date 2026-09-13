"""
Invariant Mass Calculation Pipeline.

Processes parsed ROOT events to calculate invariant masses for particle
combinations. Used by MassCalculationHandler.
"""
import sys
import os
import logging
from typing import Dict, List, Optional, Tuple

import awkward as ak
import numpy as np

from domain.events import (
    MC_EVENT_INFO_FIELD,
    MC_EVENT_WEIGHT_FIELD,
    MC_CHANNEL_NUMBER_FIELD,
)
from services.calculations.im_calculator import IMCalculator
from services.calculations.combinatorics import get_count, get_start

# Signature suffix of the per-event MC weight array stored next to an IM array.
MC_WEIGHT_SUFFIX = "_mcw"


def process_final_state(
    final_state: str,
    fs_events: ak.Array,
    filename: str,
    all_combinations: List[Dict],
    config: Dict,
    output_dir: str,
    logger: logging.Logger,
    calculator: IMCalculator,
    worker_num: Optional[int] = None
) -> Tuple[Dict, List[str]]:
    """
    Process all combinations for a given final state.

    Returns:
        Tuple of (statistics dict, list of created artifact identifiers)
    """
    if len(fs_events) == 0:
        return None, []

    prefix = f"[Worker {worker_num}]" if worker_num is not None else ""
    num_combinations = sum(
        1 for c in all_combinations
        if calculator.does_final_state_contain_combination(final_state, c)
    )

    logger.info(
        f"{prefix} [{filename}] Computing final state '{final_state}': "
        f"{len(fs_events):,} events, {num_combinations} combinations"
    )

    fs_mapping_threshold_bytes = config["fs_chunk_threshold_bytes"]
    output_mode = config.get("output_mode", "npy")
    sqlite_writer = config.get("sqlite_writer")
    fs_im_mapping: Dict[str, Dict[str, ak.Array]] = {}

    stats = {
        'calculated': 0,
        'skipped': 0,
        'skip_reasons': {
            'no_matching_combination': 0,
            'no_events_after_filter': 0,
            'no_events_after_slice': 0,
            'empty_inv_mass': 0
        }
    }

    created_im_files = []

    milestones = set()
    if num_combinations > 1:
        milestones = {1, num_combinations // 4, num_combinations // 2,
                      3 * num_combinations // 4, num_combinations}
        milestones = {m for m in milestones if m > 0}
    elif num_combinations == 1:
        milestones = {1}

    combination_count = 0
    for combination in all_combinations:
        if not calculator.does_final_state_contain_combination(final_state, combination):
            stats['skipped'] += 1
            stats['skip_reasons']['no_matching_combination'] += 1
            continue

        combination_count += 1

        if combination_count in milestones:
            pct = (combination_count / num_combinations) * 100 if num_combinations > 0 else 0
            logger.info(
                f"{prefix} [{filename}] '{final_state}': "
                f"{combination_count}/{num_combinations} ({pct:.0f}%) - "
                f"{stats['calculated']} calculated, {stats['skipped']} skipped"
            )

        inv_mass, mc_event_weights, skip_reason = _calculate_combination_invariant_mass(
            fs_events, combination, config, calculator, logger, final_state, worker_num
        )

        if inv_mass is None:
            stats['skipped'] += 1
            if skip_reason:
                stats['skip_reasons'][skip_reason] += 1
            continue

        inv_mass = _convert_array_to_gev(inv_mass)

        stats['calculated'] += 1
        combination_name = prepare_im_combination_name(filename, final_state, combination)
        saved_files = _accumulate_invariant_mass(
            fs_im_mapping, final_state, combination_name, inv_mass,
            fs_mapping_threshold_bytes, output_dir, logger,
            output_mode=output_mode, sqlite_writer=sqlite_writer
        )
        # Store per-event MC weights as a parallel signature (_mcw suffix)
        # through the same accumulate/flush path: same length and event order
        # as the IM array, so every event carries its final weight and nothing
        # downstream needs to know its dataset.
        if mc_event_weights is not None:
            saved_files += _accumulate_invariant_mass(
                fs_im_mapping, final_state, combination_name + MC_WEIGHT_SUFFIX,
                ak.Array(mc_event_weights),
                fs_mapping_threshold_bytes, output_dir, logger,
                output_mode=output_mode, sqlite_writer=sqlite_writer
            )
        created_im_files.extend(_without_weight_arrays(saved_files))

    remaining_files = _save_remaining_accumulated_data(
        fs_im_mapping,
        output_dir,
        logger,
        output_mode=output_mode,
        sqlite_writer=sqlite_writer,
    )
    created_im_files.extend(_without_weight_arrays(remaining_files))

    logger.info(
        f"{prefix} [{filename}] Completed '{final_state}': "
        f"{stats['calculated']} calculated, {stats['skipped']} skipped"
    )

    return stats, created_im_files


def _without_weight_arrays(names: List[str]) -> List[str]:
    """Drop the MC weight siblings so only IM arrays are counted as created artifacts."""
    return [n for n in names if not n.endswith(MC_WEIGHT_SUFFIX)]


def _convert_array_to_gev(inv_mass: ak.Array) -> ak.Array:
    return inv_mass * 1e-3


def _calculate_combination_invariant_mass(
    fs_events: ak.Array,
    combination: Dict,
    config: Dict,
    calculator: IMCalculator,
    logger: logging.Logger,
    final_state: str,
    worker_num: Optional[int] = None
) -> Tuple[Optional[ak.Array], Optional[np.ndarray], Optional[str]]:
    """
    Returns:
        (inv_mass, mc_event_weights, skip_reason)
        mc_event_weights is a numpy array of final per-event MC weights
        (generator weight x dataset normalization), or None when MC
        weighting is disabled or the events carry no MC info (data).
    """
    logger.debug(f"Processing combination: {combination} for final state: {final_state}")

    filtered_events = calculator.filter_by_particle_counts(
        events=fs_events, particle_counts=combination, is_exact_count=True
    )
    if len(filtered_events) == 0:
        return None, None, 'no_events_after_filter'

    field_to_slice_by = config["field_to_slice_by"]
    sliced_events = calculator.slice_by_field(
        events=filtered_events, particle_counts=combination,
        field_to_slice_by=field_to_slice_by
    )
    if len(sliced_events) == 0:
        return None, None, 'no_events_after_slice'

    inv_mass = calculator.calculate_invariant_mass(sliced_events)
    if not ak.any(inv_mass):
        return None, None, 'empty_inv_mass'

    # The MC info field survives the same filtering/slicing as the particle
    # arrays (ak.Array[mask] preserves all fields), so weights stay aligned.
    # Weights are only emitted when MC weighting is switched on.
    mc_event_weights = None
    if config.get("mc_weighting_enabled") and MC_EVENT_INFO_FIELD in sliced_events.fields:
        mc_event_weights = _event_weights(
            sliced_events[MC_EVENT_INFO_FIELD],
            config.get("mc_norm_by_dsid"),
            config.get("mc_norm_default", 1.0),
        )

    return inv_mass, mc_event_weights, None


def _event_weights(
    mc_info: ak.Array,
    norm_by_dsid: Optional[Dict[int, float]],
    default_norm: float = 1.0,
) -> np.ndarray:
    """
    Final per-event MC weight: generator weight x per-dataset normalization.

    ``norm_by_dsid`` maps dataset number -> w_norm (computed once per DSID by
    the mass-calculation handler). Events whose dataset is not in the map, or
    that carry no dataset number, get ``default_norm`` (the file-level factor,
    or 1 when unknown).
    """
    n = len(mc_info)
    if MC_EVENT_WEIGHT_FIELD in mc_info.fields:
        weights = np.asarray(ak.to_numpy(mc_info[MC_EVENT_WEIGHT_FIELD]), dtype=np.float64)
    else:
        weights = np.ones(n, dtype=np.float64)

    norm = np.full(n, float(default_norm), dtype=np.float64)
    if norm_by_dsid and MC_CHANNEL_NUMBER_FIELD in mc_info.fields:
        dsids = np.asarray(ak.to_numpy(mc_info[MC_CHANNEL_NUMBER_FIELD]))
        for dsid in np.unique(dsids):
            factor = norm_by_dsid.get(int(dsid))
            if factor is not None:
                norm[dsids == dsid] = factor
    return weights * norm


def _accumulate_invariant_mass(
    fs_im_mapping: Dict[str, Dict[str, ak.Array]],
    final_state: str,
    combination_name: str,
    inv_mass: ak.Array,
    threshold_bytes: int,
    output_dir: str,
    logger: logging.Logger,
    output_mode: str = "npy",
    sqlite_writer=None,
) -> List[str]:
    if final_state not in fs_im_mapping:
        fs_im_mapping[final_state] = {}

    if combination_name in fs_im_mapping[final_state]:
        existing_im = fs_im_mapping[final_state][combination_name]
        fs_im_mapping[final_state][combination_name] = ak.concatenate([existing_im, inv_mass])
    else:
        fs_im_mapping[final_state][combination_name] = inv_mass

    saved_files = []
    if _fs_dict_exceeding_threshold(fs_im_mapping, threshold_bytes):
        logger.info(f"Memory threshold exceeded. Saving accumulated arrays for {final_state}")
        saved_files = _save_fs_mapping(
            fs_im_mapping[final_state],
            output_dir,
            final_state,
            output_mode=output_mode,
            sqlite_writer=sqlite_writer,
        )
        fs_im_mapping[final_state].clear()

    return saved_files


def _save_remaining_accumulated_data(
    fs_im_mapping: Dict[str, Dict[str, ak.Array]],
    output_dir: str,
    logger: logging.Logger,
    output_mode: str = "npy",
    sqlite_writer=None,
) -> List[str]:
    all_saved_files = []
    for fs, combinations_dict in fs_im_mapping.items():
        if combinations_dict:
            logger.debug(f"Saving remaining {len(combinations_dict)} combinations for final state: {fs}")
            saved_files = _save_fs_mapping(
                combinations_dict,
                output_dir,
                fs,
                output_mode=output_mode,
                sqlite_writer=sqlite_writer,
            )
            if saved_files:
                all_saved_files.extend(saved_files)
    return all_saved_files


def _save_fs_mapping(
    fs_mapping: Dict[str, ak.Array],
    output_dir: str,
    final_state: str,
    output_mode: str = "npy",
    sqlite_writer=None,
) -> List[str]:
    saved_files = []
    if output_mode == "sqlite":
        if sqlite_writer is None:
            raise ValueError("output_mode='sqlite' requires sqlite_writer in config")
        to_write = {}
        for combination_name, im_arr in fs_mapping.items():
            to_write[combination_name] = ak.to_numpy(im_arr)
            saved_files.append(combination_name)
        if to_write:
            sqlite_writer.append_many(to_write)
            sqlite_writer.commit()
        return saved_files

    for combination_name, im_arr in fs_mapping.items():
        filename = f"{combination_name}.npy"
        output_path = os.path.join(output_dir, filename)
        if os.path.exists(output_path):
            existing_data = np.load(output_path)
            combined_data = np.concatenate([existing_data, ak.to_numpy(im_arr)])
            np.save(output_path, combined_data)
        else:
            np.save(output_path, ak.to_numpy(im_arr))
        saved_files.append(filename)
    return saved_files


def _fs_dict_exceeding_threshold(fs_im_mapping: Dict, threshold: int) -> bool:
    if not fs_im_mapping:
        return False

    total_size = sys.getsizeof(fs_im_mapping)

    for fs, combinations in fs_im_mapping.items():
        if not isinstance(combinations, dict):
            continue
        total_size += sys.getsizeof(fs) + sys.getsizeof(combinations)
        for name, arr in combinations.items():
            total_size += sys.getsizeof(name)
            if hasattr(arr, 'layout'):
                total_size += arr.layout.nbytes
            elif hasattr(arr, 'nbytes'):
                total_size += arr.nbytes
            else:
                total_size += sys.getsizeof(arr)

    return total_size >= threshold


def prepare_im_combination_name(
    filename: str,
    final_state: str,
    combination: Dict,
) -> str:
    """
    Build the SQLite signature name for a given IM combination.

    IM part encodes each selected particle as letter + rank index.
    Works with both plain-int values (start=0) and (count, start_index) tuples.

    Examples (leading only, start=0):
        {"Electrons": (1, 0), "Jets": (1, 0)}       → IM_e0j0
        {"Electrons": (2, 0), "Jets": (1, 0)}       → IM_e0e1j0
        {"Electrons": (1, 0), "Muons": (1, 0)}      → IM_e0m0

    Examples (sub-leading, start>0):
        {"Electrons": (1, 1), "Jets": (1, 0)}       → IM_e1j0
        {"Electrons": (1, 0), "Jets": (1, 1)}       → IM_e0j1
        {"Electrons": (1, 1), "Jets": (1, 1)}       → IM_e1j1
        {"Electrons": (2, 1), "Jets": (1, 0)}       → IM_e1e2j0

    The name is unambiguous: IM_e1j0 always means sub-leading electron
    + leading jet (2-body invariant mass).
    """
    base_filename = filename.replace(".root", "")

    # Canonical particle order matches BumpNet convention
    PARTICLE_ORDER = [
        ("Electrons", "e"),
        ("Muons",     "m"),
        ("Jets",      "j"),
        ("BJets",     "b"),
        ("Photons",   "g"),
        ("Taus",      "t"),
    ]

    im_parts = []
    for ptype, letter in PARTICLE_ORDER:
        if ptype not in combination:
            continue
        value = combination[ptype]
        count = get_count(value)
        start = get_start(value)
        for idx in range(start, start + count):
            im_parts.append(f"{letter}{idx}")

    im_str = "".join(im_parts)   # e.g. "e0j0", "e1j0", "e0e1j0"
    return f"{base_filename}_FS_{final_state}_IM_{im_str}"
