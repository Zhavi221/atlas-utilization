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

from services.calculations.im_calculator import IMCalculator


def process_final_state(
    final_state: str,
    fs_events: ak.Array,
    filename: str,
    all_combinations: List[Dict[str, int]],
    config: Dict,
    output_dir: str,
    logger: logging.Logger,
    calculator: IMCalculator,
    worker_num: Optional[int] = None
) -> Tuple[Dict, List[str]]:
    """
    Process all combinations for a given final state.

    Returns:
        Tuple of (statistics dict, list of created .npy filenames)
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

        inv_mass, skip_reason = _calculate_combination_invariant_mass(
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
            fs_mapping_threshold_bytes, output_dir, logger
        )
        if saved_files:
            created_im_files.extend(saved_files)

    remaining_files = _save_remaining_accumulated_data(fs_im_mapping, output_dir, logger)
    if remaining_files:
        created_im_files.extend(remaining_files)

    logger.info(
        f"{prefix} [{filename}] Completed '{final_state}': "
        f"{stats['calculated']} calculated, {stats['skipped']} skipped"
    )

    return stats, created_im_files


def _convert_array_to_gev(inv_mass: ak.Array) -> ak.Array:
    return inv_mass * 1e-3


def _calculate_combination_invariant_mass(
    fs_events: ak.Array,
    combination: Dict[str, int],
    config: Dict,
    calculator: IMCalculator,
    logger: logging.Logger,
    final_state: str,
    worker_num: Optional[int] = None
) -> Tuple[Optional[ak.Array], Optional[str]]:
    logger.debug(f"Processing combination: {combination} for final state: {final_state}")

    filtered_events = calculator.filter_by_particle_counts(
        events=fs_events, particle_counts=combination, is_exact_count=True
    )
    if len(filtered_events) == 0:
        return None, 'no_events_after_filter'

    field_to_slice_by = config["field_to_slice_by"]
    sliced_events = calculator.slice_by_field(
        events=filtered_events, particle_counts=combination,
        field_to_slice_by=field_to_slice_by
    )
    if len(sliced_events) == 0:
        return None, 'no_events_after_slice'

    inv_mass = calculator.calculate_invariant_mass(sliced_events)
    if not ak.any(inv_mass):
        return None, 'empty_inv_mass'

    return inv_mass, None


def _accumulate_invariant_mass(
    fs_im_mapping: Dict[str, Dict[str, ak.Array]],
    final_state: str,
    combination_name: str,
    inv_mass: ak.Array,
    threshold_bytes: int,
    output_dir: str,
    logger: logging.Logger
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
        saved_files = _save_fs_mapping(fs_im_mapping[final_state], output_dir, final_state)
        fs_im_mapping[final_state].clear()

    return saved_files


def _save_remaining_accumulated_data(
    fs_im_mapping: Dict[str, Dict[str, ak.Array]],
    output_dir: str,
    logger: logging.Logger
) -> List[str]:
    all_saved_files = []
    for fs, combinations_dict in fs_im_mapping.items():
        if combinations_dict:
            logger.debug(f"Saving remaining {len(combinations_dict)} combinations for final state: {fs}")
            saved_files = _save_fs_mapping(combinations_dict, output_dir, fs)
            if saved_files:
                all_saved_files.extend(saved_files)
    return all_saved_files


def _save_fs_mapping(
    fs_mapping: Dict[str, ak.Array],
    output_dir: str,
    final_state: str
) -> List[str]:
    saved_files = []
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
    combination: Dict[str, int]
) -> str:
    base_filename = filename.replace(".root", "")

    e_count = combination.get("Electrons", 0)
    j_count = combination.get("Jets", 0)
    m_count = combination.get("Muons", 0)
    g_count = combination.get("Photons", 0)

    combination_part = f"{e_count}e_{m_count}m_{j_count}j_{g_count}g"
    return f"{base_filename}_FS_{final_state}_IM_{combination_part}"
