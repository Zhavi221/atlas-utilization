"""
Invariant Mass Calculation Pipeline

Processes ROOT files to calculate invariant masses for particle combinations.
Supports multiprocessing and batch separation for large-scale processing.
"""
import logging
import sys
import os
import multiprocessing as mp
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import math

import awkward as ak
import numpy as np
from tqdm import tqdm
import ROOT

from src.utils.calculations import combinatorics, physics_calcs
from src.parse_atlas import parser
from src.im_calculator.im_calculator import IMCalculator
from src.utils import memory_utils


def mass_calculate(config: Dict, file_list: Optional[List[str]] = None) -> List[str]:
    """
    Main entry point for invariant mass calculation pipeline.
    
    Args:
        config: Configuration dictionary with keys:
            - input_dir: Directory containing ROOT files
            - output_dir: Directory to save .npy files
            - objects_to_calculate: List of particle types
            - min_particles, max_particles: Range for particle counts
            - min_count, max_count: Range for combination counts
            - field_to_slice_by: Field to sort by (default: "pt")
            - limit_combinations: Optional limit on combinations
            - use_multiprocessing: Whether to use multiprocessing
            - max_workers: Number of worker processes
            - batch_job_index: Optional batch job index
            - total_batch_jobs: Optional total batch jobs
            - fs_mapping_threshold_bytes: Memory threshold for saving
        file_list: Optional explicit list of ROOT filenames to process.
                   If provided, only these files will be processed (must exist in input_dir).
                   If None, scans input_dir for all .root files.
    
    Returns:
        List of created IM array filenames (.npy files)
    """
    logger = init_logging()
    
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all combinations
    all_combinations: List[Dict[str, int]] = combinatorics.get_all_combinations(
        config["objects_to_calculate"],
        min_particles=config["min_particles"],
        max_particles=config["max_particles"],
        min_count=config["min_count"],
        max_count=config["max_count"],
        limit=config.get("limit_combinations")
    )
    
    logger.info(f"Generated {len(all_combinations)} combinations to process")
    
    # Get list of ROOT files
    if file_list is not None:
        # Use explicit file list provided
        root_files = file_list.copy()
        logger.info(f"Using explicit file list with {len(root_files)} files")
        
        # Validate that files exist
        existing_files = []
        for filename in root_files:
            file_path = os.path.join(input_dir, filename)
            if os.path.exists(file_path):
                existing_files.append(filename)
            else:
                logger.warning(f"File {filename} not found in {input_dir}, skipping")
        
        root_files = existing_files
        
        if not root_files:
            logger.warning(f"None of the {len(file_list)} specified files exist in {input_dir}")
            return
    else:
        # Scan directory for ROOT files (original behavior)
        if not os.path.exists(input_dir) or len(os.listdir(input_dir)) == 0:
            logger.warning(f"Input directory '{input_dir}' is empty or doesn't exist.")
            return
        
        root_files = [
            f for f in os.listdir(input_dir) 
            if f.endswith(".root")
        ]
        
        if not root_files:
            logger.warning(f"No ROOT files found in {input_dir}")
            return
        
        # Handle batch separation if configured (only when scanning directory)
        batch_job_index = config.get("batch_job_index")
        total_batch_jobs = config.get("total_batch_jobs")
        
        if batch_job_index is not None and total_batch_jobs is not None:
            root_files = get_batch_files(root_files, batch_job_index, total_batch_jobs)
            logger.info(f"Batch {batch_job_index}/{total_batch_jobs}: Processing {len(root_files)} files")
    
    # Process files
    use_multiprocessing = config["use_multiprocessing"]
    max_workers = config.get("max_workers")
    if max_workers is None:
        max_workers = mp.cpu_count()
    
    # if config["concat_root_files"]: #FEATURE
    #     pass
    
    created_im_files = []
    
    if use_multiprocessing and len(root_files) > 1:
        logger.info(f"Using multiprocessing with {max_workers} workers")
        created_im_files = process_files_multiprocessing(
            root_files, input_dir, output_dir, all_combinations, config, logger, max_workers
        )
    else:
        logger.info("Processing files sequentially")
        created_im_files = process_files_sequential(
            root_files, input_dir, output_dir, all_combinations, config, logger
        )
    
    return created_im_files
        

def process_files_sequential(
    root_files: List[str],
    input_dir: str,
    output_dir: str,
    all_combinations: List[Dict[str, int]],
    config: Dict,
    logger: logging.Logger
) -> List[str]:
    """Process files sequentially."""
    all_created_files = []
    for filename in tqdm(root_files, desc="Processing files"):
        try:
            created_files = process_single_file(
                filename, input_dir, output_dir, all_combinations, config, logger
            )
            if created_files:
                all_created_files.extend(created_files)
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}", exc_info=True)
    return all_created_files


def process_files_multiprocessing(
    root_files: List[str],
    input_dir: str,
    output_dir: str,
    all_combinations: List[Dict[str, int]],
    config: Dict,
    logger: logging.Logger,
    max_workers: int
) -> List[str]:
    """Process files using multiprocessing."""
    # Prepare arguments for workers with worker index
    worker_args = [
        (i, filename, input_dir, output_dir, all_combinations, config)
        for i, filename in enumerate(root_files)
    ]
    
    with Pool(processes=max_workers) as pool:
        results = pool.starmap(process_single_file_worker, worker_args)
    
    # Log summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "error")
    logger.info(f"Processing complete: {successful} successful, {failed} failed")
    
    # Collect all created files from results
    all_created_files = []
    for r in results:
        if r["status"] == "success" and "created_files" in r:
            all_created_files.extend(r["created_files"])
    
    return all_created_files


def process_single_file_worker(
    worker_id: int,
    filename: str,
    input_dir: str,
    output_dir: str,
    all_combinations: List[Dict[str, int]],
    config: Dict
) -> Dict:
    """
    Worker function for multiprocessing.
    
    Args:
        worker_id: Worker identifier (file index, used for logging)
        filename: Name of ROOT file
        input_dir: Input directory path
        output_dir: Output directory path
        all_combinations: List of particle combinations to process
        config: Configuration dictionary
        
    Returns:
        Dictionary with status and metadata
    """
    logger = init_logging()
    # Get process name for better worker identification
    import multiprocessing
    process_name = multiprocessing.current_process().name
    # Extract worker number from process name (e.g., "ForkPoolWorker-1" -> 1)
    try:
        worker_num = int(process_name.split('-')[-1]) if '-' in process_name else worker_id
    except (ValueError, IndexError):
        worker_num = worker_id
    
    try:
        created_files = process_single_file(
            filename, input_dir, output_dir, all_combinations, config, logger, worker_num
        )
        return {"status": "success", "filename": filename, "worker": worker_num, "created_files": created_files}
    except Exception as e:
        logger.error(f"[Worker {worker_num}] Error processing {filename}: {e}", exc_info=True)
        return {"status": "error", "filename": filename, "error": str(e), "worker": worker_num}


def process_single_file(
    filename: str,
    input_dir: str,
    output_dir: str,
    all_combinations: List[Dict[str, int]],
    config: Dict,
    logger: logging.Logger,
    worker_num: Optional[int] = None
) -> List[str]:
    """
    Process a single ROOT file to calculate invariant masses.
    
    Args:
        filename: Name of ROOT file
        input_dir: Input directory path
        output_dir: Output directory path
        all_combinations: List of particle combinations to process
        config: Configuration dictionary
        logger: Logger instance
        worker_num: Optional worker number for multiprocessing (None for sequential)
    """
    # Create prefix for logging
    prefix = f"[Worker {worker_num}]" if worker_num is not None else ""
    file_path = os.path.join(input_dir, filename)
    
    # Get file size for progress indication
    file_size_mb = os.path.getsize(file_path) / (1024**2) if os.path.exists(file_path) else 0.0
    logger.info(f"{prefix} [{filename}] Starting parsing (size: {file_size_mb:.2f} MB)...")
    
    # Extract release year from filename
    release_year = parser.AtlasOpenParser.extract_release_year_from_filename(filename)
    logger.info(f"{prefix} [{filename}] Extracted release year: {release_year}")
    
    # Parse ROOT file with inferred release year
    particle_arrays: Optional[ak.Array] = parser.AtlasOpenParser.parse_root_file(
        file_path, release_year=release_year)
    
    if particle_arrays is None or len(particle_arrays) == 0:
        logger.info(f"{prefix} [{filename}] File is empty or could not be parsed")
        return
    
    num_events = len(particle_arrays)
    logger.info(f"{prefix} [{filename}] Parsed successfully: {num_events:,} events")
    
    # Initialize calculator
    calculator = IMCalculator(
        particle_arrays, 
        min_events_per_fs=config["min_events_per_fs"],
        min_k=config["min_count"],
        max_k=config["max_count"],
        min_n=config["min_particles"],
        max_n=config["max_particles"],
    )

    # Track statistics for this file
    file_stats = {
        'calculated': 0,
        'skipped': 0,
        'skip_reasons': {
            'no_matching_combination': 0,
            'no_events_after_filter': 0,
            'no_events_after_slice': 0,
            'empty_inv_mass': 0
        },
        'final_states_processed': 0
    }
    
    # Track created IM array files
    created_im_files = []
    
    # Process each final state
    for cur_fs in calculator.group_by_final_state():
        fs_events = calculator.get_events_for_final_state(cur_fs)
        result = process_final_state(
            cur_fs, fs_events, filename, all_combinations, config, output_dir, logger, calculator, worker_num
        )
        if result is None:
            continue
        
        fs_stats, fs_created_files = result
        # Aggregate statistics
        if fs_stats:
            file_stats['calculated'] += fs_stats['calculated']
            file_stats['skipped'] += fs_stats['skipped']
            file_stats['final_states_processed'] += 1
            for reason, count in fs_stats['skip_reasons'].items():
                file_stats['skip_reasons'][reason] += count
        # Collect created files
        if fs_created_files:
            created_im_files.extend(fs_created_files)
    
    # Print summary after each file
    logger.info(f"{prefix} [{filename}] ===== FILE PROCESSING COMPLETE =====")
    logger.info(f"{prefix} [{filename}] Events: {num_events:,} | Final States: {file_stats['final_states_processed']}")
    logger.info(f"{prefix} [{filename}] Mass arrays calculated: {file_stats['calculated']}")
    logger.info(f"{prefix} [{filename}] Mass arrays skipped: {file_stats['skipped']}")
    if file_stats['skipped'] > 0:
        logger.info(f"{prefix} [{filename}] Skip reasons:")
        for reason, count in file_stats['skip_reasons'].items():
            if count > 0:
                logger.info(f"{prefix} [{filename}]   - {reason}: {count}")
    logger.info(f"{prefix} [{filename}] =====================================")
    
    return created_im_files


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
    
    Orchestrates the processing of all combinations for a final state,
    managing memory through threshold-based saving.
    
    Args:
        final_state: Final state string (e.g., "2e_3m_5j_1p")
        fs_events: Events matching this final state
        filename: Source filename
        all_combinations: List of combinations to process
        config: Configuration dictionary
        output_dir: Output directory
        logger: Logger instance
        calculator: IMCalculator instance
        worker_num: Optional worker number for multiprocessing
        
    Returns:
        Dictionary with statistics about processed combinations
    """
    if len(fs_events) == 0:
        return None, []
    
    # Create prefix for logging
    prefix = f"[Worker {worker_num}]" if worker_num is not None else ""
    num_combinations = sum(1 for c in all_combinations 
                          if calculator.does_final_state_contain_combination(final_state, c))
    
    logger.info(f"{prefix} [{filename}] Computing final state '{final_state}': "
                f"{len(fs_events):,} events, {num_combinations} combinations")
    
    fs_mapping_threshold_bytes = config["fs_mapping_threshold_bytes"]
    fs_im_mapping: Dict[str, Dict[str, ak.Array]] = {}
    
    # Track statistics for this final state
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
    
    # Track created IM array files for this final state
    created_im_files = []
    
    combination_count = 0
    # Calculate milestone points for progress logging (only log at key milestones)
    milestones = set()
    if num_combinations > 1:
        milestones = {1, num_combinations // 4, num_combinations // 2, 
                     3 * num_combinations // 4, num_combinations}
        milestones = {m for m in milestones if m > 0}  # Remove zeros
    elif num_combinations == 1:
        milestones = {1}
    
    for combination in all_combinations:
        if not calculator.does_final_state_contain_combination(final_state, combination):
            stats['skipped'] += 1
            stats['skip_reasons']['no_matching_combination'] += 1
            continue
        
        combination_count += 1
        
        # Log progress only at milestones (start, 25%, 50%, 75%, 100%)
        if combination_count in milestones:
            pct = (combination_count / num_combinations) * 100 if num_combinations > 0 else 0
            logger.info(f"{prefix} [{filename}] '{final_state}': "
                       f"{combination_count}/{num_combinations} ({pct:.0f}%) - "
                       f"{stats['calculated']} calculated, {stats['skipped']} skipped")
        
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
    
    # Save any remaining accumulated data
    remaining_files = _save_remaining_accumulated_data(fs_im_mapping, output_dir, logger)
    if remaining_files:
        created_im_files.extend(remaining_files)
    
    logger.info(f"{prefix} [{filename}] ✓ Completed '{final_state}': "
               f"{stats['calculated']} calculated, {stats['skipped']} skipped")
    
    return stats, created_im_files

def _convert_array_to_gev(inv_mass: ak.Array) -> ak.Array:
    """
    Convert invariant mass array to GeV.
    Args:
        inv_mass: Invariant mass array
    """
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
    """
    Calculate invariant mass for a single combination.
    
    Filters, slices, and calculates invariant mass for the given combination.
    Returns None if no valid events found.
    
    Args:
        fs_events: Events for the final state
        combination: Particle combination dictionary
        config: Configuration dictionary
        calculator: IMCalculator instance
        logger: Logger instance
        final_state: Final state string (for logging)
        
    Returns:
        Tuple of (invariant mass array or None, skip reason string or None)
    """
    # Logging is handled at higher level, just debug here
    logger.debug(f"Processing combination: {combination} for final state: {final_state}")
    
    # Filter events by exact particle counts
    # When is_exact_count=True, this also extracts only the specified particle types
    filtered_events = calculator.filter_by_particle_counts(
        events=fs_events,
        particle_counts=combination,
        is_exact_count=True  # Use exact count matching for combinations
    )
    
    if len(filtered_events) == 0:
        return None, 'no_events_after_filter'
    
    # Slice events by field (e.g., top N by pt)
    field_to_slice_by = config["field_to_slice_by"]
    sliced_events = calculator.slice_by_field(
        events=filtered_events,
        particle_counts=combination,
        field_to_slice_by=field_to_slice_by
    )
    
    if len(sliced_events) == 0:
        return None, 'no_events_after_slice'
    
    # Calculate invariant mass
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
    """
    Accumulate invariant mass in memory mapping and save if threshold exceeded.
    
    Args:
        fs_im_mapping: Dictionary to accumulate results
        final_state: Final state identifier
        combination_name: Name of the combination
        inv_mass: Invariant mass array to accumulate
        threshold_bytes: Memory threshold in bytes
        output_dir: Output directory for saving
        logger: Logger instance
    """
    if final_state not in fs_im_mapping:
        fs_im_mapping[final_state] = {}
    
    # Add to mapping (concatenate if combination already exists)
    if combination_name in fs_im_mapping[final_state]:
        existing_im = fs_im_mapping[final_state][combination_name]
        fs_im_mapping[final_state][combination_name] = ak.concatenate([existing_im, inv_mass])
    else:
        fs_im_mapping[final_state][combination_name] = inv_mass
    
    # Check memory threshold and save if needed
    saved_files = []
    if fs_dict_exceeding_threshold(fs_im_mapping, threshold_bytes):
        logger.info(f"Memory threshold exceeded. Saving accumulated arrays for {final_state}")
        saved_files = save_fs_mapping(fs_im_mapping[final_state], output_dir, final_state)
        fs_im_mapping[final_state].clear()
    
    return saved_files


def _save_remaining_accumulated_data(
    fs_im_mapping: Dict[str, Dict[str, ak.Array]],
    output_dir: str,
    logger: logging.Logger
) -> List[str]:
    """
    Save any remaining accumulated data after processing all combinations.
    
    Args:
        fs_im_mapping: Dictionary of accumulated results
        output_dir: Output directory
        logger: Logger instance
    
    Returns:
        List of saved filenames
    """
    all_saved_files = []
    for fs, combinations_dict in fs_im_mapping.items():
        if combinations_dict:
            logger.debug(f"Saving remaining {len(combinations_dict)} combinations for final state: {fs}")
            saved_files = save_fs_mapping(combinations_dict, output_dir, fs)
            if saved_files:
                all_saved_files.extend(saved_files)
    return all_saved_files


def save_fs_mapping(
    fs_mapping: Dict[str, ak.Array],
    output_dir: str,
    final_state: str
) -> List[str]:
    """
    Save all arrays in a final state mapping to disk.
    
    Args:
        fs_mapping: Dictionary mapping combination names to arrays
        output_dir: Output directory
        final_state: Final state identifier (for logging)
    
    Returns:
        List of saved filenames (just the .npy filenames, not full paths)
    """
    saved_files = []
    for combination_name, im_arr in fs_mapping.items():
        filename = f"{combination_name}.npy"
        output_path = os.path.join(output_dir, filename)
        # If file exists, load and concatenate (for batch processing)
        if os.path.exists(output_path):
            existing_data = np.load(output_path)
            combined_data = np.concatenate([existing_data, ak.to_numpy(im_arr)])
            np.save(output_path, combined_data)
        else:
            np.save(output_path, ak.to_numpy(im_arr))
        saved_files.append(filename)
    return saved_files


def fs_dict_exceeding_threshold(fs_im_mapping: Dict, threshold: int) -> bool:
    """
    Check if final state mapping exceeds memory threshold.
    
    Args:
        fs_im_mapping: Dictionary of final states to combination mappings
        threshold: Memory threshold in bytes
        
    Returns:
        True if threshold exceeded
    """
    if not fs_im_mapping:
        return False
    
    total_size = sys.getsizeof(fs_im_mapping)
    
    # Recursively calculate size of nested dictionaries and arrays
    for fs, combinations in fs_im_mapping.items():
        if not isinstance(combinations, dict):
            continue
            
        total_size += sys.getsizeof(fs) + sys.getsizeof(combinations)
        
        for name, arr in combinations.items():
            total_size += sys.getsizeof(name)
            # Calculate actual memory size of awkward array
            if hasattr(arr, 'layout'):
                total_size += arr.layout.nbytes
            elif hasattr(arr, 'nbytes'):
                total_size += arr.nbytes
            else:
                # Fallback: estimate from sys.getsizeof
                total_size += sys.getsizeof(arr)
    
    return total_size >= threshold


def get_batch_files(
    root_files: List[str],
    batch_index: int,
    total_batches: int
) -> List[str]:
    """
    Split files into batches for distributed processing.
    
    Args:
        root_files: List of all ROOT file names
        batch_index: Current batch index (1-indexed)
        total_batches: Total number of batches
        
    Returns:
        List of files for this batch
    """
    batch_index = int(batch_index)
    total_batches = int(total_batches)
    
    total_files = len(root_files)
    files_per_batch = total_files // total_batches
    start_idx = (batch_index - 1) * files_per_batch
    
    if batch_index == total_batches:
        end_idx = total_files
    else:
        end_idx = start_idx + files_per_batch
    
    return root_files[start_idx:end_idx]


def prepare_im_combination_name(
    filename: str,
    final_state: str,
    combination: Dict[str, int]
) -> str:
    """
    Generate a filename for an invariant mass combination.
    
    Format: {base_filename}_FS_{final_state}_IM_{e}e_{j}j_{m}m_{g}g
    Always includes all particle types in consistent order: e, j, m, g
    
    Args:
        filename: Source ROOT filename (without extension)
        final_state: Final state string (e.g., "2e_3m_5j_1g")
        combination: Dictionary mapping particle types to counts
        
    Returns:
        Combination name string (e.g., "file_FS_2e_3m_5j_1g_IM_2e_0j_0m_0g")
    """
    # Remove .root extension if present
    base_filename = filename.replace(".root", "")
    
    # Map canonical particle type names to their single-letter codes
    particle_type_map = {
        "Electrons": "e",
        "Jets": "j",
        "Muons": "m",
        "Photons": "g"  # Changed from "p" to "g" for gamma
    }
    
    # Always include all particle types in consistent order: e, j, m, g
    # Get counts from combination dict, defaulting to 0 if not present
    e_count = combination.get("Electrons", 0)
    j_count = combination.get("Jets", 0)
    m_count = combination.get("Muons", 0)
    g_count = combination.get("Photons", 0)
    
    # Build combination part in format: Ne_Nj_Nm_Ng
    combination_part = f"{e_count}e_{m_count}m_{j_count}j_{g_count}g"
    
    # Final format: {base_filename}_FS_{final_state}_IM_{combination_part}
    combination_name = f"{base_filename}_FS_{final_state}_IM_{combination_part}"
    
    return combination_name


def init_logging() -> logging.Logger:
    """Initialize and return logger instance."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    return logging.getLogger(__name__)
