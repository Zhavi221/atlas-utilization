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

import awkward as ak
import numpy as np
from tqdm import tqdm

from src.calculations import combinatorics, physics_calcs
from src.parse_atlas import parser
from src.im_calculator.im_calculator import IMCalculator
from src.utils import memory_utils


def mass_calculate(config: Dict) -> None:
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
    """
    logger = init_logging()
    
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    
    if not os.path.exists(input_dir) or len(os.listdir(input_dir)) == 0:
        logger.warning(f"Input directory '{input_dir}' is empty or doesn't exist.")
        return
    
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
    root_files = [
        f for f in os.listdir(input_dir) 
        if f.endswith(".root")
    ]
    
    if not root_files:
        logger.warning(f"No ROOT files found in {input_dir}")
        return
    
    # Handle batch separation if configured
    batch_job_index = config.get("batch_job_index")
    total_batch_jobs = config.get("total_batch_jobs")
    
    if batch_job_index is not None and total_batch_jobs is not None:
        root_files = get_batch_files(root_files, batch_job_index, total_batch_jobs)
        logger.info(f"Batch {batch_job_index}/{total_batch_jobs}: Processing {len(root_files)} files")
    
    # Process files
    use_multiprocessing = config.get("use_multiprocessing", False)
    max_workers = config.get("max_workers")
    if max_workers is None:
        max_workers = mp.cpu_count()
    
    if use_multiprocessing and len(root_files) > 1:
        logger.info(f"Using multiprocessing with {max_workers} workers")
        process_files_multiprocessing(
            root_files, input_dir, output_dir, all_combinations, config, logger, max_workers
        )
    else:
        logger.info("Processing files sequentially")
        process_files_sequential(
            root_files, input_dir, output_dir, all_combinations, config, logger
        )


def process_files_sequential(
    root_files: List[str],
    input_dir: str,
    output_dir: str,
    all_combinations: List[Dict[str, int]],
    config: Dict,
    logger: logging.Logger
) -> None:
    """Process files sequentially."""
    for filename in tqdm(root_files, desc="Processing files"):
        try:
            process_single_file(
                filename, input_dir, output_dir, all_combinations, config, logger
            )
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}", exc_info=True)


def process_files_multiprocessing(
    root_files: List[str],
    input_dir: str,
    output_dir: str,
    all_combinations: List[Dict[str, int]],
    config: Dict,
    logger: logging.Logger,
    max_workers: int
) -> None:
    """Process files using multiprocessing."""
    # Prepare arguments for workers
    worker_args = [
        (filename, input_dir, output_dir, all_combinations, config)
        for filename in root_files
    ]
    
    with Pool(processes=max_workers) as pool:
        results = pool.starmap(process_single_file_worker, worker_args)
    
    # Log summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "error")
    logger.info(f"Processing complete: {successful} successful, {failed} failed")


def process_single_file_worker(
    filename: str,
    input_dir: str,
    output_dir: str,
    all_combinations: List[Dict[str, int]],
    config: Dict
) -> Dict:
    """
    Worker function for multiprocessing.
    
    Returns:
        Dictionary with status and metadata
    """
    logger = init_logging()
    try:
        process_single_file(
            filename, input_dir, output_dir, all_combinations, config, logger
        )
        return {"status": "success", "filename": filename}
    except Exception as e:
        logger.error(f"Error in worker processing {filename}: {e}", exc_info=True)
        return {"status": "error", "filename": filename, "error": str(e)}


def process_single_file(
    filename: str,
    input_dir: str,
    output_dir: str,
    all_combinations: List[Dict[str, int]],
    config: Dict,
    logger: logging.Logger
) -> None:
    """
    Process a single ROOT file to calculate invariant masses.
    
    Args:
        filename: Name of ROOT file
        input_dir: Input directory path
        output_dir: Output directory path
        all_combinations: List of particle combinations to process
        config: Configuration dictionary
        logger: Logger instance
    """
    file_path = os.path.join(input_dir, filename)
    
    # Parse ROOT file
    particle_arrays: Optional[ak.Array] = parser.AtlasOpenParser.parse_root_file(file_path)
    
    if particle_arrays is None or len(particle_arrays) == 0:
        logger.info(f"File {filename} is empty or could not be parsed")
        return
    
    logger.info(f"Processing file: {filename} with {len(particle_arrays)} events")
    
    # Initialize calculator
    calculator = IMCalculator(particle_arrays)
    
    # Process each final state
    for cur_fs, fs_events in calculator.group_by_final_state():
        process_final_state(
            cur_fs, fs_events, filename, all_combinations, config, output_dir, logger, calculator
        )


def process_final_state(
    final_state: str,
    fs_events: ak.Array,
    filename: str,
    all_combinations: List[Dict[str, int]],
    config: Dict,
    output_dir: str,
    logger: logging.Logger,
    calculator: IMCalculator
) -> None:
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
    """
    if len(fs_events) == 0:
        return
    
    fs_mapping_threshold_bytes = config.get("fs_mapping_threshold_bytes", 50000000)  # 50MB default
    fs_im_mapping: Dict[str, Dict[str, ak.Array]] = {}
    
    for combination in all_combinations:
        if not calculator.does_final_state_contain_combination(final_state, combination):
            continue
        
        inv_mass = _calculate_combination_invariant_mass(
            fs_events, combination, config, calculator, logger, final_state
        )
        
        if inv_mass is None:
            continue
        
        combination_name = prepare_im_combination_name(filename, final_state, combination)
        _accumulate_invariant_mass(
            fs_im_mapping, final_state, combination_name, inv_mass, 
            fs_mapping_threshold_bytes, output_dir, logger
        )
    
    # Save any remaining accumulated data
    _save_remaining_accumulated_data(fs_im_mapping, output_dir, logger)


def _calculate_combination_invariant_mass(
    fs_events: ak.Array,
    combination: Dict[str, int],
    config: Dict,
    calculator: IMCalculator,
    logger: logging.Logger,
    final_state: str
) -> Optional[ak.Array]:
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
        Invariant mass array or None if no valid events
    """
    logger.debug(f"Processing combination: {combination} for final state: {final_state}")
    
    # Filter events by exact particle counts
    filtered_events = calculator.filter_by_particle_counts(
        events=fs_events,
        particle_counts=combination,
        is_exact_count=True  # Use exact count matching for combinations
    )
    
    if len(filtered_events) == 0:
        return None
    
    # Slice events by field (e.g., top N by pt)
    field_to_slice_by = config.get("field_to_slice_by", "pt")
    sliced_events = calculator.slice_by_field(
        events=filtered_events,
        particle_counts=combination,
        field_to_slice_by=field_to_slice_by
    )
    
    if len(sliced_events) == 0:
        return None
    
    # Calculate invariant mass
    inv_mass = calculator.calculate_invariant_mass(sliced_events)
    
    if not ak.any(inv_mass):
        return None
    
    return inv_mass


def _accumulate_invariant_mass(
    fs_im_mapping: Dict[str, Dict[str, ak.Array]],
    final_state: str,
    combination_name: str,
    inv_mass: ak.Array,
    threshold_bytes: int,
    output_dir: str,
    logger: logging.Logger
) -> None:
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
    if fs_dict_exceeding_threshold(fs_im_mapping, threshold_bytes):
        logger.info(f"Memory threshold exceeded. Saving accumulated arrays for {final_state}")
        save_fs_mapping(fs_im_mapping[final_state], output_dir, final_state)
        fs_im_mapping[final_state].clear()


def _save_remaining_accumulated_data(
    fs_im_mapping: Dict[str, Dict[str, ak.Array]],
    output_dir: str,
    logger: logging.Logger
) -> None:
    """
    Save any remaining accumulated data after processing all combinations.
    
    Args:
        fs_im_mapping: Dictionary of accumulated results
        output_dir: Output directory
        logger: Logger instance
    """
    for fs, combinations_dict in fs_im_mapping.items():
        if combinations_dict:
            logger.debug(f"Saving remaining {len(combinations_dict)} combinations for final state: {fs}")
            save_fs_mapping(combinations_dict, output_dir, fs)


def save_fs_mapping(
    fs_mapping: Dict[str, ak.Array],
    output_dir: str,
    final_state: str
) -> None:
    """
    Save all arrays in a final state mapping to disk.
    
    Args:
        fs_mapping: Dictionary mapping combination names to arrays
        output_dir: Output directory
        final_state: Final state identifier (for logging)
    """
    for combination_name, im_arr in fs_mapping.items():
        output_path = os.path.join(output_dir, f"{combination_name}.npy")
        # If file exists, load and concatenate (for batch processing)
        if os.path.exists(output_path):
            existing_data = np.load(output_path)
            combined_data = np.concatenate([existing_data, ak.to_numpy(im_arr)])
            np.save(output_path, combined_data)
        else:
            np.save(output_path, ak.to_numpy(im_arr))


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
    
    Args:
        filename: Source ROOT filename (without extension)
        final_state: Final state string
        combination: Dictionary mapping particle types to counts
        
    Returns:
        Combination name string
    """
    # Remove .root extension if present
    base_filename = filename.replace(".root", "")
    
    combination_name = f"{base_filename}_FS_{final_state}_IM_"
    
    for particle_type, amount in combination.items():
        # Get first letter of particle type
        letter = particle_type[0].lower()
        combination_name += f"{amount}{letter}_"
    
    # Remove trailing underscore
    return combination_name.rstrip("_")


def init_logging() -> logging.Logger:
    """Initialize and return logger instance."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    return logging.getLogger(__name__)
