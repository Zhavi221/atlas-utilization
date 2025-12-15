"""
Post-processing pipeline for IM arrays.

Processes IM arrays to:
1. Bin data using specified bin widths
2. Find the rightmost highest bin (peak)
3. Remove data before the peak
4. Split arrays by the first empty bin
"""
import logging
import sys
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import math


def process_im_arrays(config: Dict, file_list: Optional[List[str]] = None) -> List[str]:
    """
    Post-process IM arrays: find peak, remove pre-peak data, split by empty bin.
    
    Args:
        config: Configuration dictionary with keys:
            - input_dir: Directory containing IM array .npy files
            - output_dir: Directory to save processed arrays
            - bin_widths_gev: List of bin widths to use for peak detection
        file_list: Optional explicit list of .npy filenames to process.
                   If provided, only these files will be processed.
                   If None, scans input_dir for all .npy files.
    
    Returns:
        List of processed filenames
    """
    logger = init_logging()
    
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    bin_widths_gev = config["bin_widths_gev"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of IM array files
    if file_list is not None:
        # Use explicit file list provided
        im_array_files = [f for f in file_list if f.endswith(".npy")]
        logger.info(f"Using explicit file list with {len(im_array_files)} IM array files")
        
        # Validate that files exist
        existing_files = []
        for filename in im_array_files:
            file_path = os.path.join(input_dir, filename)
            if os.path.exists(file_path):
                existing_files.append(filename)
            else:
                logger.warning(f"File {filename} not found in {input_dir}, skipping")
        
        im_array_files = existing_files
        
        if not im_array_files:
            logger.warning(f"None of the {len(file_list)} specified files exist in {input_dir}")
            return []
    else:
        # Scan directory for .npy files
        if not os.path.exists(input_dir) or len(os.listdir(input_dir)) == 0:
            logger.warning(f"Input directory '{input_dir}' is empty or doesn't exist.")
            return []
        
        im_array_files = [
            f for f in os.listdir(input_dir) 
            if f.endswith(".npy")
        ]
        
        if not im_array_files:
            logger.warning(f"No .npy files found in {input_dir}")
            return []
    
    total_arrays = len(im_array_files)
    logger.info(f"Processing {total_arrays} IM arrays for peak removal and splitting...")
    
    processed_files = []
    for im_array_filename in im_array_files:
        try:
            output_files = process_single_array(
                im_array_filename, input_dir, output_dir, bin_widths_gev, logger
            )
            if output_files:
                processed_files.extend(output_files)
        except Exception as e:
            logger.error(f"Error processing {im_array_filename}: {e}", exc_info=True)
    
    logger.info(f"Post-processing complete. Created {len(processed_files)} processed arrays.")
    return processed_files


def process_single_array(
    filename: str,
    input_dir: str,
    output_dir: str,
    bin_widths_gev: List[float],
    logger: logging.Logger
) -> List[str]:
    """
    Process a single IM array file.
    
    Args:
        filename: Name of .npy file
        input_dir: Input directory path
        output_dir: Output directory path
        bin_widths_gev: List of bin widths for peak detection
        logger: Logger instance
    
    Returns:
        List of created filenames
    """
    file_path = os.path.join(input_dir, filename)
    im_array = np.load(file_path)
    
    if len(im_array) == 0:
        logger.warning(f"Array {filename} is empty, skipping")
        return []
    
    # Use the first bin width for peak detection (or could use all and take consensus)
    bin_width = bin_widths_gev[0]
    
    # Find peak using binned data
    peak_mass = find_rightmost_highest_peak(im_array, bin_width, logger)
    
    if peak_mass is None:
        logger.warning(f"Could not find peak in {filename}, keeping all data")
        # Still process for splitting by empty bin
        filtered_array = im_array
    else:
        # Remove data before peak
        filtered_array = im_array[im_array >= peak_mass]
        logger.debug(f"{filename}: Removed {len(im_array) - len(filtered_array)} values before peak at {peak_mass:.2f} GeV")
    
    if len(filtered_array) == 0:
        logger.warning(f"Array {filename} is empty after filtering, skipping")
        return []
    
    # Split by first empty bin
    main_array, outliers_array = split_by_first_empty_bin(filtered_array, bin_width, logger)
    
    # Save processed arrays
    base_name = filename.replace(".npy", "")
    output_files = []
    
    # Save main array
    if len(main_array) > 0:
        main_filename = f"{base_name}_main.npy"
        main_path = os.path.join(output_dir, main_filename)
        np.save(main_path, main_array)
        output_files.append(main_filename)
        logger.debug(f"Saved main array: {main_filename} ({len(main_array)} values)")
    
    # Save outliers array
    if len(outliers_array) > 0:
        outliers_filename = f"{base_name}_outliers.npy"
        outliers_path = os.path.join(output_dir, outliers_filename)
        np.save(outliers_path, outliers_array)
        output_files.append(outliers_filename)
        logger.debug(f"Saved outliers array: {outliers_filename} ({len(outliers_array)} values)")
    
    return output_files


def find_rightmost_highest_peak(im_array: np.ndarray, bin_width: float, logger: logging.Logger) -> Optional[float]:
    """
    Find the rightmost bin with the highest count (peak).
    
    Args:
        im_array: Array of invariant mass values
        bin_width: Bin width in GeV
        logger: Logger instance
    
    Returns:
        Peak mass value (left edge of peak bin) or None if no peak found
    """
    if len(im_array) == 0:
        return None
    
    # Create bins
    min_mass = np.min(im_array)
    max_mass = np.max(im_array)
    nbins = math.ceil((max_mass - min_mass) / bin_width)
    
    if nbins == 0:
        return None
    
    bin_edges = np.linspace(min_mass, max_mass, nbins + 1)
    counts, _ = np.histogram(im_array, bins=bin_edges)
    
    if len(counts) == 0:
        return None
    
    # Find maximum count
    max_count = np.max(counts)
    
    # Find rightmost bin with maximum count
    # Search from right to left
    peak_bin_idx = None
    for i in range(len(counts) - 1, -1, -1):
        if counts[i] == max_count:
            peak_bin_idx = i
            break
    
    if peak_bin_idx is None:
        return None
    
    # Return the left edge of the peak bin
    peak_mass = bin_edges[peak_bin_idx]
    
    logger.debug(f"Found peak at bin {peak_bin_idx} with count {max_count}, mass = {peak_mass:.2f} GeV")
    
    return peak_mass


def split_by_first_empty_bin(
    im_array: np.ndarray,
    bin_width: float,
    logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split array into two parts at the first empty bin.
    
    Args:
        im_array: Array of invariant mass values
        bin_width: Bin width in GeV
        logger: Logger instance
    
    Returns:
        Tuple of (main_array, outliers_array)
        - main_array: Values before first empty bin
        - outliers_array: Values after first empty bin
    """
    if len(im_array) == 0:
        return np.array([]), np.array([])
    
    # Create bins
    min_mass = np.min(im_array)
    max_mass = np.max(im_array)
    nbins = math.ceil((max_mass - min_mass) / bin_width)
    
    if nbins == 0:
        return im_array, np.array([])
    
    bin_edges = np.linspace(min_mass, max_mass, nbins + 1)
    counts, _ = np.histogram(im_array, bins=bin_edges)
    
    # Find first empty bin
    first_empty_bin_idx = None
    for i in range(len(counts)):
        if counts[i] == 0:
            first_empty_bin_idx = i
            break
    
    if first_empty_bin_idx is None or first_empty_bin_idx <= 1:
        # No empty bin found or at position 1, return all as main
        logger.debug("No empty bin found or at position 1, keeping all data in main array")
        return im_array, np.array([])
    
    # Get the mass value at the first empty bin (left edge)
    split_mass = bin_edges[first_empty_bin_idx]
    
    # Split array
    main_array = im_array[im_array < split_mass]
    outliers_array = im_array[im_array >= split_mass]
    
    logger.debug(f"Split at bin {first_empty_bin_idx} (mass = {split_mass:.2f} GeV): "
                f"main={len(main_array)}, outliers={len(outliers_array)}")
    
    return main_array, outliers_array


def init_logging() -> logging.Logger:
    """Initialize and return logger instance."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    return logging.getLogger(__name__)

