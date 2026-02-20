"""
Post-processing pipeline for invariant mass arrays.

Processes IM arrays to:
1. Bin data using specified bin widths
2. Find the rightmost highest bin (peak)
3. Remove data before the peak
4. Split arrays by the first empty bin into main + outliers
"""
import logging
import sys
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import math


def process_im_arrays(config: Dict, file_list: Optional[List[str]] = None) -> List[str]:
    logger = _init_logging()

    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    peak_detection_bin_width_gev = config["peak_detection_bin_width_gev"]

    os.makedirs(output_dir, exist_ok=True)

    if file_list is not None:
        im_array_files = [f for f in file_list if f.endswith(".npy")]
        logger.info(f"Using explicit file list with {len(im_array_files)} IM array files")

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
        if not os.path.exists(input_dir) or len(os.listdir(input_dir)) == 0:
            logger.warning(f"Input directory '{input_dir}' is empty or doesn't exist.")
            return []

        im_array_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]
        if not im_array_files:
            logger.warning(f"No .npy files found in {input_dir}")
            return []

        batch_job_index = config.get("batch_job_index")
        total_batch_jobs = config.get("total_batch_jobs")
        if batch_job_index is not None and total_batch_jobs is not None:
            im_array_files = _get_batch_files(im_array_files, batch_job_index, total_batch_jobs)
            logger.info(f"Batch {batch_job_index}/{total_batch_jobs}: Processing {len(im_array_files)} files")

    total_arrays = len(im_array_files)
    logger.info(f"Processing {total_arrays} IM arrays for peak removal and splitting...")

    processed_files = []
    for im_array_filename in im_array_files:
        try:
            output_files = _process_single_array(
                im_array_filename, input_dir, output_dir, peak_detection_bin_width_gev, logger
            )
            if output_files:
                processed_files.extend(output_files)
        except Exception as e:
            logger.error(f"Error processing {im_array_filename}: {e}", exc_info=True)

    logger.info(f"Post-processing complete. Created {len(processed_files)} processed arrays.")
    return processed_files


def _process_single_array(
    filename: str, input_dir: str, output_dir: str,
    peak_detection_bin_width_gev: float, logger: logging.Logger
) -> List[str]:
    file_path = os.path.join(input_dir, filename)
    im_array = np.load(file_path)

    if len(im_array) == 0:
        logger.warning(f"Array {filename} is empty, skipping")
        return []

    bin_width = peak_detection_bin_width_gev
    peak_mass = _find_rightmost_highest_peak(im_array, bin_width, logger)

    if peak_mass is None:
        logger.warning(f"Could not find peak in {filename}, keeping all data")
        filtered_array = im_array
    else:
        filtered_array = im_array[im_array >= peak_mass]
        logger.debug(
            f"{filename}: Removed {len(im_array) - len(filtered_array)} "
            f"values before peak at {peak_mass:.2f} GeV"
        )

    if len(filtered_array) == 0:
        logger.warning(f"Array {filename} is empty after filtering, skipping")
        return []

    main_array, outliers_array = _split_by_first_empty_bin(filtered_array, bin_width, logger)

    base_name = filename.replace(".npy", "")
    output_files = []

    if len(main_array) > 0:
        main_filename = f"{base_name}_main.npy"
        np.save(os.path.join(output_dir, main_filename), main_array)
        output_files.append(main_filename)
        logger.debug(f"Saved main array: {main_filename} ({len(main_array)} values)")

    if len(outliers_array) > 0:
        outliers_filename = f"{base_name}_outliers.npy"
        np.save(os.path.join(output_dir, outliers_filename), outliers_array)
        output_files.append(outliers_filename)
        logger.debug(f"Saved outliers array: {outliers_filename} ({len(outliers_array)} values)")

    return output_files


def _find_rightmost_highest_peak(
    im_array: np.ndarray, bin_width: float, logger: logging.Logger
) -> Optional[float]:
    if len(im_array) == 0:
        return None

    min_mass = np.min(im_array)
    max_mass = np.max(im_array)
    nbins = math.ceil((max_mass - min_mass) / bin_width)
    if nbins == 0:
        return None

    bin_edges = np.linspace(min_mass, max_mass, nbins + 1)
    counts, _ = np.histogram(im_array, bins=bin_edges)
    if len(counts) == 0:
        return None

    max_count = np.max(counts)
    peak_bin_idx = None
    for i in range(len(counts) - 1, -1, -1):
        if counts[i] == max_count:
            peak_bin_idx = i
            break

    if peak_bin_idx is None:
        return None

    peak_mass = bin_edges[peak_bin_idx]
    logger.debug(f"Found peak at bin {peak_bin_idx} with count {max_count}, mass = {peak_mass:.2f} GeV")
    return peak_mass


def _split_by_first_empty_bin(
    im_array: np.ndarray, bin_width: float, logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray]:
    if len(im_array) == 0:
        return np.array([]), np.array([])

    min_mass = np.min(im_array)
    max_mass = np.max(im_array)
    nbins = math.ceil((max_mass - min_mass) / bin_width)
    if nbins == 0:
        return im_array, np.array([])

    bin_edges = np.linspace(min_mass, max_mass, nbins + 1)
    counts, _ = np.histogram(im_array, bins=bin_edges)

    first_empty_bin_idx = None
    for i in range(len(counts)):
        if counts[i] == 0:
            first_empty_bin_idx = i
            break

    if first_empty_bin_idx is None or first_empty_bin_idx <= 1:
        logger.debug("No empty bin found or at position 1, keeping all data in main array")
        return im_array, np.array([])

    split_mass = bin_edges[first_empty_bin_idx]
    main_array = im_array[im_array < split_mass]
    outliers_array = im_array[im_array >= split_mass]

    logger.debug(
        f"Split at bin {first_empty_bin_idx} (mass = {split_mass:.2f} GeV): "
        f"main={len(main_array)}, outliers={len(outliers_array)}"
    )
    return main_array, outliers_array


def _get_batch_files(files: List[str], batch_index: int, total_batches: int) -> List[str]:
    batch_index = int(batch_index)
    total_batches = int(total_batches)
    total_files = len(files)
    files_per_batch = total_files // total_batches
    start_idx = (batch_index - 1) * files_per_batch

    if batch_index == total_batches:
        end_idx = total_files
    else:
        end_idx = start_idx + files_per_batch
    return files[start_idx:end_idx]


def _init_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)
