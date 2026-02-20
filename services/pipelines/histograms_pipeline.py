"""
Histogram creation pipeline.

Builds ROOT histograms from processed invariant-mass arrays.
Supports BumpNet-compatible naming and concurrent file writing.
"""
import logging
import sys
import os
import fcntl
import time
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import ROOT
import math


def create_histograms(histograms_config: Dict, file_list: Optional[List[str]] = None):
    logger = _init_logging()

    input_dir = histograms_config["input_dir"]
    output_dir = histograms_config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    bin_width_gev = histograms_config["bin_width_gev"]
    if isinstance(bin_width_gev, (int, float)):
        bin_widths_gev = [bin_width_gev]
    else:
        bin_widths_gev = bin_width_gev
    use_bumpnet_naming = histograms_config.get("use_bumpnet_naming", False)
    exclude_outliers = histograms_config.get("exclude_outliers", False)

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
            return
    else:
        if not os.path.exists(input_dir) or len(os.listdir(input_dir)) == 0:
            logger.warning(f"Input directory '{input_dir}' is empty or doesn't exist.")
            return

        im_array_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]
        if not im_array_files:
            logger.warning(f"No .npy files found in {input_dir}")
            return

        batch_job_index = histograms_config.get("batch_job_index")
        total_batch_jobs = histograms_config.get("total_batch_jobs")
        if batch_job_index is not None and total_batch_jobs is not None:
            im_array_files = _get_batch_files(im_array_files, batch_job_index, total_batch_jobs)
            logger.info(f"Batch {batch_job_index}/{total_batch_jobs}: Processing {len(im_array_files)} files")

    if exclude_outliers:
        before_count = len(im_array_files)
        im_array_files = [f for f in im_array_files if "_outliers" not in f]
        excluded_count = before_count - len(im_array_files)
        if excluded_count > 0:
            logger.info(f"Excluded {excluded_count} outlier files (exclude_outliers=true)")

    total_im_arrays = len(im_array_files)
    logger.info(f"Found {total_im_arrays} IM array files to process")

    single_output_file = histograms_config.get("single_output_file", False)
    output_filename = histograms_config.get("output_filename", "all_histograms.root")

    if use_bumpnet_naming:
        logger.info("Using BumpNet naming mode: grouping files by signature and merging")
        _process_im_arrays_bumpnet(
            input_dir, output_dir, bin_widths_gev, logger, im_array_files,
            single_output_file=single_output_file, output_filename=output_filename
        )
    else:
        logger.info("Using standard naming mode")
        _process_im_arrays_standard(
            input_dir, output_dir, bin_widths_gev, logger, im_array_files,
            single_output_file=single_output_file, output_filename=output_filename
        )


def _group_im_files_by_signature(im_files: List[str]) -> Dict[str, List[str]]:
    groups = defaultdict(list)
    unmatched_files = []
    for filename in im_files:
        match = re.search(r'_FS_(\d+e_\d+m_\d+j_\d+g)_IM_(\d+e_\d+m_\d+j_\d+g)', filename)
        if match:
            fs_str, im_str = match.groups()
            bumpnet_name = _convert_to_bumpnet_name(fs_str, im_str)
            groups[bumpnet_name].append(filename)
        else:
            unmatched_files.append(filename)

    if unmatched_files:
        print(f"WARNING: {len(unmatched_files)} files didn't match FS/IM pattern and will be skipped")
    return dict(groups)


def _convert_to_bumpnet_name(fs_str: str, im_str: str) -> str:
    particles = re.findall(r'(\d+)([emjg])', im_str)
    combo_parts = [f"{p}{c}" for c, p in particles if c != '0']
    combo = "".join(combo_parts) if combo_parts else "none"

    fs_particles = re.findall(r'(\d+)([emjg])', fs_str)
    fs_formatted = "_".join(f"{c}{p}x" for c, p in fs_particles)

    result = f"mass_{combo}_cat_{fs_formatted}"

    if 'cat' not in result and 'hCat' not in result:
        raise ValueError(
            f"Generated histogram name '{result}' doesn't contain 'cat' or 'hCat' "
            "- this will cause UnboundLocalError in BumpNet"
        )
    return result


def _process_im_arrays_bumpnet(
    im_arrays_dir: str, output_dir: str, bin_widths_gev: list,
    logger: logging.Logger, im_array_files: List[str],
    single_output_file: bool = False, output_filename: str = "all_histograms.root"
):
    os.makedirs(output_dir, exist_ok=True)
    grouped_files = _group_im_files_by_signature(im_array_files)
    logger.info(f"Grouped {len(im_array_files)} files into {len(grouped_files)} unique signatures")

    if single_output_file:
        root_filepath = os.path.join(output_dir, output_filename)
        hist_count = 0

        for bumpnet_name, matching_files in grouped_files.items():
            logger.info(f"Processing {bumpnet_name}: merging {len(matching_files)} files")
            hists = _create_merged_histograms_streaming(
                matching_files, im_arrays_dir, bumpnet_name, bin_widths_gev, logger
            )
            if hists:
                _write_hists_to_shared_file(hists, root_filepath, logger)
                hist_count += len(hists)

        logger.info(f"Wrote {hist_count} histograms to shared file {root_filepath}")
    else:
        for bumpnet_name, matching_files in grouped_files.items():
            logger.info(f"Processing {bumpnet_name}: merging {len(matching_files)} files")
            hists = _create_merged_histograms_streaming(
                matching_files, im_arrays_dir, bumpnet_name, bin_widths_gev, logger
            )
            if hists:
                root_filename = f"{bumpnet_name}_hists.root"
                root_filepath = os.path.join(output_dir, root_filename)
                root_file = ROOT.TFile(root_filepath, "RECREATE")
                for hist in hists:
                    hist.Write()
                root_file.Close()
                logger.debug(f"Saved {len(hists)} histograms to {root_filepath}")


def _create_merged_histograms_streaming(
    files: List[str], directory: str, hist_name_base: str,
    bin_widths_gev: list, logger: logging.Logger
) -> List[ROOT.TH1F]:
    global_min, global_max = float('inf'), float('-inf')
    total_entries = 0

    for f in files:
        try:
            arr = np.load(os.path.join(directory, f))
            if len(arr) > 0:
                global_min = min(global_min, np.min(arr))
                global_max = max(global_max, np.max(arr))
                total_entries += len(arr)
            del arr
        except Exception as e:
            logger.warning(f"Error reading {f} for min/max: {e}")
            continue

    if global_min == float('inf') or global_max == float('-inf'):
        logger.warning(f"No valid data found for {hist_name_base}")
        return []

    logger.debug(f"{hist_name_base}: {total_entries} entries, range [{global_min:.2f}, {global_max:.2f}]")

    histograms = []
    for bin_width in bin_widths_gev:
        nbins = max(1, math.ceil((global_max - global_min) / bin_width))
        hist_name = f"ROI_{hist_name_base}_width_{bin_width}"
        if 'cat' not in hist_name_base and 'hCat' not in hist_name_base:
            logger.error(
                f"CRITICAL: Histogram name base '{hist_name_base}' doesn't contain 'cat' or 'hCat' "
                "- this will cause UnboundLocalError in BumpNet!"
            )
            raise ValueError(
                f"Invalid histogram name base '{hist_name_base}': must contain 'cat' for BumpNet compatibility"
            )
        hist = ROOT.TH1F(hist_name, hist_name, nbins, global_min, global_max)
        histograms.append(hist)

    for f in files:
        try:
            arr = np.load(os.path.join(directory, f))
            for hist in histograms:
                for val in arr:
                    hist.Fill(val)
            del arr
        except Exception as e:
            logger.warning(f"Error filling from {f}: {e}")
            continue

    return histograms


def _process_im_arrays_standard(
    im_arrays_dir: str, output_dir: str, bin_widths_gev: list,
    logger: logging.Logger, im_array_files: Optional[List[str]] = None,
    single_output_file: bool = False, output_filename: str = "all_histograms.root"
):
    if im_array_files is None:
        im_array_files = [f for f in os.listdir(im_arrays_dir) if f.endswith(".npy")]

    os.makedirs(output_dir, exist_ok=True)

    if single_output_file:
        root_filepath = os.path.join(output_dir, output_filename)
        hist_count = 0

        for im_array_filename in im_array_files:
            hists = _make_histograms_single_file(im_array_filename, im_arrays_dir, bin_widths_gev, logger)
            if hists:
                _write_hists_to_shared_file(hists, root_filepath, logger)
                hist_count += len(hists)

        logger.info(f"Wrote {hist_count} histograms to shared file {root_filepath}")
    else:
        for im_array_filename in im_array_files:
            hists = _make_histograms_single_file(im_array_filename, im_arrays_dir, bin_widths_gev, logger)
            _save_hists(hists, output_dir, im_array_filename, logger)


def _make_histograms_single_file(
    im_array_filename: str, im_arrays_dir: str,
    bin_widths_gev: list, logger: logging.Logger
):
    im_array = np.load(os.path.join(im_arrays_dir, im_array_filename))
    hists = []
    for bin_width in bin_widths_gev:
        hist = _create_histogram_single_array(im_array_filename, im_array, bin_width)
        hists.append(hist)
    return hists


def _create_histogram_single_array(im_array_filename, im_array, bin_width) -> ROOT.TH1F:
    nbins = math.ceil((np.max(im_array) - np.min(im_array)) / bin_width)
    bin_edges = np.linspace(np.min(im_array), np.max(im_array), nbins + 1)

    hist_name = f"ROI_{im_array_filename}_width_{bin_width}"
    hist = ROOT.TH1F(hist_name, hist_name, len(bin_edges) - 1, bin_edges)

    for mass in im_array:
        hist.Fill(mass)
    return hist


def _save_hists(hists: List[ROOT.TH1F], output_dir: str,
                im_array_filename: str, logger: logging.Logger) -> None:
    if not hists:
        logger.warning(f"No histograms to save for {im_array_filename}")
        return

    os.makedirs(output_dir, exist_ok=True)
    base_name = im_array_filename.replace(".npy", "")
    root_filename = f"{base_name}_hists.root"
    root_filepath = os.path.join(output_dir, root_filename)

    root_file = ROOT.TFile(root_filepath, "RECREATE")
    for hist in hists:
        hist.Write()
    root_file.Close()
    logger.debug(f"Saved {len(hists)} histograms to {root_filepath}")


def _write_hists_to_shared_file(
    hists: List[ROOT.TH1F], root_filepath: str,
    logger: logging.Logger, max_retries: int = 10, retry_delay: float = 0.5
) -> None:
    if not hists:
        return

    lock_filepath = root_filepath + ".lock"

    for attempt in range(max_retries):
        try:
            lock_file = open(lock_filepath, 'w')
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                if os.path.exists(root_filepath):
                    root_file = ROOT.TFile(root_filepath, "UPDATE")
                else:
                    root_file = ROOT.TFile(root_filepath, "RECREATE")

                if not root_file or root_file.IsZombie():
                    logger.error(f"Failed to open ROOT file: {root_filepath}")
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                    lock_file.close()
                    return

                for hist in hists:
                    hist.Write()
                root_file.Close()

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                logger.debug(f"Wrote {len(hists)} histograms to {root_filepath}")
                return

            except BlockingIOError:
                lock_file.close()
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"Failed to acquire lock after {max_retries} attempts for {root_filepath}")
                    return

        except Exception as e:
            logger.error(f"Error writing to shared ROOT file: {e}")
            if 'lock_file' in locals() and not lock_file.closed:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                    lock_file.close()
                except Exception:
                    pass
            return


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
