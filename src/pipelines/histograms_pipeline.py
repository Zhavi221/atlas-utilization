import logging
import sys
import os
import fcntl
import time
import re
from typing import Dict
from collections import defaultdict
import numpy as np
import ROOT
import math
from typing import List, Tuple, Optional


def create_histograms(histograms_config: Dict, file_list: Optional[List[str]] = None):
    logger = init_logging()

    input_dir = histograms_config["input_dir"]
    output_dir = histograms_config["output_dir"]    
    os.makedirs(output_dir, exist_ok=True)

    bin_widths_gev = histograms_config["bin_widths_gev"]
    use_bumpnet_naming = histograms_config.get("use_bumpnet_naming", False)
    exclude_outliers = histograms_config.get("exclude_outliers", False)
    
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
            return
    else:
        # Scan directory for .npy files (original behavior)
        if not os.path.exists(input_dir) or len(os.listdir(input_dir)) == 0:
            logger.warning(f"Input directory '{input_dir}' is empty or doesn't exist.")
            return
        
        im_array_files = [
            f for f in os.listdir(input_dir) 
            if f.endswith(".npy")
        ]
        
        if not im_array_files:
            logger.warning(f"No .npy files found in {input_dir}")
            return
        
        # Handle batch separation if configured (only when scanning directory)
        batch_job_index = histograms_config.get("batch_job_index")
        total_batch_jobs = histograms_config.get("total_batch_jobs")
        
        if batch_job_index is not None and total_batch_jobs is not None:
            im_array_files = get_batch_files(im_array_files, batch_job_index, total_batch_jobs)
            logger.info(f"Batch {batch_job_index}/{total_batch_jobs}: Processing {len(im_array_files)} files")
    
    # Filter out outlier files if configured
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
        process_im_arrays_bumpnet(
            input_dir, output_dir, bin_widths_gev, logger, im_array_files,
            single_output_file=single_output_file, output_filename=output_filename
        )
    else:
        logger.info("Using standard naming mode")
        process_im_arrays(
            input_dir, output_dir, bin_widths_gev, logger, im_array_files,
            single_output_file=single_output_file, output_filename=output_filename
        )

def group_im_files_by_signature(im_files: List[str]) -> Dict[str, List[str]]:
    """
    Group .npy files by their FS + IM signature (ignoring hash/batch info).
    
    Args:
        im_files: List of .npy filenames
        
    Returns:
        Dictionary mapping BumpNet-compatible names to list of source files
    """
    groups = defaultdict(list)
    
    for filename in im_files:
        # Extract FS and IM parts: ..._FS_<final_state>_IM_<combination>...
        match = re.search(r'_FS_(\d+e_\d+m_\d+j_\d+g)_IM_(\d+e_\d+m_\d+j_\d+g)', filename)
        if match:
            fs_str, im_str = match.groups()
            bumpnet_name = convert_to_bumpnet_name(fs_str, im_str)
            groups[bumpnet_name].append(filename)
    
    return dict(groups)


def convert_to_bumpnet_name(fs_str: str, im_str: str) -> str:
    """
    Convert FS/IM strings to BumpNet naming format.
    
    Args:
        fs_str: Final state string (e.g., "0e_2m_3j_1g")
        im_str: Invariant mass combination string (e.g., "0e_2m_0j_1g")
        
    Returns:
        BumpNet-compatible name (e.g., "mass_m2g1_cat_0ex_2mx_3jx_1gx")
    """
    # Parse IM for combination (e.g., "0e_2m_0j_1g" -> "m2g1")
    # Only include particles with non-zero counts
    particles = re.findall(r'(\d+)([emjg])', im_str)
    combo_parts = [f"{p}{c}" for c, p in particles if c != '0']
    combo = "".join(combo_parts) if combo_parts else "none"
    
    # Parse FS for final state (e.g., "0e_2m_3j_1g" -> "0ex_2mx_3jx_1gx")
    fs_particles = re.findall(r'(\d+)([emjg])', fs_str)
    fs_formatted = "_".join(f"{c}{p}x" for c, p in fs_particles)
    
    return f"mass_{combo}_cat_{fs_formatted}"


def process_im_arrays_bumpnet(
    im_arrays_dir: str, 
    output_dir: str, 
    bin_widths_gev: list, 
    logger: logging.Logger, 
    im_array_files: List[str],
    single_output_file: bool = False,
    output_filename: str = "all_histograms.root"
):
    """
    Process IM arrays with BumpNet-compatible naming and merging.
    
    Groups files by signature, merges arrays, and creates histograms with
    BumpNet naming format (mass_<combo>_cat_<final_state>).
    
    Args:
        im_arrays_dir: Directory containing .npy files
        output_dir: Directory to save ROOT files
        bin_widths_gev: List of bin widths to use
        logger: Logger instance
        im_array_files: List of .npy filenames to process
        single_output_file: If True, write all histograms to single ROOT file
        output_filename: Filename for single output file mode
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Group files by their FS+IM signature
    grouped_files = group_im_files_by_signature(im_array_files)
    logger.info(f"Grouped {len(im_array_files)} files into {len(grouped_files)} unique signatures")
    
    if single_output_file:
        root_filepath = os.path.join(output_dir, output_filename)
        hist_count = 0
        
        for bumpnet_name, matching_files in grouped_files.items():
            logger.info(f"Processing {bumpnet_name}: merging {len(matching_files)} files")
            
            hists = create_merged_histograms_streaming(
                matching_files, im_arrays_dir, bumpnet_name, bin_widths_gev, logger
            )
            
            if hists:
                write_hists_to_shared_file(hists, root_filepath, logger)
                hist_count += len(hists)
        
        logger.info(f"Wrote {hist_count} histograms to shared file {root_filepath}")
    else:
        # One ROOT file per signature
        for bumpnet_name, matching_files in grouped_files.items():
            logger.info(f"Processing {bumpnet_name}: merging {len(matching_files)} files")
            
            hists = create_merged_histograms_streaming(
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


def create_merged_histograms_streaming(
    files: List[str], 
    directory: str, 
    hist_name_base: str, 
    bin_widths_gev: list,
    logger: logging.Logger
) -> List[ROOT.TH1F]:
    """
    Create histograms by streaming through files - memory efficient.
    
    Loads one array at a time to avoid memory issues with large datasets.
    
    Args:
        files: List of .npy filenames to merge
        directory: Directory containing the files
        hist_name_base: Base name for histograms (BumpNet format)
        bin_widths_gev: List of bin widths
        logger: Logger instance
        
    Returns:
        List of ROOT.TH1F histograms (one per bin width)
    """
    # First pass: find global min/max for consistent binning
    global_min, global_max = float('inf'), float('-inf')
    total_entries = 0
    
    for f in files:
        try:
            arr = np.load(os.path.join(directory, f))
            if len(arr) > 0:
                global_min = min(global_min, np.min(arr))
                global_max = max(global_max, np.max(arr))
                total_entries += len(arr)
            del arr  # Free memory
        except Exception as e:
            logger.warning(f"Error reading {f} for min/max: {e}")
            continue
    
    if global_min == float('inf') or global_max == float('-inf'):
        logger.warning(f"No valid data found for {hist_name_base}")
        return []
    
    logger.debug(f"{hist_name_base}: {total_entries} entries, range [{global_min:.2f}, {global_max:.2f}]")
    
    # Create histograms for each bin width
    histograms = []
    for bin_width in bin_widths_gev:
        nbins = max(1, math.ceil((global_max - global_min) / bin_width))
        hist_name = f"ROI_{hist_name_base}_width_{bin_width}"
        hist = ROOT.TH1F(hist_name, hist_name, nbins, global_min, global_max)
        histograms.append(hist)
    
    # Second pass: fill histograms one file at a time
    for f in files:
        try:
            arr = np.load(os.path.join(directory, f))
            for hist in histograms:
                for val in arr:
                    hist.Fill(val)
            del arr  # Free memory immediately
        except Exception as e:
            logger.warning(f"Error filling from {f}: {e}")
            continue
    
    return histograms


def process_im_arrays(
    im_arrays_dir: str, 
    output_dir: str, 
    bin_widths_gev: list, 
    logger: logging.Logger, 
    im_array_files: Optional[List[str]] = None,
    single_output_file: bool = False,
    output_filename: str = "all_histograms.root"
):
    if im_array_files is None:
        # Fallback to scanning directory if no file list provided
        im_array_files = [f for f in os.listdir(im_arrays_dir) if f.endswith(".npy")]
    
    os.makedirs(output_dir, exist_ok=True)
    
    if single_output_file:
        # Single output file mode with file locking for concurrent writes
        root_filepath = os.path.join(output_dir, output_filename)
        hist_count = 0
        
        for im_array_filename in im_array_files:
            hists = make_histograms_single_file(
                im_array_filename, im_arrays_dir, bin_widths_gev, logger)
            
            if hists:
                write_hists_to_shared_file(hists, root_filepath, logger)
                hist_count += len(hists)
        
        logger.info(f"Wrote {hist_count} histograms to shared file {root_filepath}")
    else:
        # Original behavior: one ROOT file per input file
        for im_array_filename in im_array_files:
            hists = make_histograms_single_file(
                im_array_filename, im_arrays_dir, bin_widths_gev, logger)
            
            save_hists(
                hists, output_dir, im_array_filename, logger)

def make_histograms_single_file(im_array_filename: str, im_arrays_dir: str, bin_widths_gev: list, logger: logging.Logger):
    im_array = np.load(os.path.join(im_arrays_dir, im_array_filename))

    hists = []
    for bin_width in bin_widths_gev:
        # Create histogram directly from array (no splitting needed - already done in post-processing)
        hist = create_histogram_single_array(im_array_filename, im_array, bin_width)
        hists.append(hist)

    return hists

def create_histogram_single_array(im_array_filename, im_array, bin_width) -> ROOT.TH1F:
    nbins = math.ceil((np.max(im_array) - np.min(im_array)) / bin_width)
    bin_edges = np.linspace(np.min(im_array), np.max(im_array), nbins + 1)

    # Create ROOT histogram
    hist_name = f"ROI_{im_array_filename}_width_{bin_width}"
    hist = ROOT.TH1F(
        hist_name, hist_name, len(bin_edges)-1, bin_edges)

    for mass in im_array:
        hist.Fill(mass)

    return hist

def split_histogram_at_bin_idx(hist, original_hist_name, bin_idx) -> Tuple[ROOT.TH1F, ROOT.TH1F]:
    first_part_edges = []
    for i in range(1, bin_idx + 1):
        first_part_edges.append(hist.GetXaxis().GetBinLowEdge(i))
    first_part_edges.append(hist.GetXaxis().GetBinUpEdge(bin_idx))
    
    hist1_name = f"{original_hist_name}_main"
    hist1 = ROOT.TH1F(
        hist1_name, hist1_name,
        len(first_part_edges) - 1, 
        np.array(first_part_edges, dtype='float64')
    )
    
    # Copy bin contents and errors for first part
    for i in range(1, bin_idx):
        hist1.SetBinContent(i, hist.GetBinContent(i))
        hist1.SetBinError(i, hist.GetBinError(i))
    
    second_part_edges = []
    total_bins = hist.GetXaxis().GetNbins()
    for i in range(bin_idx, total_bins):
        second_part_edges.append(hist.GetXaxis().GetBinLowEdge(i))
    second_part_edges.append(hist.GetXaxis().GetBinUpEdge(total_bins))
    
    hist2_name = f"{original_hist_name}_outliers"
    hist2 = ROOT.TH1F(
        hist2_name, hist2_name,
        len(second_part_edges) - 1, 
        np.array(second_part_edges, dtype='float64')
    )
    
    # Copy bin contents and errors for second part
    for i in range(bin_idx, total_bins):
        hist2.SetBinContent(i, hist.GetBinContent(i))
        hist2.SetBinError(i, hist.GetBinError(i)) 

    return hist1, hist2


def save_hists(hists: List[ROOT.TH1F], output_dir: str, im_array_filename: str, logger: logging.Logger) -> None:
    """
    Save ROOT histograms to a ROOT file.
    
    Args:
        hists: List of ROOT.TH1F histograms to save
        output_dir: Directory to save the ROOT file
        im_array_filename: Original .npy filename (used to name the ROOT file)
        logger: Logger instance
    """
    if not hists:
        logger.warning(f"No histograms to save for {im_array_filename}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create ROOT file name based on the im_array_filename
    # Remove .npy extension and add .root
    base_name = im_array_filename.replace(".npy", "")
    root_filename = f"{base_name}_hists.root"
    root_filepath = os.path.join(output_dir, root_filename)
    
    # Open/create ROOT file
    root_file = ROOT.TFile(root_filepath, "RECREATE")
    
    # Write all histograms to the file
    for hist in hists:
        hist.Write()
    
    # Close the file
    root_file.Close()
    
    logger.debug(f"Saved {len(hists)} histograms to {root_filepath}")


def write_hists_to_shared_file(
    hists: List[ROOT.TH1F], 
    root_filepath: str, 
    logger: logging.Logger,
    max_retries: int = 10,
    retry_delay: float = 0.5
) -> None:
    """
    Write histograms to a shared ROOT file with file locking for concurrent access.
    
    Uses fcntl file locking to ensure only one process writes at a time.
    Uses UPDATE mode to append to existing file or create if doesn't exist.
    
    Args:
        hists: List of ROOT.TH1F histograms to save
        root_filepath: Path to the shared ROOT file
        logger: Logger instance
        max_retries: Maximum number of lock acquisition attempts
        retry_delay: Delay between retries in seconds
    """
    if not hists:
        return
    
    lock_filepath = root_filepath + ".lock"
    
    for attempt in range(max_retries):
        try:
            # Create/open lock file
            lock_file = open(lock_filepath, 'w')
            
            try:
                # Acquire exclusive lock (blocking with timeout via retries)
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                # Open ROOT file in UPDATE mode (creates if doesn't exist, appends if exists)
                if os.path.exists(root_filepath):
                    root_file = ROOT.TFile(root_filepath, "UPDATE")
                else:
                    root_file = ROOT.TFile(root_filepath, "RECREATE")
                
                if not root_file or root_file.IsZombie():
                    logger.error(f"Failed to open ROOT file: {root_filepath}")
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                    lock_file.close()
                    return
                
                # Write histograms
                for hist in hists:
                    hist.Write()
                
                root_file.Close()
                
                # Release lock
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                
                logger.debug(f"Wrote {len(hists)} histograms to {root_filepath}")
                return
                
            except BlockingIOError:
                # Lock not available, close and retry
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
                except:
                    pass
            return


def get_batch_files(
    files: List[str],
    batch_index: int,
    total_batches: int
) -> List[str]:
    """
    Split files into batches for distributed processing.
    
    Args:
        files: List of all file names
        batch_index: Current batch index (1-indexed)
        total_batches: Total number of batches
        
    Returns:
        List of files for this batch
    """
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


def init_logging() -> logging.Logger:
    """Initialize and return logger instance."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    return logging.getLogger(__name__)
