import logging
import sys
import os
from typing import Dict
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
    
    total_im_arrays = len(im_array_files)
    logger.info(f"Making {total_im_arrays} histograms...")
    
    process_im_arrays(
        input_dir, output_dir, bin_widths_gev, logger, im_array_files
    )

def process_im_arrays(im_arrays_dir: str, output_dir: str, bin_widths_gev: list, logger: logging.Logger, im_array_files: Optional[List[str]] = None):
    if im_array_files is None:
        # Fallback to scanning directory if no file list provided
        im_array_files = [f for f in os.listdir(im_arrays_dir) if f.endswith(".npy")]
    
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

def init_logging() -> logging.Logger:
    """Initialize and return logger instance."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    return logging.getLogger(__name__)
