#!/usr/bin/env python3
"""
Convert .npy invariant mass arrays to ROOT histograms for BumpNet pipeline.

This script:
1. Scans all .npy files to find global mass range
2. Creates consistent binning (1 GeV default)
3. Converts each array to a ROOT TH1 histogram
4. Saves all histograms to a ROOT file

Usage:
    python convert_npy_to_root.py --input_dir /path/to/npy/files --output_file output.root
"""

import numpy as np
import ROOT
import math
import os
import argparse
from pathlib import Path
from tqdm import tqdm


def find_global_mass_range(input_dir, verbose=True):
    """
    Scan all .npy files to find global min and max mass values.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing .npy files
    verbose : bool
        Print progress information
    
    Returns:
    --------
    global_min : float
        Global minimum mass in GeV
    global_max : float
        Global maximum mass in GeV
    """
    npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    
    if len(npy_files) == 0:
        raise ValueError(f"No .npy files found in {input_dir}")
    
    if verbose:
        print(f"Scanning {len(npy_files)} files to find global mass range...")
    
    all_mins = []
    all_maxs = []
    
    for im_array_npy in tqdm(npy_files, desc="Scanning arrays", disable=not verbose):
        try:
            im_array = np.load(os.path.join(input_dir, im_array_npy))
            im_array_gev = im_array * 1e-3  # Convert MeV to GeV
            all_mins.append(np.min(im_array_gev))
            all_maxs.append(np.max(im_array_gev))
        except Exception as e:
            if verbose:
                print(f"Warning: Error loading {im_array_npy}: {e}")
    
    global_min = np.min(all_mins)
    global_max = np.max(all_maxs)
    
    if verbose:
        print(f"\nGlobal mass range: [{global_min:.2f}, {global_max:.2f}] GeV")
        print(f"Range span: {global_max - global_min:.2f} GeV")
    
    return global_min, global_max


def create_bin_edges(global_min, global_max, bin_width):
    """
    Create bin edges with specified bin width.
    
    Parameters:
    -----------
    global_min : float
        Minimum mass value (GeV)
    global_max : float
        Maximum mass value (GeV)
    bin_width : float
        Desired bin width (GeV)
    
    Returns:
    --------
    bin_edges : np.ndarray
        Array of bin edges
    nbins : int
        Number of bins
    """
    nbins = math.ceil((global_max - global_min) / bin_width)
    bin_edges = np.linspace(global_min, global_max, nbins + 1)
    return bin_edges, nbins


def convert_npy_to_root(
    input_dir,
    output_file,
    bin_width=1.0,
    hist_name_prefix="ROI_",
    min_entries=100,
    verbose=True
):
    """
    Convert all .npy files to ROOT histograms with consistent binning.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing .npy files
    output_file : str
        Path to output ROOT file
    bin_width : float
        Bin width in GeV (default: 1.0)
    hist_name_prefix : str
        Prefix for histogram names (default: "ROI_")
    min_entries : int
        Minimum number of entries to include a histogram
    verbose : bool
        Print progress information
    """
    # Step 1: Find global mass range
    global_min, global_max = find_global_mass_range(input_dir, verbose=verbose)
    
    # Step 2: Create consistent bin edges
    bin_edges, nbins = create_bin_edges(global_min, global_max, bin_width)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Binning Configuration")
        print(f"{'='*60}")
        print(f"Bin width: {bin_width} GeV")
        print(f"Number of bins: {nbins}")
        print(f"Bin edges: [{bin_edges[0]:.2f}, ..., {bin_edges[-1]:.2f}] GeV")
        print(f"{'='*60}\n")
    
    # Step 3: Create ROOT file
    if verbose:
        print(f"Creating ROOT file: {output_file}")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    root_file = ROOT.TFile(output_file, "RECREATE")
    
    # Step 4: Convert each array
    npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    n_created = 0
    n_skipped = 0
    skipped_reasons = {}
    
    for im_array_npy in tqdm(npy_files, desc="Converting to ROOT", disable=not verbose):
        try:
            # Load and convert to GeV
            im_array = np.load(os.path.join(input_dir, im_array_npy))
            im_array_gev = im_array * 1e-3
            masses = im_array_gev.flatten()
            
            # Remove invalid values
            masses = masses[np.isfinite(masses)]
            
            # Check minimum entries
            if len(masses) < min_entries:
                n_skipped += 1
                skipped_reasons['low_entries'] = skipped_reasons.get('low_entries', 0) + 1
                continue
            
            # Create histogram name
            base_name = Path(im_array_npy).stem
            hist_name = f"{hist_name_prefix}{base_name}"
            
            # Create ROOT histogram with consistent binning
            hist = ROOT.TH1F(
                hist_name,
                hist_name,
                len(bin_edges) - 1,  # number of bins
                bin_edges  # bin edges array
            )
            
            # Fill histogram
            for mass in masses:
                hist.Fill(mass)
            
            # Set errors (sqrt of content)
            for i in range(1, len(bin_edges)):
                content = hist.GetBinContent(i)
                hist.SetBinError(i, np.sqrt(content) if content > 0 else 0)
            
            # Write to ROOT file
            hist.Write()
            n_created += 1
            
        except Exception as e:
            if verbose:
                print(f"Error processing {im_array_npy}: {e}")
            n_skipped += 1
            skipped_reasons['error'] = skipped_reasons.get('error', 0) + 1
            continue
    
    root_file.Close()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Conversion Complete!")
    print(f"{'='*60}")
    print(f"Created: {n_created} histograms")
    print(f"Skipped: {n_skipped} files")
    if skipped_reasons:
        print(f"\nSkipped reasons:")
        for reason, count in skipped_reasons.items():
            print(f"  {reason}: {count}")
    print(f"\nOutput: {output_file}")
    print(f"All histograms use consistent {bin_width} GeV binning")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert .npy invariant mass arrays to ROOT histograms for BumpNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default 1 GeV binning
  python convert_npy_to_root.py --input_dir /path/to/npy --output_file output.root
  
  # Custom bin width
  python convert_npy_to_root.py --input_dir /path/to/npy --output_file output.root --bin_width 2.0
  
  # Minimum entries threshold
  python convert_npy_to_root.py --input_dir /path/to/npy --output_file output.root --min_entries 200
        """
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/storage/agrp/netalev/data/inv_masses/",
        help="Directory containing .npy files (default: /storage/agrp/netalev/data/inv_masses/)"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output ROOT file path (e.g., rebinned.root)"
    )
    
    parser.add_argument(
        "--bin_width",
        type=float,
        default=1.0,
        help="Bin width in GeV (default: 1.0)"
    )
    
    parser.add_argument(
        "--prefix",
        type=str,
        default="ROI_",
        help="Histogram name prefix (default: ROI_)"
    )
    
    parser.add_argument(
        "--min_entries",
        type=int,
        default=100,
        help="Minimum entries per histogram (default: 100)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    
    if args.bin_width <= 0:
        raise ValueError("Bin width must be positive")
    
    # Run conversion
    convert_npy_to_root(
        input_dir=args.input_dir,
        output_file=args.output_file,
        bin_width=args.bin_width,
        hist_name_prefix=args.prefix,
        min_entries=args.min_entries,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()

