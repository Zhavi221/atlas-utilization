# NPY to ROOT Conversion

This folder contains tools to convert `.npy` invariant mass arrays into ROOT TH1 histograms for use with the BumpNet pipeline.

## Files

- **`tutorial_npy_to_root.ipynb`**: Interactive tutorial notebook with explanations and visualizations
- **`convert_npy_to_root.py`**: Python script to perform the conversion

## Quick Start

### 1. Explore the Tutorial

Open `tutorial_npy_to_root.ipynb` to:
- Understand the conversion process
- Visualize the binning strategy
- See examples of how arrays are converted

### 2. Run the Conversion Script

```bash
cd /srv01/agrp/netalev/atlas_utilization/testing/npy_to_root_conversion
python convert_npy_to_root.py \
    --input_dir /storage/agrp/netalev/data/inv_masses/ \
    --output_file rebinned.root \
    --bin_width 1.0
```

## Script Options

```bash
python convert_npy_to_root.py --help
```

**Required:**
- `--output_file`: Path to output ROOT file

**Optional:**
- `--input_dir`: Directory with .npy files (default: `/storage/agrp/netalev/data/inv_masses/`)
- `--bin_width`: Bin width in GeV (default: 1.0)
- `--prefix`: Histogram name prefix (default: "ROI_")
- `--min_entries`: Minimum entries per histogram (default: 100)
- `--quiet`: Suppress progress output

## Examples

### Basic conversion with 1 GeV bins
```bash
python convert_npy_to_root.py --output_file rebinned_1gev.root
```

### Custom bin width
```bash
python convert_npy_to_root.py --output_file rebinned_2gev.root --bin_width 2.0
```

### Higher minimum entries threshold
```bash
python convert_npy_to_root.py --output_file rebinned.root --min_entries 200
```

### Custom input directory
```bash
python convert_npy_to_root.py \
    --input_dir /path/to/your/npy/files \
    --output_file output.root
```

## How It Works

1. **Scan Phase**: Finds global min/max mass values across all arrays
2. **Binning Phase**: Creates consistent bin edges with specified bin width
3. **Conversion Phase**: Converts each array to ROOT TH1 histogram
4. **Output**: Saves all histograms to a single ROOT file

## Key Features

- **Consistent Binning**: All histograms use the same bin edges for direct comparison
- **Automatic Range Detection**: Finds global range automatically
- **Error Handling**: Skips problematic files and reports issues
- **Progress Tracking**: Shows progress bars during conversion
- **ROOT Compatibility**: Creates TH1 histograms compatible with BumpNet

## Requirements

- Python 3.7+
- numpy
- ROOT (PyROOT)
- tqdm (for progress bars)

## Notes

- Empty bins are perfectly fine - they represent "no data" regions
- All arrays are converted from MeV to GeV automatically
- Histogram names are prefixed with "ROI_" by default (BumpNet convention)
- Files with fewer than `min_entries` entries are skipped

