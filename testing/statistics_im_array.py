# Statistics about invariant mass distributions
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import heapq
from collections import defaultdict, Counter

inv_mass_path = "/storage/agrp/netalev/data/inv_masses/"

# Configuration: Set to None to analyze all arrays, or set to an integer to analyze top X arrays
top_X = None  # Change to e.g., 10 to analyze only top 10 arrays by entry count

# Parse filename format: 2024r-pp_0f796fe10c1327d1_FS_1e_1m_2j_0p_IM_0e_2j_1m_0p.npy
def parse_filename(filename):
    """Extract background file, FS, and IM from filename"""
    # Remove .npy extension
    name = filename.replace('.npy', '')
    
    # Pattern: {background}_FS_{fs}_IM_{im}
    pattern = r'(.+?)_FS_(.+?)_IM_(.+?)$'
    match = re.match(pattern, name)
    
    if match:
        background = match.group(1)
        fs = match.group(2)
        im = match.group(3)
        return background, fs, im
    return None, None, None

# First pass: Get entry counts, keeping only top X in memory
print("Scanning files to determine entry counts...")
file_info = []  # Will be a list of (entry_count, filename) tuples
total_files = 0

for im_array_npy in os.listdir(inv_mass_path):
    if im_array_npy.endswith('.npy'):
        total_files += 1
        try:
            # Load temporarily just to get length, then let it be garbage collected
            im_array = np.load(inv_mass_path + im_array_npy)
            entry_count = len(im_array)
            del im_array  # Free memory immediately
            
            if top_X is not None and top_X > 0:
                # Use min-heap to keep only top X (min-heap keeps smallest at root)
                if len(file_info) < top_X:
                    heapq.heappush(file_info, (entry_count, im_array_npy))
                elif entry_count > file_info[0][0]:
                    heapq.heapreplace(file_info, (entry_count, im_array_npy))
            else:
                # Store all if top_X is None
                file_info.append((entry_count, im_array_npy))
        except Exception as e:
            print(f"Error loading {im_array_npy}: {e}")

# Process results
if top_X is not None and top_X > 0:
    # Convert heap to sorted list (descending by entry count)
    file_info = sorted(file_info, reverse=True)
    print(f"Selected top {len(file_info)} arrays by entry count (out of {total_files} total)")
    files_to_load = [name for _, name in file_info]
else:
    # Sort by entry count for consistency
    file_info.sort(key=lambda x: x[0], reverse=True)
    print(f"Analyzing all {len(file_info)} arrays...")
    files_to_load = [name for _, name in file_info]

# Second pass: Load only the selected arrays and collect statistics
all_arrays = []
all_masses = []
background_files = set()
final_states = set()
im_combinations = set()
file_stats = []  # List of dicts with file info

for im_array_npy in files_to_load:
    try:
        im_array = np.load(inv_mass_path + im_array_npy)
        background, fs, im_comb = parse_filename(im_array_npy)
        
        all_arrays.append(im_array)
        all_masses.extend(im_array.flatten())
        
        if background:
            background_files.add(background)
        if fs:
            final_states.add(fs)
        if im_comb:
            im_combinations.add(im_comb)
        
        file_stats.append({
            'filename': im_array_npy,
            'background': background,
            'fs': fs,
            'im': im_comb,
            'entries': len(im_array),
            'min_mass': np.min(im_array),
            'max_mass': np.max(im_array),
            'mean_mass': np.mean(im_array),
            'median_mass': np.median(im_array)
        })
    except Exception as e:
        print(f"Error processing {im_array_npy}: {e}")

# Print statistics
print("=" * 80)
print("INVARIANT MASS DISTRIBUTION STATISTICS")
print("=" * 80)
print(f"\nTotal number of .npy files: {len(file_stats)}")
print(f"Number of unique background files: {len(background_files)}")
print(f"Number of unique final states (FS): {len(final_states)}")
print(f"Number of unique IM combinations: {len(im_combinations)}")

print("\n" + "-" * 80)
print("BACKGROUND FILES:")
print("-" * 80)
for bg in sorted(background_files):
    count = sum(1 for f in file_stats if f['background'] == bg)
    print(f"  {bg}: {count} files")

print("\n" + "-" * 80)
print("FINAL STATES (FS):")
print("-" * 80)
for fs in sorted(final_states):
    count = sum(1 for f in file_stats if f['fs'] == fs)
    print(f"  {fs}: {count} files")

print("\n" + "-" * 80)
print("IM COMBINATIONS:")
print("-" * 80)
for im in sorted(im_combinations):
    count = sum(1 for f in file_stats if f['im'] == im)
    print(f"  {im}: {count} files")

print("\n" + "-" * 80)
print("ENTRIES PER FILE:")
print("-" * 80)
entries_list = [f['entries'] for f in file_stats]
print(f"  Total entries across all files: {sum(entries_list):,}")
print(f"  Mean entries per file: {np.mean(entries_list):.2f}")
print(f"  Median entries per file: {np.median(entries_list):.2f}")
print(f"  Min entries: {min(entries_list):,}")
print(f"  Max entries: {max(entries_list):,}")
print(f"  Std entries: {np.std(entries_list):.2f}")

print("\n" + "-" * 80)
print("MASS RANGES:")
print("-" * 80)
if all_masses:
    all_masses_array = np.array(all_masses)
    print(f"  Overall mass range: [{np.min(all_masses_array):.2f}, {np.max(all_masses_array):.2f}] GeV")
    print(f"  Overall mean mass: {np.mean(all_masses_array):.2f} GeV")
    print(f"  Overall median mass: {np.median(all_masses_array):.2f} GeV")
    print(f"  Overall std mass: {np.std(all_masses_array):.2f} GeV")
    
    print("\n  Per-file mass ranges:")
    for f in sorted(file_stats, key=lambda x: x['min_mass'])[:10]:  # Show first 10
        print(f"    {f['filename']}: [{f['min_mass']:.2f}, {f['max_mass']:.2f}] GeV (mean: {f['mean_mass']:.2f}, entries: {f['entries']:,})")
    if len(file_stats) > 10:
        print(f"    ... and {len(file_stats) - 10} more files")

print("\n" + "-" * 80)
print("DETAILED FILE INFORMATION:")
print("-" * 80)
for f in file_stats:
    print(f"\n  File: {f['filename']}")
    print(f"    Background: {f['background']}")
    print(f"    FS: {f['fs']}")
    print(f"    IM: {f['im']}")
    print(f"    Entries: {f['entries']:,}")
    print(f"    Mass range: [{f['min_mass']:.2f}, {f['max_mass']:.2f}] GeV")
    print(f"    Mean: {f['mean_mass']:.2f} GeV, Median: {f['median_mass']:.2f} GeV")

print("\n" + "=" * 80)

# Create visualization with improved number formatting
def format_number_with_scale(value, decimals=2):
    """Format number with scientific notation and bold scale indicator"""
    if value == 0:
        return "0"
    
    # Determine the appropriate scale
    abs_value = abs(value)
    if abs_value >= 1e9:
        scale = 1e9
        scale_label = "× 10⁹"
    elif abs_value >= 1e6:
        scale = 1e6
        scale_label = "× 10⁶"
    elif abs_value >= 1e3:
        scale = 1e3
        scale_label = "× 10³"
    elif abs_value >= 1:
        scale = 1
        scale_label = ""
    elif abs_value >= 1e-3:
        scale = 1e-3
        scale_label = "× 10⁻³"
    elif abs_value >= 1e-6:
        scale = 1e-6
        scale_label = "× 10⁻⁶"
    else:
        scale = 1e-9
        scale_label = "× 10⁻⁹"
    
    scaled_value = value / scale
    if scale_label:
        # Use bold formatting for the scale
        return f"{scaled_value:.{decimals}f} **{scale_label}**"
    else:
        return f"{scaled_value:.{decimals}f}"

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Entry counts distribution
ax1 = fig.add_subplot(gs[0, 0])
entries_list = [f['entries'] for f in file_stats]
if entries_list:
    ax1.hist(entries_list, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Entries per File', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Files', fontsize=11, fontweight='bold')
    ax1.set_title('Distribution of Entry Counts', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    # Format x-axis with scientific notation
    ax1.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    ax1.xaxis.get_offset_text().set_fontweight('bold')

# 2. Mass range distribution
ax2 = fig.add_subplot(gs[0, 1])
if all_masses:
    all_masses_array = np.array(all_masses)
    ax2.hist(all_masses_array, bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax2.set_xlabel('Invariant Mass (GeV)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('Overall Mass Distribution', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)

# 3. Summary statistics text
ax3 = fig.add_subplot(gs[1, :])
ax3.axis('off')
stats_text = []
stats_text.append("SUMMARY STATISTICS")
stats_text.append("")

if entries_list:
    total_entries = sum(entries_list)
    mean_entries = np.mean(entries_list)
    median_entries = np.median(entries_list)
    min_entries = min(entries_list)
    max_entries = max(entries_list)
    
    stats_text.append(f"Total Files: {len(file_stats)}")
    stats_text.append(f"Total Entries: {format_number_with_scale(total_entries)}")
    stats_text.append(f"Mean Entries/File: {format_number_with_scale(mean_entries)}")
    stats_text.append(f"Median Entries/File: {format_number_with_scale(median_entries)}")
    stats_text.append(f"Min Entries: {format_number_with_scale(min_entries)}")
    stats_text.append(f"Max Entries: {format_number_with_scale(max_entries)}")
    stats_text.append("")

if all_masses:
    all_masses_array = np.array(all_masses)
    stats_text.append(f"Overall Mass Range: [{np.min(all_masses_array):.2f}, {np.max(all_masses_array):.2f}] GeV")
    stats_text.append(f"Overall Mean Mass: {np.mean(all_masses_array):.2f} GeV")
    stats_text.append(f"Overall Median Mass: {np.median(all_masses_array):.2f} GeV")
    stats_text.append(f"Overall Std Mass: {np.std(all_masses_array):.2f} GeV")
    stats_text.append("")

stats_text.append(f"Unique Background Files: {len(background_files)}")
stats_text.append(f"Unique Final States: {len(final_states)}")
stats_text.append(f"Unique IM Combinations: {len(im_combinations)}")

# Create text with bold scale indicators
text_content = "\n".join(stats_text)
# Replace **text** with bold formatting and make title bold
text_content = re.sub(r'\*\*(.*?)\*\*', r'$\mathbf{\1}$', text_content)
# Make the title bold
text_content = re.sub(r'^(SUMMARY STATISTICS)$', r'$\mathbf{\1}$', text_content, flags=re.MULTILINE)

ax3.text(0.5, 0.5, text_content, transform=ax3.transAxes, 
         fontsize=11, verticalalignment='center', horizontalalignment='center',
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4. Top files by entry count
ax4 = fig.add_subplot(gs[2, :])
ax4.axis('off')
top_files = sorted(file_stats, key=lambda x: x['entries'], reverse=True)[:15]
files_text = ["TOP 15 FILES BY ENTRY COUNT", ""]
for i, f in enumerate(top_files, 1):
    files_text.append(f"{i:2d}. {f['filename'][:60]}...")
    files_text.append(f"    Entries: {format_number_with_scale(f['entries'])}, "
                     f"Mass: [{f['min_mass']:.2f}, {f['max_mass']:.2f}] GeV")

files_content = "\n".join(files_text)
files_content = re.sub(r'\*\*(.*?)\*\*', r'$\mathbf{\1}$', files_content)
# Make the title bold
files_content = re.sub(r'^(TOP 15 FILES BY ENTRY COUNT)$', r'$\mathbf{\1}$', files_content, flags=re.MULTILINE)

ax4.text(0.05, 0.95, files_content, transform=ax4.transAxes,
         fontsize=9, verticalalignment='top', horizontalalignment='left',
         family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.suptitle('Invariant Mass Distribution Statistics', fontsize=16, fontweight='bold', y=0.995)

# Save plot to the same directory as the script
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, 'statistics_im_array_plot.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")
plt.close()
