"""
Calculate number of batch jobs needed for IM pipeline.

Based on number of ROOT files, combinations, and estimated final states per file.
The complexity is: files * combinations * average_final_states_per_file
"""
import os
import math
import argparse
import yaml
import sys
from pathlib import Path
import random

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.calculations import combinatorics
from src.parse_atlas import parser
from src.im_calculator.im_calculator import IMCalculator
from typing import List

CONFIG_PATH = "configs/pipeline_config.yaml"
TIME_FOR_WORK_UNIT_SEC = 2  # Estimated time per file+combination+final_state unit
WALLTIME_PER_JOB_SEC = 24 * 3600  # 24 hours
SAMPLE_FILES_FOR_FS_ESTIMATE = 5  # Number of files to sample for final state estimation


def _estimate_avg_final_states(
    input_dir: str,
    root_files: List[str],
    sample_size: int,
    config: dict
) -> float:
    """
    Estimate average number of final states per file by sampling.
    
    Args:
        input_dir: Directory containing ROOT files
        root_files: List of all ROOT file names
        sample_size: Number of files to sample
        config: Configuration dictionary
        
    Returns:
        Average number of final states per file
    """
    sample_files = random.sample(root_files, min(sample_size, len(root_files)))
    final_state_counts = []
    
    for filename in sample_files:
        try:
            file_path = os.path.join(input_dir, filename)
            particle_arrays = parser.AtlasOpenParser.parse_root_file(file_path)
            
            if particle_arrays is None or len(particle_arrays) == 0:
                continue
            
            calculator = IMCalculator(particle_arrays)
            # Count unique final states
            unique_fs = set()
            for fs, _ in calculator.group_by_final_state():
                unique_fs.add(fs)
            
            final_state_counts.append(len(unique_fs))
        except Exception as e:
            # Skip files that can't be parsed
            continue
    
    if not final_state_counts:
        # Fallback: use a conservative estimate based on particle count ranges
        # Typical final states: combinations of e, m, j, p counts
        # With limits (0-6e, 0-6m, 2-8j, 0-5p), we might have ~100-200 unique states
        # But most files won't have all of them, so estimate conservatively
        return 20.0  # Conservative default
    
    avg_fs = sum(final_state_counts) / len(final_state_counts)
    return max(avg_fs, 1.0)  # At least 1 final state per file


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", help="Config file", default=CONFIG_PATH)
    arg_parser.add_argument("--time_for_work_unit_sec", type=float, default=TIME_FOR_WORK_UNIT_SEC,
                           help="Estimated time per work unit (file*combination*final_state)")
    arg_parser.add_argument("--walltime_per_job_sec", type=int, default=WALLTIME_PER_JOB_SEC)
    arg_parser.add_argument("--input_dir", help="Input directory with ROOT files", default=None)
    arg_parser.add_argument("--sample_size", type=int, default=SAMPLE_FILES_FOR_FS_ESTIMATE,
                           help="Number of files to sample for final state estimation")
    
    args = arg_parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Get input directory
    input_dir = args.input_dir or config["mass_calculate"]["input_dir"]
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Count ROOT files
    root_files = [f for f in os.listdir(input_dir) if f.endswith(".root")]
    num_files = len(root_files)
    
    if num_files == 0:
        print(f"Error: No ROOT files found in '{input_dir}'", file=sys.stderr)
        sys.exit(1)
    
    # Get number of combinations
    im_config = config["mass_calculate"]
    all_combinations = combinatorics.get_all_combinations(
        im_config["objects_to_calculate"],
        min_particles=im_config["min_particles"],
        max_particles=im_config["max_particles"],
        min_count=im_config["min_count"],
        max_count=im_config["max_count"],
        limit=im_config.get("limit_combinations")
    )
    num_combinations = len(all_combinations)
    
    # Estimate average final states per file by sampling
    avg_final_states_per_file = _estimate_avg_final_states(
        input_dir, root_files, args.sample_size, config
    )
    
    # Total work units = files * combinations * avg_final_states_per_file
    # This represents the actual computational complexity
    total_work_units = num_files * num_combinations * avg_final_states_per_file
    
    # Calculate jobs needed
    work_units_per_job = args.walltime_per_job_sec / args.time_for_work_unit_sec
    num_jobs = math.ceil(total_work_units / work_units_per_job)
    
    print(f"Files: {num_files}")
    print(f"Combinations: {num_combinations}")
    print(f"Avg final states per file: {avg_final_states_per_file:.1f} (estimated from {min(args.sample_size, num_files)} samples)")
    print(f"Total work units (files*combinations*final_states): {total_work_units:.0f}")
    print(f"Work units per job: {work_units_per_job:.1f}")
    print(num_jobs)  # Last line is the number of jobs (for script parsing)

