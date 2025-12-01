import os
import math
import argparse
import atlasopenmagic as atom
import yaml
import itertools
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.parse_atlas import parser
import io
import contextlib

CONFIG_PATH = "configs/pipeline_config.yaml"

# ===== OPTIMIZED DEFAULTS FROM YOUR TESTING =====
# Based on thread_heavy_approach results
DEFAULT_EVENTS_PER_SECOND = 23490  # From your best config
DEFAULT_MB_PER_SECOND = 2.34
DEFAULT_WALLTIME_HOURS = 24
DEFAULT_WORKERS = 4
DEFAULT_THREADS = 8

# Average file size from your tests (1000 files = 16.88 GB)
AVERAGE_FILE_SIZE_MB = 16878 / 1000  # ~16.88 MB per file

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--config", help="Config file", default=CONFIG_PATH)
arg_parser.add_argument("--walltime_hours", type=int, default=DEFAULT_WALLTIME_HOURS,
                       help="Walltime per PBS job in hours")
arg_parser.add_argument("--events_per_sec", type=float, default=DEFAULT_EVENTS_PER_SECOND,
                       help="Processing speed (events/second)")
arg_parser.add_argument("--mb_per_sec", type=float, default=DEFAULT_MB_PER_SECOND,
                       help="Processing speed (MB/second)")
arg_parser.add_argument("--safety_factor", type=float, default=0.8,
                       help="Safety factor for walltime (0.8 = use 80% of walltime)")
arg_parser.add_argument("--cores_per_node", type=int, default=32,
                       help="Number of cores available per PBS node")
arg_parser.add_argument("--memory_per_node_gb", type=int, default=128,
                       help="Memory available per PBS node in GB")

args = arg_parser.parse_args()

# Load config
with open(args.config) as f:
    config = yaml.safe_load(f)

# Fetch file IDs
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf):
    release_year_file_ids: dict = parser.AtlasOpenParser.fetch_record_ids_for_release_years(
        release_years=config["parsing_config"]["atlasparser_config"]["release_years"],
        timeout=60
    )

all_file_ids = list(itertools.chain.from_iterable(release_year_file_ids.values()))
num_files = len(all_file_ids)
total_size_mb = num_files * AVERAGE_FILE_SIZE_MB

print(f"{'='*70}")
print(f"PBS JOB CALCULATOR - OPTIMIZED FOR PARALLEL PROCESSING")
print(f"{'='*70}")
print(f"\n📊 Dataset Information:")
print(f"   Total files: {num_files:,}")
print(f"   Estimated total size: {total_size_mb:,.1f} MB ({total_size_mb/1024:.1f} GB)")
print(f"   Average file size: {AVERAGE_FILE_SIZE_MB:.2f} MB")

# Calculate processing capacity per job
walltime_seconds = args.walltime_hours * 3600
effective_walltime = walltime_seconds * args.safety_factor

# Calculate how much data ONE job can process
mb_per_job = args.mb_per_sec * effective_walltime
files_per_job = int(mb_per_job / AVERAGE_FILE_SIZE_MB)

# Calculate job requirements
num_jobs = math.ceil(num_files / files_per_job)
estimated_total_time_hours = (total_size_mb / args.mb_per_sec) / 3600

# Get worker/thread config from pipeline config
workers = config["parsing_config"]["pipeline_config"]["max_parallel_workers"]
threads = config["parsing_config"]["atlasparser_config"]["max_threads"]
cores_per_job = workers * threads

# Memory estimation (from your tests: ~2.4GB per instance)
memory_per_instance_gb = 3  # Conservative estimate
memory_per_job_gb = memory_per_instance_gb

# Calculate how many instances can run per node
instances_per_node_by_cores = args.cores_per_node // cores_per_job
instances_per_node_by_memory = args.memory_per_node_gb // memory_per_instance_gb
instances_per_node = min(instances_per_node_by_cores, instances_per_node_by_memory)

# If we can run multiple instances per node
if instances_per_node > 1:
    print(f"\n💡 MULTI-INSTANCE OPTIMIZATION:")
    print(f"   Each node can run {instances_per_node} instances in parallel")
    print(f"   Limited by: {'cores' if instances_per_node_by_cores < instances_per_node_by_memory else 'memory'}")
    
    # Adjust jobs for multi-instance
    effective_jobs_per_node = instances_per_node
    adjusted_num_jobs = math.ceil(num_jobs / instances_per_node)
    
    print(f"\n⚡ Adjusted Job Calculation:")
    print(f"   Original jobs needed: {num_jobs:,}")
    print(f"   With {instances_per_node}x instances/node: {adjusted_num_jobs:,} PBS jobs")
    print(f"   (Each PBS job spawns {instances_per_node} parallel instances)")
else:
    adjusted_num_jobs = num_jobs
    effective_jobs_per_node = 1

print(f"\n📋 PBS Job Configuration:")
print(f"   Jobs to submit: {adjusted_num_jobs:,}")
print(f"   Files per job: {files_per_job:,}")
print(f"   Data per job: {mb_per_job:,.1f} MB ({mb_per_job/1024:.1f} GB)")
print(f"   Walltime per job: {args.walltime_hours} hours")
print(f"   Effective processing time: {effective_walltime/3600:.1f} hours ({args.safety_factor*100:.0f}% of walltime)")

print(f"\n🖥️  Resource Requirements per PBS Job:")
print(f"   Cores: {cores_per_job} (workers={workers}, threads={threads})")
print(f"   Memory: ~{memory_per_job_gb} GB")
print(f"   Instances per job: {instances_per_node}")

print(f"\n⏱️  Time Estimates:")
print(f"   Total processing time (serial): {estimated_total_time_hours:,.1f} hours ({estimated_total_time_hours/24:.1f} days)")
print(f"   With {adjusted_num_jobs:,} parallel PBS jobs: {args.walltime_hours} hours")
print(f"   If cluster runs all jobs simultaneously: {args.walltime_hours} hours")
print(f"   If cluster queues jobs sequentially: {adjusted_num_jobs * args.walltime_hours:,.1f} hours ({adjusted_num_jobs * args.walltime_hours/24:.1f} days)")

# Calculate optimal cluster size
print(f"\n🎯 Cluster Recommendations:")
nodes_for_24h = math.ceil(adjusted_num_jobs)
nodes_for_7d = math.ceil(adjusted_num_jobs * args.walltime_hours / (7*24))
nodes_for_1d = math.ceil(adjusted_num_jobs * args.walltime_hours / 24)

print(f"   For 24-hour completion: {nodes_for_24h:,} nodes")
print(f"   For 7-day completion: {nodes_for_7d:,} nodes")
print(f"   For 1-day completion: {nodes_for_1d:,} nodes")

print(f"\n📝 PBS Script Template:")
print(f"   #PBS -l select=1:ncpus={cores_per_job}:mem={memory_per_job_gb}gb")
print(f"   #PBS -l walltime={args.walltime_hours:02d}:00:00")
print(f"   #PBS -J 0-{adjusted_num_jobs-1}")

# Generate batch file splitting
print(f"\n📦 File Distribution:")
file_batches = []
for i in range(adjusted_num_jobs):
    start_idx = i * files_per_job * instances_per_node
    end_idx = min(start_idx + files_per_job * instances_per_node, num_files)
    batch_files = all_file_ids[start_idx:end_idx]
    file_batches.append(batch_files)
    if i < 3:  # Show first 3 batches
        print(f"   Job {i}: files {start_idx:,} to {end_idx:,} ({len(batch_files):,} files)")

if adjusted_num_jobs > 3:
    print(f"   ... ({adjusted_num_jobs-3:,} more jobs)")

print(f"\n{'='*70}")

# Output job count for downstream scripts
print(f"\n{adjusted_num_jobs}")  # For backward compatibility

# Optionally save batch assignments
output_dir = Path("batch_jobs")
output_dir.mkdir(exist_ok=True)

batch_file_path = output_dir / "job_file_assignments.json"
import json
with open(batch_file_path, 'w') as f:
    json.dump({
        'total_jobs': adjusted_num_jobs,
        'files_per_job': files_per_job,
        'instances_per_node': instances_per_node,
        'batches': [[str(fid) for fid in batch] for batch in file_batches]
    }, f, indent=2)

print(f"📁 Batch assignments saved to: {batch_file_path}")