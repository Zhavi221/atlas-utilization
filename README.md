# ATLAS Open Data Pipeline

End-to-end pipeline that downloads ATLAS Open Data ROOT files, computes invariant masses for particle combinations, and produces ROOT histograms ready for anomaly detection (BumpNet).

## Pipeline stages

```
1. Parsing          Download ROOT files from CERN, extract particle arrays
2. Mass Calculation  Compute invariant masses for all valid final-state combinations
3. Post-Processing   Remove known-mass peaks, split main / outlier distributions
4. Histograms        Build ROOT histograms (one file, BumpNet-compatible naming)
```

Each run writes to an isolated timestamped directory:

```
{base_output_dir}/{run_name}_{YYYYMMDD_HHMMSS}/
  parsed_data/            ROOT files with particle arrays
  im_arrays/              invariant-mass .npy files
  im_arrays_processed/    peak-removed arrays
  histograms/             final ROOT histograms
  plots/                  summary plots
  logs/                   job logs
  metadata_cache.json     cached ATLAS file URLs
```

## Setup

The pipeline requires a ROOT-enabled Python environment. On an ATLAS cluster node:

```bash
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_106a_ATLAS_1 x86_64-el9-gcc14-opt"
```

## Usage

Everything is controlled through one file: **`config.yaml`**.

### Local — quick test

1. Limit the number of files so it finishes fast:

```yaml
# in config.yaml
parsing_task_config:
  max_files_to_process: 3       # just 3 files
mass_calculation_task_config:
  min_events_per_fs: 10         # keep rare final states too
```

2. Run:

```bash
python main.py
```

Output appears in `./output/{run_name}_{timestamp}/`.

### Local — full run

Set `max_files_to_process: null` (or remove the line) to process all files, then run the same command.

### Cluster (PBS) — single job

```bash
bash submit.sh
```

This submits one PBS job that runs the full pipeline. To write output to a storage volume, edit `config.yaml`:

```yaml
run_metadata:
  base_output_dir: "/storage/agrp/netalev/data"
```

### Cluster (PBS) — multi-job batch

Split the work across N jobs (e.g. 4). Each job parses its slice, computes invariant masses, and writes histograms. A final merge job combines everything.

```bash
NUM_JOBS=4 bash submit.sh
```

`submit.sh` automatically:
- Creates a shared run directory on the storage volume
- Submits a PBS array of N jobs
- Submits a merge job that runs after all N finish

### Re-generate plots from an existing run

```bash
python main.py --plots-only --run-dir /path/to/existing/run
```

### Validate config without running

```bash
python main.py --dry-run
```

## Config reference

| Section | Key | What it does |
|---|---|---|
| `run_metadata` | `run_name` | Name prefix for the output directory |
| `run_metadata` | `base_output_dir` | Root directory for all runs (`./output` local, `/storage/...` cluster) |
| `tasks` | `do_parsing` ... `do_histogram_creation` | Toggle each stage on/off |
| `parsing_task_config` | `max_files_to_process` | `null` = all, or an integer for testing |
| `parsing_task_config` | `release_years` | Which ATLAS data releases to use |
| `mass_calculation_task_config` | `min_events_per_fs` | Drop final states with fewer events than this |
| `histogram_creation_task_config` | `use_bumpnet_naming` | `true` for BumpNet-compatible histogram names |

All paths (`output_path`, `input_dir`, `output_dir`, etc.) are defined once in the `paths:` block at the top of `config.yaml` and reused via YAML anchors. At runtime they are overridden to point inside the timestamped run directory.

## CLI options

```
python main.py --help

  --config FILE              Config file (default: config.yaml)
  --dry-run                  Validate config, don't run
  --log-level LEVEL          DEBUG / INFO / WARNING / ERROR
  --batch-job-index N        Batch job index (1-based, set by submit.sh)
  --total-batch-jobs N       Total batch jobs (set by submit.sh)
  --run-dir DIR              Use this directory instead of creating a new one
  --merge-only               Merge batch outputs (hadd + stats + plots)
  --plots-only               Re-generate plots from existing run
```
