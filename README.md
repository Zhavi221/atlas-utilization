# ATLAS Open Data Pipeline

An end-to-end pipeline for downloading ATLAS Open Data ROOT files from CERN, computing invariant masses for all valid final-state particle combinations, and producing ROOT histograms suitable for anomaly detection with [BumpNet](../BumpNet-main/).

The primary use case is bulk processing of **ATLAS Open Data releases** (e.g. 13 TeV Run-2 datasets), but the pipeline also supports targeting **individual CERN record IDs** — for example CMS NanoAOD records or any other record hosted on [opendata.cern.ch](https://opendata.cern.ch). In both modes the processing flow is identical: fetch file URLs, parse ROOT trees, compute invariant masses, remove known-mass peaks, and write histograms.

## How it works

The pipeline is implemented as a state machine. Each stage runs in sequence, and individual stages can be toggled on or off in the config. The stages are:

### 1. Fetch metadata

Retrieves the list of ROOT file URLs to process. Two input modes are supported:

- **By release year** — queries the ATLAS Open Data portal (via `atlasopenmagic`) for all available datasets in the configured release years (e.g. `2024r-pp`, `2020e-13tev`).
- **By record ID** — fetches file listings directly from `opendata.cern.ch/record/{id}/filepage/...` for one or more specific CERN record IDs.

Results are cached to `metadata_cache.json` so subsequent runs skip the network call.

### 2. Parse ROOT files

Downloads each ROOT file, opens it with `uproot`, and extracts per-event particle arrays (electrons, muons, jets, photons) with their kinematics (`pt`, `eta`, `phi`, `mass`). Branch naming is release-dependent — schemas exist for ATLAS 2024, 2020, 2016 releases and CMS NanoAOD.

Parsing is multi-threaded. Events are accumulated into chunks (controlled by a memory threshold) and written as new ROOT files under `parsed_data/`.

### 3. Compute invariant masses

Reads the parsed ROOT files and computes invariant masses for every valid combination of final-state objects. A "final state" is a particle-type multiplicity pattern like `2e_2m_4j_2g`. The combinatorics and Lorentz-vector math live in `services/calculations/`.

Output: `.npy` arrays written to `im_arrays/`, one file per final-state / combination pair.

### 4. Post-processing

Removes known-mass peaks (Z, Higgs, etc.) from the invariant-mass distributions and splits each distribution into a main component and an outlier tail.

Output: cleaned `.npy` arrays in `im_arrays_processed/`.

### 5. Histogram creation

Converts the processed arrays into ROOT histograms. When `use_bumpnet_naming` is enabled (the default), histogram names follow BumpNet's expected convention (`mass_<combo>_cat_<final_state>_width_<bin_width>`). All histograms are written to a single ROOT file by default.

Output: ROOT histogram file(s) in `histograms/`.

## Output structure

Each run writes to an isolated timestamped directory:

```
{base_output_dir}/{run_name}_{YYYYMMDD_HHMMSS}/
  parsed_data/            ROOT files with normalised particle arrays
  im_arrays/              invariant-mass .npy files
  im_arrays_processed/    peak-removed / outlier-split arrays
  histograms/             final ROOT histograms (BumpNet-compatible)
  plots/                  summary plots (performance, data overview, IM distributions)
  logs/                   job logs and per-batch statistics JSON
  metadata_cache.json     cached file URLs from CERN
```

## Setup

The pipeline requires a ROOT-enabled Python environment. There are two ways to set one up:

### Option A — Docker (local development)

The repository includes a ready-made Docker configuration. You will need [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed.

1. Build the image from the `docker/` directory:

```bash
cd docker
docker build -t atlas-pipeline .
```

2. If you use VS Code, install the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension. Open the project and click **"Reopen in Container"** when prompted — it will use the `.devcontainer/devcontainer.json` that ships with the repo.

   Alternatively, in the **Remote Explorer** tab, select the Dev Containers section and connect to the image you just built.

The Docker image includes ROOT and all Python dependencies. Everything is self-contained.

### Option B — ATLAS cluster (CVMFS / LCG)

On a cluster node with CVMFS access:

```bash
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "views LCG_106a_ATLAS_1 x86_64-el9-gcc14-opt"
```

## Usage

Everything is controlled through one file: **`config.yaml`**.

### Local — quick test

Limit the number of files so it finishes fast:

```yaml
parsing_task_config:
  max_files_to_process: 3
mass_calculation_task_config:
  min_events_per_fs: 10
```

```bash
python main.py
```

Output appears in `./output/{run_name}_{timestamp}/`.

### Local — full run

Set `max_files_to_process: null` (or remove the line) to process every file in the selected releases, then run the same command.

### Cluster (PBS) — single job

```bash
bash submit.sh
```

Submits one PBS job that runs the full pipeline. To write output to a shared storage volume, set `base_output_dir` in `config.yaml`:

```yaml
run_metadata:
  base_output_dir: "/storage/agrp/netalev/data"
```

### Cluster (PBS) — multi-job batch

Split the work across N jobs (e.g. 4). Each job processes its slice of files through the full pipeline. A dependent merge job combines all outputs afterward.

```bash
NUM_JOBS=4 bash submit.sh
```

`submit.sh` automatically:
- Creates a shared run directory on the storage volume
- Submits a PBS array of N parsing/processing jobs
- Submits a merge job (runs `hadd` on ROOT files, aggregates statistics, generates plots) that starts after all N jobs finish

### Re-generate plots from an existing run

```bash
python main.py --plots-only --run-dir /path/to/existing/run
```

### Validate config without running

```bash
python main.py --dry-run
```

## Config reference

| Section | Key | Description |
|---|---|---|
| `run_metadata` | `run_name` | Name prefix for the output directory |
| `run_metadata` | `base_output_dir` | Root directory for all runs (`./output` locally, `/storage/...` on cluster) |
| `tasks` | `do_parsing` … `do_histogram_creation` | Toggle each pipeline stage on or off |
| `parsing_task_config` | `release_years` | Which ATLAS data releases to fetch (e.g. `[2024r-pp, 2020e-13tev]`) |
| `parsing_task_config` | `specific_record_ids` | List of CERN record IDs to process instead of (or in addition to) releases |
| `parsing_task_config` | `max_files_to_process` | `null` = all files, or an integer to cap file count for testing |
| `parsing_task_config` | `threads` | Number of threads for parallel file download and parsing |
| `mass_calculation_task_config` | `min_events_per_fs` | Drop final states with fewer events than this threshold |
| `mass_calculation_task_config` | `objects_to_calculate` | Which particle types to include in combinations |
| `mass_calculation_task_config` | `min/max_particles_in_combination` | Bounds on combination size |
| `histogram_creation_task_config` | `use_bumpnet_naming` | `true` for BumpNet-compatible histogram names |
| `histogram_creation_task_config` | `bin_width_gev` | Histogram bin width in GeV |
| `post_processing_task_config` | `peak_detection_bin_width_gev` | Bin width used during known-peak detection |

All paths (`output_path`, `input_dir`, `output_dir`, etc.) are defined once in the `paths:` block at the top of `config.yaml` and reused via YAML anchors. At runtime they are overridden to point inside the timestamped run directory.

## CLI options

```
python main.py --help

  --config FILE              Config file (default: config.yaml)
  --dry-run                  Validate config and exit
  --log-level LEVEL          DEBUG / INFO / WARNING / ERROR
  --batch-job-index N        Batch job index (1-based, set by submit.sh)
  --total-batch-jobs N       Total batch jobs (set by submit.sh)
  --tasks TASKS              Override config task toggles (comma-separated:
                             parsing, mass_calculating, post_processing,
                             histogram_creation)
  --stage-input STAGE:PATH   Override input directory for a stage (repeatable)
  --run-dir DIR              Use this directory instead of creating a new one
  --merge-only               Merge batch outputs (hadd + aggregate stats + plots)
  --plots-only               Re-generate plots from an existing run
```
