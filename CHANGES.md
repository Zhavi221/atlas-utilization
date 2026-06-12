# ATLAS Open Data → BumpNet Pipeline

This branch contains fixes and extensions to the ATLAS utilization pipeline
for processing ATLAS Open Data PHYSLITE files and producing invariant mass
histograms for the BumpNet anomaly detection algorithm.

## Branch: `atlas-opendata-bumpnet`

Based on: `master` of [atlas-utilization](https://github.com/Zhavi221/atlas-utilization)

---

## Summary of Changes

### Fix 17 — Histogram batch jobs not splitting SQLite files correctly (`histograms_pipeline.py`, `orchestration/handlers/histogram_creation_handler.py`, `main.py`, `domain/config.py`)

**Problem:** Multiple issues caused all histogram batch jobs to process all
processed SQLite files instead of their assigned slice, producing identical
histograms in each batch that hadd summed incorrectly:

1. `total_batch_jobs` was missing from the `config_dict` passed to
   `create_histograms` — without it, `_create_histograms_from_sqlite` could
   not split SQLite files by batch index.

2. `global_ranges_path` was not passed through `HistogramCreationHandler` to
   `create_histograms`, so `global_ranges` was always `None` and histograms
   used batch-local bin edges instead of the global ones from `global_ranges.json`.

3. `_create_histograms_from_sqlite` had no SQLite file splitting logic at all —
   it always read all SQLite files regardless of `batch_job_index`.

4. `batch_job_index` and `total_batch_jobs` were injected into
   `histogram_creation_task_config` in `main.py` but only after
   `update_config_paths_with_run_dir` was called, so the split never reached
   `PipelineConfig`.

5. `also_save_pre_postproc`, `pre_postproc_filename`, and `global_ranges_path`
   were not defined as fields in `HistogramCreationConfig` or read in
   `from_dict`, so YAML values were silently ignored.

**Fixes:**

- `histograms_pipeline.py`: added SQLite file splitting in
  `_create_histograms_from_sqlite` using `_get_batch_files` when
  `batch_job_index` and `total_batch_jobs` are present in config.

- `histogram_creation_handler.py`: added `total_batch_jobs` and
  `global_ranges_path` to the `config_dict` passed to `create_histograms`.

- `main.py`: inject `batch_job_index` and `total_batch_jobs` into
  `histogram_creation_task_config` alongside `output_filename`.

- `domain/config.py`: added `global_ranges_path`, `also_save_pre_postproc`,
  and `pre_postproc_filename` fields to `HistogramCreationConfig` dataclass
  and wired them through `from_dict`.


### Fix 12 — Histogram bin-edge mismatch across batches (`histograms_pipeline.py`, `executor.py`, `utils/paths.py`)

**Problem:** Each histogram batch job computed `global_min`/`global_max` from its
own data slice, producing histograms with different bin edges per batch. `hadd`
silently kept only the first batch's histogram and discarded the rest — e.g. for
`mass_m0m1j0_cat_2mx_1jx`, batch_1 had `nbins=33 min=59.8 max=397.5` while
batch_3 had `nbins=254 min=62.3 max=2598.7`. The merged file contained only
batch_1's 33 events instead of the true ~52,000.

**Fix:** Introduced a mandatory scan step between post-processing and histogram
creation. The scan reads all processed SQLite files, computes the true global
min/max per bumpnet signature across all batches and chunks, and writes
`logs/global_ranges.json`. All histogram batch jobs load this file and use
identical bin edges. `hadd` now merges correctly.

**New `--scan-only` mode** in `main.py` + `scan_global_ranges()` in `executor.py`:
```bash
python main.py --scan-only --run-dir /path/to/run_dir
```

**New helper functions** in `histograms_pipeline.py`:
```python
compute_global_ranges(sqlite_files, input_dir, exclude_outliers)
save_global_ranges(ranges, output_path)
load_global_ranges(path)
```

**`utils/paths.py`:** `update_config_paths_with_run_dir` now always sets
`global_ranges_path = {run_dir}/logs/global_ranges.json` automatically — no
manual config change needed.

**PBS job chain updated** (`submit_data.sh`, `submit_mc.sh`):

---

### Fix 13 — MC/data contamination in pipeline runs (`submit_data.sh`, `submit_mc.sh`, `main.py`)

**Problem:** Data and MC files were parsed into the same run directory. With
`parse_mc=False` the metadata cache still contained MC URLs (see Fix 11), and
the pipeline had no clean mechanism to enforce full separation at the run level.

**Fix:** Separated into two independent submission scripts:
- `submit_data.sh` — creates `{run_name}_data_{TIMESTAMP}/`, passes `--parse-mc false`
- `submit_mc.sh` — creates `{run_name}_mc_{TIMESTAMP}/`, passes `--parse-mc true`

Both scripts use the same config file. A new `--parse-mc true/false` CLI flag
overrides `parse_mc` in `parsing_task_config` at runtime without editing YAML:

```bash
# in submit_data.sh:
python -u main.py --config "${CONFIG}" --parse-mc false ...

# in submit_mc.sh:
python -u main.py --config "${CONFIG}" --parse-mc true ...
```

Both scripts implement the full 4-stage PBS chain from Fix 12.

---

### Fix 14 — Batch file deletion after hadd destroyed retry runs (`executor.py`)

**Problem:** `merge_outputs` called `bf.unlink()` on each `batch_N.root` file
after a successful `hadd`. On retry runs (e.g. after a failed batch job was
resubmitted), only the new batch file existed — `hadd` merged just 1 file instead
of all N, silently overwriting the previously correct merged file.

**Fix:** Batch files are now moved to `histograms/batch_files_archive/` instead
of deleted:
```python
archive_dir = Path(hist_dir) / "batch_files_archive"
archive_dir.mkdir(exist_ok=True)
for bf in batch_roots:
    bf.rename(archive_dir / bf.name)
```

---

### Fix 15 — Stats aggregation crash on float-valued fields (`executor.py`)

**Problem:** `_aggregate_batch_stats` called `int(p[key])` on fields like
`total_size_mb` which are stored as floats (e.g. `'0.00'`), raising
`ValueError: invalid literal for int() with base 10: '0.00'`.

**Fix:** `int(float(p[key]))` — converts to float first, then truncates to int.

---

### Fix 16 — Memory crash in plot generation (`executor.py`)

**Problem:** `_read_parsed_data_stats` accumulated full per-event particle count
arrays (`particle_dists[ptype].extend(counts.tolist())`) across all 671 parsed
ROOT files. With millions of events per file this exhausted memory on the login
node, killing the process silently with no traceback.

**Fix:** Replaced array accumulation with immediate summation:
```python
counts = tree[branch_name].array(library="np")
particle_counts[ptype] = particle_counts.get(ptype, 0) + int(np.sum(counts))
del counts  # free immediately
```
`distributions` dict is now always empty — too large to store for runs of this size.

---

### Feature — Pre-post-processing histograms (`executor.py`, `histogram_creation_task_config`)

**Motivation:** Comparing histograms before and after post-processing is essential
for validating peak removal. 
**New config options:**
```yaml
histogram_creation_task_config:
  also_save_pre_postproc: true
  pre_postproc_filename: "atlas_opendata_nopostproc.root"
```

When enabled, the merge job reads all `im_arrays/*.sqlite` files (raw invariant
mass arrays, before peak removal) and writes a second ROOT file alongside the
main post-processed one. Produced in the merge step to avoid the bin-edge problem
— all data is visible at once, no batching required.

## Fix 11 — prevent MC/data contamination in metadata cache

**Problem:**The previous `_separate_mc_files` used `"mc" in url.lower()` which is
ambiguous — ATLAS Open Data URLs all contain "opendata", causing MC URLs
to be misclassified. With `parse_mc=False`, no separation happened at
all, so MC events were silently parsed as collision data.

**Modified files:**:
- fetcher.py: replace loose substring match with regex-based classifier
  (_classify_url) anchored to the dataset namespace component (mc<N>_,
  data<N>_). Separation now always happens unconditionally. Unclassifiable
  URLs raise ValueError immediately — no silent misrouting.
- fetcher.py: add _assert_no_cross_contamination() post-separation check
  as a safety net.
- fetch_metadata_handler.py: validate cache integrity on load; raise
  RuntimeError with first 5 violations if contaminated, forcing cache
  rebuild. Remove separate_mc argument from fetcher.fetch() call.
- parsing_handler.py: skip keys ending in _mc when parse_mc=False.
  parse_mc now means "include MC in the parsing run", not "separate in
  cache" (separation is always performed).
  
## Fix 10 — Sub-leading particle support

**Problem:** All combinations used only leading particles (highest pT).
`{"Electrons": 1}` always meant e₀. There was no way to compute e₁+j₀
(sub-leading electron + leading jet) as a separate invariant mass signature.

**Solution:** Combination values are now `(count, start_index)` tuples.
`start_index=0` = leading (unchanged behaviour); `start_index=1` = sub-leading.
Plain `int` values are still accepted via `get_count()` / `get_start()` helpers
for full backward compatibility.

**New config flags** (in `mass_calculation_task_config`):
```yaml
include_subleading: false   # set true to activate
max_subleading_index: 1     # highest rank index (1 = up to sub-leading)
```

**Impact:** With `max_subleading_index: 1`, enabling sub-leading expands
combinations from 65 → 308 (~4.7×). Walltime should be scaled accordingly.

**Modified files:** `combinatorics.py`, `physics_calcs.py`,
`im_calculator.py`, `im_pipeline.py`, `domain/config.py`, `config.yaml`

############################

### Fix 5 — max_total_particles_in_combination cap (`combinatorics.py`, `domain/config.py`, `mass_calculation_handler.py`)

**Problem:** With `min_particles_in_combination: 1`, the number of IM combinations
explodes to 624 (4 particle types, counts 1–4 each, no cap). Most high-multiplicity
combinations (e.g. 4e+4μ+4j+4γ = 16 particles) are unphysical and produce
negligible statistics, wasting mass calculation time.

**Fix:** Added `max_total_particles_in_combination` parameter that skips any
combination where the sum of particle counts exceeds the cap:

```python
# services/calculations/combinatorics.py
for counts in itertools.product(*count_ranges):
    if sum(counts) > max_total_particles:   # skip combinations exceeding particle cap
        continue
```

Configurable via yaml:
```yaml
mass_calculation_task_config:
  max_total_particles_in_combination: 4  # default; reduces 624 → 69 combinations
```

| Config | Combinations |
|--------|-------------|
| No cap | 624 |
| cap=2  | 14  |
| cap=3  | 34  |
| cap=4  | 69  |

---

### Fix 6 — Post-processing: merge all chunks by FS_IM key before peak detection (`post_processing_pipeline.py`)

**Problem:** The post-processing pipeline iterated over individual signatures like
`parsed_2024r-pp_batch1_chunk0_FS_0e_0m_3j_0g_IM_j0j1j2`, processing each chunk
independently. This caused the peak finder to see partial distributions (e.g.
250k events from one chunk) rather than the full merged distribution (~2.5M
events across all chunks and batches). On sparse per-chunk distributions the
highest bin was often the low-mass turn-on, not the Z peak — leading to incorrect
peak removal.

**Fix:** Group all signatures by their `FS_IM` key (stripping the
`parsed_..._batch{N}_chunk{M}` prefix) before peak detection, then load and
concatenate all chunk arrays for that key:

```python
from collections import defaultdict
fs_im_groups = defaultdict(list)
for sig in signatures:
    m = re.search(r'(_FS_.+)', sig)
    fs_im_key = m.group(1) if m else sig
    fs_im_groups[fs_im_key].append(sig)

for fs_im_key in sorted(fs_im_groups.keys()):
    chunks = []
    for sig in fs_im_groups[fs_im_key]:
        for filename in sqlite_files:
            chunks.extend(iter_arrays_for_signature(..., sig))
    arr = np.concatenate(chunks)
    # peak detection on full merged array
```

**Result:** Peak detection now operates on the complete merged distribution,
correctly identifying the Z peak at ~91 GeV even for sparse final states.

---

### Fix 7 — Post-processing: use right edge of peak bin (`post_processing_pipeline.py`)

remove Fix 7 — peak bin right-edge shift had no effect

The peak_bin_idx + 1 change (right edge instead of left edge of peak bin)
did not solve the post-processing problem. Root cause is elsewhere.
Reverting to original behavior pending proper fix.

---

### Fix 8 — Post-processing: periodic SQLite commits (`post_processing_pipeline.py`)

**Problem:** The original code called `writer.commit()` only once after processing
all signatures. With 3.7M signatures this produced an 8 GB SQLite WAL file that
was rolled back entirely on job kill — losing all progress.

**Fix:** Commit every 1000 signatures:

```python
COMMIT_EVERY = 1000
processed = 0
for fs_im_key in sorted(fs_im_groups.keys()):
    # ... process ...
    processed += 1
    if processed % COMMIT_EVERY == 0:
        writer.commit()
        logger.info(f"Post-processing progress: {processed}/{len(fs_im_groups)} signatures committed")
writer.commit()  # final commit for remainder
```

**Result:** WAL stays small (~few MB), and at most 1000 signatures worth of work
is lost on job kill.

### Fix 9 — Skip single-particle combinations in combinatorics (`combinatorics.py`)

Added a guard in `get_all_combinations` to reject combinations where the total
particle count is less than 2. A single-particle "combination" produces a trivial
invariant mass equal to the particle's own mass, which is physically meaningless
for bump hunting. Previously, such combinations could pass the `max_total_particles`
filter and propagate through the mass calculation, generating spurious signatures
(e.g. `1e_0m_0j_0g`) that inflate the histogram count without any signal sensitivity.

**Change:** `if sum(counts) < 2: continue`
**Result:** Combinatorics now only generates signatures with ≥ 2 particles

---

### Fix 1 — Index-based IM naming (`im_pipeline.py`, `histograms_pipeline.py`)

**Problem:** The pipeline used count-based naming for invariant mass combinations,
e.g. `IM_2e_0m_1j_0g`, which is ambiguous — it does not specify *which* particles
were used (leading vs sub-leading). BumpNet expects index-based naming starting
from 0 (e.g. `e0`, `e1`, `j0`).

**Fix:** Two functions were updated:
- `prepare_im_combination_name` in `services/pipelines/im_pipeline.py`
- `_convert_to_bumpnet_name` in `services/pipelines/histograms_pipeline.py`

**Result:**

| Old signature | New signature | Meaning |
|---|---|---|
| `IM_1e_0m_1j_0g` | `IM_e0j0` | leading e + leading j (2-body) |
| `IM_2e_0m_1j_0g` | `IM_e0e1j0` | both e + leading j (3-body) |
| `IM_1e_0m_2j_0g` | `IM_e0j0j1` | leading e + both j (3-body) |

Histogram names follow the same convention:
`ROI_mass_e0j0_cat_1ex_0mx_1jx_0gx_width_10.0`

---

### Fix 2 — Kinematic cuts in MeV (`config.yaml`)

**Problem:** ATLAS PHYSLITE stores all 4-vector quantities (pt, mass, energy) in
MeV. The original config used `pt_min: 25.0`, which meant 25 MeV — effectively
no cut, passing all soft particles and producing unphysical invariant masses.

**Fix:** Config updated to use MeV values:

```yaml
kinematic_cuts:
  electrons: { pt_min: 25000.0, eta_max: 2.47 }  # 25 GeV in MeV
  muons:     { pt_min: 25000.0, eta_max: 2.5  }
  jets:      { pt_min: 30000.0, eta_max: 4.5  }
  photons:   { pt_min: 25000.0, eta_max: 2.37 }
```

The `_convert_array_to_gev()` function (`* 1e-3`) in `im_pipeline.py` correctly
converts MeV → GeV for the final invariant mass values stored in SQLite.

---

### Fix 3 — Memory leak in `parsing_handler.py`

**Problem:** After writing each parsed ROOT chunk to disk, the Python `EventChunk`
object (holding large awkward arrays) was not explicitly freed. Python's garbage
collector does not release memory promptly, causing memory to grow continuously
throughout a long parsing job — reaching 150+ GB for a 72-hour run over 70,201
files.

**Fix:** Added explicit memory release after each chunk write:

```python
import gc

# After _save_chunk_to_root(chunk, ...):
del chunk
gc.collect()

# After _save_chunk_to_root(final_chunk, ...):
del final_chunk
gc.collect()
```

The ROOT files on disk are unaffected — only the in-memory Python object is freed.

---

### Fix 4 — Parsing config: reduced memory thresholds

**Problem:** `chunk_yield_threshold_bytes: 2147483648` (2 GB) combined with
`threads: 8` allowed up to 16 GB of awkward arrays to accumulate before any
flush, leading to memory explosions when running all pipeline stages in one job.

**Fix** in `config.yaml`:

```yaml
parsing_task_config:
  chunk_yield_threshold_bytes: 268435456  # 256 MB — flush frequently
  threads: 2                              # reduce from 8
```

---

## New Files

### `config.yaml`
Main config for the ATLAS Open Data BumpNet run. Key settings:
- 10 GeV bin width for histograms
- Correct MeV kinematic cuts
- Reduced memory thresholds
- `max_particles_in_combination: 2` for initial validation run
  (increase to 4 for full combination set)

### `submit_parsing_only.sh`
Submits a single PBS job that runs **only the parsing stage**.
Mass calculation is intentionally excluded to prevent memory explosion.

### `submit_mass_calc.sh`
Submits one PBS job **per parsed ROOT chunk** for mass calculation.
Each job processes exactly one ~2 GB chunk, keeping memory under 16 GB.
A dependent post-processing + histogram job is submitted automatically
after all chunk jobs complete.

### `submit_mass_postproc.sh`
Submits mass calculation array job + dependent post-processing + histogram
creation job for an **existing** parsed data directory (no re-parsing).

Usage:
```bash
RUN_DIR=/path/to/existing/run bash submit_mass_postproc.sh
```

### `submit_hist_asNeta.sh`
Submits histogram creation only from existing processed SQLite files.
Bypasses mass calculation and post-processing.

### `submit_minEvt10.sh`
Full pipeline submission for `min_events_per_fs=10` run, reusing existing
parsed data. Handles data and MC separately to avoid SQLite locking conflicts.
No `hadd` — histogram creation reads all SQLite files in one job.

### `configAsNeta.yaml`
Config matching Neta's parsing settings (25 MeV cuts, min_events=100).

### `configWmaxTotal.yaml`
Config with `max_total_particles_in_combination: 4`, `min_particles: 1`.

### `configWmaxTotal_up4j.yaml`
Config with `jets: max: 4`, `parse_mc: true` for data+MC runs.

### `configWmaxTotal_up4j_minEvt10.yaml`
Config with `min_events_per_fs: 10` to recover rare electron final states.

### `env.sh`
Environment setup script for the pipeline (LCG view + atlasenv activation).

---

## Pipeline Architecture

```
Raw PHYSLITE files (CERN EOS, streamed via XRootD)
         ↓
  ParsingHandler
  - Reads AnalysisElectronsAuxDyn.pt etc. (values in MeV)
  - Applies kinematic cuts (pt_min in MeV)
  - Writes parsed_data/parsed_2024r-pp_chunk{N}.root
         ↓
  MassCalculationHandler  (one PBS job per chunk)
  - Computes invariant masses for all particle combinations
  - Converts MeV → GeV via _convert_array_to_gev() × 1e-3
  - Writes im_arrays/im_batch_{N}.sqlite
         ↓
  PostProcessingHandler  (FIXED: merge by FS_IM key, right edge cut)
  - Groups all chunks for same FS_IM key, merges before peak detection
  - Removes Z peak using right edge of peak bin
  - Periodic commits every 1000 signatures
  - Splits into main / outlier arrays
         ↓
  HistogramCreationHandler
  - Builds ROOT histograms with BumpNet index-based naming
  - Output: histograms/atlas_opendata_full_bumpnet.root
```

---

## Known Issues / Pending Work

### Sub-leading particle combinations (not yet implemented)
The current code always selects leading particles (highest pT) for each
combination. `{"Electrons": 1}` always means e₀ (leading electron).
There is no mechanism to compute e₁+j₀ (sub-leading electron + leading jet)
as a separate 2-body invariant mass.

Files to modify: `services/calculations/combinatorics.py`,
`services/calculations/physics_calcs.py`,
`services/pipelines/mass_calculation_handler.py`

### Electron isolation cuts (not yet implemented)
Without isolation cuts, fake electrons from jets dominate the electron
channel. The `2ex_0mx_1jx_0gx` final state has only ~1 event across
all 524 data chunks because real Z→ee+1j events are buried under
high-multiplicity fake-electron final states. Standard ATLAS isolation:
- `ptvarcone30_Nonprompt_All_MaxWeightTTVALooseCone_pt1000 / pt < 0.06`
- `topoetcone20 / pt < 0.06`

### SM background MC not available for 2024r-pp
The `2024r-pp_mc` release via atlasopenmagic contains only BSM signal
samples (Z', W', graviton — DSIDs 301xxx). No SM background MC (Z+jets,
tt̄, W+jets, QCD) is available for the Run 3 open data release.
Confirmed with ATLAS Open Data support.

### _split_by_first_empty_bin truncates sparse distributions
For final states with low statistics (e.g. 2μ+1j), the first empty bin
occurs early in the high-mass tail, causing the postproc histogram to
be truncated. Use `exclude_outliers: false` in histogram creation to
include the full distribution above the Z peak cut.

---

## Data Notes

- ATLAS Open Data 2024r-pp release: 70,201 files, ~65 TB
- pT stored in MeV in all PHYSLITE branches
- `2024r-pp_mc`: 14,581 files — BSM signal samples only (not SM background)
- Luminosity: data ~140 fb⁻¹ (Run 3), MC weighted to match