# ATLAS Open Data → BumpNet Pipeline

This branch contains fixes and extensions to the ATLAS utilization pipeline
for processing ATLAS Open Data PHYSLITE files and producing invariant mass
histograms for the BumpNet anomaly detection algorithm.

## Branch: `atlas-opendata-bumpnet`

Based on: `master` of [atlas-utilization](https://github.com/Zhavi221/atlas-utilization)

---

## Summary of Changes

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

Usage:
```bash
bash submit_parsing_only.sh
# prints RUN_DIR when submitted — use it for mass calculation step
```

After parsing completes:
```bash
RUN_DIR=/path/to/run_dir bash submit_mass_calc.sh
```

### `submit_mass_calc.sh`
Submits one PBS job **per parsed ROOT chunk** for mass calculation.
Each job processes exactly one ~2 GB chunk, keeping memory under 16 GB.
A dependent post-processing + histogram job is submitted automatically
after all chunk jobs complete.

Usage:
```bash
# Use default RUN_DIR (hardcoded in script):
bash submit_mass_calc.sh

# Or override RUN_DIR:
RUN_DIR=/path/to/run_dir bash submit_mass_calc.sh
```

Key settings per job:
- `io=31` (measured with `iothrottle -v -l 0`)
- `walltime=00:15:00` (one chunk takes ~52 seconds)
- `mem=16gb`
- `use_multiprocessing: false`, `parallel_processes: 1`

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
  PostProcessingHandler
  - Removes known SM resonance peaks (Z, W)
  - Splits into main / outlier arrays
         ↓
  HistogramCreationHandler
  - Builds ROOT histograms with BumpNet index-based naming
  - Output: histograms/atlas_opendata_full_bumpnet.root
```

---

## Known Issues / Pending Work

### Sub-leading particle combinations (Fix 2 — not yet implemented)
The current code always selects leading particles (highest pT) for each
combination. `{"Electrons": 1}` always means e₀ (leading electron).
There is no mechanism to compute e₁+j₀ (sub-leading electron + leading jet)
as a separate 2-body invariant mass.

Files to modify: `services/calculations/combinatorics.py`,
`services/calculations/physics_calcs.py`,
`services/pipelines/mass_calculation_handler.py`

### Electron isolation cuts (not yet implemented)
The electron validation plot shows a flat tail from 1000–5000 GeV due to
the absence of isolation cuts. Standard ATLAS isolation requirements:
- `ptvarcone30_Nonprompt_All_MaxWeightTTVALooseCone_pt1000 / pt < 0.06`
- `topoetcone20 / pt < 0.06`

These branches are available in PHYSLITE and need to be added to the
parsing stage `filter_events_by_kinematics`.

### Z peak removal for multi-body final states
`_find_rightmost_highest_peak` fails for 3-body distributions (e.g. μ₀+μ₁+j₀)
where the highest bin is the low-mass turn-on, not the Z+jet peak at ~117 GeV.
BumpNet flags this as a 5σ excess. The peak removal algorithm needs to be
updated for multi-body final states.

---

## Data Notes

- ATLAS Open Data 2024r-pp release: 70,201 files, ~65 TB
- pT stored in MeV in all PHYSLITE branches

---