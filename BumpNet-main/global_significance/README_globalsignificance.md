# Global-Significance Study (with Split-Dataset A/B validation)

This pipeline estimates the **global (look-elsewhere) significance** of BumpNet on ATLAS-style mass histograms and validates it with a **split-dataset A/B test**:

- **ZLR — the analytical local significance**  
  Computed from a likelihood-ratio (LR) test against a smoothed background.

- **Zpred — BumpNet’s estimate of Z**  
  Trained to reproduce ZLR. Comparing Zpred to ZLR tells us
  1) how accurate BumpNet is and  
  2) how often background-only PEs produce apparently significant bumps.

Everything lives in `global_significance/`:

```
global_significance/
├── analyze_split_ds.py           # NEW: A/B split validation & global-significance plots
├── generate_toys_bkg_only.py     # (legacy) make background-only PEs
├── predict_toys_bkg_only.py      # run BumpNet (+ optional ZLR)
├── plot_toys_bkg_only.py         # (legacy) basic global-significance plots
└── toys_atlas.yaml               # master configuration (both flows)
```

---

## Two ways to study global significance

### 1) Split-Dataset A/B analysis (recommended)
`analyze_split_ds.py` assumes you already have:
- **Predictions** for paired PEs (`PE_XXXXXXX_A.npz`, `PE_XXXXXXX_B.npz`) produced with the same selection, where **A** is used to *pick* a candidate bin and **B** is used to *check* it.
- The corresponding **raw histogram NPZs** (for bin counts and edges).

What it does (high level):
- Pairs A/B PEs per shard; asserts 1-to-1 mapping and name consistency.
- In **A**: finds the histogram/bin of the **maximum positive Zpred** (within a configurable window).
- In **B @ (A-selected bin)**: reads Zpred and ZLR at *that exact* histo+bin to estimate the global false-positive rate under background-only.
- Produces:
  - Distributions of `Z(B@A)` for **BumpNet** and **ZLR**, each with robust Gaussian fits and **empirical 3σ global thresholds** (one-sided tail).
  - Where A’s maxima tend to sit in the histograms (relative position + optional absolute index).
  - Optional **ΔZmax vs relative position** heatmaps for A and B (single configuration flag controls both).
  - **Triplet plots** (per slice in relative position or per-bin event count): aligned panels for Zpred, ZLR, and ΔZ with Gaussian overlays and χ²/ndof.
  - Side-by-side **example figures** for selected PEs, with shared A/B y-scales on mass panels and the A-selected bin marked.

### 2) Toys-only flow (legacy but still useful)
- `generate_toys_bkg_only.py` → `predict_toys_bkg_only.py` → `plot_toys_bkg_only.py`
- Good for quick end-to-end checks and non-split studies.
- The A/B logic and the stronger global-significance test now live in `analyze_split_ds.py` and supersede most of `plot_toys_bkg_only.py`.

---

## Quick start (cluster)

1. **Edit the YAML**  
   Configure paths and options in `global_significance/toys_atlas.yaml`.

2. **Submit**  
   ```bash
   bash submit_rorqual.sh global_significance/toys_atlas.yaml
   ```
   The submission script reads the `tasks:` block and runs only the steps you set to `true`.

---

## YAML cheat sheet (`toys_atlas.yaml`)

### Global switches
```yaml
tasks:
  do_generate_toys_bkg_only: true      # optional
  do_predict_toys_bkg_only:  true      # optional
  do_plot_toys_bkg_only:     false     # legacy plotting
  do_analyze_split_ds:       true      # <<< recommended analysis
```

### Section `analyze_split_ds` (new, main focus)
```yaml
analyze_split_ds:
  input_dir: "/.../predictions_split_DS_AB_signal0p5_optimizedtraining"  # A/B prediction NPZs
  input_dir_hist: "/.../split_DS_AB"                                     # matching HIST NPZs
  output_dir: "/.../plots_split_ds_signal0p5_out_optimizedtraining_test"

  # Basic selection
  edge: [0.25, 1.00]        # analyze only this fractional range of each histogram
  min_nbins: 25             # skip histograms shorter than this after trimming
  max_pes: 500              # cap for speed/debug (omit for all)
  examples: 20              # number of seeded PE example figures to write
  seed: 42                  # reproducible PE/example selection

  # I/O and figure format
  dpi: 300
  format: "png"

  # Triplet plots (per-slice overlays of Zpred, ZLR, ΔZ, each with Gaussian fit)
  triplet_plots:
    logy: true
    relpos_edges:   [0.10, 0.25, 0.35, 0.50, 0.70, 0.90, 1.00]   # slices in relative position
    nevents_edges:  [100000000, 5000, 500, 50, 20, 5, 2, 1, 0]   # slices in per-bin counts (descending OK)
    plot_relpos_triplets:  true     # make the relpos triplets
    plot_nevents_triplets: true     # make the Nevents triplets

  # ΔZmax heatmaps (A and B) — single switch for both samples
  plot_deltaZmax: true
```

**Notes & expectations**
- `input_dir` must contain **paired** files per PE: `PE_XXXXXXX_A.npz` and `PE_XXXXXXX_B.npz`, possibly under shard subfolders (e.g. `shard_0003/`).
- `input_dir_hist` must contain the **HIST NPZs** with arrays `HIST` (counts) and optionally `bin_edges`. The analyzer will find the matching HIST file(s) by shard and filename.  
- The analyzer **asserts** A↔B consistency and histogram-name matching; it fails fast if something is off.
- `edge: [low, high]` applies independently to A and B (e.g. ignore the low-mass region).
- `triplet_plots.nevents_edges` may be in descending order; the code handles open/closed last bin correctly.

### Section `generate_toys_bkg_only` (legacy)
- `output_dir` – where shards and PEs are written  
- `backgrounds` – template histograms (`hist`, `binning`, `name`)  
- `n_pseudo_experiments`, `pe_per_shard`, `seed`, `pool` (workers), …

### Section `predict_toys_bkg_only`
- `checkpoint` – trained BumpNet `.ckpt`  
- `input_dir` / `output_dir` – PE source / predictions destination  
- `smooth_*csv` – smoothed backgrounds (required if `compute_Z_LR: true`)  
- `compute_Z_LR` – toggle ZLR  
- `max_pe`, `pool`, `verbose`

### Section `plot_toys_bkg_only` (legacy plotting)
- `input_dir`, `input_dir_hist`, `output_dir`, `edge`, `max_pes`, `min_nbins`  
- `do_zmax_analysis`, `check_coincident_mass_bumps`, `coincidence_threshold`, `do_top10_analysis`  
- `dpi`, `format`, `show`

---

## What `analyze_split_ds.py` produces

In `<output_dir>/` you can expect:

- **Global distributions**
  - `B_at_A_bin_Z_distribution.<fmt>`  
    Overlay of `Z(B@A)` for **BumpNet** (red) and **ZLR** (blue), with robust Gaussian fits and **empirical 3σ lines**.  
    *Implementation note:* label boxes live in the **figure header** to avoid overlaps with vertical annotations and legends.
  - `maxZmax_distribution_A.<fmt>`  
    Distribution of **A-side max Z per PE** (BumpNet vs ZLR) with **empirical global‑3σ thresholds** (based on those distributions).
  - `maxzmax_position_distribution_A.<fmt>` and `maxzmax_position_by_binindex_A.<fmt>`  
    Where the A maxima occur (relative position and absolute index variants).

- **ΔZmax heatmaps** (optional, single switch):
  - `deltaZmax_vs_position_A.<fmt>`
  - `deltaZmax_vs_position_B.<fmt>`  
    Each shows ΔZmax = (max Zpred − max ZLR) vs the LR‑max relative position, with μ and σ printed.

- **Triplet families** (per slice):
  - `triplet_by_relpos_*.{fmt}`  
  - `triplet_by_nevents_*.{fmt}`  
    For each slice you get three aligned panels: Zpred, ZLR, and ΔZ, each with a histogram, a Gaussian overlay, and a μ/σ/χ²/ndof box. Y‑headroom is hard‑coded (separate for log/linear) for readable peaks independent of autoscale.

- **Examples**
  - `examples/PE_<id>_<histname>_maxZ_example.<fmt>`  
    Side-by-side A/B mass spectra and Z curves with the **A-selected** bin highlighted; A and B mass panels share y‑scales (linear or log based on a simple heuristic).

---

## Typical A/B directory structure

```
<input_dir>/                          # predictions
└── shard_0003/
    ├── PE_0000123_A.npz
    └── PE_0000123_B.npz

<input_dir_hist>/                     # raw histograms
└── shard_0003/
    ├── PE_0000123.npz                # contains HIST (+ optional bin_edges)
    └── ...
```
The analyzer also handles “flat” layouts when shard names don’t exist under `input_dir_hist`, as long as files are present and unambiguous.

---

## FAQ

**How is global significance read from `B_at_A_bin_Z_distribution`?**  
The **dashed vertical lines** are *empirical* **3σ one-sided tail thresholds** of the shown distributions. If BumpNet’s line is, say, at 3.1, then “`global 3σ` ⇒ `Zpred > 3.1` at B@A”. The same is computed for ZLR.

**What does the A/B split buy us?**  
It transforms the “scan penalty” into **real‑world practice**: select a candidate using only **A**, and ask **B** to confirm at the **same bin**. This directly estimates the global false‑positive rate under background‑only.

**What if ZLR isn’t available?**  
The analyzer still runs; ZLR panels just won’t be drawn where not available. For robust comparisons and thresholds, include ZLR when possible.

**Why do some triplet slices look sparse?**  
Your `relpos_edges`/`nevents_edges` may be too fine; merge slices or increase statistics.

---

## Minimal commands (examples)

Run toys (optional):
```bash
bash submit_rorqual.sh global_significance/toys_atlas.yaml
```

If you already have split A/B predictions + histograms, just enable:
```yaml
tasks:
  do_analyze_split_ds: true
```
and fill `analyze_split_ds:` with your folders and options.

---

Happy (and globally safe) bump hunting!
