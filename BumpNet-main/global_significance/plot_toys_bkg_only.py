# ================================================================
# Plot helpers for the “global-significance” study:
#   * scans BumpNet/ZLR outputs
#   * finds extreme Z values / coincidences
#   * generates PNG summaries
# ================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path
from tqdm import tqdm  # Provides progress bars for loops
from collections import Counter, defaultdict
import argparse, json
import os
import re     
import shutil 

# ------------------------------------------------------------
# Helper to turn an arbitrary string into a file-system-safe slug
# ------------------------------------------------------------
_slug = lambda s: re.sub(r'[^A-Za-z0-9._-]+', '_', str(s))

# ------------------------------------------------------------
# Helper: make sure a directory exists **and** is empty
# ------------------------------------------------------------
def _ensure_empty_dir(path: Path):
    """
    Remove all files/subdirs inside `path` (if it exists) and recreate it.
    """
    path = Path(path)
    if path.exists():
        for child in path.iterdir():
            try:
                child.unlink()          # regular file, symlink
            except IsADirectoryError:   # directory
                shutil.rmtree(child)
    path.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------
# PE / histogram discovery
# ----------------------------------------------------------------
def find_all_pe_files(root_dir):
    """Recursively find all PE_*.npz files under root_dir."""
    return sorted(Path(root_dir).rglob('PE_*.npz'))


def find_matching_hist_file(pe_path, input_dir, input_dir_hist):
    """Return corresponding histogram file path."""
    rel_path = pe_path.relative_to(input_dir)
    hist_name = rel_path.name.replace('.pz.npz', '.npz')
    return Path(input_dir_hist) / rel_path.parent / hist_name

# ------------------------------------------------------------------
# Mother-histogram library  (loaded once, reused everywhere)
# ------------------------------------------------------------------
_smooth_cache = None
def _get_smooth_library(smooth_dir):
    """
    Build dict  name -> (edges, counts)  from three *.npy files in `smooth_dir`:
        background.npy, bin_edges.npy, names.npy

    Arrays may be ragged (dtype=object). If bin_edges[i] length equals nbins,
    treat it as centers and convert to true edges (nbins+1).
    """
    global _smooth_cache
    if _smooth_cache is not None:
        return _smooth_cache

    if not smooth_dir:
        _smooth_cache = {}           # overlay disabled
        return _smooth_cache

    d = Path(smooth_dir)
    bkg_arr   = np.load(d / "background.npy", allow_pickle=True)
    edges_arr = np.load(d / "bin_edges.npy",  allow_pickle=True)
    names_arr = np.load(d / "names.npy",      allow_pickle=True)

    # Ensure 1-D object arrays (ragged-safe)
    bkg_arr   = np.asarray(bkg_arr,   dtype=object).reshape(-1)
    edges_arr = np.asarray(edges_arr, dtype=object).reshape(-1)
    names_arr = np.asarray(names_arr, dtype=object).reshape(-1)

    if not (len(bkg_arr) == len(edges_arr) == len(names_arr)):
        raise ValueError("smooth_dir arrays differ in length: "
                         f"{len(bkg_arr)} vs {len(edges_arr)} vs {len(names_arr)}")

    lib = {}
    for i, nm in enumerate(names_arr):
        name   = str(nm)
        counts = np.asarray(bkg_arr[i],   dtype=float)
        edges  = np.asarray(edges_arr[i], dtype=float)

        # Strip any non-finite tails (paranoia)
        counts = counts[np.isfinite(counts)]
        edges  = edges [np.isfinite(edges )]

        nb = len(counts)
        if len(edges) == nb + 1:
            edges_use = edges
        elif len(edges) == nb:
            # centers → edges
            c = edges
            mids  = 0.5 * (c[:-1] + c[1:])
            left  = c[0]  - 0.5 * (c[1]  - c[0])
            right = c[-1] + 0.5 * (c[-1] - c[-2])
            edges_use = np.concatenate(([left], mids, [right]))
        else:
            raise ValueError(f"{name}: len(edges)={len(edges)} not in {{nbins, nbins+1}} (nbins={nb})")

        # IMPORTANT: return (edges, counts) to match _mother_hist_for
        lib[name] = (edges_use, counts)

    _smooth_cache = lib
    return lib

def _mother_hist_for(name, smooth_lib, edges_needed):
    """
    Return mother-hist counts on the requested binning
    (interpolated if needed) or None if not available.
    """
    if name not in smooth_lib:
        return None

    edges_src, counts_src = smooth_lib[name]

    # --- if the binning matches exactly (same length *and* values) ----
    if edges_src.shape == edges_needed.shape \
       and np.allclose(edges_src, edges_needed, atol=1e-6):
        return counts_src

    # --- otherwise: interpolate to the requested centres -------------
    from scipy.interpolate import interp1d
    centres_src = 0.5 * (edges_src[1:] + edges_src[:-1])
    centres_dst = 0.5 * (edges_needed[1:] + edges_needed[:-1])

    f = interp1d(centres_src, counts_src,
                 kind='linear', bounds_error=False,
                 fill_value=0.0)
    return f(centres_dst)

def _draw_left_cut(ax, edges, left_frac, color="grey", alpha=0.15):
    """
    Shade the mass region that is ignored in the bump search.
    """
    n_bins = len(edges) - 1
    cut_idx = max(0, int(left_frac * n_bins + .5))
    cut_x = edges[cut_idx]
    ax.axvspan(edges[0], cut_x, color=color, alpha=alpha, zorder=0)

def load_zmax_per_pe(input_dir,
                     input_dir_hist,
                     edge=(0.0, 1.0),
                     max_pes=None,
                     min_nbins=1):
    """
    Loop over PE_*.npz, keep only bins in `edge` window, and
    record “star” histograms for BumpNet & Z_LR.
    Returns many arrays used by downstream plots.
    """
    from collections import Counter
    input_dir   = Path(input_dir)
    pe_files    = find_all_pe_files(input_dir)
    if max_pes is not None:
        pe_files = pe_files[:max_pes]

    # ── PER-HISTOGRAM containers (returned to caller) ────────────────
    zmax_pred_hist, zmax_lr_hist = [], []
    rel_pos_pred_hist, rel_pos_lr_hist = [], []
    delta_z_at_same_bin = []

    # ── Star-hist bookkeeping (same as before) ───────────────────────
    eff_len_pred, eff_len_lr   = [], []
    maxzmax_hist_names         = []
    example_data               = []
    all_eff_bins               = []
    delta_maxzmax_values = []  # stores max(Z_pred) – max(Z_LR in same histogram)

    # ── PE-level (star-histogram) summaries – needed only for Top-10 printout
    zmax_pred_values, zmax_lr_values = [], []   # Zmax per PE   (BumpNet / LR)
    rel_pos_pred_pe , rel_pos_lr_pe  = [], []   # rel. bin pos  per PE

    for idx, fpath in enumerate(tqdm(pe_files, desc="Loading PEs")):
        with np.load(fpath, allow_pickle=True) as zdat:
            z_arr    = zdat['z']
            z_lr_arr = zdat.get('z_lr')            # may be None
            names    = zdat.get('names')
            bin_edgesL = zdat.get('bin_edges')

        # companion histogram file ------------------------------------
        hfile = find_matching_hist_file(fpath, input_dir, input_dir_hist)
        if not hfile.exists():
            print(f"[WARN] {fpath.name}: missing histogram file – skipped")
            continue
        with np.load(hfile, allow_pickle=True) as hdat:
            hist_arrs = hdat['HIST']

        # local holders to find PE-level winners ----------------------
        zmax_per_hist_pred, zmax_per_hist_lr = [], []
        kept_rows, eff_bins_list = [], []

        # ---------- loop over histograms in this PE ------------------
        for j_orig, (z_row, hist_row) in enumerate(zip(z_arr, hist_arrs)):

            # ---- keep histogram if it has at least one non-zero bin ----
            nonzero = np.flatnonzero(hist_row != 0)
            if nonzero.size == 0:
                continue

            eff_n = nonzero[-1] + 1                 # effective length
            if eff_n < min_nbins:
                continue

            kept_rows.append(j_orig)
            eff_bins_list.append(eff_n)
            all_eff_bins.append(eff_n)

            # ---- reference-style edge cut on the *full* length ----------
            n_bins = len(z_row)                       #  ✱ use prediction length ✱
            sb = max(0, int(edge[0] * n_bins + 0.5))
            lb = max(sb + 1, min(n_bins,
                                int(edge[1] * n_bins)))

            if sb >= lb:      # paranoia guard, prevents zero-length slice
                continue

            # ---- BumpNet -------------------------------------------------
            z_slice_pred  = z_row[sb:lb]            # NO eff_n truncation
            zmax_pred_val = np.max(z_slice_pred)
            zmax_pred_bin = sb + int(np.argmax(z_slice_pred)) 
            pos_pred_val  = zmax_pred_bin / n_bins

            zmax_pred_hist.append(zmax_pred_val)
            rel_pos_pred_hist.append(pos_pred_val)
            zmax_per_hist_pred.append(zmax_pred_val)

            # ---- ΔZ at the same bin as Zmax^pred -------------------
            if z_lr_arr is not None:
                z_lr_row = z_lr_arr[j_orig]
                if zmax_pred_bin < len(z_lr_row):
                    delta_z_val = z_row[zmax_pred_bin] - z_lr_row[zmax_pred_bin]
                else:
                    delta_z_val = np.nan
            else:
                delta_z_val = np.nan
            delta_z_at_same_bin.append(delta_z_val)

            # ---- LR ------------------------------------------------------
            if z_lr_arr is not None:
                z_slice_lr  = z_lr_arr[j_orig][sb:lb]
                zmax_lr_val = np.max(z_slice_lr)
                pos_lr_val  = (sb + np.argmax(z_slice_lr)) / n_bins
            else:
                zmax_lr_val = np.nan
                pos_lr_val  = np.nan

            zmax_lr_hist.append(zmax_lr_val)
            rel_pos_lr_hist.append(pos_lr_val)
            zmax_per_hist_lr.append(zmax_lr_val)

        # nothing useful in this PE?
        if not zmax_per_hist_pred:
            continue

        # ── star-histogram for BumpNet ─────────────────────────────────
        j_loc_star_pred = int(np.argmax(zmax_per_hist_pred))
        j_star_pred     = kept_rows[j_loc_star_pred]
        eff_n_pred      = eff_bins_list[j_loc_star_pred]
        eff_len_pred.append(eff_n_pred)

        sb = max(0, int(edge[0] * eff_n_pred + .5))
        lb = max(sb + 1, min(eff_n_pred, int(edge[1] * eff_n_pred)))
        z_trim_pred = z_arr[j_star_pred][:eff_n_pred]
        max_local_pred = int(np.argmax(z_trim_pred[sb:lb]))

        relbin = (sb + max_local_pred) / eff_n_pred
        rel_pos_pred_pe.append(relbin) 
        zmax_pred_values.append(zmax_per_hist_pred[j_loc_star_pred])

        # Compute Δ(max Z_pred − max Z_LR) in same histogram
        if z_lr_arr is not None:
            z_lr_star_hist = z_lr_arr[j_star_pred][:eff_n_pred]
            zmax_lr_in_star = np.max(z_lr_star_hist)
            delta_maxzmax = zmax_pred_values[-1] - zmax_lr_in_star
        else:
            delta_maxzmax = np.nan
        delta_maxzmax_values.append(delta_maxzmax)

        # name bookkeeping stays keyed to BumpNet star
        hname = "(no histogram name)"
        if names is not None:
            e = names[j_star_pred]
            hname = e if isinstance(e, str) else e[0]
        maxzmax_hist_names.append(hname)

        # ── star-histogram for Z_LR (may differ) ───────────────────────
        if np.isfinite(zmax_per_hist_lr).any():
            j_loc_star_lr = int(np.argmax(zmax_per_hist_lr))
            j_star_lr     = kept_rows[j_loc_star_lr]
            eff_n_lr      = eff_bins_list[j_loc_star_lr]
            eff_len_lr.append(eff_n_lr)

            sb_lr = max(0, int(edge[0] * eff_n_lr + .5))
            lb_lr = max(sb_lr + 1, min(eff_n_lr, int(edge[1] * eff_n_lr)))
            z_trim_lr = z_lr_arr[j_star_lr][:eff_n_lr]
            max_local_lr = int(np.argmax(z_trim_lr[sb_lr:lb_lr]))

            relbin_lr = (sb_lr + max_local_lr) / eff_n_lr
            rel_pos_lr_pe.append(relbin_lr)
            zmax_lr_values.append(zmax_per_hist_lr[j_loc_star_lr])
        else:
            eff_len_lr.append(np.nan)
            rel_pos_lr_pe.append(np.nan)
            zmax_lr_values.append(np.nan)

        # ── save example for first 10 PEs (same as before) ─────────────
        if idx < 10:
            hist_trim = hist_arrs[j_star_pred][:eff_n_pred]

            # ---------- build trimmed bin-edges -----------------------
            if bin_edgesL is not None:
                if bin_edgesL.dtype == object:          # ragged 1-D
                    raw_edges = bin_edgesL[j_star_pred]
                elif bin_edgesL.ndim > 1:               # 2-D
                    raw_edges = bin_edgesL[j_star_pred]
                else:                                   # single common array
                    raw_edges = bin_edgesL
            else:
                raw_edges = np.arange(eff_n_pred + 1)   # fallback
            edges_trim = np.asarray(raw_edges[:eff_n_pred + 1], dtype=float)

            # ---------- store *both* Z-arrays -------------------------
            z_trim_pred = z_arr[j_star_pred][:eff_n_pred]
            z_trim_lr   = (z_lr_arr[j_star_pred][:eff_n_pred]
                           if z_lr_arr is not None else None)

            example_data.append((hist_trim,
                                 edges_trim,
                                 (z_trim_pred, z_trim_lr),   # ← tuple!
                                 hname,
                                 fpath.stem))

            print(f"[PE {idx:02d}] {fpath.name} → {hname} "
                  f"(eff_n={eff_n_pred}) → "
                  f"Zmax_pred={zmax_per_hist_pred[j_loc_star_pred]:.2f}")

    # ──  summary printout ─────────────────────────────────
    name_counter = Counter(maxzmax_hist_names)
    print("\nTop-10 histograms producing max(Zmax_pred) most frequently:")
    for rank, (hname, n) in enumerate(name_counter.most_common(10), 1):
        print(f"{rank:2d}) {hname:<60s} → {n:4d} occurrences")
        # ---- one example per offending PE, like the legacy version -------
        for k, (nm, zmax, relpos) in enumerate(
                zip(maxzmax_hist_names, zmax_pred_values, rel_pos_pred_pe)):
            if nm == hname:
                print(f"     • PE {k:04d}  Zmax={zmax:5.2f}  rel.bin.pos={relpos:6.3f}")

    # ── first_occurrence dict (keyed to pred stars) ───────
    hot_names        = {n for n, _ in name_counter.most_common(10)}
    first_occurrence = {}
    for hname_winner, fpath in zip(maxzmax_hist_names, pe_files):
        if hname_winner not in hot_names or hname_winner in first_occurrence:
            continue
        with np.load(fpath, allow_pickle=True) as zdat:
            z_arr = zdat['z']; z_lr_arr = zdat.get('z_lr'); names = zdat['names']; edgesL = zdat.get('bin_edges')
        hfile = find_matching_hist_file(fpath, input_dir, input_dir_hist)
        with np.load(hfile, allow_pickle=True) as hdat:
            hists = hdat['HIST']

        def _lab(e): return e if isinstance(e, str) else e[0]
        cand = [j for j, e in enumerate(names) if _lab(e) == hname_winner]
        j_star = max(cand, key=lambda j: np.max(z_arr[j]))

        nz = np.where(hists[j_star] > 0)[0]; eff_n = nz[-1] + 1 if nz.size else 1
        hist_t = hists[j_star][:eff_n]; z_t = z_arr[j_star][:eff_n]
        z_lr_t   = z_lr_arr[j_star][:eff_n] if z_lr_arr is not None else None
        if edgesL is not None:
            if edgesL.dtype == object:          # 1-D object array (ragged)
                raw_edges = edgesL[j_star]
            elif edgesL.ndim > 1:               # regular 2-D float array
                raw_edges = edgesL[j_star]
            else:                               # single common binning
                raw_edges = edgesL
        else:
            raw_edges = np.arange(eff_n + 1)    # fallback: 0,1,2,…

        edges_t = np.asarray(raw_edges[:eff_n + 1], dtype=float)

        first_occurrence[hname_winner] = (hist_t, edges_t, (z_t, z_lr_t),
                                          hname_winner, fpath.stem)
        if len(first_occurrence) == 10:
            break

    # ---------- RETURN per-hist arrays + star-hist extras ------------
    return (np.asarray(zmax_pred_hist),
            np.asarray(zmax_lr_hist),
            example_data,
            np.asarray(rel_pos_pred_hist),
            np.asarray(rel_pos_lr_hist),
            maxzmax_hist_names,
            np.asarray(zmax_pred_values),
            np.asarray(zmax_lr_values),
            np.asarray(rel_pos_pred_pe),
            np.asarray(rel_pos_lr_pe),
            all_eff_bins,
            eff_len_pred,
            eff_len_lr,
            first_occurrence,
            np.asarray(delta_z_at_same_bin),
            np.asarray(delta_maxzmax_values))



# ──────────────────────────────────────────────────────────────────────
#  2.  Top-level driver that scans all PEs, collects totals, and prints
#      the overall probability.
# ──────────────────────────────────────────────────────────────────────
def find_coincident_mass_bumps(input_dir,
                               input_dir_hist,
                               threshold=5.0,
                               edge=[0.0, 1.0],
                               min_nbins=1,
                               max_pes=None,
                               output_dir=None,
                               smooth_lib=None,
                               verbose=True):
    """
    Scan prediction ensembles (PEs) and count how many exhibit at least
    one pair of coincident bumps above `threshold`.
    """
    input_dir  = Path(input_dir)
    pe_files   = sorted(input_dir.rglob('PE_*.npz'))
    if max_pes is not None:
        pe_files = pe_files[:max_pes]

    total_matches = 0   # how many coincident pairs overall
    n_pe_scanned  = 0   # how many PE files were successfully analysed

    for fpath in tqdm(pe_files, desc="Checking coincidences"):
        fpath_hist = find_matching_hist_file(fpath, input_dir, input_dir_hist)

        if not fpath.exists() or not fpath_hist.exists():
            if verbose:
                print(f"[WARN] Skipping {fpath.name} (missing prediction or histogram file)")
            continue

        try:
            with np.load(fpath, allow_pickle=True) as zdata, np.load(fpath_hist, allow_pickle=True) as hdata:
                z_array    = zdata['z']
                names      = zdata['names']
                bin_edges  = zdata['bin_edges']
                hists      = hdata['HIST']
        except Exception as e:
            print(f"[ERROR] Could not load {fpath.name}: {e}")
            continue

        n_pe_scanned += 1

        # ---------- build PE-local list --------------------------------
        pe_data = []
        for z_row, bins_row, name, hist in zip(z_array, bin_edges, names, hists):
            nonzero_bins      = np.where(hist > 0)[0]
            effective_n_bins  = nonzero_bins[-1] + 1 if nonzero_bins.size else 1
            if effective_n_bins < min_nbins:
                continue

            sb = max(0, int(edge[0] * effective_n_bins + .5))
            lb = min(effective_n_bins, int(edge[1] * effective_n_bins))
            if sb >= lb:
                continue

            z_slice = z_row[sb:lb]
            if z_slice.size == 0 or np.max(z_slice) < threshold:
                continue

            trimmed_z     = z_row[:effective_n_bins]
            trimmed_edges = bins_row[:effective_n_bins + 1]
            hist_name     = name if isinstance(name, str) else name[0]

            pe_data.append((hist_name, trimmed_z, trimmed_edges, hist))

        # ---------- count coincidences in this PE ----------------------
        total_matches += _find_matches_within_pe(
            pe_data,
            pe_filename=fpath.name,
            output_dir=output_dir,
            threshold=threshold,
            min_nbins=min_nbins,
            verbose=verbose,
            smooth_lib=smooth_lib,
            left_frac=edge[0] 
        )

    # ---------- final summary -----------------------------------------
    if n_pe_scanned:
        prob = total_matches / n_pe_scanned
        print("\n══════════════════════════════════════════════════════")
        print(f"Found {total_matches} coincident mass-bump matches "
              f"in {n_pe_scanned} PEs")
        print(f"⇒  P(match) = {prob:.4f}  ({prob*100:.2f} %)")
        print("══════════════════════════════════════════════════════")
    else:
        print("\n[WARN] No PE files processed – probability undefined.")

# ──────────────────────────────────────────────────────────────────────
#  1.  Helper that scans ONE PE and returns how many tight coincidences
#      it found (0, 1, 2, …).
# ──────────────────────────────────────────────────────────────────────
def _find_matches_within_pe(zmax_data,
                            pe_filename,
                            output_dir,
                            threshold=5.0,
                            min_nbins=1,
                            verbose=True, smooth_lib=None, left_frac=0.0):
    """
    Parameters
    ----------
    zmax_data : list[(name, z, bins, hist)]
        Output list constructed in the calling routine.
    Returns
    -------
    int
        Number of coincident mass-bump matches found in this PE.
    """
    # keep only rows long enough
    filtered_data = [
        (name, z, bins, hist)
        for (name, z, bins, hist) in zmax_data
        if len(z) >= min_nbins
    ]

    match_counter = 0       

    n = len(filtered_data)
    for i in range(n):
        name_i, z_i, bins_i, hist_i = filtered_data[i]
        if np.max(z_i) < threshold:
            continue
        peak_bin_i = np.argmax(z_i)
        cat_i = name_i.split("cat")[-1]

        for j in range(i + 1, n):
            name_j, z_j, bins_j, hist_j = filtered_data[j]
            if np.max(z_j) < threshold:
                continue
            cat_j = name_j.split("cat")[-1]
            if cat_i == cat_j:
                continue

            peak_bin_j = np.argmax(z_j)

            # ---------- build ±1-bin windows --------------------------
            start_i = max(0, peak_bin_i - 1)
            end_i   = min(len(bins_i) - 2, peak_bin_i + 1)
            start_j = max(0, peak_bin_j - 1)
            end_j   = min(len(bins_j) - 2, peak_bin_j + 1)

            window_i = (bins_i[start_i], bins_i[end_i + 1])
            window_j = (bins_j[start_j], bins_j[end_j + 1])

            edges_i  = bins_i[start_i:end_i + 2]
            edges_j  = bins_j[start_j:end_j + 2]

            # ---------- tight-match requirement ----------------------
            width_i = window_i[1] - window_i[0]
            width_j = window_j[1] - window_j[0]

            # peak masses (bin centres)
            peak_mass_i = 0.5 * (bins_i[peak_bin_i] + bins_i[peak_bin_i + 1])
            peak_mass_j = 0.5 * (bins_j[peak_bin_j] + bins_j[peak_bin_j + 1])

            if width_i <= width_j:          # i is narrower
                if not (window_j[0] <= peak_mass_i <= window_j[1]):
                    continue
            else:                            # j is narrower
                if not (window_i[0] <= peak_mass_j <= window_i[1]):
                    continue

            # optional overlap safeguard
            if window_i[1] < window_j[0] or window_j[1] < window_i[0]:
                continue

            # ---------- record & report the match --------------------
            match_counter += 1            

            if output_dir:
                plot_coincidence_match(
                    pe_filename, name_i, hist_i, z_i, bins_i,
                    name_j, hist_j, z_j, bins_j, output_dir, smooth_lib=smooth_lib, left_frac=left_frac
                )

            if verbose:
                print(f"\n🔁 Match in PE file {pe_filename}:")
                print(f"  ➤ {name_i} (Zmax={np.max(z_i):.2f}, mass bin edges={edges_i})")
                print(f"  ➤ {name_j} (Zmax={np.max(z_j):.2f}, mass bin edges={edges_j})")

    return match_counter

def plot_coincidence_match(pe_file, name_i, hist_i, z_i, bins_i, name_j, hist_j, z_j, bins_j, output_dir, smooth_lib=None, left_frac=0.0):
    fig, (ax1, axr, ax2) = plt.subplots(3, 1, figsize=(8, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    
    centers_i = 0.5 * (bins_i[1:] + bins_i[:-1])
    centers_j = 0.5 * (bins_j[1:] + bins_j[:-1])

    _draw_left_cut(ax1, bins_i, left_frac, color="tab:blue",   alpha=0.12)
    _draw_left_cut(ax1, bins_j, left_frac, color="tab:orange", alpha=0.12)

    # Top panel — mass histograms
    ax1.step(centers_i, hist_i[:len(centers_i)], where='mid', label=name_i, color='tab:blue')
    ax1.step(centers_j, hist_j[:len(centers_j)], where='mid', label=name_j, color='tab:orange')

    mother_i = _mother_hist_for(name_i, smooth_lib, bins_i)
    mother_j = _mother_hist_for(name_j, smooth_lib, bins_j)
    if mother_i is not None:
        ax1.step(centers_i, mother_i, where='mid',
                color='green', linestyle='--', label=f'{name_i} mother')
        res_i = hist_i[:len(mother_i)] - mother_i
        axr.step(centers_i, res_i, where='mid', color='tab:blue')
    if mother_j is not None:
        ax1.step(centers_j, mother_j, where='mid',
                color='green', linestyle='--', label=f'{name_j} mother')
        res_j = hist_j[:len(mother_j)] - mother_j
        axr.step(centers_j, res_j, where='mid', color='tab:orange')

    axr.axhline(0, color='grey', lw=0.5)
    axr.set_ylabel("Resid.")

    ax1.set_yscale("log")
    ax1.set_ylabel("Entries")
    ax1.legend()
    ax1.legend(fontsize=8, loc='best')
    ax1.set_title(f"Mass histograms for match in {pe_file}")

    # Bottom panel — predicted Z
    ax2.step(centers_i, z_i[:len(centers_i)], where='mid', color='tab:blue')
    ax2.step(centers_j, z_j[:len(centers_j)], where='mid', color='tab:orange')
    ax2.set_xlabel("Mass [GeV]")
    ax2.set_ylabel("Predicted Z")

    plt.tight_layout()

    match_dir = Path(output_dir) / "coincident_matches"
    match_dir.mkdir(parents=True, exist_ok=True)
    outfile = match_dir / f"{pe_file}_match_{name_i}_vs_{name_j}.png"
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"📈 Saved coincidence plot: {outfile}")

def analyze_top10_worst_offenders(input_dir, input_dir_hist, output_dir, edge=(0.0, 1.0), min_nbins=1, dpi=300, fmt='png', show=False):
    """
    Analyze all PEs but only for a specific list of histogram names stored in
    'list_top10worstOffenders_maxZmax.txt' located in the current directory.

    Produces:
      - ΔZmax vs position of Zmax_pred
      - Histogram of ΔZmax
    """
    from matplotlib.ticker import AutoMinorLocator
    from matplotlib import colormaps
    from seaborn import kdeplot
    import matplotlib.pyplot as plt
    import numpy as np

    # Load offending histograms
    with open("list_top10worstOffenders_maxZmax.txt") as f:
        top10_names = set(line.strip() for line in f if line.strip())

    delta_zmax_vals = []
    rel_pos_lr_vals = []

    for fpath in find_all_pe_files(input_dir):
        zdat = np.load(fpath, allow_pickle=True)
        z_arr, names = zdat["z"], zdat["names"]
        z_lr_arr = zdat.get("z_lr")
        bin_edgesL = zdat.get("bin_edges")

        f_hist = find_matching_hist_file(fpath, input_dir, input_dir_hist)
        if not f_hist.exists():
            continue
        hist_arrs = np.load(f_hist, allow_pickle=True)["HIST"]

        for j, (z_row, name_entry, hist_row) in enumerate(zip(z_arr, names, hist_arrs)):
            hname = name_entry if isinstance(name_entry, str) else name_entry[0]
            if hname not in top10_names:
                continue

            nonzero = np.flatnonzero(hist_row)
            if nonzero.size == 0:
                continue
            eff_n = nonzero[-1] + 1
            if eff_n < min_nbins:
                continue

            z_row = z_row[:eff_n]
            z_lr_row = z_lr_arr[j][:eff_n] if z_lr_arr is not None else None

            sb = max(0, int(edge[0] * eff_n + 0.5))
            lb = min(eff_n, int(edge[1] * eff_n + 0.5))
            if sb >= lb:
                continue

            zmax_pred = np.max(z_row[sb:lb])
            zmax_lr = np.max(z_lr_row[sb:lb]) if z_lr_row is not None else np.nan
            delta = zmax_pred - zmax_lr
            rel_bin_lr = (sb + np.argmax(z_row[sb:lb])) / eff_n

            delta_zmax_vals.append(delta)
            rel_pos_lr_vals.append(rel_bin_lr)

    # ---------- PLOTTING ----------
    delta = np.asarray(delta_zmax_vals)
    relpos = np.asarray(rel_pos_lr_vals)
    good = np.isfinite(delta) & np.isfinite(relpos)
    if not good.any():
        print("[WARN] No good entries for Top10 ΔZmax plot.")
        return

    delta = delta[good]
    relpos = relpos[good]
    cmap = colormaps["Spectral_r"]
    cmap.set_under('white', alpha=0)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 2D plot
    fig, ax = plt.subplots(figsize=(10/2.54, 7/2.54), dpi=dpi)
    hb = ax.hist2d(relpos, delta, bins=(100, 100), cmap=cmap, vmin=1e-3)
    kdeplot(x=relpos, y=delta, ax=ax, fill=False, cmap='Spectral_r')
    mu, sigma = np.nanmean(delta), np.nanstd(delta)
    ax.annotate(fr"$\mu = {mu:+.3f}$\n$\sigma = {sigma:.2f}$",
                xy=(0.02, 0.97), xycoords='axes fraction',
                ha='left', va='top', fontsize=9)
    ax.set_xlabel(r"$Z^{\mathrm{LR}}_{\max}$ bin / number of bins", loc="right")
    ax.set_ylabel(r"$\Delta Z^{\mathrm{top-10\ offenders}}_{\max}$", loc="center")
    ax.axhline(0, color="black", lw=0.6)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    fig.colorbar(hb[3], ax=ax, label="Entries")

    fout1 = outdir / f"top10_deltaZmax_vs_position.{fmt}"
    fig.savefig(fout1, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)
    print(f"Saved {fout1}")

    # 1D plot
    fig2, ax2 = plt.subplots(figsize=(10/2.54, 7/2.54), dpi=dpi)
    ax2.hist(delta, bins=80, histtype='step', color='black')
    ax2.annotate(fr"$\mu = {mu:+.3f}$" "\n" fr"$\sigma = {sigma:.2f}$",
                 xy=(0.02, 0.97), xycoords='axes fraction',
                 ha='left', va='top', fontsize=9)
    ax2.set_xlabel(r"$\Delta Z^{\mathrm{top-10\ offenders}}_{\max}$", loc="right")
    ax2.set_ylabel("Entries", loc="top")
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.tick_params(axis='both', which='both', direction='in', top=True, right=True)

    fout2 = outdir / f"top10_deltaZmax_1d_distribution.{fmt}"
    fig2.savefig(fout2, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig2)
    print(f"Saved {fout2}")


def plot_max_zmax_distribution(zmax_pred,
                               zmax_lr,
                               output_path,
                               dpi=300,
                               fmt='png',
                               show=False):

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator
    from pathlib import Path

    zmax_pred = np.asarray(zmax_pred, dtype=float)
    zmax_lr   = np.asarray(zmax_lr  , dtype=float)

    # ---------- helper -------------------------------------------------
    def _stats(arr):
        mean   = float(np.mean(arr))
        thresh = float(np.quantile(arr, 0.99865))   # 3 σ one-sided
        return mean, thresh

    mean_pred, thr_pred = _stats(zmax_pred)
    mean_lr  , thr_lr   = _stats(zmax_lr)

    # ---------- figure -------------------------------------------------
    fig, ax = plt.subplots(figsize=(10/2.54, 10/2.54), dpi=dpi)

    # common bins
    bins = np.histogram_bin_edges(np.concatenate([zmax_pred, zmax_lr]), bins=100)

    ax.hist(zmax_pred, bins=bins, histtype='step', color='red',
            label=r'BumpNet')
    ax.hist(zmax_lr,   bins=bins, histtype='step', color='blue',
            label=r'$Z_{\mathrm{LR}}$')

    # leave some head-room
    ax.set_ylim(0, max(ax.get_yticks()) * 1.15)

    ax.set_xlabel(r'$\max(Z_{\max})$ per PE', loc='right')
    ax.set_ylabel('Number of PEs',           loc='top')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)

    # ---------- annotation (upper-left) ---------------------------------
    annotation = (
        rf'BumpNet  $\langle\max(Z^{{\mathrm{{pred}}}}_{{\max}})\rangle = {mean_pred:.2f}\,\sigma$'
        '\n'
        rf'$Z^{{\mathrm{{pred}}}}_{{\mathrm{{global}}}}>3\sigma'
        rf'\;\Rightarrow\;\max(Z^{{\mathrm{{pred}}}}_{{\max}})>{thr_pred:.2f}\,\sigma$'
        '\n\n'
        rf'$Z_{{\mathrm{{LR}}}}$  $\langle\max(Z^{{\mathrm{{LR}}}}_{{\max}})\rangle = {mean_lr:.2f}\,\sigma$'
        '\n'
        rf'$Z^{{\mathrm{{LR}}}}_{{\mathrm{{global}}}}>3\sigma'
        rf'\;\Rightarrow\;\max(Z^{{\mathrm{{LR}}}}_{{\max}})>{thr_lr:.2f}\,\sigma$'
    )

    # Place annotation a bit farther to the right so the legend fits
    ax.text(0.02, 0.97,            # slightly lower
            annotation,
            transform=ax.transAxes,
            ha='left', va='top',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.2',
                    fc='white', ec='none', alpha=0.9),
            zorder=3)

    # ---------- thresholds ---------------------------------------------
    y0, y1 = ax.get_ylim()
    ax.vlines([thr_pred], y0, y0 + 0.33*(y1 - y0),
              color='red',  linestyle='--', alpha=0.7)
    ax.vlines([thr_lr],   y0, y0 + 0.33*(y1 - y0),
              color='blue', linestyle='--', alpha=0.7)

    # ---------- legend (upper-right, opposite corner) ------------------
    ax.legend(loc='upper right',
            bbox_to_anchor=(0.98, 0.98),  # x, y in axes-fraction coords
            fontsize=8,
            frameon=True)

    if show:
        plt.show()

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    fout = output_path / f'maxZmax_distribution.{fmt}'
    fig.savefig(fout, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {fout}')


def plot_maxzmax_position_distribution(rel_pos_pred,
                                    rel_pos_lr,
                                    output_path,
                                    dpi=300,
                                    fmt='png',
                                    show=False):
    """
    Overlay the distribution of the relative position (in %) of the max-Zmax
    bin for
      • BumpNet   – red
      • Z_LR      – blue
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator
    from pathlib import Path

    rel_pos_pred = 100.0 * np.asarray(rel_pos_pred, dtype=float)
    rel_pos_lr   = 100.0 * np.asarray(rel_pos_lr,   dtype=float)

    fig, ax = plt.subplots(figsize=(10/2.54, 10/2.54), dpi=dpi)

    bins = np.linspace(0, 100, 51)   # 2-% bins

    ax.hist(rel_pos_pred, bins=bins, histtype='step',
            color='red',  label='BumpNet')
    ax.hist(rel_pos_lr,   bins=bins, histtype='step',
            color='blue', label='Z$_{\\mathrm{LR}}$')

    ax.set_xlabel('Position of max $Z$ in histogram [%]', loc='right')
    ax.set_ylabel('Number of PEs',                     loc='top')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both', which='both',
                   direction='in', top=True, right=True)

    ax.legend(fontsize=8)

    if show:
        plt.show()

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    outfile = output_path / f"zmax_position_distribution.{fmt}"
    fig.savefig(outfile, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {outfile}")

def plot_histogram_length_distribution(all_lengths,
                                       maxz_pred_lengths,
                                       maxz_lr_lengths,
                                       output_path,
                                       dpi=300,
                                       fmt='png',
                                       show=False):
    """
    Compare histogram effective lengths:
      • grey  – all histograms
      • red   – histograms hosting BumpNet max(Zmax)
      • blue  – histograms hosting Z_LR   max(Zmax)
    Areas of red & blue curves are normalised to the *same* area
    (area of red), so shapes can be compared directly.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator
    from pathlib import Path

    all_lengths       = np.asarray(all_lengths,        dtype=int)
    maxz_pred_lengths = np.asarray(maxz_pred_lengths, dtype=int)
    maxz_lr_lengths   = np.asarray(maxz_lr_lengths,   dtype=int)

    # common binning
    bins = np.histogram_bin_edges(
        np.concatenate([all_lengths, maxz_pred_lengths, maxz_lr_lengths]),
        bins=50
    )

    c_all,  _ = np.histogram(all_lengths,       bins=bins)
    c_pred, _ = np.histogram(maxz_pred_lengths, bins=bins)
    c_lr,  _ = np.histogram(maxz_lr_lengths,   bins=bins)

    # normalise red & blue to same area (area of red curve)
    area_target = c_pred.sum() if c_pred.sum() else 1
    scale_lr    = area_target / c_lr.sum() if c_lr.sum() else 1
    c_all_scaled  = c_all  * (area_target / c_all.sum()) if c_all.sum() else c_all
    c_lr_scaled   = c_lr   * scale_lr

    centres = 0.5 * (bins[:-1] + bins[1:])

    fig, ax = plt.subplots(figsize=(10/2.54, 10/2.54), dpi=dpi)
    ax.step(centres, c_all_scaled,  where='mid', color='grey', alpha=0.6,
            label='All histograms')
    ax.step(centres, c_pred,        where='mid', color='red',
            label='BumpNet max$(Z_{\\max})$')
    ax.step(centres, c_lr_scaled,   where='mid', color='blue',
            label='Z$_{\\mathrm{LR}}$ max$(Z_{\\max})$')

    ax.set_xlabel('Effective histogram length (non-zero bins)', loc='right')
    ax.set_ylabel('Number of histograms (area-norm.)',          loc='top')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both', which='both',
                   direction='in', top=True, right=True)
    ax.legend(fontsize=8)

    if show:
        plt.show()

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    outfile = output_path / f"histogram_length_distribution.{fmt}"
    fig.savefig(outfile, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {outfile}")

def plot_top_examples(example_data, output_dir, smooth_lib=None, left_frac=0.0):
    example_dir = Path(output_dir) / "examples"
    example_dir.mkdir(exist_ok=True)

    for i, (hist, edges, zvals, name, pe_file_name) in enumerate(example_data):
        z_pred, z_lr = zvals if isinstance(zvals, tuple) else (zvals, None)
        edges = np.asarray(edges, dtype=float) 
        centers = 0.5 * (edges[1:] + edges[:-1])
        use_log_scale = hist[0] >= 500

        fig, (ax1, axr, ax2) = plt.subplots(3, 1, figsize=(8, 7), sharex=True,
                                      gridspec_kw={'height_ratios': [3, 1, 1]})

        ax1.step(centers, hist, where='mid', color='black')
        if use_log_scale:
            ax1.set_yscale("log")
        ax1.set_ylabel("Entries")
        ax1.set_title(f"max($Z_{{max}}$) for {pe_file_name} is in {name}")
        _draw_left_cut(ax1, edges, left_frac)

        ax2.step(centers, z_pred, where='mid', color='red',  label='BumpNet')
        if z_lr is not None and np.isfinite(z_lr).any():
            ax2.step(centers, z_lr,   where='mid', color='blue', label='Z$_{\\mathrm{LR}}$')
        ax2.set_xlabel("Mass [GeV]")
        ax2.set_ylabel("Z")
        ax2.legend(fontsize=8, loc='best')

        mother = _mother_hist_for(name, smooth_lib, edges)
        if mother is not None:
            ax1.step(centers, mother, where='mid', color='green',
                    linestyle='--', label='Mother')
            ax1.legend(fontsize=8, loc='best')
            residual = hist - mother              # PE – mother
            axr.step(centers, residual, where='mid', color='black')
            axr.axhline(0, color='grey', lw=0.5)
            axr.set_ylabel("Resid.")

        plt.tight_layout()
        fig_path = example_dir / f"{pe_file_name}_maxZ_example.png"
        plt.savefig(fig_path)
        plt.close()
        print(f"Saved example plot: {fig_path}")

# ------------------------------------------------------------------
def plot_top_maxzmax_histograms(top_examples_dict,
                                output_dir,
                                dpi=300,
                                fmt='png', smooth_lib=None, left_frac=0.0):
    """
    top_examples_dict maps  histogram-name  →  (hist, edges, zvals, name, pe_stem)
    """
    output_path = Path(output_dir) / "top10_maxZmax_histos"
    output_path.mkdir(parents=True, exist_ok=True)

    for hname, (hist, edges, zvals, _, pe_stem) in top_examples_dict.items():
        z_pred, z_lr = zvals if isinstance(zvals, tuple) else (zvals, None)
        edges = np.asarray(edges, dtype=float)    
        centres = 0.5 * (edges[1:] + edges[:-1])
        use_log = hist[0] >= 500    

        fig, (ax1, axr, ax2) = plt.subplots(
            3, 1, figsize=(8, 7), sharex=True,
            gridspec_kw={'height_ratios': [3, 1, 1]}
        )

        # top panel – histogram
        ax1.step(centres, hist, where='mid', color='black')
        mother = _mother_hist_for(hname, smooth_lib, edges)
        if mother is not None:
            ax1.step(centres, mother, where='mid', color='green',
                    linestyle='--', label='Mother')
            ax1.legend(fontsize=8, loc='best')
            residual = hist - mother
            axr.step(centres, residual, where='mid', color='black')
            axr.axhline(0, color='grey', lw=0.5)
            axr.set_ylabel("Resid.")
        _draw_left_cut(ax1, edges, left_frac)

        if use_log:
            ax1.set_yscale('log')
        ax1.set_ylabel("Entries")
        ax1.set_title(f"Top-10: {hname}")

        ax2.step(centres, z_pred, where='mid', color='red', label='BumpNet')
        if z_lr is not None and np.isfinite(z_lr).any():
            ax2.step(centres, z_lr,   where='mid', color='blue', label='Z$_{\\mathrm{LR}}$')
        ax2.set_xlabel("Mass [GeV]")
        ax2.set_ylabel("Z")
        ax2.legend(fontsize=8, loc='best')

        plt.tight_layout()

        # -------------------- build a unique, readable filename
        #    e.g.  stack_mass_e0j0_cat_2ex_0mx_0gx_2jx_PE0050_top10Zmax.png
        # --------------------
        safe_hname = _slug(hname)[:120]      # keep it short-ish
        safe_pestem = _slug(pe_stem)
        fout = output_path / f"{safe_hname}_{safe_pestem}_top10Zmax.{fmt}"
        fig.savefig(fout, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved top-Zmax plot: {fout}")


def plot_deltaZmax_and_positions(zmax_pred,
                              zmax_lr,
                              rel_pos_pred,
                              rel_pos_lr,
                              output_path,
                              deltaZ_same_bin=None,   
                              use_predbin_lr=False,   
                              dpi=300,
                              fmt='png',
                              show=False):
    """
    Produces three files in `output_path`:
      • deltaZ_vs_position.<fmt>       – ΔZ vs rel bin position (pred or LR)
      • position_correlation.<fmt>     – rel. bin pos (LR vs BumpNet)
      • zmax_1d_distribution.<fmt>     – 1-D Z_max overlay
    """
    import numpy as np, matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator
    from matplotlib import colormaps
    from seaborn import kdeplot
    from pathlib import Path

    # ---------- data prep --------------------------------------------
    zmax_pred    = np.asarray(zmax_pred, dtype=float)
    zmax_lr      = np.asarray(zmax_lr,   dtype=float)
    rel_pos_pred = np.asarray(rel_pos_pred, dtype=float)
    rel_pos_lr   = np.asarray(rel_pos_lr,   dtype=float)

    if use_predbin_lr:
        if deltaZ_same_bin is None:
            raise ValueError("`deltaZ_same_bin` must be provided when use_predbin_lr=True")
        deltaZ = np.asarray(deltaZ_same_bin, dtype=float)
        x_pos = rel_pos_pred  
        xlabel = r"$Z^{\mathrm{pred}}_{\max}$ bin / number of bins"
    else:
        deltaZ = zmax_pred - zmax_lr
        x_pos = rel_pos_lr     # ← fallback: LR position
        xlabel = r"$Z^{\mathrm{LR}}_{\max}$ bin / number of bins"

    # Apply mask to all
    good = np.isfinite(zmax_pred) & np.isfinite(zmax_lr) \
         & np.isfinite(rel_pos_pred) & np.isfinite(rel_pos_lr) \
         & np.isfinite(deltaZ)
    if not good.any():
        print("[WARN] No finite entries for ΔZ / position plot")
        return

    zmax_pred    = zmax_pred[good]
    zmax_lr      = zmax_lr[good]
    rel_pos_pred = rel_pos_pred[good]
    rel_pos_lr   = rel_pos_lr[good]
    deltaZ       = deltaZ[good]
    x_pos        = x_pos[good]

    cmap = colormaps["Spectral_r"]
    cmap.set_under('white', alpha=0)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # ---------- FIRST FIGURE — ΔZ vs bin position --------------------
    fig1, ax1 = plt.subplots(figsize=(10/2.54, 7/2.54), dpi=dpi)
    hb = ax1.hist2d(x_pos, deltaZ, bins=(100, 100), cmap=cmap, vmin=1e-3)
    kdeplot(x=x_pos, y=deltaZ, ax=ax1, fill=False, cmap='Spectral_r')

    mu, sigma = np.nanmean(deltaZ), np.nanstd(deltaZ)
    ax1.annotate(rf"$\mu = {mu:+.3f}$\n$\sigma = {sigma:.2f}$",
                 xy=(0.02, 0.97), xycoords='axes fraction',
                 ha='left', va='top', fontsize=9)
    ax1.set_xlabel(xlabel, loc='right')
    ax1.set_ylabel(r"$\Delta Z_{\max}$"+"\n(BumpNet – LR)", loc='center')
    ax1.axhline(0, color='black', lw=0.6)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    fig1.colorbar(hb[3], ax=ax1, label="Entries")

    fout1 = output_path / f"deltaZmax_vs_position.{fmt}"
    fig1.savefig(fout1, bbox_inches='tight')
    if show: plt.show()
    plt.close(fig1)
    print(f"Saved {fout1}")

    # ---------- SECOND FIGURE — correlation of max-bin positions -----
    fig2, ax2 = plt.subplots(figsize=(10/2.54, 7/2.54), dpi=dpi)
    hb2 = ax2.hist2d(rel_pos_lr, rel_pos_pred, bins=(100, 100), cmap=cmap, vmin=1e-3)
    kdeplot(x=rel_pos_lr, y=rel_pos_pred, ax=ax2, fill=False, cmap='Spectral_r')
    ax2.plot([0, 1], [0, 1], ls='--', lw=0.7, color='black')

    ax2.set_xlabel(r"True max-bin position  (LR)", loc='right')
    ax2.set_ylabel(r"BumpNet max-bin position", loc='center')
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    fig2.colorbar(hb2[3], ax=ax2, label="Entries")

    fout2 = output_path / f"position_correlation.{fmt}"
    fig2.savefig(fout2, bbox_inches='tight')
    if show: plt.show()
    plt.close(fig2)
    print(f"Saved {fout2}")

    # ---------- THIRD FIGURE – 1-D Z_max overlay ---------------------
    fig3, ax3 = plt.subplots(figsize=(10/2.54, 7/2.54), dpi=dpi)
    bins = np.histogram_bin_edges(np.concatenate([zmax_pred, zmax_lr]), bins=80)
    ax3.hist(zmax_pred, bins=bins, histtype='step', color='red',  label='BumpNet')
    ax3.hist(zmax_lr,   bins=bins, histtype='step', color='blue', label=r'$Z_{\mathrm{LR}}$')

    ax3.set_xlabel(r'$Z_{\max}$', loc='right')
    ax3.set_ylabel('Entries', loc='top')
    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    ax3.yaxis.set_minor_locator(AutoMinorLocator())
    ax3.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax3.legend(frameon=False, fontsize=9)

    fout3 = output_path / f"zmax_1d_distribution.{fmt}"
    fig3.savefig(fout3, bbox_inches='tight')
    if show: plt.show()
    plt.close(fig3)
    print(f"Saved {fout3}")

    # ---------- EXTRA FIGURE – 1-D ΔZ histogram ---------------------
    fig_dz, ax_dz = plt.subplots(figsize=(10/2.54, 7/2.54), dpi=dpi)
    ax_dz.hist(deltaZ, bins=80, histtype='step', color='black')

    mu, sigma = np.nanmean(deltaZ), np.nanstd(deltaZ)
    ax_dz.annotate(fr"$\mu = {mu:+.3f}$" "\n" fr"$\sigma = {sigma:.2f}$",
                xy=(0.02, 0.97), xycoords='axes fraction',
                ha='left', va='top', fontsize=9)

    ax_dz.set_xlabel(r"$\Delta Z_{\max}$", loc='right')
    ax_dz.set_ylabel("Entries", loc='top')
    ax_dz.xaxis.set_minor_locator(AutoMinorLocator())
    ax_dz.yaxis.set_minor_locator(AutoMinorLocator())
    ax_dz.tick_params(axis='both', which='both', direction='in', top=True, right=True)

    fout_dz = output_path / f"deltaZmax_1d_distribution.{fmt}"
    fig_dz.savefig(fout_dz, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig_dz)
    print(f"Saved {fout_dz}")

# ------------------------------------------------------------------
#  collect & plot per-bin ΔZ (BumpNet−LR) for the first N PEs
# ------------------------------------------------------------------
def plot_binwise_deltaZ(input_dir,
                        max_pes=20,
                        output_dir=".", *,
                        dpi=300,
                        fmt="png",
                        show=False):
    """
    Build a 2-D histogram of  ΔZ_bin = Z_pred – Z_LR
    versus the *relative bin position*  i / n_bins
    using the first `max_pes` pseudo-experiments only.

    Saved as  <output_dir>/binwise_deltaZ_vs_pos.<fmt>
    """
    import numpy as np, matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.ticker import AutoMinorLocator
    from pathlib import Path

    deltaz, relpos = [], []

    for fpath in find_all_pe_files(input_dir)[:max_pes]:
        with np.load(fpath, allow_pickle=True) as npz:
            z_arr   = npz["z"]
            zlr_arr = npz.get("z_lr")       # may be None → skip
        if zlr_arr is None:
            continue

        # -------- per histogram in this PE --------------------------
        for z_row, zlr_row in zip(z_arr, zlr_arr):
            if zlr_row is None or not np.isfinite(zlr_row).any():
                continue

            n_bins = len(z_row)             # ← full length as saved
            dz  = z_row - zlr_row                # ΔZ for every bin
            pos = np.arange(n_bins) / n_bins     # relative position

            deltaz.extend(dz)
            relpos.extend(pos)

    if not deltaz:
        print("[WARN] No ΔZ entries collected – skipping plot")
        return

    deltaz = np.asarray(deltaz, dtype=float)
    relpos = np.asarray(relpos, dtype=float)

    # ------------------- plotting ----------------------------------
    cmap = colormaps["Spectral_r"]; cmap.set_under("white", alpha=0)
    fig, ax = plt.subplots(figsize=(10/2.54, 7/2.54), dpi=dpi)

    hb = ax.hist2d(relpos, deltaz, bins=(100, 100), cmap=cmap, vmin=1e-3)
    mu, sigma = np.nanmean(deltaz), np.nanstd(deltaz)
    annotation = (
        fr"$\mu = {mu:+.3f}$"      # ← math on
        "\n"                       # ← plain newline
        fr"$\sigma = {sigma:.2f}$" # ← math on again
    )

    ax.annotate(annotation,
                xy=(0.02, 0.97), xycoords='axes fraction',
                ha='left', va='top', fontsize=9)

    ax.set_xlabel("Bin index / n_bins", loc="right")
    ax.set_ylabel(r"$\Delta Z_{\mathrm{bin}}$  (BumpNet – LR)", loc="center")
    ax.axhline(0, color="black", lw=0.6)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    fig.colorbar(hb[3], ax=ax, label="Entries")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fout = out_dir / f"binwise_deltaZ_vs_pos.{fmt}"
    fig.savefig(fout, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved {fout}")

    # ---------- EXTRA FIGURE – 1-D ΔZ histogram ---------------------
    fig_dz, ax_dz = plt.subplots(figsize=(10/2.54, 7/2.54), dpi=dpi)
    ax_dz.hist(deltaz, bins=80, histtype='step', color='black')

    mu, sigma = np.nanmean(deltaz), np.nanstd(deltaz)
    ax_dz.annotate(fr"$\mu = {mu:+.3f}$" "\n" fr"$\sigma = {sigma:.2f}$",
                xy=(0.02, 0.97), xycoords='axes fraction',
                ha='left', va='top', fontsize=9)

    ax_dz.set_xlabel(r"$\Delta Z_{\max}$", loc='right')
    ax_dz.set_ylabel("Entries", loc='top')
    ax_dz.xaxis.set_minor_locator(AutoMinorLocator())
    ax_dz.yaxis.set_minor_locator(AutoMinorLocator())
    ax_dz.tick_params(axis='both', which='both', direction='in', top=True, right=True)

    fout_dz = out_dir / f"deltaZ_1d_distribution.{fmt}"
    fig_dz.savefig(fout_dz, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig_dz)
    print(f"Saved {fout_dz}")

def plot_deltamaxZmax_vs_position(delta_maxzmax, rel_pos, output_path, dpi=300, fmt='png', show=False):
    """
    Plots two figures:
      • deltamaxZmax_vs_position.<fmt>     – ΔZ (max pred − max LR in same hist) vs position of LR Zmax in that histogram
      • deltamaxZmax_1d_distribution.<fmt> – 1-D histogram of the same quantity
    """
    import numpy as np, matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator
    from matplotlib import colormaps
    from seaborn import kdeplot
    from pathlib import Path

    delta_maxzmax = np.asarray(delta_maxzmax, dtype=float)
    rel_pos = np.asarray(rel_pos, dtype=float)

    good = np.isfinite(delta_maxzmax) & np.isfinite(rel_pos)
    if not good.any():
        print("[WARN] No finite entries for ΔZ_maxZmax plot")
        return

    delta_maxzmax = delta_maxzmax[good]
    rel_pos = rel_pos[good]

    cmap = colormaps["Spectral_r"]
    cmap.set_under("white", alpha=0)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. 2D histogram: ΔmaxZmax vs LR rel. bin pos
    fig1, ax1 = plt.subplots(figsize=(10/2.54, 7/2.54), dpi=dpi)
    hb = ax1.hist2d(rel_pos, delta_maxzmax, bins=(100, 100), cmap=cmap, vmin=1e-3)
    kdeplot(x=rel_pos, y=delta_maxzmax, ax=ax1, fill=False, cmap="Spectral_r")

    mu, sigma = np.nanmean(delta_maxzmax), np.nanstd(delta_maxzmax)
    ax1.annotate(fr"$\mu = {mu:+.3f}$" "\n" fr"$\sigma = {sigma:.2f}$",
                 xy=(0.02, 0.97), xycoords='axes fraction',
                 ha='left', va='top', fontsize=9)
    ax1.set_xlabel(r"$Z^{\mathrm{LR}}_{\max}$ bin / number of bins", loc="right")
    ax1.set_ylabel(r"$\Delta Z^{\mathrm{maxZmax}}_{\max}$", loc="center")
    ax1.axhline(0, color="black", lw=0.6)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    fig1.colorbar(hb[3], ax=ax1, label="Entries")

    fout1 = output_path / f"deltamaxZmax_vs_position.{fmt}"
    fig1.savefig(fout1, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig1)
    print(f"Saved {fout1}")

    # 2. 1D histogram
    fig2, ax2 = plt.subplots(figsize=(10/2.54, 7/2.54), dpi=dpi)
    ax2.hist(delta_maxzmax, bins=80, histtype='step', color='black')
    ax2.annotate(fr"$\mu = {mu:+.3f}$" "\n" fr"$\sigma = {sigma:.2f}$",
                 xy=(0.02, 0.97), xycoords='axes fraction',
                 ha='left', va='top', fontsize=9)

    ax2.set_xlabel(r"$\Delta Z^{\mathrm{maxZmax}}_{\max}$", loc="right")
    ax2.set_ylabel("Entries", loc="top")
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.tick_params(axis='both', which='both', direction='in', top=True, right=True)

    fout2 = output_path / f"deltamaxZmax_1d_distribution.{fmt}"
    fig2.savefig(fout2, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig2)
    print(f"Saved {fout2}")


def plot_toys_bkg_only(config):
    input_dir = config['input_dir']
    input_dir_hist = config.get('input_dir_hist')
    output_dir = config['output_dir']
    # ------------------------------------------------------------------
    # wipe previous PNGs so each run writes a fresh set
    # ------------------------------------------------------------------
    _ensure_empty_dir(Path(output_dir) / "coincident_matches")
    _ensure_empty_dir(Path(output_dir) / "top10_maxZmax_histos")
    dpi = config.get('dpi', 300)
    fmt = config.get('format', 'png')
    show = config.get('show', False)
    edge = config.get('edge', [0.0, 1.0])
    max_pes = config.get('max_pes', None)
    min_nbins = config.get('min_nbins', 1)

    smooth_lib = _get_smooth_library(config.get('smooth_dir'))

    # Run main Zmax analysis
    if config.get('do_zmax_analysis', True):
        print(">>>>> Running Zmax analysis and plotting...")
        (z_pred, z_lr, examples,
        relpos_pred, relpos_lr,
        hist_names,   
        zmax_pred_values, zmax_lr_values,
        relpos_pred_pe, relpos_lr_pe, 
        all_lens,
        eff_pred, eff_lr,
        first_occ, deltaZ_same_bin,
        delta_maxzmax_values) = load_zmax_per_pe(
            input_dir, input_dir_hist, edge, max_pes, min_nbins
        )
        plot_max_zmax_distribution(zmax_pred_values, zmax_lr_values, output_dir, dpi, fmt, show)
        plot_maxzmax_position_distribution(relpos_pred_pe, relpos_lr_pe, output_dir, dpi, fmt, show)
        plot_histogram_length_distribution(all_lens, eff_pred, eff_lr, output_dir, dpi, fmt, show)
        plot_deltaZmax_and_positions(z_pred, z_lr,
                                relpos_pred, relpos_lr,
                                output_dir,
                                deltaZ_same_bin=deltaZ_same_bin,
                                use_predbin_lr=config.get("use_predbin_lr", False),
                                dpi=dpi, fmt=fmt, show=show)
        plot_top_examples(examples, output_dir, smooth_lib=smooth_lib, left_frac=edge[0])

        name_counts = Counter(hist_names)
        top_10_names = [name for name, _ in name_counts.most_common(10)]
        top_examples_dict = {
            name: first_occ[name]
            for name in top_10_names if name in first_occ
        }
        plot_top_maxzmax_histograms(top_examples_dict, output_dir, dpi, fmt, smooth_lib=smooth_lib, left_frac=edge[0])

        # plot_binwise_deltaZ(
        #     input_dir,
        #     max_pes=20,
        #     output_dir=output_dir,
        #     dpi=dpi,
        #     fmt=fmt,
        #     show=show
        # )    
        plot_deltamaxZmax_vs_position(delta_maxzmax_values,
                              relpos_pred_pe,
                              output_path=output_dir,
                              dpi=dpi,
                              fmt=fmt,
                              show=show)

        if config.get("do_top10_analysis", False):
            print(">>>>> Running Top-10 maxZmax histogram analysis...")
            analyze_top10_worst_offenders(
                input_dir=input_dir,
                input_dir_hist=input_dir_hist,
                output_dir=output_dir,
                edge=edge,
                min_nbins=min_nbins,
                dpi=dpi,
                fmt=fmt,
                show=show
            )

    # Optional: run coincidence check
    run_coincidence = config.get('check_coincident_mass_bumps', False)
    coincidence_threshold = config.get('coincidence_threshold', 5.0)
    if run_coincidence:
        print(f"\n>>>>> Running coincidence check (Z threshold = {coincidence_threshold})...")
        find_coincident_mass_bumps(
            input_dir=input_dir,
            input_dir_hist=input_dir_hist,
            threshold=coincidence_threshold,
            edge=edge,
            min_nbins=min_nbins,
            max_pes=max_pes,
            output_dir=output_dir,
            smooth_lib=smooth_lib
        )


if __name__ == '__main__':
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to JSON config file')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    plot_toys_bkg_only(config['plot_toys_bkg_only'])
