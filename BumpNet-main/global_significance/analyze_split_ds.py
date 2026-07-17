# global_significance/analyze_split_ds.py
# ================================================================
# Split-DS analysis:
#   • Pair PE_XXXXXXX_A.npz with PE_XXXXXXX_B.npz per shard
#   • In A: find (histo, bin) of max Z_pred (masked, positive)
#   • In B: read Z_pred and Z_LR at that exact (histo, bin)
#   • Plots:
#       - A:   maxZmax distributions (Z_pred & Z_LR), positions
#       - B@:  overlay of Z_pred(B@A) & Z_LR(B@A)
#       - A/B: ΔZmax vs relative position, separately
#       - Examples: 20 seeded PEs, A & B side-by-side, selected bin marked
#   • All A↔B mapping is asserted (fail fast if mismatch)
# ================================================================

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import re, shutil, random
import math

# --- consistent colors ---
COL_PRED  = "red"    # Z_pred
COL_LR    = "blue"   # Z_LR
COL_DELTA = "green"  # Delta Z

# ---------------- helpers ----------------
_slug = lambda s: re.sub(r'[^A-Za-z0-9._-]+', '_', str(s))

def _robust_gaussian_fit(vals, clip_sigma=3.0):
    """Return (mu, sigma) using a robust core (MAD 3σ-clip)."""
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan, np.nan
    if a.size < 8:
        # Fallback to plain sample stats
        mu = float(np.mean(a))
        sig = float(np.std(a, ddof=1)) if a.size > 1 else np.nan
        return mu, sig

    med = float(np.median(a))
    mad = 1.4826 * float(np.median(np.abs(a - med)))  # ≈ robust σ
    if not np.isfinite(mad) or mad <= 0:
        core = a
    else:
        core = a[(a >= med - clip_sigma * mad) & (a <= med + clip_sigma * mad)]
        if core.size < 8:
            core = a

    mu  = float(np.mean(core))
    sig = float(np.std(core, ddof=1)) if core.size > 1 else np.nan
    # avoid zero/NaN for plotting
    if not np.isfinite(sig) or sig <= 0:
        sig = float(np.std(a, ddof=1)) if a.size > 1 else np.nan
    return mu, sig

def _fit_gaussian_with_uncertainties(vals):
    """
    Wrapper that reimplements the same robust clipping logic to recover n_core.
    Returns (mu, sigma, mu_err, sigma_err, n_core).
    """
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, 0

    # --- reproduce the clipping from _robust_gaussian_fit to get core & n_core ---
    if n < 8:
        mu  = float(np.mean(a))
        sig = float(np.std(a, ddof=1)) if n > 1 else np.nan
        n_core = n
    else:
        med = float(np.median(a))
        mad = 1.4826 * float(np.median(np.abs(a - med)))
        if not np.isfinite(mad) or mad <= 0:
            core = a
        else:
            core = a[(a >= med - 3.0 * mad) & (a <= med + 3.0 * mad)]
            if core.size < 8:
                core = a
        n_core = core.size
        mu  = float(np.mean(core))
        sig = float(np.std(core, ddof=1)) if n_core > 1 else np.nan
        if not np.isfinite(sig) or sig <= 0:
            sig = float(np.std(a, ddof=1)) if n > 1 else np.nan
            n_core = n

    if not np.isfinite(sig) or sig <= 0 or n_core <= 1:
        return mu, sig, np.nan, np.nan, n_core

    mu_err    = sig / np.sqrt(n_core)
    sigma_err = sig / np.sqrt(2.0 * n_core)
    return mu, sig, mu_err, sigma_err, n_core

def _ensure_empty_dir(path: Path):
    path = Path(path)
    if path.exists():
        for child in path.iterdir():
            try:
                child.unlink()
            except IsADirectoryError:
                shutil.rmtree(child)
    path.mkdir(parents=True, exist_ok=True)

def _draw_left_cut(ax, edges, left_frac, color="grey", alpha=0.15):
    n_bins = len(edges) - 1
    cut_idx = max(0, int(left_frac * n_bins + .5))
    cut_x = edges[cut_idx]
    ax.axvspan(edges[0], cut_x, color=color, alpha=alpha, zorder=0)

# If a separate hist repo exists (like your previous flow), allow pairing.
def _find_all_pe_files(root_dir: Path):
    return sorted(Path(root_dir).rglob("PE_*_[AB].npz"))

def _pe_id_side(fname: str):
    # e.g., "PE_0000123_A.npz" → ("0000123", "A")
    base = Path(fname).name
    m = re.match(r"PE_(\d+)_([AB])\.npz$", base)
    if not m:
        raise ValueError(f"Unexpected PE filename: {fname}")
    return m.group(1), m.group(2)

def _pair_pe_files_in_dir(shard_dir: Path):
    # returns dict pe_id -> {"A": Path, "B": Path}
    pairs = defaultdict(dict)
    for p in _find_all_pe_files(shard_dir):
        peid, side = _pe_id_side(p.name)
        pairs[peid][side] = p
    # Keep only those with both sides present
    complete = {k: v for k, v in pairs.items() if "A" in v and "B" in v}
    return complete

def _load_npz(path: Path):
    # read-and-close to avoid leaking file descriptors
    with np.load(path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}

def _get_names_array(names_entry):
    # names is sometimes array of strings or array of singleton arrays
    if isinstance(names_entry, str):
        return names_entry
    try:
        return names_entry[0]
    except Exception:
        return str(names_entry)

def _nbins_from_edges(edges_all, j, z_row):
    # Prefer explicit bin_edges; fall back to Z length
    if edges_all is None:
        return len(z_row)
    if getattr(edges_all, "dtype", None) == object or np.ndim(edges_all) > 1:
        # ragged per-hist array
        return len(edges_all[j]) - 1
    # common 1D edges array
    return len(edges_all) - 1

def _edge_slice(eff_n: int, edge: tuple[float, float]) -> tuple[int, int]:
    sb = max(0, int(edge[0] * eff_n + 0.5))
    lb = max(sb + 1, min(eff_n, int(edge[1] * eff_n + 0.5)))
    return sb, lb

def _list_shard_dirs(root: Path):
    """
    Return all shard subdirectories under `root`. If PE files live directly
    under `root`, include `root` as well.
    """
    shards = [p for p in root.iterdir() if p.is_dir()]
    # Heuristic: only keep those that actually contain PE files (or have subdirs that do)
    good = []
    for s in shards:
        if any(s.rglob("PE_*_[AB].npz")):
            good.append(s)
    # Also include root itself if it directly contains PE files
    if any(root.glob("PE_*_[AB].npz")):
        good.append(root)
    # Deduplicate and sort
    return sorted(set(good), key=lambda p: p.as_posix())

def _hist_file_for_prediction(pe_path: Path, pred_root: Path, hist_root: Path) -> Path:
    rel = pe_path.relative_to(pred_root)              # shard_xxxx/PE_...
    # If hist_root is already the shard dir, use it; else append the shard.
    d = hist_root if (hist_root.name == rel.parent.name) else (hist_root / rel.parent)

    base = rel.name
    base_no_side = base.replace("_A", "").replace("_B", "")

    cand_names = [
        base,                                   # identical name
        base.replace("_A.npz", ".npz").replace("_B.npz", ".npz"),  # side-stripped (.npz)
        base.replace(".pz.npz", ".npz"),        # legacy .pz.npz → .npz
        # ↓ combined legacy + side‐stripped (missing before)
        base_no_side.replace(".pz.npz", ".npz"),
    ]

    for nm in cand_names:
        p = d / nm
        if p.exists():
            return p

    tried = [str(d / nm) for nm in cand_names]
    raise FileNotFoundError(
        "Could not locate histogram NPZ for prediction file:\n"
        f"  pred: {pe_path}\n"
        "  tried:\n    " + "\n    ".join(tried)
    )

# ---------------- core analysis ----------------
def analyze_split_ds(config: dict):
    input_dir = Path(config["input_dir"])
    input_dir_hist = Path(config["input_dir_hist"]) if config.get("input_dir_hist") else None
    outdir = Path(config["output_dir"])
    dpi = config.get("dpi", 300)
    fmt = config.get("format", "png")
    edge = tuple(config.get("edge", [0.0, 1.0]))
    min_nbins = config.get("min_nbins", 1)
    n_examples = int(config.get("examples", 20))
    seed = int(config.get("seed", 42))
    max_pes = config.get("max_pes", None)
    if max_pes is not None:
        max_pes = int(max_pes)
    # plotting options (only log/linear comes from YAML)
    tp   = config.get("triplet_plots", {}) or {}
    logy = bool(tp.get("logy", False))

    random.seed(seed)
    np.random.seed(seed)

    # Clean a few subfolders up-front to mirror your hygiene
    _ensure_empty_dir(outdir / "examples")

    # Accumulators
    # A (one entry per PE)
    zmaxA_pred_vals, zmaxA_lr_vals = [], []
    relposA_pred_vals, relposA_lr_vals = [], []
    eff_lenA_pred, eff_lenA_lr = [], []

    # B-at-A-bin (one entry per PE)
    B_atA_pred_vals, B_atA_lr_vals = [], []

    # ΔZmax vs position (ALL histograms across all PEs), separately for A and B
    deltaZmax_vs_pos_A = []  # tuples (rel_pos_lr, deltaZmax)
    deltaZmax_vs_pos_B = []

    # Example material: list of tuples
    # (edgesA, histA, zA_pred, zA_lr, edgesB, histB, zB_pred, zB_lr, name, peid, bin_idx_A)
    examples = []

    # ---- Pooled A∪B, all bins (no A/B distinction) ----
    # Collect as NumPy chunks to avoid Python floats / GC thrash, concatenate once at the end
    all_pred_chunks = []     # list[np.ndarray(float32)]
    all_lr_chunks = []       # list[np.ndarray(float32)]
    all_relpos_chunks = []   # list[np.ndarray(float32)]
    all_nevt_chunks = []     # list[np.ndarray, int32] — per-bin counts aligned with Z arrays

    # Gather pairs across all input dirs
    # Discover shards under the input root(s)
    shard_dirs = _list_shard_dirs(input_dir)
    if not shard_dirs:
        raise RuntimeError(f"No shards with PE files found under {input_dir}")

    # Optional hist root: map by shard directory name (exact match)
    shard_hist_map = {}
    if input_dir_hist is not None:
        hist_dirs = list(input_dir_hist.iterdir())
        hist_shards = {p.name: p for p in hist_dirs if p.is_dir()}

        for s in shard_dirs:
            # exact shard subdir match
            m = hist_shards.get(s.name)
            if m is not None:
                shard_hist_map[s] = m
                continue
            # flat layout fallback: if PE files are directly in s, point to input_dir_hist itself
            if any(s.glob("PE_*_[AB].npz")) and any(input_dir_hist.glob("PE_*.npz")):
                shard_hist_map[s] = input_dir_hist
            else:
                shard_hist_map[s] = None
    else:
        for s in shard_dirs:
            shard_hist_map[s] = None

    # Pair A/B per shard
    all_pairs = []
    for s in shard_dirs:
        pairs = _pair_pe_files_in_dir(s)
        all_pairs.append((s, pairs))
    total_pes = sum(len(p) for _, p in all_pairs)
    if total_pes == 0:
        raise RuntimeError("No complete A/B PE pairs were found under the provided input_dir.")

    # Prepare a deterministic subset of PEs for examples
    # We'll fill as we go until we hit n_examples.
    ex_left = n_examples

    # Process shards
    cap = min(total_pes, max_pes) if max_pes is not None else total_pes
    pbar = tqdm(total=cap, desc="Analyzing split-DS PEs")

    seen = 0
    for shard_dir, pairs in all_pairs:
        if max_pes is not None and seen >= max_pes:
            break
        hist_dir = shard_hist_map.get(shard_dir)

        for peid, sides in pairs.items():
            if max_pes is not None and seen >= max_pes:
                break
            seen += 1
            pbar.update(1)
            pathA, pathB = sides["A"], sides["B"]
            assert pathA.exists() and pathB.exists(), f"Missing A or B for PE {peid} in {shard_dir}"

            npzA = _load_npz(pathA)
            npzB = _load_npz(pathB)

            # Required fields
            zA    = npzA["z"]
            zA_lr = npzA.get("z_lr")
            namesA = npzA["names"]
            edgesA_all = npzA.get("bin_edges")  # may be common or ragged
            zB    = npzB["z"]
            zB_lr = npzB.get("z_lr")
            namesB = npzB["names"]
            edgesB_all = npzB.get("bin_edges")

            # Assert same number of histograms & identity mapping by name
            assert len(zA) == len(zB), f"PE {peid}: A and B have different #histos ({len(zA)} vs {len(zB)})"
            assert len(namesA) == len(namesB) == len(zA), f"PE {peid}: names length mismatch"

            # --- HIST and bin_edges: always load from input_dir_hist via shard map ---
            HA = HB = None
            edgesA_all = edgesB_all = None

            assert hist_dir is not None, (
                f"PE {peid}: histogram shard directory not found for {shard_dir.name} "
                f"under input_dir_hist. Make sure {input_dir_hist} contains {shard_dir.name}"
            )

            # A-side hist npz
            hfileA = _hist_file_for_prediction(pathA, input_dir, hist_dir)
            with np.load(hfileA, allow_pickle=True) as hnpzA:
                assert "HIST" in hnpzA, f"{hfileA} missing 'HIST' array"
                HA = hnpzA["HIST"]
                edgesA_all = hnpzA.get("bin_edges")   # may be ragged or common

            # B-side hist npz
            hfileB = _hist_file_for_prediction(pathB, input_dir, hist_dir)
            with np.load(hfileB, allow_pickle=True) as hnpzB:
                assert "HIST" in hnpzB, f"{hfileB} missing 'HIST' array"
                HB = hnpzB["HIST"]
                edgesB_all = hnpzB.get("bin_edges")   # may be ragged or common

            # ---------- per-hist A/B ΔZmax vs pos (all histos) ----------
            for j in range(len(zA)):
                nameA = _get_names_array(namesA[j]); nameB = _get_names_array(namesB[j])
                assert nameA == nameB, f"PE {peid}: histogram name mismatch at idx {j}: {nameA} vs {nameB}"

                # Effective lengths
                if HA is not None:
                    effA = _nbins_from_edges(edgesA_all, j, zA[j])
                else:
                    effA = len(zA[j])
                if HB is not None:
                    effB = _nbins_from_edges(edgesB_all, j, zB[j])
                else:
                    effB = len(zB[j])

                if effA < min_nbins or effB < min_nbins:
                    continue

                # Apply edge window on each side independently
                sbA, lbA = _edge_slice(effA, edge)
                sbB, lbB = _edge_slice(effB, edge)
                if sbA >= lbA or sbB >= lbB:
                    continue

                # Z arrays trimmed to effective lengths
                zA_t    = np.asarray(zA[j][:effA], dtype=float)
                zB_t    = np.asarray(zB[j][:effB], dtype=float)
                zA_lr_t = np.asarray(zA_lr[j][:effA], dtype=float) if zA_lr is not None else None
                zB_lr_t = np.asarray(zB_lr[j][:effB], dtype=float) if zB_lr is not None else None

                # ---- Pool ALL bins (A and B) in the analysis window ----
                # A side
                if (lbA - sbA) > 0:
                    relpos_A = np.arange(sbA, lbA, dtype=float) / float(effA)
                    zpred_A  = zA_t[sbA:lbA]
                    zlr_A    = zA_lr_t[sbA:lbA] if zA_lr_t is not None else np.full_like(zpred_A, np.nan)
                    mA = np.isfinite(zpred_A) & np.isfinite(zlr_A) & np.isfinite(relpos_A)
                    if mA.any():
                        all_pred_chunks.append(   zpred_A[mA].astype(np.float32, copy=True))
                        all_lr_chunks.append(       zlr_A[mA].astype(np.float32, copy=True))
                        all_relpos_chunks.append(relpos_A[mA].astype(np.float32, copy=True))

                # B side
                if (lbB - sbB) > 0:
                    relpos_B = np.arange(sbB, lbB, dtype=float) / float(effB)
                    zpred_B  = zB_t[sbB:lbB]
                    zlr_B    = zB_lr_t[sbB:lbB] if zB_lr_t is not None else np.full_like(zpred_B, np.nan)
                    mB = np.isfinite(zpred_B) & np.isfinite(zlr_B) & np.isfinite(relpos_B)
                    if mB.any():
                        all_pred_chunks.append(   zpred_B[mB].astype(np.float32, copy=True))
                        all_lr_chunks.append(       zlr_B[mB].astype(np.float32, copy=True))
                        all_relpos_chunks.append(relpos_B[mB].astype(np.float32, copy=True))

                # per-bin event counts for A (must align 1-1 with zpred_A, zlr_A, relpos_A)
                nevents_edges = tp.get("nevents_edges", tp.get("nevents_edges"))
                want_nevents = nevents_edges is not None
                if want_nevents:
                    if HA is None:
                        raise RuntimeError("HIST arrays not available; cannot bin by Nevents.")
                    nevt_A = np.asarray(HA[j][:effA], dtype=np.int64)[sbA:lbA]
                    mA = mA & np.isfinite(nevt_A)   # keep alignment; nevt_A are integers but this is safe
                    if mA.any():
                        all_nevt_chunks.append(nevt_A[mA].astype(np.int32, copy=True))

                    nevt_B = np.asarray(HB[j][:effB], dtype=np.int64)[sbB:lbB]
                    mB = mB & np.isfinite(nevt_B)
                    if mB.any():
                        all_nevt_chunks.append(nevt_B[mB].astype(np.int32, copy=True))

                # ---- ΔZmax in A (per histogram) ----
                winA = zA_t[sbA:lbA]
                if winA.size and zA_lr_t is not None and (lbA - sbA) > 0:
                    zmaxA_pred = float(np.max(winA))
                    zlr_winA   = zA_lr_t[sbA:lbA]
                    zmaxA_lr   = float(np.max(zlr_winA))
                    relpos_lrA = (sbA + int(np.argmax(zlr_winA))) / effA
                    deltaA     = zmaxA_pred - zmaxA_lr
                    if np.isfinite(deltaA) and np.isfinite(relpos_lrA):
                        deltaZmax_vs_pos_A.append((relpos_lrA, deltaA))

                # ---- ΔZmax in B (per histogram) ----
                winB = zB_t[sbB:lbB]
                if winB.size and zB_lr_t is not None and (lbB - sbB) > 0:
                    zmaxB_pred = float(np.max(winB))
                    zlr_winB   = zB_lr_t[sbB:lbB]
                    zmaxB_lr   = float(np.max(zlr_winB))
                    relpos_lrB = (sbB + int(np.argmax(zlr_winB))) / effB
                    deltaB     = zmaxB_pred - zmaxB_lr
                    if np.isfinite(deltaB) and np.isfinite(relpos_lrB):
                        deltaZmax_vs_pos_B.append((relpos_lrB, deltaB))


            # ---------- PE-level: pick the A star by Z_pred -------------
            # Search only positive Z within masked region
            best_val, best_idx, best_eff = -np.inf, None, None
            best_bin = None

            zmax_lr_for_PE = -np.inf
            best_lr_idx, best_lr_eff = None, None
            best_lr_bin = None

            for j in range(len(zA)):
                # effective length via HIST if available, else Z length
                if HA is not None:
                    eff = _nbins_from_edges(edgesA_all, j, zA[j])
                else:
                    eff = len(zA[j])
                if eff < min_nbins:
                    continue
                sb, lb = _edge_slice(eff, edge)
                if sb >= lb:
                    continue

                zpred_t = np.asarray(zA[j][:eff], dtype=float)
                zpred_win = zpred_t[sb:lb]
                if zpred_win.size == 0:
                    continue
                # positive only
                local_max = np.max(zpred_win)
                if local_max <= 0:
                    continue
                arg_loc = int(np.argmax(zpred_win))
                # leftmost tie behavior is automatic via np.argmax
                # keep PE-level best
                if local_max > best_val:
                    best_val = float(local_max)
                    best_idx = j
                    best_eff = eff
                    best_bin = sb + arg_loc

                # record LR PE-level maximum as well (for A overlays)
                if zA_lr is not None:
                    zlr_t = np.asarray(zA_lr[j][:eff], dtype=float)
                    zlr_win = zlr_t[sb:lb]
                    if zlr_win.size:
                        lr_local = float(np.max(zlr_win))
                        if np.isfinite(lr_local) and lr_local > zmax_lr_for_PE:
                            zmax_lr_for_PE = lr_local
                            best_lr_idx = j
                            best_lr_eff = eff
                            best_lr_bin = sb + int(np.argmax(zlr_win))

            # If no positive Z_pred in A, skip this PE entirely for PE-level plots
            if best_idx is None:
                continue

            # A PE-level overlays (values and relative positions)
            nameA_star = _get_names_array(namesA[best_idx])

            # Raw BumpNet Z on A-star histogram
            zA_star_raw = np.asarray(zA[best_idx][:best_eff], dtype=float)

            # PE-level accumulators (A side)
            relposA_pred_vals.append(best_bin / best_eff)
            zmaxA_pred_vals.append(best_val)  # uses adjusted if enabled
            eff_lenA_pred.append(best_eff)

            # LR side (unchanged)
            if zA_lr is not None:
                if np.isfinite(zmax_lr_for_PE):
                    zmaxA_lr_vals.append(zmax_lr_for_PE)
                else:
                    zmaxA_lr_vals.append(np.nan)
                if (best_lr_idx is not None) and (best_lr_eff is not None):
                    relposA_lr_vals.append(best_lr_bin / best_lr_eff)
                    eff_lenA_lr.append(best_lr_eff)
                else:
                    relposA_lr_vals.append(np.nan); eff_lenA_lr.append(np.nan)
            else:
                zmaxA_lr_vals.append(np.nan)
                relposA_lr_vals.append(np.nan)
                eff_lenA_lr.append(np.nan)

            # ---------- B at A-selected (histo, bin) ----------
            nameB_star = _get_names_array(namesB[best_idx])
            assert nameA_star == nameB_star, f"PE {peid}: star hist name mismatch: {nameA_star} vs {nameB_star}"

            # Effective length in B for that histogram
            if HB is not None:
                effB_star = _nbins_from_edges(edgesB_all, best_idx, zB[best_idx])
            else:
                effB_star = len(zB[best_idx])
            assert best_bin < effB_star, f"PE {peid} hist {nameA_star}: A-selected bin {best_bin} ≥ effB {effB_star}"

            zB_star_raw  = np.asarray(zB[best_idx][:effB_star], dtype=float)
            val_pred_B_atA = float(zB_star_raw[best_bin])
            B_atA_pred_vals.append(val_pred_B_atA)

            # LR value at same B@A bin (unchanged)
            if zB_lr is not None:
                zB_star_lr_t = np.asarray(zB_lr[best_idx][:effB_star], dtype=float)
                val_lr_B_atA = float(zB_star_lr_t[best_bin])
                B_atA_lr_vals.append(val_lr_B_atA)
            else:
                B_atA_lr_vals.append(np.nan)

            # ---------- prepare one example figure if we still want more
            if ex_left > 0:
                # Build bin edges and hist arrays for A and B if available
                def _edges_for(npz_edges, j, eff):
                    if npz_edges is None:
                        return np.arange(eff + 1, dtype=float)
                    # npz_edges can be ragged (object) or a common 1-D array
                    if getattr(npz_edges, "dtype", None) is not None and npz_edges.dtype == object:
                        return np.asarray(npz_edges[j][:eff+1], dtype=float)
                    elif np.ndim(npz_edges) > 1:
                        return np.asarray(npz_edges[j][:eff+1], dtype=float)
                    else:
                        return np.asarray(npz_edges[:eff+1], dtype=float)

                edgesA = _edges_for(edgesA_all, best_idx, best_eff)
                edgesB = _edges_for(edgesB_all, best_idx, effB_star)

                # pull HIST if present
                histA = HA[best_idx][:best_eff] if HA is not None else None
                histB = HB[best_idx][:effB_star] if HB is not None else None

                zA_lr_star = (np.asarray(zA_lr[best_idx][:best_eff], dtype=float)
                              if zA_lr is not None else None)
                zB_lr_star = (np.asarray(zB_lr[best_idx][:effB_star], dtype=float)
                              if zB_lr is not None else None)

                examples.append((
                    edgesA, histA, zA_star_raw, zA_lr_star,
                    edgesB, histB, zB_star_raw, zB_lr_star,
                    nameA_star, peid, best_bin))
                ex_left -= 1

    pbar.close()

    # ---------------- plotting outputs ----------------
    outdir.mkdir(parents=True, exist_ok=True)

    # A: maxZmax distribution overlay
    _plot_maxZmax_distribution_A(zmaxA_pred_vals, zmaxA_lr_vals, outdir, dpi, fmt)
    # A: positions overlay
    _plot_maxZmax_positions_A(relposA_pred_vals, relposA_lr_vals, eff_lenA_pred, eff_lenA_lr, outdir, dpi, fmt)
    # B-at-A-bin overlay
    _plot_B_atA_bin_overlay(B_atA_pred_vals, B_atA_lr_vals, outdir, dpi, fmt)
    # ΔZmax vs pos for A and B separately
    # ΔZmax vs pos (A & B) — single flag
    if bool(config.get("plot_deltaZmax", True)):
        _plot_deltaZmax_vs_pos(deltaZmax_vs_pos_A, outdir / f"deltaZmax_vs_position_A.{fmt}", dpi)
        _plot_deltaZmax_vs_pos(deltaZmax_vs_pos_B, outdir / f"deltaZmax_vs_position_B.{fmt}", dpi)

    # Final pooled arrays (compact, single contiguous blocks)
    all_pred_vals   = (np.concatenate(all_pred_chunks).astype(np.float32, copy=False)
                    if all_pred_chunks else np.array([], dtype=np.float32))
    all_lr_vals     = (np.concatenate(all_lr_chunks).astype(np.float32, copy=False)
                    if all_lr_chunks else np.array([], dtype=np.float32))
    all_relpos_vals = (np.concatenate(all_relpos_chunks).astype(np.float32, copy=False)
                    if all_relpos_chunks else np.array([], dtype=np.float32))
    all_nevt_vals   = (np.concatenate(all_nevt_chunks).astype(np.int32,   copy=False)
                    if all_nevt_chunks else np.array([], dtype=np.int32))
    
    # --- Pooled 1D plots (all bins, A∪B) with fits ---
    #_plot_significance_1d_all(all_pred_vals, all_lr_vals, outdir, dpi, fmt)

    # --- ΔZ per relative-position slice (edges can come from JSON if provided) ---
    relpos_edges = tp.get("relpos_edges", None)
    nevents_edges = tp.get("nevents_edges", None)
    plot_relpos  = bool(tp.get("plot_relpos_triplets", True))
    plot_nevents = bool(tp.get("plot_nevents_triplets", True))
    if plot_relpos:
        _plot_relpos_triplets(all_pred_vals, all_lr_vals, all_relpos_vals, relpos_edges, outdir, dpi, fmt, logy=logy)
    if plot_nevents:
        _plot_triplets_by_nevents(all_pred_vals, all_lr_vals, all_nevt_vals, nevents_edges, outdir, dpi, fmt, logy=logy)

    # Examples
    _plot_examples_AB(examples, outdir / "examples", dpi, left_frac=edge[0])


def _one_sided_tail_p(sigma: float) -> float:
    return 0.5 * math.erfc(sigma / math.sqrt(2.0))

def _tail_threshold_from_empirical(vals, sigma=3.0):
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    q = 1.0 - _one_sided_tail_p(sigma)
    q = min(max(q, 0.0), 1.0)
    return float(np.quantile(a, q))

def _plot_maxZmax_distribution_A(zmax_pred, zmax_lr, outdir: Path, dpi=300, fmt="png"):
    zmax_pred = np.asarray(zmax_pred, dtype=float)
    zmax_lr   = np.asarray(zmax_lr,   dtype=float)

    fig, ax = plt.subplots(figsize=(10/2.54, 10/2.54), dpi=dpi)
    # Build bins from finite-only values (avoid -999 sentinel widening the range)
    finite_combo = np.concatenate([
        zmax_pred[np.isfinite(zmax_pred)],
        zmax_lr[np.isfinite(zmax_lr)]
    ])
    if finite_combo.size == 0:
        print("[WARN] No finite entries to build bins in _plot_maxZmax_distribution_A")
        return
    bins = np.histogram_bin_edges(finite_combo, bins=100)

    ax.hist(zmax_pred[np.isfinite(zmax_pred)], bins=bins, histtype='step',
            color='red',  label='BumpNet')
    ax.hist(zmax_lr[np.isfinite(zmax_lr)],     bins=bins, histtype='step',
            color='blue', label=r'$Z_{\mathrm{LR}}$')

    # Compute means and empirical 3σ-global thresholds
    mu_pred = float(np.nanmean(zmax_pred)) if np.isfinite(zmax_pred).any() else np.nan
    mu_lr   = float(np.nanmean(zmax_lr))   if np.isfinite(zmax_lr).any()   else np.nan
    thr_pred = _tail_threshold_from_empirical(zmax_pred, sigma=3.0)
    thr_lr   = _tail_threshold_from_empirical(zmax_lr,   sigma=3.0)

    # Draw threshold lines
    ytop = ax.get_ylim()[1]
    if thr_pred is not None:
        ax.axvline(thr_pred, color='red', ls='--')
    if thr_lr is not None:
        ax.axvline(thr_lr, color='blue', ls='--')

    # Text blocks (top-left), with arrows pointing to the dashed lines
    y0 = 0.95; dy = 0.18  # spacing in axes fraction
    if np.isfinite(mu_pred):
        txt_pred = (r'BumpNet  $(\langle \max(Z_{\max}^{\mathrm{pred}})\rangle) = %.2f\,\sigma$' % mu_pred
                    + (('\n$Z_{\mathrm{global}}^{\mathrm{pred}} > 3\sigma \Rightarrow '
                        r'\max(Z_{\max}^{\mathrm{pred}}) > %.2f\,\sigma$') % thr_pred if thr_pred is not None else ''))
        t1 = ax.text(0.05, y0, txt_pred, transform=ax.transAxes, ha='left', va='top',
                     color='red', fontsize=9,
                     bbox=dict(facecolor='white', edgecolor='0.8', alpha=0.9, pad=3))
        if thr_pred is not None:
            ax.annotate('', xy=(thr_pred, 0.90*ytop), xytext=(thr_pred, 0.96*ytop),
                        arrowprops=dict(arrowstyle='-|>', lw=1.0, color='red'))
    if np.isfinite(mu_lr):
        txt_lr = (r'$Z_{\mathrm{LR}}$  $(\langle \max(Z_{\max}^{\mathrm{LR}})\rangle) = %.2f\,\sigma$' % mu_lr
                  + (('\n$Z_{\mathrm{global}}^{\mathrm{LR}} > 3\sigma \Rightarrow '
                      r'\max(Z_{\max}^{\mathrm{LR}}) > %.2f\,\sigma$') % thr_lr if thr_lr is not None else ''))
        t2 = ax.text(0.05, y0 - dy, txt_lr, transform=ax.transAxes, ha='left', va='top',
                     color='blue', fontsize=9,
                     bbox=dict(facecolor='white', edgecolor='0.8', alpha=0.9, pad=3))
        if thr_lr is not None:
            ax.annotate('', xy=(thr_lr, 0.60*ytop), xytext=(thr_lr, 0.72*ytop),
                        arrowprops=dict(arrowstyle='-|>', lw=1.0, color='blue'))

    ax.set_xlabel(r'$\max(Z_{\max})$ per PE (A)', loc='right')
    ax.set_ylabel('Number of PEs', loc='top')
    ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9)

    out = outdir / f"maxZmax_distribution_A.{fmt}"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


def _plot_maxZmax_positions_A(rel_pos_pred, rel_pos_lr, eff_len_pred, eff_len_lr, outdir: Path, dpi=300, fmt="png"):
    rel_pos_pred = np.asarray(rel_pos_pred, dtype=float)
    rel_pos_lr   = np.asarray(rel_pos_lr,   dtype=float)

    fig, ax = plt.subplots(figsize=(10/2.54, 10/2.54), dpi=dpi)
    bins = np.linspace(0, 100, 51)
    ax.hist(100.0 * rel_pos_pred[np.isfinite(rel_pos_pred)], bins=bins, histtype='step', color='red',  label='BumpNet (A)')
    ax.hist(100.0 * rel_pos_lr[np.isfinite(rel_pos_lr)],     bins=bins, histtype='step', color='blue', label=r'$Z_{\mathrm{LR}}$ (A)')
    ax.set_xlabel('Position of max $Z$ in histogram [%] (A)', loc='right')
    ax.set_ylabel('Number of PEs', loc='top')
    ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax.legend(fontsize=8)
    out = outdir / f"maxzmax_position_distribution_A.{fmt}"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")

    # Optional second panel by absolute index (mirroring your style)
    eff_len_pred = np.asarray(eff_len_pred, dtype=float)
    eff_len_lr   = np.asarray(eff_len_lr,   dtype=float)
    m_pred = np.isfinite(rel_pos_pred) & np.isfinite(eff_len_pred) & (eff_len_pred > 0)
    m_lr   = np.isfinite(rel_pos_lr)   & np.isfinite(eff_len_lr)   & (eff_len_lr > 0)
    if m_pred.any() or m_lr.any():
        idx_pred = np.floor(rel_pos_pred[m_pred] * eff_len_pred[m_pred]).astype(int)
        idx_lr   = np.floor(rel_pos_lr[m_lr]     * eff_len_lr[m_lr]).astype(int)
        if idx_pred.size: idx_pred = np.clip(idx_pred, 0, (eff_len_pred[m_pred] - 1).astype(int))
        if idx_lr.size:   idx_lr   = np.clip(idx_lr,   0, (eff_len_lr[m_lr]   - 1).astype(int))
        xmax = int(max(idx_pred.max() if idx_pred.size else 0, idx_lr.max() if idx_lr.size else 0))
        edges = np.arange(-0.5, xmax + 1.5, 1)
        fig2, ax2 = plt.subplots(figsize=(10/2.54, 10/2.54), dpi=dpi)
        if idx_pred.size: ax2.hist(idx_pred, bins=edges, histtype='step', color='red',  label='BumpNet (A)')
        if idx_lr.size:   ax2.hist(idx_lr,   bins=edges, histtype='step', color='blue', label=r'$Z_{\mathrm{LR}}$ (A)')
        ax2.set_xlabel('Index of max $Z$ bin (A)', loc='right')
        ax2.set_ylabel('Number of PEs', loc='top')
        ax2.xaxis.set_minor_locator(AutoMinorLocator()); ax2.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        ax2.legend(fontsize=8)
        out2 = outdir / f"maxzmax_position_by_binindex_A.{fmt}"
        fig2.savefig(out2, bbox_inches="tight"); plt.close(fig2)
        print(f"Saved {out2}")

def _plot_B_atA_bin_overlay(B_pred, B_lr, outdir: Path, dpi=300, fmt="png", sigma=3.0):
    B_pred = np.asarray(B_pred, dtype=float)
    B_lr   = np.asarray(B_lr,   dtype=float)

    fig, ax = plt.subplots(figsize=(10/2.54, 10/2.54), dpi=dpi)
    # Build bins from finite-only values (avoid -999 sentinel widening the range)
    finite_combo = np.concatenate([
        B_pred[np.isfinite(B_pred)],
        B_lr[np.isfinite(B_lr)]
    ])
    if finite_combo.size == 0:
        print("[WARN] No finite entries to build bins in _plot_B_atA_bin_overlay")
        return
    bins = np.histogram_bin_edges(finite_combo, bins=100)

    # Draw histograms
    ax.hist(B_pred[np.isfinite(B_pred)], bins=bins, histtype='step', color='red',  label='BumpNet (B@A)')
    ax.hist(B_lr[np.isfinite(B_lr)],     bins=bins, histtype='step', color='blue', label=r'$Z_{\mathrm{LR}}$ (B@A)')

    # --- 3σ thresholds on these very distributions (with anti-overlap labels) ---
    def _tail_threshold(vals, sigma):
        vals = np.asarray(vals, dtype=float); vals = vals[np.isfinite(vals)]
        if not vals.size: return None
        p_tail = 0.5 * math.erfc(sigma / math.sqrt(2.0))  # one-sided tail
        q = 1.0 - p_tail
        return float(np.quantile(vals, q))

    thr_pred = _tail_threshold(B_pred, sigma)
    thr_lr   = _tail_threshold(B_lr,   sigma)

    # Draw the vertical lines first
    if thr_pred is not None:
        ax.axvline(thr_pred, color='red', ls='--')
    if thr_lr is not None:
        ax.axvline(thr_lr, color='blue', ls='--')

    # Prepare non-overlapping vertical labels near the top
    ytop = ax.get_ylim()[1]
    x0, x1 = ax.get_xlim()
    span   = x1 - x0
    dx     = 0.03 * span            # base horizontal offset for label text (data units)
    pad    = 0.01 * span            # keep inside axes

    def _safe_x(x):
        return min(max(x, x0 + pad), x1 - pad)

    def _vlabel(x, txt, color, prefer_side="right"):
        # choose side and alignment
        if prefer_side == "right":
            x_text, ha = _safe_x(x + dx), 'left'
        else:
            x_text, ha = _safe_x(x - dx), 'right'
        ax.text(x_text, 0.92*ytop, txt, color=color, rotation=90,
                va='top', ha=ha, fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    if (thr_pred is not None) or (thr_lr is not None):
        # If thresholds are too close, push labels to opposite sides
        close = (thr_pred is not None and thr_lr is not None and abs(thr_pred - thr_lr) < 1.2*dx)

        if thr_pred is not None:
            txt = f"3σ ⇒ Zpred > {thr_pred:.2f}"
            side = "right"
            if close and thr_pred <= thr_lr:  # send the leftmost one to the left
                side = "left"
            _vlabel(thr_pred, txt, 'red', prefer_side=side)

        if thr_lr is not None:
            txt = f"3σ ⇒ ZLR > {thr_lr:.2f}"
            side = "left"
            if close and thr_lr < thr_pred:   # send the leftmost one to the left
                side = "left"
            elif not close:
                side = "left"                 # default: go left of the line
            _vlabel(thr_lr, txt, 'blue', prefer_side=side)


    # ================== NEW: Gaussian fits & overlays ==================
    pred_finite = B_pred[np.isfinite(B_pred)]
    lr_finite   = B_lr[np.isfinite(B_lr)]

    mu_p,  sig_p  = _robust_gaussian_fit(pred_finite, clip_sigma=3.0)
    mu_lr, sig_lr = _robust_gaussian_fit(lr_finite,   clip_sigma=3.0)

    # Smooth x-grid for lines
    xline = np.linspace(bins[0], bins[-1], 600)
    # Convert Normal pdf to "counts per bin" scale: N * Δx * pdf(x)
    Np = float(pred_finite.size)
    Nl = float(lr_finite.size)
    dx = float(np.median(np.diff(bins))) if (len(bins) > 1) else 1.0

    def _gauss_counts(x, mu, sig, N):
        if not np.isfinite(mu) or not np.isfinite(sig) or sig <= 0: 
            return np.full_like(x, np.nan)
        pdf = (1.0 / (sig * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu)/sig)**2)
        return N * dx * pdf

    y_pred_fit = _gauss_counts(xline, mu_p,  sig_p,  Np)
    y_lr_fit   = _gauss_counts(xline, mu_lr, sig_lr, Nl)

    # Plot fit curves
    if np.isfinite(y_pred_fit).any():
        ax.plot(xline, y_pred_fit, color='red', lw=1.8, alpha=0.9)
    if np.isfinite(y_lr_fit).any():
        ax.plot(xline, y_lr_fit,   color='blue', lw=1.8, alpha=0.9)

    # Reserve a little header room at the top of the figure
    fig.subplots_adjust(top=0.85)

    # Put fit summaries in the figure header so they never overlap or clip
    fig.text(0.01, 0.985, f"BumpNet fit: μ={mu_p:.2f}, σ={sig_p:.2f}",
            ha='left', va='top', color='red', fontsize=9,
            bbox=dict(facecolor='white', edgecolor='0.8', alpha=0.95))

    fig.text(0.99, 0.985, r"$Z_{\mathrm{LR}}$ fit: " + f"μ={mu_lr:.2f}, σ={sig_lr:.2f}",
            ha='right', va='top', color='blue', fontsize=9,
            bbox=dict(facecolor='white', edgecolor='0.8', alpha=0.95))

    print(f"[B@A Gaussian fits] BumpNet: mu={mu_p:.3f}, sigma={sig_p:.3f} | "
      f"Z_LR: mu={mu_lr:.3f}, sigma={sig_lr:.3f}")
    # ===================================================================

    ax.set_xlabel(r'$Z$ at A-selected histo and bin (B sample)', loc='right')
    ax.set_ylabel('Number of PEs', loc='top')
    ax.legend(fontsize=8, loc='upper left', framealpha=0.9)

    out = outdir / f"B_at_A_bin_Z_distribution.{fmt}"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)

def _plot_deltaZmax_vs_pos(pairs, outfile: Path, dpi=300):
    if not pairs:
        print(f"[WARN] No entries for {outfile.name}")
        return
    pairs = np.asarray(pairs, dtype=float)
    x = pairs[:,0]; y = pairs[:,1]
    from matplotlib import colormaps
    cmap = colormaps["Spectral_r"]; cmap.set_under("white", alpha=0)
    fig, ax = plt.subplots(figsize=(10/2.54, 7/2.54), dpi=dpi)
    hb = ax.hist2d(x, y, bins=(100, 100), cmap=cmap, vmin=1e-3, range=((0.0, 1.0), (-5, 5)))
    ax.set_ylim(-5, 5)
    mu, sigma = np.nanmean(y), np.nanstd(y)
    ax.annotate(fr"$\mu = {mu:+.3f}$" "\n" fr"$\sigma = {sigma:.2f}$",
                xy=(0.02, 0.97), xycoords='axes fraction',
                ha='left', va='top', fontsize=9)
    ax.set_xlabel(r"$Z^{\mathrm{LR}}_{\max}$ bin / number of bins", loc='right')
    ax.set_ylabel(r"$\Delta Z_{\max}$ (BumpNet − LR)", loc='center')
    ax.axhline(0, color='black', lw=0.6)
    ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator())
    fig.colorbar(hb[3], ax=ax, label="Entries")
    fig.savefig(outfile, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {outfile}")

def _plot_significance_1d_all(all_pred, all_lr, outdir: Path, dpi=300, fmt="png"):
    """Overlay 1D hist of Z_pred vs Z_LR (all bins), plus ΔZ = Z_pred - Z_LR, each with Gaussian fit."""
    all_pred = np.asarray(all_pred, dtype=float)
    all_lr   = np.asarray(all_lr,   dtype=float)
    m = np.isfinite(all_pred) & np.isfinite(all_lr)
    if not m.any():
        print("[WARN] No finite entries for pooled 1D hist")
        return

    pred  = all_pred[m]
    lr    = all_lr[m]
    delta = pred - lr

    # ---------------- Figure 1: overlay of Z_pred & Z_LR ----------------
    fig1, ax1 = plt.subplots(figsize=(10/2.54, 10/2.54), dpi=dpi)
    bins = np.histogram_bin_edges(np.concatenate([pred, lr]), bins=120)
    ax1.hist(pred, bins=bins, histtype='step', label='BumpNet (all bins)', lw=1.2)
    ax1.hist(lr,   bins=bins, histtype='step', label=r'$Z_{\mathrm{LR}}$ (all bins)', lw=1.2)

    # Fits (reusing robust fit) + uncertainties
    mu_p, sg_p, mu_p_e, sg_p_e, n_p = _fit_gaussian_with_uncertainties(pred)
    mu_l, sg_l, mu_l_e, sg_l_e, n_l = _fit_gaussian_with_uncertainties(lr)

    # Smooth grid + scaling of pdf to "counts per bin"
    xline = np.linspace(bins[0], bins[-1], 800)
    dx = float(np.median(np.diff(bins))) if (len(bins) > 1) else 1.0

    def _gauss_counts(x, mu, sig, N):
        if not np.isfinite(mu) or not np.isfinite(sig) or sig <= 0:
            return np.full_like(x, np.nan)
        pdf = (1.0 / (sig * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu)/sig)**2)
        return N * dx * pdf

    y_pred = _gauss_counts(xline, mu_p, sg_p, n_p)
    y_lr   = _gauss_counts(xline, mu_l, sg_l, n_l)

    if np.isfinite(y_pred).any():
        ax1.plot(xline, y_pred, color='C0', lw=1.8, alpha=0.9)  # same color family as first hist
    if np.isfinite(y_lr).any():
        ax1.plot(xline, y_lr,   color='C1', lw=1.8, alpha=0.9)

    ax1.set_xlabel('Significance Z', loc='right'); ax1.set_ylabel('Entries', loc='top')
    ax1.xaxis.set_minor_locator(AutoMinorLocator()); ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax1.legend(fontsize=8, loc='best')

    # Keep text off the axes: put fit stats in the figure header
    fig1.subplots_adjust(top=0.86)
    fig1.text(0.01, 0.985, f"BumpNet: μ={mu_p:.3f}±{mu_p_e:.3f}, σ={sg_p:.3f}±{sg_p_e:.3f} (N={n_p})",
              ha='left', va='top', color='C0', fontsize=9,
              bbox=dict(facecolor='white', edgecolor='0.8', alpha=0.95))
    fig1.text(0.99, 0.985, r"$Z_{\mathrm{LR}}$: " + f"μ={mu_l:.3f}±{mu_l_e:.3f}, σ={sg_l:.3f}±{sg_l_e:.3f} (N={n_l})",
              ha='right', va='top', color='C1', fontsize=9,
              bbox=dict(facecolor='white', edgecolor='0.8', alpha=0.95))

    f1 = outdir / f"pooled_allbins_Zpred_ZLR.{fmt}"
    fig1.savefig(f1, bbox_inches="tight"); plt.close(fig1); print(f"Saved {f1}")

    # ---------------- Figure 2: ΔZ = Z_pred - Z_LR ----------------
    fig2, ax2 = plt.subplots(figsize=(10/2.54, 10/2.54), dpi=dpi)
    bins_d = np.histogram_bin_edges(delta, bins=120)
    ax2.hist(delta, bins=bins_d, histtype='step', lw=1.2)

    mu_d, sg_d, mu_d_e, sg_d_e, n_d = _fit_gaussian_with_uncertainties(delta)
    xline_d = np.linspace(bins_d[0], bins_d[-1], 800)
    dxd     = float(np.median(np.diff(bins_d))) if (len(bins_d) > 1) else 1.0
    y_d     = _gauss_counts(xline_d, mu_d, sg_d, n_d)

    if np.isfinite(y_d).any():
        ax2.plot(xline_d, y_d, color='C2', lw=1.8, alpha=0.9, label='Gaussian fit')

    ax2.set_xlabel(r'$\Delta Z = Z_{\mathrm{pred}} - Z_{\mathrm{LR}}$', loc='right')
    ax2.set_ylabel('Entries', loc='top')
    ax2.xaxis.set_minor_locator(AutoMinorLocator()); ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax2.legend(fontsize=8, loc='best')

    fig2.subplots_adjust(top=0.86)
    fig2.text(0.01, 0.985, f"ΔZ fit: μ={mu_d:.3f}±{mu_d_e:.3f}, σ={sg_d:.3f}±{sg_d_e:.3f} (N={n_d})",
              ha='left', va='top', color='C2', fontsize=9,
              bbox=dict(facecolor='white', edgecolor='0.8', alpha=0.95))

    f2 = outdir / f"pooled_allbins_deltaZ.{fmt}"
    fig2.savefig(f2, bbox_inches="tight"); plt.close(fig2); print(f"Saved {f2}")

def _plot_all_by_relpos(all_pred, all_lr, all_relpos, relpos_edges, outdir: Path, dpi=300, fmt="png"):
    """Make 5 (or user-given) relative-position slices; plot ΔZ per slice with Gaussian fit/summary."""
    pred = np.asarray(all_pred, dtype=float)
    lr   = np.asarray(all_lr,   dtype=float)
    rpos = np.asarray(all_relpos, dtype=float)
    m = np.isfinite(pred) & np.isfinite(lr) & np.isfinite(rpos)
    if not m.any():
        print("[WARN] No finite entries for relpos-sliced plots")
        return
    pred = pred[m]; lr = lr[m]; rpos = rpos[m]
    delta = pred - lr

    # Default to 5 equal bins if not provided
    if relpos_edges is None:
        relpos_edges = np.linspace(0.0, 1.0, 6)  # 0..1 in 5 bins
    else:
        relpos_edges = np.asarray(relpos_edges, dtype=float)

    for i in range(len(relpos_edges)-1):
        lo, hi = relpos_edges[i], relpos_edges[i+1]
        sel = (rpos >= lo) & (rpos < hi)
        if not np.any(sel):
            print(f"[INFO] relpos slice [{lo:.2f}, {hi:.2f}): empty")
            continue
        d = delta[sel]
        fig, ax = plt.subplots(figsize=(10/2.54, 10/2.54), dpi=dpi)
        bins = np.histogram_bin_edges(d, bins=100)
        ax.hist(d, bins=bins, histtype='step', lw=1.2)

        # Fit + overlay
        mu, sg, mu_e, sg_e, n = _fit_gaussian_with_uncertainties(d)
        xline = np.linspace(bins[0], bins[-1], 600)
        dx    = float(np.median(np.diff(bins))) if (len(bins) > 1) else 1.0

        def _gauss_counts(x, mu, sig, N):
            if not np.isfinite(mu) or not np.isfinite(sig) or sig <= 0:
                return np.full_like(x, np.nan)
            pdf = (1.0 / (sig * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu)/sig)**2)
            return N * dx * pdf

        y = _gauss_counts(xline, mu, sg, n)
        if np.isfinite(y).any():
            ax.plot(xline, y, color='C3', lw=1.8, alpha=0.9, label='Gaussian fit')

        ax.set_title(f"ΔZ in relpos ∈ [{lo:.2f}, {hi:.2f})")
        ax.set_xlabel(r'$\Delta Z$', loc='right'); ax.set_ylabel('Entries', loc='top')
        ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        ax.legend(fontsize=8, loc='best')

        # Header text to avoid overlap
        fig.subplots_adjust(top=0.86)
        fig.text(0.01, 0.985, f"μ={mu:.3f}±{mu_e:.3f}, σ={sg:.3f}±{sg_e:.3f} (N={n})",
                 ha='left', va='top', color='C3', fontsize=9,
                 bbox=dict(facecolor='white', edgecolor='0.8', alpha=0.95))

        f = outdir / f"deltaZ_by_relpos_{i}_{lo:.2f}_{hi:.2f}.{fmt}"
        fig.savefig(f, bbox_inches="tight"); plt.close(fig); print(f"Saved {f}")

def _plot_relpos_triplets(all_pred, all_lr, all_relpos, relpos_edges,
                          outdir: Path, dpi=300, fmt="png", logy=False):
    """
    For each relative-position slice: Z_pred (red), Z_LR (blue), ΔZ (green),
    each with histogram, Gaussian overlay, and μ, σ, χ²/ndof box.
    """
    # ---- styling & hardcoded headroom ----
    COL_PRED, COL_LR, COL_DELTA = "red", "blue", "green"
    HEADROOM_LINEAR = 0.35   # +35% on peak (linear y)
    HEADROOM_LOG    = 3.00   # ×3 on peak (log y)

    pred = np.asarray(all_pred, dtype=float)
    lr   = np.asarray(all_lr,   dtype=float)
    rpos = np.asarray(all_relpos, dtype=float)
    m = np.isfinite(pred) & np.isfinite(lr) & np.isfinite(rpos)
    if not m.any():
        print("[WARN] No finite entries for relpos triplets"); return
    pred = pred[m]; lr = lr[m]; rpos = rpos[m]
    delta = pred - lr

    edges = (np.linspace(0.0, 1.0, 6) if relpos_edges is None
             else np.asarray(relpos_edges, dtype=float).ravel())
    if edges.size < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("relpos_edges must be strictly increasing with ≥2 values.")

    def _gauss_bin_expectation(edges, mu, sig, N):
        if not (np.isfinite(mu) and np.isfinite(sig) and sig > 0 and N > 0):
            return np.zeros(len(edges)-1, dtype=float)
        centers = 0.5*(edges[:-1] + edges[1:])
        widths  = edges[1:] - edges[:-1]
        pdf     = (1.0/(sig*np.sqrt(2.0*np.pi))) * np.exp(-0.5*((centers-mu)/sig)**2)
        return N * widths * pdf

    def _chi2_ndof(counts, h_edges, mu, sig):
        Ntot  = counts.sum()
        model = _gauss_bin_expectation(h_edges, mu, sig, Ntot)
        denom = np.maximum(model, 1.0)
        chi2  = np.sum((counts - model)**2 / denom)
        ndof  = max(int(np.count_nonzero(denom) - 3), 1)  # (N, μ, σ)
        return chi2, ndof

    for i in range(len(edges)-1):
        lo, hi = edges[i], edges[i+1]
        sel = (rpos >= lo) & (rpos < hi)
        if not np.any(sel):
            print(f"[INFO] relpos slice [{lo:.2f}, {hi:.2f}): empty"); continue

        p = pred[sel]; l = lr[sel]; d = delta[sel]
        fig, axes = plt.subplots(1, 3, figsize=(18/2.54, 10/2.54), dpi=dpi)
        panels = [(axes[0], p, r"$Z_{\mathrm{pred}}$", COL_PRED),
                  (axes[1], l, r"$Z_{\mathrm{LR}}$",   COL_LR),
                  (axes[2], d, r"$\Delta Z = Z_{\mathrm{pred}}-Z_{\mathrm{LR}}$", COL_DELTA)]

        for ax, dat, xlabel, col in panels:
            dat = np.asarray(dat, dtype=float)
            h_edges = np.histogram_bin_edges(dat, bins=120)
            counts, h_edges = np.histogram(dat, bins=h_edges)
            ax.hist(dat, bins=h_edges, histtype='step', lw=1.2, color=col)

            # ---- y-scale & headroom ----
            if logy:
                ax.set_yscale('log')
                pos = counts[counts > 0]
                ymin = float(pos.min()*0.8) if pos.size else 0.5
                ymax = float(max(counts.max(), 1.0) * HEADROOM_LOG)
                ax.set_ylim(ymin, ymax)
            else:
                peak = float(counts.max()) if counts.size else 1.0
                y0, y1 = ax.get_ylim()
                y1_new = max(y1, peak) * (1.0 + HEADROOM_LINEAR)
                ax.set_ylim(0.0, y1_new)

            # ---- robust fit + overlay ----
            mu, sg, _, _, _ = _fit_gaussian_with_uncertainties(dat)
            if np.isfinite(mu) and np.isfinite(sg) and sg > 0:
                xline = np.linspace(h_edges[0], h_edges[-1], 800)
                dxm   = float(np.median(np.diff(h_edges))) if len(h_edges) > 1 else 1.0
                pdf   = (1.0/(sg*np.sqrt(2.0*np.pi))) * np.exp(-0.5*((xline-mu)/sg)**2)
                yfit  = counts.sum() * dxm * pdf
                ax.plot(xline, yfit, color=col, lw=1.8, alpha=0.95)

            chi2, ndof = _chi2_ndof(counts, h_edges, mu, sg)
            ax.set_xlabel(xlabel, loc='right'); ax.set_ylabel('Entries', loc='top')
            ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            ax.text(0.03, 0.97, f"μ={mu:.3f}\nσ={sg:.3f}\nχ²/ndof={chi2/ndof:.2f}",
                    transform=ax.transAxes, ha='left', va='top', fontsize=9, color=col,
                    bbox=dict(facecolor='white', edgecolor='0.8', alpha=0.95, pad=3))

        fig.suptitle(f"Triplet in relpos ∈ [{lo:.2f}, {hi:.2f})", y=0.96)
        fig.subplots_adjust(top=0.89, wspace=0.35)
        fout = outdir / f"triplet_by_relpos_{i}_{lo:.2f}_{hi:.2f}.{fmt}"
        fig.savefig(fout, bbox_inches="tight"); plt.close(fig); print(f"Saved {fout}")


def _plot_triplets_by_nevents(all_pred, all_lr, all_nevt, nevt_edges,
                              outdir: Path, dpi=300, fmt="png", logy=False):
    """
    One 3-panel figure per Nevents slice with μ, σ, χ²/ndof boxes.
    """
    COL_PRED, COL_LR, COL_DELTA = "red", "blue", "green"
    HEADROOM_LINEAR = 0.35   # +35% on peak (linear y)
    HEADROOM_LOG    = 3.00   # ×3 on peak (log y)

    pred = np.asarray(all_pred, dtype=float)
    lr   = np.asarray(all_lr,   dtype=float)
    nevt = np.asarray(all_nevt, dtype=float)
    m = np.isfinite(pred) & np.isfinite(lr) & np.isfinite(nevt)
    if not m.any():
        print("[WARN] No finite entries for Nevents triplets"); return
    pred = pred[m]; lr = lr[m]; nevt = nevt[m]
    delta = pred - lr

    if nevt_edges is None:
        print("[WARN] nevents_edges not provided; skipping Nevents triplets."); return
    edgesN = np.unique(np.sort(np.asarray(nevt_edges, dtype=float).ravel()))
    if edgesN.size < 2:
        print("[WARN] nevents_edges malformed; skipping."); return

    def _gauss_bin_expectation(edges, mu, sig, N):
        if not (np.isfinite(mu) and np.isfinite(sig) and sig > 0 and N > 0):
            return np.zeros(len(edges)-1, dtype=float)
        centers = 0.5*(edges[:-1] + edges[1:])
        widths  = edges[1:] - edges[:-1]
        pdf     = (1.0/(sig*np.sqrt(2.0*np.pi))) * np.exp(-0.5*((centers-mu)/sig)**2)
        return N * widths * pdf

    def _chi2_ndof(counts, h_edges, mu, sig):
        Ntot  = counts.sum()
        model = _gauss_bin_expectation(h_edges, mu, sig, Ntot)
        denom = np.maximum(model, 1.0)
        chi2  = np.sum((counts - model)**2 / denom)
        ndof  = max(int(np.count_nonzero(denom) - 3), 1)
        return chi2, ndof

    def _near_int(x, tol=1e-9): return np.isfinite(x) and np.isclose(x, round(x), atol=tol)

    for i in range(len(edgesN) - 1):
        lo, hi = edgesN[i], edgesN[i+1]
        sel = (nevt >= lo) & (nevt < hi) if i < len(edgesN)-2 else (nevt >= lo) & (nevt <= hi)
        if not np.any(sel):
            print(f"[INFO] Nevents slice [{lo:.0f}, {hi:.0f}): empty"); continue

        p = pred[sel]; l = lr[sel]; d = delta[sel]
        fig, axes = plt.subplots(1, 3, figsize=(18/2.54, 10/2.54), dpi=dpi)
        panels = [(axes[0], p, r"$Z_{\mathrm{pred}}$", COL_PRED),
                  (axes[1], l, r"$Z_{\mathrm{LR}}$",   COL_LR),
                  (axes[2], d, r"$\Delta Z = Z_{\mathrm{pred}}-Z_{\mathrm{LR}}$", COL_DELTA)]

        for ax, dat, xlabel, col in panels:
            dat = np.asarray(dat, dtype=float)
            h_edges = np.histogram_bin_edges(dat, bins=120)
            counts, h_edges = np.histogram(dat, bins=h_edges)
            ax.hist(dat, bins=h_edges, histtype='step', lw=1.2, color=col)

            if logy:
                ax.set_yscale('log')
                pos  = counts[counts > 0]
                ymin = float(pos.min()*0.8) if pos.size else 0.5
                ymax = float(max(counts.max(), 1.0) * HEADROOM_LOG)
                ax.set_ylim(ymin, ymax)
            else:
                peak = float(counts.max()) if counts.size else 1.0
                y0, y1 = ax.get_ylim()
                y1_new = max(y1, peak) * (1.0 + HEADROOM_LINEAR)
                ax.set_ylim(0.0, y1_new)

            mu, sg, _, _, _ = _fit_gaussian_with_uncertainties(dat)
            if np.isfinite(mu) and np.isfinite(sg) and sg > 0:
                xline = np.linspace(h_edges[0], h_edges[-1], 800)
                dxm   = float(np.median(np.diff(h_edges))) if len(h_edges) > 1 else 1.0
                pdf   = (1.0/(sg*np.sqrt(2.0*np.pi))) * np.exp(-0.5*((xline-mu)/sg)**2)
                yfit  = counts.sum() * dxm * pdf
                ax.plot(xline, yfit, color=col, lw=1.8, alpha=0.95)

            chi2, ndof = _chi2_ndof(counts, h_edges, mu, sg)
            ax.set_xlabel(xlabel, loc='right'); ax.set_ylabel('Entries', loc='top')
            ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            ax.text(0.03, 0.97, f"μ={mu:.3f}\nσ={sg:.3f}\nχ²/ndof={chi2/ndof:.2f}",
                    transform=ax.transAxes, ha='left', va='top', fontsize=9, color=col,
                    bbox=dict(facecolor='white', edgecolor='0.8', alpha=0.95, pad=3))

        lo_i, hi_i = int(round(lo)), int(round(hi))
        is_single  = _near_int(lo) and _near_int(hi) and (hi_i - lo_i == 1)
        closing    = ')' if i < len(edgesN)-2 else ']'
        title      = f"Triplet by Nevents — N={lo_i} only" if is_single \
                     else f"Triplet by Nevents ∈ [{lo:,.0f}, {hi:,.0f}{closing}"
        fig.suptitle(title, y=0.96)
        fig.subplots_adjust(top=0.89, wspace=0.35)
        fout = (outdir / f"triplet_by_nevents_N{lo_i}.{fmt}") if is_single \
               else (outdir / f"triplet_by_nevents_{i}_{int(lo)}_{int(hi)}.{fmt}")
        fig.savefig(fout, bbox_inches="tight"); plt.close(fig); print(f"Saved {fout}")


def _plot_examples_AB(examples, outdir: Path, dpi=300, left_frac=0.0):
    outdir.mkdir(parents=True, exist_ok=True)
    for (edgesA, histA, zA_pred, zA_lr,
         edgesB, histB, zB_pred, zB_lr,
         name, peid, bin_idx) in examples:

        centresA = 0.5 * (edgesA[1:] + edgesA[:-1])
        centresB = 0.5 * (edgesB[1:] + edgesB[:-1])

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
        axA_top, axA_bot, axB_top, axB_bot = axes[0,0], axes[1,0], axes[0,1], axes[1,1]

        # --- Mass histograms (lock B to A's scale & limits) ---
        if histA is not None:
            axA_top.step(centresA, histA, where='mid', color='black')

            use_log = (histA[0] >= 500)  # your heuristic
            if use_log:
                # strictly positive lower bound for log
                pos = np.asarray(histA, float)
                pos = pos[pos > 0]
                ylo = 0.8 * (pos.min() if pos.size else 1.0)
                yhi = 1.2 * (pos.max() if pos.size else 1.0)
                axA_top.set_yscale('log')
                axB_top.set_yscale('log')
            else:
                ylo = 0.0
                yhi = 1.1 * (float(np.max(histA)) if np.max(histA) > 0 else 1.0)

            # set limits AFTER scale, and freeze autoscaling
            axA_top.set_ylim(ylo, yhi)
            axB_top.set_ylim(ylo, yhi)
            axA_top.autoscale(enable=False, axis='y')
            axB_top.autoscale(enable=False, axis='y')

        if histB is not None:
            axB_top.step(centresB, histB, where='mid', color='black')


        axA_top.set_ylabel("Entries")
        axA_top.set_title(f"PE_{peid} A — {name}")
        _draw_left_cut(axA_top, edgesA, left_frac)

        axB_top.set_ylabel("Entries")
        axB_top.set_title(f"PE_{peid} B — {name}")
        _draw_left_cut(axB_top, edgesB, left_frac)

        # --- Z distributions ---
        axA_bot.step(centresA, zA_pred, where='mid', label='BumpNet (raw)', color='red', alpha=0.85)
        if zA_lr is not None and np.isfinite(zA_lr).any():
            axA_bot.step(centresA, zA_lr, where='mid', label=r'$Z_{\mathrm{LR}}$', color='blue')
        axA_bot.axvline(centresA[bin_idx], color='k', ls='--', lw=0.8)

        axB_bot.step(centresB, zB_pred, where='mid', label='BumpNet (raw)', color='red', alpha=0.85)
        if zB_lr is not None and np.isfinite(zB_lr).any():
            axB_bot.step(centresB, zB_lr, where='mid', label=r'$Z_{\mathrm{LR}}$', color='blue')
        axB_bot.axvline(centresB[bin_idx], color='k', ls='--', lw=0.8)

        # --- lock Z-axis range to that of sample A ---
        valsA = np.array([], dtype=float)
        if zA_pred is not None:
            valsA = np.concatenate([valsA, np.asarray(zA_pred, dtype=float)])
        if (zA_lr is not None) and np.isfinite(zA_lr).any():
            valsA = np.concatenate([valsA, np.asarray(zA_lr, dtype=float)])

        valsA = valsA[np.isfinite(valsA)]
        if valsA.size:
            ylo, yhi = float(valsA.min()), float(valsA.max())
            if yhi == ylo:  # avoid zero-height axis
                ylo, yhi = ylo - 1.0, yhi + 1.0
            pad = 0.08 * (yhi - ylo)
            ylo -= pad; yhi += pad
            axA_bot.set_ylim(ylo, yhi)
            axB_bot.set_ylim(ylo, yhi)

        axA_bot.set_xlabel("Mass [GeV]"); axA_bot.set_ylabel("Z"); axA_bot.legend(fontsize=8, loc='best')
        axB_bot.set_xlabel("Mass [GeV]"); axB_bot.set_ylabel("Z"); axB_bot.legend(fontsize=8, loc='best')

        for ax in (axA_top, axA_bot, axB_top, axB_bot):
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)

        plt.tight_layout()
        fout = outdir / f"PE_{peid}_{_slug(name)}_maxZ_example.png"
        fig.savefig(fout, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved example plot: {fout}")

