# global_significance/generate_toys_bkg_only.py
import numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from typing import Tuple

def _split_counts_exact_k(counts: np.ndarray, frac: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact-size split using a multivariate hypergeometric draw.
    - counts: per-bin integer counts (sum = N)
    - frac:   target fraction for A
    Returns A, B with:
      sum(A) = round(frac * N), sum(B) = N - sum(A), and A[i]+B[i] = counts[i].
    """
    N = int(counts.sum())
    if N == 0:
        # Keep structure, nothing to split
        z = np.zeros_like(counts, dtype=int)
        return z.copy(), z.copy()

    kA = int(round(frac * N))
    kA = max(0, min(kA, N))

    A = np.zeros_like(counts, dtype=int)
    remaining_A  = kA
    remaining_tot = N

    # Sequential multivariate hypergeometric sampling
    # (Equivalent to choosing exactly kA events uniformly from the N, then counting by bin.)
    for i in range(len(counts) - 1):
        ngood = int(counts[i])
        nbad  = int(remaining_tot - ngood)
        if remaining_A == 0 or ngood == 0:
            take_i = 0
        elif remaining_A >= remaining_tot:
            take_i = ngood
        else:
            take_i = int(rng.hypergeometric(ngood, nbad, remaining_A))
            take_i = min(take_i, ngood)  # numerical safety

        A[i] = take_i
        remaining_A  -= take_i
        remaining_tot -= ngood

    # Last bin gets what's left
    A[-1] = remaining_A
    B = counts - A

    # Sanity checks
    assert int(A.sum()) == kA, f"Exact-k split failed: sum(A)={A.sum()} != {kA}"
    assert np.all(A >= 0) and np.all(B >= 0), "Negative bin after split"
    assert np.all(A + B == counts), "Per-bin totals not preserved"

    return A, B

# ---------------------------------------------------------------------------
# Multiprocessing helpers – must live at module scope so they’re picklable
# ---------------------------------------------------------------------------
def _init_worker(bkg_template):
    """
    Run once in every worker process; stash the template histograms
    in a global so the per-task function doesn’t need to pickle them.
    """
    global BKG_TEMPLATE
    BKG_TEMPLATE = bkg_template          # object-dtype NumPy array

def _worker_chunk(args):
    """
    Generate one “shard” of background-only toy pseudo-experiments (PEs).

    Parameters
    ----------
    start_idx  : int       Index of the first PE in this shard.
    seeds      : 1-D int   One RNG seed per PE for reproducibility.
    shard_dir  : Path      Where to write *.npz files for this shard.
    mmap_path  : Path      Path to the template .npy file (already mem-mapped
                           by _init_worker, kept here only for completeness).                       
    split_DS_AB   : bool   Emit A/B files per PE if True
    ds_fraction_A : float  Fraction assigned to A in [0,1]
    """
    (start_idx, seeds, shard_dir, mmap_path, split_DS_AB, ds_fraction_A) = args

    # Local scratch list reused for every PE to avoid reallocations
    hist_list = []

    bar_desc = f"Worker starting at PE {start_idx}"
    for i, seed in enumerate(tqdm(seeds, desc=bar_desc, position=0, leave=False)):
        pe_idx = start_idx + i
        rng = np.random.default_rng(seed)

        hist_list.clear()
        for row in BKG_TEMPLATE:
            lam_vec = np.asarray(row, dtype=float)
            lam_vec = np.nan_to_num(lam_vec, nan=0.0)

            # Draw Poisson counts for this histogram (background-only)
            counts = rng.poisson(lam=lam_vec)

            if split_DS_AB:
                A, B = _split_counts_exact_k(counts, ds_fraction_A, rng)
                hist_list.append((A, B))
            else:
                hist_list.append(counts)

        tqdm.write(f"[PE {pe_idx:07d}] Generated {len(hist_list)} histograms")

        if split_DS_AB:
            hist_A = np.array([ab[0] for ab in hist_list], dtype=object)
            hist_B = np.array([ab[1] for ab in hist_list], dtype=object)

            np.savez_compressed(
                shard_dir / f"PE_{pe_idx:07d}_A.npz",
                HIST=hist_A,
                bin_edges=BIN_EDGES_GLOBAL,
                names=NAMES_GLOBAL,
            )
            np.savez_compressed(
                shard_dir / f"PE_{pe_idx:07d}_B.npz",
                HIST=hist_B,
                bin_edges=BIN_EDGES_GLOBAL,
                names=NAMES_GLOBAL,
            )
        else:
            hist_obj = np.array(hist_list, dtype=object)
            np.savez_compressed(
                shard_dir / f"PE_{pe_idx:07d}.npz",
                HIST=hist_obj,
                bin_edges=BIN_EDGES_GLOBAL,
                names=NAMES_GLOBAL,
            )

    return start_idx


def generate_toys_bkg_only(cfg):
    """
    Driver: divide work into shards, spawn a Pool, and write diagnostics.
    `cfg` may be a dict or a SimpleNamespace with at least:
        cfg.backgrounds, cfg.output_dir, cfg.name,
        cfg.n_pseudo_experiments, cfg.seed, cfg.pool
    """

    # Allow plain dicts for convenience
    from types import SimpleNamespace
    if isinstance(cfg, dict):
        cfg = SimpleNamespace(**cfg)

    # ---- user-tunable ----------------------------------------------------
    PE_PER_SHARD = getattr(cfg, "pe_per_shard", 1000)
    SPLIT_DS_AB   = bool(getattr(cfg, "split_DS_AB", False))
    DS_FRACTION_A = float(getattr(cfg, "ds_fraction_A", 0.5))
    if not (0.0 <= DS_FRACTION_A <= 1.0):
        raise ValueError(f"ds_fraction_A must be in [0,1], got {DS_FRACTION_A}")
    # ---------------------------------------------------------------------

    out_root = Path(cfg.output_dir) / cfg.name
    out_root.mkdir(parents=True, exist_ok=True)

    rng_master = np.random.default_rng(cfg.seed)
 
    # --------- 1. Load template histograms, bin edges, and names ----------
    # New inputs are (N,) object arrays: each element i is a 1-D array for histogram i
    bkg_list, bins_list, name_list = [], [], []
    for entry in cfg.backgrounds.values():
        d = Path(entry["dir"])
        bkg_arr   = np.load(d / "background.npy", allow_pickle=True)  # (N,) object
        bins_arr  = np.load(d / "bin_edges.npy",  allow_pickle=True)  # (N,) object (centers or edges)
        names_arr = np.load(d / "names.npy",      allow_pickle=True)  # (N,) object or (N,) str

        # Ensure 1-D object arrays
        bkg_list.append(np.asarray(bkg_arr, dtype=object).reshape(-1))
        bins_list.append(np.asarray(bins_arr, dtype=object).reshape(-1))
        name_list.append(np.asarray(names_arr, dtype=object).reshape(-1))

    # Concatenate across multiple background dirs (still 1-D object arrays)
    bkg       = np.concatenate(bkg_list,  dtype=object)
    bin_edges = np.concatenate(bins_list, dtype=object)
    names     = np.concatenate(name_list, dtype=object)

    # --- sanity: number of histograms ---
    N_HIST = len(bkg)
    assert len(bin_edges) == N_HIST, f"bin_edges length {len(bin_edges)} != {N_HIST}"
    assert len(names)     == N_HIST, f"names length {len(names)} != {N_HIST}"

    # --- per-histogram validation + fix centers→edges if needed ---
    fixed_edges = [None] * N_HIST
    for i in range(N_HIST):
        h   = np.asarray(bkg[i], dtype=float)         # 1-D counts for hist i
        bed = np.asarray(bin_edges[i], dtype=float)   # 1-D centers or edges for hist i
        nb  = len(h)

        if len(bed) == nb + 1:
            # already true edges
            fixed_edges[i] = bed
        elif len(bed) == nb:
            # treat as centers; build edges by midpoint + extrapolated ends
            c = bed
            mids = 0.5 * (c[:-1] + c[1:])
            left  = c[0]  - 0.5 * (c[1]  - c[0])
            right = c[-1] + 0.5 * (c[-1] - c[-2])
            e = np.empty(nb + 1, dtype=c.dtype)
            e[0], e[1:-1], e[-1] = left, mids, right
            fixed_edges[i] = e
        else:
            raise ValueError(f"Histogram {i}: len(edges)={len(bed)} not in {{nbins, nbins+1}} with nbins={nb}")

    # Replace with a (N,) object array of true edges
    bin_edges = np.asarray(fixed_edges, dtype=object)

    # Normalize names to str (avoid np.str_ issues later)
    names = np.asarray([str(n) for n in names], dtype=object)

    print(f"Loaded {N_HIST} template histograms")

    # make globals for workers
    global BIN_EDGES_GLOBAL, NAMES_GLOBAL
    BIN_EDGES_GLOBAL = bin_edges
    NAMES_GLOBAL     = names

    # --- Save the background template to a file so all workers can access it efficiently ---
    # Workers are independent processes running in parallel to speed up toy generation.
    # Rather than passing this large array (possibly hundreds of MB) to each worker directly,
    # we save it once to disk. Workers will then load it from this file as needed,
    # which avoids copying the full data and reduces memory usage.
    mmap_path = out_root / "bkg_template.npy"
    np.save(mmap_path, bkg)    # This large (~200 MB) file will be read by workers

    # ---- Divide pseudo-experiments (PEs) into shards for multiprocessing --
    # Each shard gets its own set of random seeds and output directory
    N_PE   = cfg.n_pseudo_experiments
    n_shards = (N_PE + PE_PER_SHARD - 1)//PE_PER_SHARD
    tasks = []
    for shard in range(n_shards):
        start = shard * PE_PER_SHARD
        stop  = min(start + PE_PER_SHARD, N_PE)
        shard_seeds = rng_master.integers(0, 2**32, size=stop-start)
        shard_dir   = out_root / f"shard_{shard:04d}"
        shard_dir.mkdir(exist_ok=True)
        tasks.append((start, shard_seeds, shard_dir, mmap_path, SPLIT_DS_AB, DS_FRACTION_A))

    print(f"Generating {N_PE} PE in {n_shards} shards …")

    # ------------------------------------------------------------------------
    # Use Python's multiprocessing Pool to run multiple "workers" in parallel.
    # Each worker is like a helper doing part of the job — generating toy histograms
    # for one "shard" (a group of pseudo-experiments, e.g. 1000 at a time).
    #
    # This block sends all the work to the workers, and as results come back,
    # we update a live progress bar so users can track how many pseudo-experiments (PEs)
    # have been completed in real time.
    #
    # For example: if we're generating 2500 PEs, and each shard has 1000 PEs,
    # we will launch 3 shards. This loop will update the progress bar as each shard finishes.
    # ------------------------------------------------------------------------
    with Pool(processes=cfg.pool, initializer=_init_worker, initargs=(bkg,)) as pool:
        with tqdm(total=N_PE, ncols=80, desc="Generating PEs") as pbar:
            for start_idx in pool.imap_unordered(_worker_chunk, tasks):
                # Each start_idx means PE_PER_SHARD PEs were completed
                # But last shard might have fewer
                this_shard_count = min(PE_PER_SHARD, N_PE - start_idx)
                pbar.update(this_shard_count)

    # --------- Optional diagnostic plots: first 5 PEs × first 5 histograms --
    # Shows Poisson-fluctuated distributions with correct binning and labels
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator
    except ModuleNotFoundError:
        print("matplotlib not available – skipping example plots")
    else:
        ex_dir = out_root / "examples"
        ex_dir.mkdir(exist_ok=True)

        N_EX_PE    = min(5, N_PE)            # first 5 pseudo-experiments
        N_EX_HISTS = 5                       # first 5 histograms in each PE
        fontsize   = 11
        ticksize   = fontsize - 2

        for pe_idx in range(N_EX_PE):
            shard = pe_idx // PE_PER_SHARD

            if SPLIT_DS_AB:
                pe_file_A = out_root / f"shard_{shard:04d}" / f"PE_{pe_idx:07d}_A.npz"
                pe_file_B = out_root / f"shard_{shard:04d}" / f"PE_{pe_idx:07d}_B.npz"

                with np.load(pe_file_A, allow_pickle=True) as npzA, \
                     np.load(pe_file_B, allow_pickle=True) as npzB:
                    hA  = npzA["HIST"]
                    eA  = npzA["bin_edges"]
                    nms = npzA["names"].astype(str)
                    hB  = npzB["HIST"]
                    eB  = npzB["bin_edges"]

                for h_idx in range(min(N_EX_HISTS, len(hA))):
                    # A
                    hist  = hA[h_idx]
                    edges = eA[h_idx]
                    name  = nms[h_idx]
                    centers = 0.5 * (edges[1:] + edges[:-1])

                    plt.figure(figsize=(6, 4))
                    plt.step(centers, hist, where="mid")
                    if hist[0] > 500: plt.yscale("log")
                    plt.xlabel("mass [GeV]", fontsize=fontsize)
                    plt.ylabel("entries", fontsize=fontsize)
                    plt.title(f"PE {pe_idx:07d} – A – {name}", fontsize=ticksize)
                    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
                    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
                    plt.tick_params(axis='both', which='both', direction='in', labelsize=ticksize)
                    plt.tight_layout()
                    plt.savefig(ex_dir / f"PE_{pe_idx:07d}_H{h_idx:02d}_A.png", dpi=150)
                    plt.close()

                    # B
                    hist  = hB[h_idx]
                    edges = eB[h_idx]
                    centers = 0.5 * (edges[1:] + edges[:-1])

                    plt.figure(figsize=(6, 4))
                    plt.step(centers, hist, where="mid")
                    if hist[0] > 500: plt.yscale("log")
                    plt.xlabel("mass [GeV]", fontsize=fontsize)
                    plt.ylabel("entries", fontsize=fontsize)
                    plt.title(f"PE {pe_idx:07d} – B – {name}", fontsize=ticksize)
                    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
                    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
                    plt.tick_params(axis='both', which='both', direction='in', labelsize=ticksize)
                    plt.tight_layout()
                    plt.savefig(ex_dir / f"PE_{pe_idx:07d}_H{h_idx:02d}_B.png", dpi=150)
                    plt.close()

            else:
                pe_file = out_root / f"shard_{shard:04d}" / f"PE_{pe_idx:07d}.npz"
                with np.load(pe_file, allow_pickle=True) as npz:
                    hists  = npz["HIST"]
                    edgesA = npz["bin_edges"]
                    namesA = npz["names"].astype(str)

                for h_idx in range(min(N_EX_HISTS, len(hists))):
                    hist  = hists[h_idx]
                    edges = edgesA[h_idx]
                    name  = namesA[h_idx]
                    centers = 0.5 * (edges[1:] + edges[:-1])

                    plt.figure(figsize=(6, 4))
                    plt.step(centers, hist, where="mid")
                    if hist[0] > 500: plt.yscale("log")
                    plt.xlabel("mass [GeV]", fontsize=fontsize)
                    plt.ylabel("entries", fontsize=fontsize)
                    plt.title(f"PE {pe_idx:07d} – {name}", fontsize=ticksize)
                    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
                    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
                    plt.tick_params(axis='both', which='both', direction='in', labelsize=ticksize)
                    plt.tight_layout()
                    plt.savefig(ex_dir / f"PE_{pe_idx:07d}_H{h_idx:02d}.png", dpi=150)
                    plt.close()

    print(f"Finished: wrote {N_PE} pseudo-experiments " f"in {n_shards} shards under {out_root}")
