"""
predict_toys_bkg_only.py
------------------------
Run a trained Znet on background-only pseudo-experiments (PEs) and,
optionally, compute the likelihood-ratio significance Z_LR using a
smoothed MC background.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import torch
torch.set_num_threads(1) 
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.Znet3 import Znet
import sys, pathlib
sys.path.append( str(pathlib.Path(__file__).resolve().parents[1] / "sample_generation") )
from workspace import Workspace
from sample_generation import signals

def _ab_label_from_path(p: Path):
    s = p.stem  # e.g. "PE_0000714_A"
    parts = s.rsplit("_", 1)
    return parts[1] if len(parts) == 2 and parts[1] in {"A","B"} else None

def _pe_base_and_label(pe_path: Path) -> tuple[str, str | None]:
    """
    Return (base_id, label) from a PE filename.
    e.g. PE_0000714_A.npz -> ("PE_0000714", "A")
         PE_0000714_B.npz -> ("PE_0000714", "B")
         PE_0000714.npz   -> ("PE_0000714", None)
    """
    stem = pe_path.stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in {"A", "B"}:
        return parts[0], parts[1]
    return stem, None

# ---------------------------------------------------------------------------
# Lazy-loading helpers – keep model / smoothed library in globals so every
# worker doesn’t reload them for each task.
# ---------------------------------------------------------------------------
_MODEL = None
def _init_model(ckpt_path):
    """Executed once per worker; loads checkpoint onto CPU."""
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_model(ckpt_path).to(torch.device("cpu"))

_smooth_cache: dict[str, tuple[np.ndarray, np.ndarray]] | None = None

def _load_smooth_library(smooth_dir: str | Path | None
                         ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Build {hist_name → (counts, edges)} from *.npy in a directory:
      background.npy, bin_edges.npy, names.npy

    Accepts per-hist variable-length arrays (dtype=object). If bin_edges[i] has
    len == nbins it is treated as centers and converted to true edges (len == nbins+1).
    Results cached globally for reuse across PEs.
    """
    global _smooth_cache
    if _smooth_cache is not None:
        return _smooth_cache

    if not smooth_dir:
        _smooth_cache = {}
        return _smooth_cache

    d = Path(smooth_dir)
    bkg_arr   = np.load(d / "background.npy", allow_pickle=True)
    edges_arr = np.load(d / "bin_edges.npy",  allow_pickle=True)
    names_arr = np.load(d / "names.npy",      allow_pickle=True)

    # Ensure 1-D object arrays
    bkg_arr   = np.asarray(bkg_arr,   dtype=object).reshape(-1)
    edges_arr = np.asarray(edges_arr, dtype=object).reshape(-1)
    names_arr = np.asarray(names_arr, dtype=object).reshape(-1)

    if not (len(bkg_arr) == len(edges_arr) == len(names_arr)):
        raise ValueError(f"Smoothed library arrays length mismatch: "
                         f"{len(bkg_arr)} vs {len(edges_arr)} vs {len(names_arr)}")

    lib: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for i, name in enumerate(names_arr):
        counts = np.asarray(bkg_arr[i],   dtype=float)
        edges  = np.asarray(edges_arr[i], dtype=float)

        # strip non-finite tails if any
        counts = counts[np.isfinite(counts)]
        edges  = edges [np.isfinite(edges )]

        nb = len(counts)
        if len(edges) == nb + 1:
            edges_use = edges
        elif len(edges) == nb:
            # treat as centers → build edges
            c = edges
            mids  = 0.5 * (c[:-1] + c[1:])
            left  = c[0]  - 0.5 * (c[1]  - c[0])
            right = c[-1] + 0.5 * (c[-1] - c[-2])
            edges_use = np.concatenate(([left], mids, [right]))
        else:
            raise ValueError(f"{name}: len(edges)={len(edges)} not in {{nbins, nbins+1}} (nbins={nb})")

        lib[str(name)] = (counts, edges_use)

    _smooth_cache = lib
    return lib

def _load_model(checkpoint_path: str | Path) -> Znet:
    """Load a trained Znet3 model on CPU and return it in eval mode."""
    net = Znet()
    ckpt = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    state = {k.replace("net.", ""): v for k, v in ckpt["state_dict"].items()}
    net.load_state_dict(state)
    net.eval()
    return net

# ---------------------------------------------------------------------------
# Worker — processes one PE file
# ---------------------------------------------------------------------------
def _predict_single_pe_worker(args):
    pe_path, compute_zlr, smooth_dir, ds_fraction_A = args
    label = _ab_label_from_path(Path(pe_path))

    smoothed_lib = _load_smooth_library(smooth_dir) if compute_zlr else {}

    # ---------------- load PE -----------------
    with np.load(pe_path, allow_pickle=True) as npz:
        hists   = npz["HIST"].astype(object)      # len = N_HIST
        edges_A = npz["bin_edges"]
        names   = np.asarray(npz["names"]).reshape(-1).astype(str)

    N = len(hists)

    # -------- align / trim histograms ---
    hist_batch, edges_list, z_lr_list = [], [], []
    bkg_meta = []     # (counts, edges) or None per hist

    for hist, edges, name in zip(hists, edges_A, names):
        # ----- always keep full PE histogram for Znet -----
        hist_for_znet  = hist.astype(np.float32)
        edges_for_znet = edges

        # ----- if LR is requested, prepare LR-only aligned versions -----
        data_for_lr = None
        bkg_counts_scaled = None
        bkg_edges = None

        ref = smoothed_lib.get(name) if compute_zlr else None
        if ref is not None:
            bkg_counts, bkg_edges = ref
            L = len(bkg_counts)
            # LR sees the PE truncated to the template length:
            data_for_lr = hist[:L].astype(np.float32)

            # scale smoothed bkg for A/B halves (LR only)
            scale = 1.0
            if label in {"A","B"}:
                if ds_fraction_A is not None:
                    scale = ds_fraction_A if label == "A" else (1.0 - ds_fraction_A)
                else:
                    tot_b = bkg_counts.sum()
                    tot_d = float(data_for_lr.sum())
                    scale = (tot_d / tot_b) if tot_b > 0 else 1.0
            bkg_counts_scaled = bkg_counts * scale

        # collect Znet inputs
        hist_batch.append(hist_for_znet)
        edges_list.append(edges_for_znet)

        # collect LR metadata (may be Nones)
        bkg_meta.append((bkg_counts_scaled, bkg_edges, data_for_lr))

    # -------------- single Znet forward pass -------------------------

    from collections import defaultdict
    buckets = defaultdict(list)
    for i, h in enumerate(hist_batch):
        buckets[h.size].append(i)

    z_pred_list = [None]*len(hist_batch)

    with torch.no_grad():
        for L, idxs in buckets.items():
            x = torch.stack([torch.from_numpy(hist_batch[i]).float() for i in idxs])  # shape (N,L)
            z, _ = _MODEL(x)                     # z is already (N,L) with no padding
            z = z.numpy()
            for k, i in enumerate(idxs):
                z_pred_list[i] = z[k]

    # -------------- compute Z_LR where possible ----------------------
    for i, (bkg_c, bkg_e, data_lr) in enumerate(bkg_meta):
        # If we have no LR materials for this histogram, emit NaNs matching Znet length
        if bkg_c is None or data_lr is None:
            z_lr_list.append(np.full_like(hist_batch[i], np.nan, dtype=float))
            continue

        ws = Workspace()              # ← instance every time
        ws.sig_func = signals.gaussian
        ws.W_bins   = 1.0

        ws.bkg_hist  = bkg_c
        ws.bin_edges = bkg_e
        ws.update()

        ws.data = data_lr
        z_lr = ws.z_scan()            # length equals len(data_lr) (template length)

        # Optional: up-pad with NaNs so LR vector aligns to Znet vector element-wise
        znet_len = hist_batch[i].size
        if z_lr.size < znet_len:
            z_lr = np.pad(z_lr, (0, znet_len - z_lr.size), constant_values=np.nan)

        z_lr_list.append(z_lr)


    return pe_path, {
        "z":         np.array(z_pred_list, dtype=object), 
        "z_lr":      np.array(z_lr_list, dtype=object),
        "bin_edges": np.array(edges_list, dtype=object),
        "names":     names,
    }

# -----------------------------------------------------------------------------
# Plotting helper (called for the first few PEs)
# -----------------------------------------------------------------------------
def _plot_examples_from_disk(input_root: Path, output_root: Path, rel_path: Path, *,
                             max_examples: int = 5, show_zlr: bool = True):
    in_npz  = np.load(input_root  / rel_path, allow_pickle=True)
    out_npz = np.load(output_root / rel_path, allow_pickle=True)

    hists  = in_npz["HIST"].astype(object)
    edges  = in_npz["bin_edges"]
    names  = in_npz["names"].astype(str)
    z_pred = out_npz["z"]
    z_lr   = out_npz.get("z_lr")

    ex_dir = output_root / "examples"
    ex_dir.mkdir(exist_ok=True)
    stem = rel_path.stem

    for i in range(min(max_examples, len(hists))):
        hist, edg, name = hists[i], edges[i], names[i]
        centers = 0.5*(edg[1:]+edg[:-1])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7,5), sharex=True,
                                       gridspec_kw={"height_ratios":[3,1]})
        ax1.step(centers, hist, where="mid", color="black")
        if hist[0] > 500:
            ax1.set_yscale("log"); ax1.set_ylim(bottom=1)
        ax1.set_ylabel("Entries"); ax1.set_title(f"{stem} – {name}")

        ax2.step(centers, z_pred[i], where="mid", color="red", label="Z_pred")
        if show_zlr and z_lr is not None:
            ax2.step(centers, z_lr[i], where="mid", color="blue", label="Z_LR")
        ax2.set_xlabel("mass [GeV]"); ax2.set_ylabel("Z"); ax2.legend(frameon=False, fontsize=8, ncol=2)

        plt.tight_layout()
        (output_root / "examples").mkdir(exist_ok=True)
        plt.savefig(output_root / "examples" / f"{stem}_H{i:02d}.png", dpi=150)
        plt.close()


def _plot_examples_AB_from_disk(input_root: Path, output_root: Path, base_rel: Path, *,
                                max_examples: int = 5, show_zlr: bool = True):
    # base_rel like: shard_xxxx/PE_0000123
    relA = base_rel.with_name(base_rel.name + "_A.npz")
    relB = base_rel.with_name(base_rel.name + "_B.npz")

    inA  = np.load(input_root  / relA, allow_pickle=True)
    inB  = np.load(input_root  / relB, allow_pickle=True)
    outA = np.load(output_root / relA, allow_pickle=True)
    outB = np.load(output_root / relB, allow_pickle=True)

    hA, eA, nms = inA["HIST"].astype(object), inA["bin_edges"], inA["names"].astype(str)
    hB, eB      = inB["HIST"].astype(object), inB["bin_edges"]

    zA, zB = outA["z"], outB["z"]
    zlrA   = outA.get("z_lr"); zlrB = outB.get("z_lr")

    ex_dir = output_root / "examples"; ex_dir.mkdir(exist_ok=True)
    base_id = base_rel.name  # e.g. "PE_0000123"

    n = min(max_examples, len(hA), len(hB))
    for i in range(n):
        histA, histB = hA[i], hB[i]
        centA = 0.5*(eA[i][1:]+eA[i][:-1]); centB = 0.5*(eB[i][1:]+eB[i][:-1])
        name = nms[i]

        fig, axes = plt.subplots(2, 2, figsize=(10,6), sharex='col', gridspec_kw={"height_ratios":[3,1]})
        (axAt, axBt), (axAb, axBb) = axes

        axAt.step(centA, histA, where="mid", color="black")
        if histA[0] > 500: axAt.set_yscale("log"); axAt.set_ylim(bottom=1)
        axAt.set_ylabel("Entries"); axAt.set_title(f"{base_id}_A – {name}")

        axBt.step(centB, histB, where="mid", color="black")
        if histB[0] > 500: axBt.set_yscale("log"); axBt.set_ylim(bottom=1)
        axBt.set_ylabel("Entries"); axBt.set_title(f"{base_id}_B – {name}")

        axAb.step(centA, zA[i], where="mid", color="red", label="Z_pred")
        if show_zlr and zlrA is not None: axAb.step(centA, zlrA[i], where="mid", color="blue", label="Z_LR")
        axAb.set_xlabel("mass [GeV]"); axAb.set_ylabel("Z"); axAb.legend(frameon=False, fontsize=8)

        axBb.step(centB, zB[i], where="mid", color="red", label="Z_pred")
        if show_zlr and zlrB is not None: axBb.step(centB, zlrB[i], where="mid", color="blue", label="Z_LR")
        axBb.set_xlabel("mass [GeV]"); axBb.set_ylabel("Z"); axBb.legend(frameon=False, fontsize=8)

        plt.tight_layout()
        plt.savefig(ex_dir / f"{base_id}_H{i:02d}_AB.png", dpi=150)
        plt.close()


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------

def predict_toys_bkg_only(cfg: dict):
    """Run Znet + optional Z_LR on a set of PE files."""

    # ------------------------------------------------------------------
    # config parsing & defaults
    # ------------------------------------------------------------------
    checkpoint_path   = cfg["checkpoint"]
    input_dir         = Path(cfg["input_dir"])
    output_dir        = Path(cfg["output_dir"])

    compute_zlr = bool(cfg.get("compute_Z_LR", True))

    smooth_dir = cfg.get("smooth_dir")

    max_pe_files = cfg.get("max_pe")
    n_threads    = cfg.get("pool", 4)
    ds_fraction_A = cfg.get("ds_fraction_A")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # build smoothed‑background dictionary
    # ------------------------------------------------------------------
    smoothed_dict = {}
    if compute_zlr:
        n_lib = len(_load_smooth_library(smooth_dir))
        print(f"Loaded {n_lib} smoothed background histograms from {smooth_dir}")

    # ------------------------------------------------------------------
    # gather PE files
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # gather PE files  (interpret max_pe as "number of A+B PAIRS")
    # ------------------------------------------------------------------
    all_files = sorted(input_dir.rglob("PE_*.npz"))
    if not all_files:
        raise RuntimeError(f"No PE_*.npz files found under {input_dir}")

    # index by (base, label) and by base → labels
    from collections import defaultdict
    by_key = {}  # (base, label) -> Path
    labels_by_base: dict[str, set] = defaultdict(set)

    def _stem_parts(p: Path):
        # "PE_0000714_A" -> ("PE_0000714", "A"),  "PE_0000714" -> ("PE_0000714", None)
        stem = p.stem
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in {"A", "B"}:
            return parts[0], parts[1]
        return stem, None

    for p in all_files:
        base, lab = _stem_parts(p)
        labels_by_base[base].add(lab)
        by_key[(base, lab)] = p

    # split case: only count complete A+B bases as eligible "pairs"
    paired_bases = sorted([b for b, labs in labels_by_base.items() if labs == {"A", "B"}])

    # unsplit bases (no A/B label) — treated as single-file "pairs" only if no split files exist
    unsplit_bases = sorted([b for b, labs in labels_by_base.items() if labs == {None}])

    pe_files: list[Path] = []
    total_pairs = 0

    if paired_bases:
        # Apply max_pe to PAIRS (bases with A and B)
        max_pairs = int(cfg.get("max_pe")) if cfg.get("max_pe") is not None else None
        selected_bases = paired_bases[:max_pairs] if max_pairs is not None else paired_bases
        for b in selected_bases:
            pe_files.append(by_key[(b, "A")])
            pe_files.append(by_key[(b, "B")])
        total_pairs = len(selected_bases)
        # Limit example pairing to the same selected bases
        paired_bases = set(selected_bases)
    else:
        # No A/B files present: fall back to file-level cap (legacy behavior)
        max_files = int(cfg.get("max_pe")) if cfg.get("max_pe") is not None else None
        selected_bases = unsplit_bases[:max_files] if max_files is not None else unsplit_bases
        for b in selected_bases:
            pe_files.append(by_key[(b, None)])
        total_pairs = len(selected_bases)  # "pairs" == files in unsplit case

    n_files = len(pe_files)
    if n_files == 0:
        raise RuntimeError(f"No eligible PE files found under {input_dir}")

    print(f"[predict] Selected {total_pairs} "
          f"{'A+B pairs' if paired_bases else 'unsplit PEs'} "
          f"→ {n_files} files to process.")

    # ------------------------------------------------------------------
    # set up thread pool
    # ------------------------------------------------------------------
    args_iterable = [(pe_file, compute_zlr, smooth_dir, ds_fraction_A) for pe_file in pe_files]
    examples_done = 0
    EXAMPLES_LIMIT = 5

    with Pool(processes=n_threads,
            initializer=_init_model,
            initargs=(checkpoint_path,),
            maxtasksperchild=200) as pool:  # recycle workers as a safety valve
        with tqdm(total=n_files,
                  desc=(f"Predicting PEs (A+B pairs: {total_pairs})" if total_pairs else "Predicting PEs"),
                  ncols=80) as pbar:
            for pe_path, result in pool.imap_unordered(
                    _predict_single_pe_worker,
                    args_iterable,  # <-- use the 4-tuple iterable you already built
                    chunksize=4):

                rel_path = pe_path.relative_to(input_dir)
                out_path = output_dir / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)

                to_save = {k: result[k] for k in ["z", "z_lr", "bin_edges", "names"] if k in result}
                np.savez_compressed(out_path, **to_save)

                # ---- examples (A/B only) WITHOUT buffering results in memory
                if examples_done < EXAMPLES_LIMIT:
                    base_id, label = _pe_base_and_label(pe_path)
                    if label in {"A", "B"} and base_id in paired_bases:
                        other_label = "B" if label == "A" else "A"
                        # Build sibling relative path in the same shard dir
                        sibling_rel = rel_path.with_name(f"{base_id}_{other_label}.npz")
                        sibling_out = output_dir / sibling_rel
                        sibling_in  = input_dir  / sibling_rel
                        if sibling_out.exists():  # both halves processed → plot once
                            _plot_examples_AB_from_disk(
                                input_root=input_dir,
                                output_root=output_dir,
                                base_rel=rel_path.with_name(base_id)  # e.g., shard_xxxx/PE_0000123
                                , show_zlr=compute_zlr
                            )
                            examples_done += 1
                    else:
                        # unsplit case: quick single-file example from disk
                        _plot_examples_from_disk(
                            input_root=input_dir,
                            output_root=output_dir,
                            rel_path=rel_path,
                            show_zlr=compute_zlr
                        )
                        examples_done += 1

                # free per-iteration objects aggressively
                del result
                pbar.update(1)


    print(f"Finished: wrote predictions for {n_files} PE files → {output_dir}")
