#!/usr/bin/env python3
"""
Rebuild a ROOT histogram to a target bin width and save PNG plots.

This works from the binned histogram content available in a ROOT file.
When the target binning is finer than the source binning, counts are
redistributed proportionally by bin overlap (uniform-within-bin assumption).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot


def _redistribute_by_overlap(
    source_values: np.ndarray,
    source_edges: np.ndarray,
    target_edges: np.ndarray,
) -> np.ndarray:
    """Redistribute source bin counts into target bins by interval overlap."""
    target_values = np.zeros(len(target_edges) - 1, dtype=float)

    for i, count in enumerate(source_values):
        left = float(source_edges[i])
        right = float(source_edges[i + 1])
        width = right - left
        if width <= 0 or count == 0:
            continue

        start = np.searchsorted(target_edges, left, side="right") - 1
        end = np.searchsorted(target_edges, right, side="left")
        start = max(0, start)
        end = min(len(target_values), end)

        for j in range(start, end):
            t_left = float(target_edges[j])
            t_right = float(target_edges[j + 1])
            overlap = max(0.0, min(right, t_right) - max(left, t_left))
            if overlap > 0:
                target_values[j] += count * (overlap / width)

    return target_values


def _plot_hist(edges: np.ndarray, values: np.ndarray, out_path: Path, title: str, xlim: tuple[float, float] | None = None) -> None:
    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = edges[1:] - edges[:-1]

    if xlim is not None:
        mask = (centers >= xlim[0]) & (centers <= xlim[1])
        centers = centers[mask]
        widths = widths[mask]
        values = values[mask]

    plt.figure(figsize=(9, 5))
    plt.bar(centers, values, width=widths, align="center", edgecolor="black")
    plt.xlabel("Mass [GeV]")
    plt.ylabel("Entries")
    plt.title(title)
    if xlim is not None:
        plt.xlim(*xlim)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild a histogram to a new bin width and save PNGs.")
    parser.add_argument("--root", required=True, help="Path to ROOT file (e.g. all_histograms.root)")
    parser.add_argument("--hist", required=True, help="Histogram key name (without ;cycle)")
    parser.add_argument("--bin-width", type=float, default=0.5, help="Target bin width in GeV (default: 0.5)")
    parser.add_argument("--zoom-min", type=float, default=100.0, help="Zoom lower bound in GeV (default: 100)")
    parser.add_argument("--zoom-max", type=float, default=200.0, help="Zoom upper bound in GeV (default: 200)")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: same as ROOT file)")
    args = parser.parse_args()

    root_path = Path(args.root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"ROOT file not found: {root_path}")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else root_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with uproot.open(root_path) as f:
        if args.hist not in f:
            available = [k.split(";")[0] for k in f.keys()]
            if args.hist not in available:
                raise KeyError(f"Histogram key '{args.hist}' not found in {root_path}")
        values, edges = f[args.hist].to_numpy()

    min_edge = float(edges[0])
    max_edge = float(edges[-1])
    target_edges = np.arange(min_edge, max_edge + args.bin_width, args.bin_width, dtype=float)
    if target_edges[-1] < max_edge:
        target_edges = np.append(target_edges, max_edge)
    else:
        target_edges[-1] = max_edge

    rebuilt = _redistribute_by_overlap(values.astype(float), edges.astype(float), target_edges)

    safe_name = args.hist.replace("/", "_")
    full_png = out_dir / f"{safe_name}_rebinned_{args.bin_width:g}GeV.png"
    zoom_png = out_dir / (
        f"{safe_name}_rebinned_{args.bin_width:g}GeV_zoom_"
        f"{int(args.zoom_min)}_{int(args.zoom_max)}GeV.png"
    )

    _plot_hist(target_edges, rebuilt, full_png, f"{args.hist} (rebinned to {args.bin_width:g} GeV)")
    _plot_hist(
        target_edges,
        rebuilt,
        zoom_png,
        f"{args.hist} (rebinned to {args.bin_width:g} GeV, zoom {args.zoom_min:g}-{args.zoom_max:g} GeV)",
        xlim=(args.zoom_min, args.zoom_max),
    )

    print(f"Saved full plot: {full_png}")
    print(f"Saved zoom plot: {zoom_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
