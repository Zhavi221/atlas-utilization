"""
Smoke test for MC event weighting — run this in the ANALYSIS environment
(where awkward, uproot, ROOT and atlasopenmagic are installed).

It exercises each layer of the weighting feature independently and reports
PASS / FAIL / SKIP per section, so a missing dependency degrades gracefully
instead of aborting the whole run. Nothing here writes into your real output
dirs; it uses tiny in-memory data and a temp directory.

Usage:
    python -m testing.smoke_test_mc_weighting                 # all component checks
    python -m testing.smoke_test_mc_weighting --dsid 410470   # live metadata for a specific DSID
    python -m testing.smoke_test_mc_weighting --luminosity 140.1

Exit code is non-zero if any executed (non-skipped) section FAILED.
"""

import argparse
import math
import os
import sys
import tempfile

# Make the repo importable when run as a plain script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS = []  # (section, status, detail)


def _record(section, status, detail=""):
    RESULTS.append((section, status, detail))
    print(f"[{status}] {section}" + (f"  - {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
# 1. Pure logic (no heavy deps) — must PASS everywhere
# --------------------------------------------------------------------------- #
def check_pure_logic():
    section = "1. Pure logic (DSID extraction, weight math, registry round-trip)"
    try:
        from domain.metadata import MCDatasetMetadata
        from services.calculations.mc_weights import compute_event_weight
        from services.calculations.weights_registry import (
            extract_dsid_from_url, WeightsRegistry, source_prefix_from_signature,
        )

        # DSID extraction
        assert extract_dsid_from_url("mc20_13TeV.410470.ttbar.DAOD_PHYSLITE.root") == 410470
        assert extract_dsid_from_url("parsed_2024r-pp_dsid700320_chunk0.root") == 700320
        assert extract_dsid_from_url("2024r-pp_deadbeef.root") is None  # no false positive

        # weight math
        md = MCDatasetMetadata(dataset_number=410470, cross_section_pb=729.77,
                               sum_of_weights=1.104e10, k_factor=1.13975, gen_filt_eff=1.0)
        w = compute_event_weight(md, 140.1)
        expected = 729.77 * 1000 * 1.13975 * 1.0 * 140.1 / 1.104e10
        assert math.isclose(w, expected, rel_tol=1e-12), (w, expected)

        # registry round-trip
        tmp = os.path.join(tempfile.gettempdir(), "smoke_registry.json")
        WeightsRegistry({"parsed_2024r-pp_dsid410470_chunk0": w}).save(tmp)
        r = WeightsRegistry.load(tmp)
        got = r.weight_for("parsed_2024r-pp_dsid410470_chunk0_FS_2j_IM_2j")
        assert math.isclose(got, w, rel_tol=1e-12), got
        os.remove(tmp)

        _record(section, "PASS", f"weight={w:.6g}")
    except ModuleNotFoundError as e:
        _record(section, "SKIP", f"dependency missing ({e.name}); run in the analysis env")
    except Exception as e:
        _record(section, "FAIL", f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# 2. Live metadata fetch (needs atlasopenmagic) + LO-vs-NLO check
# --------------------------------------------------------------------------- #
def check_live_metadata(dsid, luminosity):
    section = f"2. Live ATLAS metadata fetch (DSID {dsid})"
    try:
        import atlasopenmagic  # noqa: F401
    except Exception as e:
        _record(section, "SKIP", f"atlasopenmagic not available ({e})")
        return
    try:
        from services.metadata.fetcher import MetadataFetcher
        from services.calculations.mc_weights import compute_event_weight

        md = MetadataFetcher().fetch_mc_metadata(dsid)
        if md is None:
            _record(section, "FAIL", "fetch_mc_metadata returned None (missing required fields?)")
            return

        w = compute_event_weight(md, luminosity)
        print(f"      cross_section_pb={md.cross_section_pb}  kFactor={md.k_factor}  "
              f"genFiltEff={md.gen_filt_eff}  sumOfWeights={md.sum_of_weights:g}  nEvents={md.n_events}")
        print(f"      generator={md.generator}  physics_short={md.physics_short}")
        print(f"      -> event weight at L={luminosity} fb^-1: {w:.6g}")

        # LO-vs-NLO empirical check (task #3): ratio ~1 => unit-weight (LO); != 1 => weighted
        if md.n_events:
            ratio = md.sum_of_weights / md.n_events
            kind = "LO / unit-weight" if math.isclose(ratio, 1.0, rel_tol=0.02) else "weighted (NLO-like)"
            print(f"      sumOfWeights/nEvents = {ratio:.4f}  -> {kind}")
        _record(section, "PASS", f"weight={w:.6g}")
    except Exception as e:
        _record(section, "FAIL", f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# 3. Real accumulator with REAL awkward arrays (needs awkward)
# --------------------------------------------------------------------------- #
def check_accumulator_real_awkward():
    section = "3. EventAccumulator per-DSID chunking with real awkward arrays"
    try:
        import awkward as ak
    except Exception as e:
        _record(section, "SKIP", f"awkward not available ({e})")
        return
    try:
        from domain.events import EventBatch
        from services.parsing.event_accumulator import EventAccumulator

        def batch(url, nev, fid):
            events = ak.Array([{"pt": [float(i)]} for i in range(nev)])
            return EventBatch(events=events, file_id=fid, release_year="2024r-pp",
                              size_bytes=nev * 100, event_count=nev,
                              processing_time_sec=0.1, source_url=url)

        acc = EventAccumulator(chunk_threshold_bytes=10**9, split_by_dataset=True)
        files = [("mc20_13TeV.410470.tt._1.root", 3), ("mc20_13TeV.410470.tt._2.root", 2),
                 ("mc20_13TeV.700320.zee._1.root", 4)]
        chunks = []
        for i, (url, nev) in enumerate(files):
            c = acc.add_batch(batch(url, nev, i))
            if c:
                chunks.append(c)
        f = acc.flush()
        if f:
            chunks.append(f)

        total_in = sum(n for _, n in files)
        total_out = sum(len(c.events) for c in chunks)
        dsids = [c.dsid for c in chunks]

        assert len(chunks) == 2, f"expected 2 single-DSID chunks, got {len(chunks)}"
        assert set(dsids) == {410470, 700320}, f"chunk DSIDs wrong: {dsids}"
        assert total_out == total_in, f"event loss: {total_out} != {total_in}"
        # each chunk must be single-DSID (label matches its events' provenance)
        _record(section, "PASS", f"chunks={len(chunks)} dsids={dsids} events {total_out}/{total_in} conserved")
    except Exception as e:
        _record(section, "FAIL", f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# 4. ROOT weighted fill (needs ROOT)
# --------------------------------------------------------------------------- #
def check_root_weighted_fill():
    section = "4. ROOT weighted histogram fill + Sumw2 errors"
    try:
        import ROOT
    except Exception as e:
        _record(section, "SKIP", f"ROOT not available ({e})")
        return
    try:
        from services.pipelines.histograms_pipeline import _fill_hist

        vals = [100.0, 150.0, 200.0]
        weight = 0.0672

        h_unit = ROOT.TH1F("h_unit", "", 10, 0, 300)
        _fill_hist(h_unit, vals, 1.0)

        h_w = ROOT.TH1F("h_w", "", 10, 0, 300)
        h_w.Sumw2()
        _fill_hist(h_w, vals, weight)

        integral_unit = h_unit.Integral()
        integral_w = h_w.Integral()
        assert math.isclose(integral_unit, 3.0, rel_tol=1e-6), integral_unit
        assert math.isclose(integral_w, 3 * weight, rel_tol=1e-6), integral_w

        # a filled bin's weighted error should be weight * sqrt(n), not sqrt(n_weighted)
        b = h_w.FindBin(150.0)
        err = h_w.GetBinError(b)
        assert math.isclose(err, weight, rel_tol=1e-6), f"bin error {err} != {weight}"

        _record(section, "PASS",
                f"unweighted_integral={integral_unit:.4g}  weighted_integral={integral_w:.4g} (=3*{weight})")
    except Exception as e:
        _record(section, "FAIL", f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# 5. Full-pipeline instructions (manual — needs your data + config)
# --------------------------------------------------------------------------- #
def print_full_pipeline_hint(luminosity):
    print("\n" + "=" * 74)
    print("5. Full pipeline (run manually with a few real MC files):")
    print("=" * 74)
    print(f"""
  a) In config.yaml:  tasks.do_parsing: true
     mc_weighting_config: {{ enabled: true, target_luminosity_fb: {luminosity} }}
     parsing_task_config.max_files_to_process: 3   # keep it small
     (use MC release_years / record_ids so files carry DSIDs)

  b) Run parsing + mass-calc + histograms:
       python main.py

  c) Verify:
     - parsed chunk filenames contain _dsid<N>:
         ls <run_dir>/parsed_data/ | grep dsid
     - build the registry and check it has real (non-1.0) weights:
         python -m utils.build_weights_registry --config config.yaml \\
             --input-dir <run_dir>/im_arrays_processed
         cat <run_dir>/im_arrays_processed/weights_registry.json
     - a weighted histogram's Integral() differs from the unweighted run.
""")


def main(argv=None):
    parser = argparse.ArgumentParser(description="MC weighting smoke test")
    parser.add_argument("--dsid", type=int, default=410470, help="DSID for live metadata fetch")
    parser.add_argument("--luminosity", type=float, default=140.1, help="Target luminosity (fb^-1)")
    parser.add_argument("--skip-network", action="store_true", help="Skip the live metadata fetch")
    args = parser.parse_args(argv)

    print("=" * 74)
    print("MC WEIGHTING SMOKE TEST")
    print("=" * 74)

    check_pure_logic()
    if not args.skip_network:
        check_live_metadata(args.dsid, args.luminosity)
    else:
        _record(f"2. Live ATLAS metadata fetch (DSID {args.dsid})", "SKIP", "--skip-network")
    check_accumulator_real_awkward()
    check_root_weighted_fill()
    print_full_pipeline_hint(args.luminosity)

    print("=" * 74)
    print("SUMMARY")
    print("=" * 74)
    passed = sum(1 for _, s, _ in RESULTS if s == "PASS")
    failed = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    skipped = sum(1 for _, s, _ in RESULTS if s == "SKIP")
    for section, status, detail in RESULTS:
        print(f"  {status:4}  {section}")
    print(f"\n  {passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
