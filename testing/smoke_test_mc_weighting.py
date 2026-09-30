"""
Smoke test for MC event weighting — run in the analysis environment
(awkward, uproot, ROOT; atlasopenmagic for the optional live section).

Each section exercises one layer of the weighting feature independently and
reports PASS / FAIL / SKIP, so a missing dependency degrades gracefully
instead of aborting the whole run. Everything uses tiny in-memory data and a
temp directory; nothing touches real output dirs.

Usage:
    python -m testing.smoke_test_mc_weighting                 # all component checks
    python -m testing.smoke_test_mc_weighting --dsid 700320   # live metadata for a specific DSID
    python -m testing.smoke_test_mc_weighting --skip-network

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
    section = "1. Pure logic (DSID extraction, weight math)"
    try:
        from domain.metadata import MCDatasetMetadata
        from services.calculations.mc_weights import compute_event_weight, compute_normalization
        from services.parsing.dsid import extract_dsid_from_url

        # DSID extraction: strict, no false positives on container / sequence numbers
        assert extract_dsid_from_url("mc20_13TeV.410470.ttbar.DAOD_PHYSLITE.root") == 410470
        assert extract_dsid_from_url("parsed_2024r-pp_mc_dsid700320_chunk0.root") == 700320
        assert extract_dsid_from_url("2024r-pp_deadbeef.root") is None
        assert extract_dsid_from_url(
            "root://eospublic.cern.ch:1094//eos/opendata/atlas/rucio/mc20_13TeV/"
            "DAOD_PHYSLITE.37621257._000071.pool.root.1"
        ) is None

        # weight math
        md = MCDatasetMetadata(dataset_number=410470, cross_section_pb=729.77,
                               sum_of_weights=1.104e10, k_factor=1.13975, gen_filt_eff=1.0)
        expected = 729.77 * 1000 * 1.13975 * 1.0 * 140.1 / 1.104e10
        assert math.isclose(compute_normalization(md, 140.1), expected, rel_tol=1e-12)
        assert math.isclose(compute_event_weight(md, 140.1, -0.5), -0.5 * expected, rel_tol=1e-12)

        _record(section, "PASS", f"w_norm={expected:.6g}")
    except ModuleNotFoundError as e:
        _record(section, "SKIP", f"dependency missing ({e.name}); run in the analysis env")
    except Exception as e:
        _record(section, "FAIL", f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# 2. Live metadata fetch (needs atlasopenmagic + network)
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
        from services.calculations.mc_weights import compute_normalization

        md = MetadataFetcher().fetch_mc_metadata(dsid)
        if md is None:
            _record(section, "FAIL", "fetch_mc_metadata returned None (missing required fields?)")
            return

        w = compute_normalization(md, luminosity)
        print(f"      cross_section_pb={md.cross_section_pb}  kFactor={md.k_factor}  "
              f"genFiltEff={md.gen_filt_eff}  sumOfWeights={md.sum_of_weights:g}  nEvents={md.n_events}")
        print(f"      generator={md.generator}  physics_short={md.physics_short}")
        print(f"      -> w_norm at L={luminosity} fb^-1: {w:.6g}")
        if md.n_events:
            ratio = md.sum_of_weights / md.n_events
            kind = "LO / unit-weight" if math.isclose(ratio, 1.0, rel_tol=0.02) else "weighted (NLO-like)"
            print(f"      sumOfWeights/nEvents = {ratio:.4g}  -> {kind}")
        _record(section, "PASS", f"w_norm={w:.6g}")
    except Exception as e:
        _record(section, "FAIL", f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# 3. Accumulator per-DSID chunking with real awkward arrays
# --------------------------------------------------------------------------- #
def check_accumulator_real_awkward():
    section = "3. EventAccumulator per-DSID chunking"
    try:
        import awkward as ak
    except Exception as e:
        _record(section, "SKIP", f"awkward not available ({e})")
        return
    try:
        from domain.events import EventBatch
        from services.parsing.event_accumulator import EventAccumulator

        def batch(dsid, nev, fid):
            events = ak.Array([{"Jets": [{"pt": float(i)}]} for i in range(nev)])
            return EventBatch(events=events, file_id=fid, release_year="2024r-pp",
                              size_bytes=nev * 100, event_count=nev,
                              processing_time_sec=0.1, file_url=f"file{fid}.root", dsid=dsid)

        acc = EventAccumulator(chunk_threshold_bytes=10**9, split_by_dataset=True)
        files = [(410470, 3), (410470, 2), (700320, 4)]
        chunks = []
        for i, (dsid, nev) in enumerate(files):
            c = acc.add_batch(batch(dsid, nev, i))
            if c:
                chunks.append(c)
        f = acc.flush()
        if f:
            chunks.append(f)

        total_in = sum(n for _, n in files)
        total_out = sum(len(c.events) for c in chunks)
        dsids = [c.dsid for c in chunks]

        assert len(chunks) == 2, f"expected 2 single-DSID chunks, got {len(chunks)}"
        assert dsids == [410470, 700320], f"chunk DSIDs wrong: {dsids}"
        assert total_out == total_in, f"event loss: {total_out} != {total_in}"

        # Disabled: size-only chunking, no DSID label
        acc = EventAccumulator(chunk_threshold_bytes=10**9, split_by_dataset=False)
        for i, (dsid, nev) in enumerate(files):
            assert acc.add_batch(batch(dsid, nev, i)) is None
        only = acc.flush()
        assert len(only.events) == total_in and only.dsid is None

        _record(section, "PASS", f"chunks={len(chunks)} dsids={dsids} events {total_out}/{total_in} conserved")
    except Exception as e:
        _record(section, "FAIL", f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# 4. Parsed-file round trip + mass calc + post-processing + weighted fill
# --------------------------------------------------------------------------- #
def check_end_to_end_synthetic():
    section = "4. Synthetic chunk -> ROOT -> mass calc -> post-processing -> weighted histograms"
    try:
        import awkward as ak
        import numpy as np
        import uproot
        import ROOT
    except Exception as e:
        _record(section, "SKIP", f"dependency not available ({e})")
        return
    try:
        import logging
        from domain.events import (
            EventBatch, EventChunk, MC_EVENT_INFO_FIELD, MC_EVENT_WEIGHT_FIELD, MC_CHANNEL_NUMBER_FIELD,
        )
        from orchestration.handlers.mass_calculation_handler import MassCalculationHandler
        from services.calculations.im_calculator import IMCalculator
        from services.pipelines.im_pipeline import process_final_state, MC_WEIGHT_SUFFIX
        from services.pipelines.post_processing_pipeline import _process_im_sqlite
        from services.pipelines.histograms_pipeline import _create_histograms_from_sqlite
        from services.storage.sqlite_shards import SqliteArrayShardWriter, list_signatures, iter_arrays_for_signature

        logger = logging.getLogger("smoke")
        rng = np.random.default_rng(7)
        n = 400
        # Two electrons per event with a broad di-electron mass spectrum; two datasets.
        pt = rng.uniform(30e3, 300e3, size=(n, 2))
        eta = rng.uniform(-2.0, 2.0, size=(n, 2))
        phi = rng.uniform(-np.pi, np.pi, size=(n, 2))
        gen_w = rng.choice([1.0, -1.0], size=n) * rng.uniform(0.5, 1.5, size=n)
        dsids = np.where(np.arange(n) < n // 2, 700320, 410470).astype(np.int64)
        events = ak.zip({
            "Electrons": ak.from_regular(ak.zip({
                "pt": ak.Array(pt), "eta": ak.Array(eta), "phi": ak.Array(phi),
                "mass": ak.Array(np.full((n, 2), 0.511)),
            }), axis=1),
            MC_EVENT_INFO_FIELD: ak.zip({
                MC_EVENT_WEIGHT_FIELD: ak.Array(gen_w),
                MC_CHANNEL_NUMBER_FIELD: ak.Array(dsids),
            }),
        }, depth_limit=1)

        with tempfile.TemporaryDirectory() as tmp:
            # --- parsed-file round trip, as ParsingHandler writes it
            batch = EventBatch(events=events, file_id=1, release_year="2024r-pp_mc", size_bytes=1,
                               event_count=n, processing_time_sec=0.0, file_url="x", dsid=None)
            chunk = EventChunk.from_batches([batch], 0, "2024r-pp_mc")
            parsed = os.path.join(tmp, "parsed_2024r-pp_mc_chunk0.root")
            with uproot.recreate(parsed) as f:
                f["events"] = {field: chunk.events[field] for field in chunk.events.fields}
            with uproot.open(parsed) as f:
                arrays = MassCalculationHandler._reconstruct_particle_arrays(f["events"])
            assert MC_EVENT_INFO_FIELD in arrays.fields, arrays.fields
            np.testing.assert_allclose(ak.to_numpy(arrays[MC_EVENT_INFO_FIELD][MC_EVENT_WEIGHT_FIELD]), gen_w)
            np.testing.assert_array_equal(ak.to_numpy(arrays[MC_EVENT_INFO_FIELD][MC_CHANNEL_NUMBER_FIELD]), dsids)

            # --- mass calculation with per-DSID normalization folded in
            norm = {700320: 0.25, 410470: 4.0}
            im_dir = os.path.join(tmp, "im")
            writer = SqliteArrayShardWriter(os.path.join(im_dir, "im_batch_1.sqlite"))
            calc = IMCalculator(arrays, min_events_per_fs=1, min_k=1, max_k=4, min_n=1, max_n=4)
            fs_list = list(calc.group_by_final_state())
            assert fs_list == ["2e_0m_0j_0g_0t_0b"], fs_list
            fs_events = calc.get_events_for_final_state(fs_list[0])
            config = {"field_to_slice_by": "pt", "fs_chunk_threshold_bytes": 10**9,
                      "output_mode": "sqlite", "sqlite_writer": writer,
                      "mc_weighting_enabled": True, "mc_norm_by_dsid": norm, "mc_norm_default": 1.0}
            process_final_state(fs_list[0], fs_events, "parsed_2024r-pp_mc_chunk0.root",
                                [{"Electrons": (2, 0)}], config, im_dir, logger, calc)
            writer.close()
            sigs = list_signatures(os.path.join(im_dir, "im_batch_1.sqlite"))
            im_sig = [s for s in sigs if s.endswith("_IM_e0e1")]
            assert im_sig and im_sig[0] + MC_WEIGHT_SUFFIX in sigs, sigs
            masses = np.concatenate(list(iter_arrays_for_signature(os.path.join(im_dir, "im_batch_1.sqlite"), im_sig[0])))
            weights = np.concatenate(list(iter_arrays_for_signature(os.path.join(im_dir, "im_batch_1.sqlite"), im_sig[0] + MC_WEIGHT_SUFFIX)))
            expected_w = gen_w * np.where(dsids == 700320, 0.25, 4.0)
            np.testing.assert_allclose(weights, expected_w)
            assert len(masses) == n

            # Disabled: same masses, no weight arrays at all (inert)
            off_dir = os.path.join(tmp, "im_off")
            off_writer = SqliteArrayShardWriter(os.path.join(off_dir, "im_batch_1.sqlite"))
            off_config = dict(config, sqlite_writer=off_writer, mc_weighting_enabled=False)
            process_final_state(fs_list[0], fs_events, "parsed_2024r-pp_mc_chunk0.root",
                                [{"Electrons": (2, 0)}], off_config, off_dir, logger, calc)
            off_writer.close()
            off_sigs = list_signatures(os.path.join(off_dir, "im_batch_1.sqlite"))
            assert off_sigs == im_sig, off_sigs
            np.testing.assert_array_equal(
                np.concatenate(list(iter_arrays_for_signature(os.path.join(off_dir, "im_batch_1.sqlite"), im_sig[0]))), masses)

            # --- post-processing keeps masses and weights aligned
            proc_dir = os.path.join(tmp, "proc")
            _process_im_sqlite({"input_dir": im_dir, "output_dir": proc_dir,
                                "peak_detection_bin_width_gev": 10.0, "z_peak_cutoff": 0.0,
                                "max_mass_cutoff": 0.0, "min_events_per_fs": 0, "batch_job_index": 1},
                               ["im_batch_1.sqlite"], logger)
            proc_db = os.path.join(proc_dir, "processed_batch_1.sqlite")
            proc_sigs = list_signatures(proc_db)
            assert not any(s.endswith("_mcw_main") or s.endswith("_mcw_outliers") for s in proc_sigs), proc_sigs
            kept_mass, kept_w = [], []
            for suffix in ("_main", "_outliers"):
                for s in proc_sigs:
                    if s.endswith(suffix) and not s.endswith(MC_WEIGHT_SUFFIX):
                        kept_mass.extend(np.concatenate(list(iter_arrays_for_signature(proc_db, s))))
                        kept_w.extend(np.concatenate(list(iter_arrays_for_signature(proc_db, s + MC_WEIGHT_SUFFIX))))
            kept_mass, kept_w = np.array(kept_mass), np.array(kept_w)
            assert 0 < len(kept_mass) <= n
            # each surviving mass must still carry its own weight
            lookup = {round(float(m), 9): float(w) for m, w in zip(masses, expected_w)}
            np.testing.assert_allclose(kept_w, [lookup[round(float(m), 9)] for m in kept_mass])

            # --- weighted histogram fill with Sumw2
            hist_dir = os.path.join(tmp, "hists")
            _create_histograms_from_sqlite(["processed_batch_1.sqlite"], proc_dir, {
                "output_dir": hist_dir, "bin_width_gev": 10.0, "use_bumpnet_naming": True,
                "exclude_outliers": False, "single_output_file": True, "output_filename": "h.root",
            }, logger)
            f = ROOT.TFile(os.path.join(hist_dir, "h.root"))
            hists = [k.ReadObj() for k in f.GetListOfKeys()]
            assert len(hists) == 1, [h.GetName() for h in hists]
            h = hists[0]
            # TH1F keeps float32 bin contents, hence the loose tolerance.
            integral = h.Integral(0, h.GetNbinsX() + 1)
            assert math.isclose(integral, float(kept_w.sum()), rel_tol=1e-5), (integral, kept_w.sum())
            sumw2 = sum(h.GetBinError(b) ** 2 for b in range(0, h.GetNbinsX() + 2))
            assert math.isclose(sumw2, float((kept_w ** 2).sum()), rel_tol=1e-5), (sumw2, (kept_w ** 2).sum())
            f.Close()

        _record(section, "PASS",
                f"sum(w)={kept_w.sum():.6g} == Integral, sum(w^2)={float((kept_w ** 2).sum()):.6g} == sum(err^2)")
    except Exception as e:
        _record(section, "FAIL", f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# 5. Masses and their _mcw weights are flushed together, whatever the threshold
# --------------------------------------------------------------------------- #
def check_flush_alignment():
    section = "5. IM / _mcw chunks stay aligned across threshold flushes"
    try:
        import logging
        import awkward as ak
        import numpy as np
        from services.pipelines.im_pipeline import (
            MC_WEIGHT_SUFFIX, _accumulate_invariant_mass, _save_remaining_accumulated_data,
        )
        from services.storage.sqlite_shards import SqliteArrayShardWriter, iter_arrays_for_signature
    except Exception as e:
        _record(section, "SKIP", f"dependency not available ({e})")
        return
    try:
        logger = logging.getLogger("smoke_flush")
        sig = "f_FS_2e_IM_2e"
        batch_sizes = (100, 100, 100, 100)
        # Thresholds chosen so a flush trips at every point of the add sequence,
        # including right after a batch's masses were added.
        thresholds = (1500, 2000, 2500, 3500, 4000, 5000, 6000)
        with tempfile.TemporaryDirectory() as tmp:
            for threshold in thresholds:
                db = os.path.join(tmp, f"im_{threshold}.sqlite")
                writer = SqliteArrayShardWriter(db)
                mapping = {}
                for n in batch_sizes:
                    _accumulate_invariant_mass(
                        mapping, "2e", sig, ak.Array(np.random.rand(n) * 100),
                        threshold, tmp, logger, output_mode="sqlite", sqlite_writer=writer,
                        mc_weights=ak.Array(np.full(n, 0.5)),
                    )
                _save_remaining_accumulated_data(mapping, tmp, logger, output_mode="sqlite", sqlite_writer=writer)
                writer.commit()
                writer.close()

                im_sizes = [len(a) for a in iter_arrays_for_signature(db, sig)]
                w_sizes = [len(a) for a in iter_arrays_for_signature(db, sig + MC_WEIGHT_SUFFIX)]
                assert sum(im_sizes) == sum(batch_sizes), f"threshold={threshold}: lost masses {im_sizes}"
                assert im_sizes == w_sizes, f"threshold={threshold}: masses {im_sizes} vs weights {w_sizes}"

        _record(section, "PASS", f"{len(thresholds)} thresholds, chunk sizes identical")
    except Exception as e:
        _record(section, "FAIL", f"{type(e).__name__}: {e}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="MC weighting smoke test")
    parser.add_argument("--dsid", type=int, default=700320, help="DSID for live metadata fetch")
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
    check_end_to_end_synthetic()
    check_flush_alignment()

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
