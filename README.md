# ATLAS Open Data Parser (concise)

Small toolkit to fetch ATLAS Open Data metadata, parse ROOT/DAOD files into awkward arrays, apply kinematic and particle-count filters, and save processed ROOT files or invariant-mass `.npy` arrays.

Quick start
1. Install dependencies (example):

```bash
pip install -r docker/requirements.txt
```

2. Configure `configs/pipeline_config.yaml` (set `release_years`, `output_path`, and limits for tests).

3. Run the pipeline:

```bash
python main_pipeline.py --config configs/pipeline_config.yaml
```

Main components
- `src/parse_atlas/parser.py`: `ATLAS_Parser` — fetch metadata, parse files (`parse_file`), yield memory-aware chunks (`parse_files`), and prepare ROOT-friendly output (`flatten_for_root`).
- `src/pipelines/parsing_pipeline.py`: threaded parse → filter → save flow.
- `src/pipelines/multiprocessing_pipeline.py`: subprocess-per-chunk mode for lower resident memory.
- `src/pipelines/im_pipeline.py`: calculate and save invariant masses from saved ROOT files.
- `src/calculations/physics_calcs.py`: filtering, grouping, and invariant-mass helpers.

Config notes
- Edit `configs/pipeline_config.yaml` for timeouts, file limits, `chunk_yield_threshold_bytes`, and `max_environment_memory_mb`.
- Enable `pipeline_config.parse_in_multiprocessing: true` to run the subprocess-per-chunk mode.

Implementation note
- Calls to `future.result(timeout=...)` raise `concurrent.futures.TimeoutError` (not the built-in `TimeoutError`). Code that counts timeouts should check for the futures exception class.

Need help?
- I can shorten further, add a top-level `requirements.txt`, or patch `ATLAS_Parser` to correctly count `concurrent.futures.TimeoutError`. Reply which you prefer.

