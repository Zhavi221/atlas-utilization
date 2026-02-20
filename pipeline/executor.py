"""
PipelineExecutor - High-level pipeline orchestrator.

Wires together all services and executes the state machine.
Supports consolidated plot generation from output data.

Architecture (multi-job mode):
  Each batch job runs the full pipeline and saves:
    - batch_N_stats.json   → logs/
    - batch_N.root         → histograms/
  The merge job (--merge-only) combines everything via:
    - hadd histograms/batch_*.root → histograms/<output_filename>
    - JSON stats aggregation
    - Plot generation
"""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

from domain.config import PipelineConfig
from orchestration import PipelineState, PipelineContext, StateMachine
from orchestration.handlers import (
    FetchMetadataHandler,
    ParsingHandler,
    MassCalculationHandler,
    PostProcessingHandler,
    HistogramCreationHandler,
)
from services.metadata.fetcher import MetadataFetcher
from services.metadata.cache import MetadataCache
from services.parsing.file_parser import FileParser
from services.parsing.event_accumulator import EventAccumulator
from services.parsing.threaded_processor import ThreadedFileProcessor
from services.analysis.statistics_plotter import StatisticsPlotter


class PipelineExecutor:
    """
    High-level pipeline executor.
    
    Responsible for:
    1. Creating all services with dependency injection
    2. Building the state machine with handlers
    3. Running the pipeline
    4. Returning results
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.state_machine = self._build_state_machine()
    
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> PipelineContext:
        """Execute the pipeline and return final context."""
        self.logger.info("Initializing pipeline execution")
        initial_context = self._create_initial_context()
        final_context = self.state_machine.run(initial_context)
        self._log_results(final_context)
        return final_context
    
    def generate_plots_from_output(self, run_dir: str):
        """
        Generate consolidated statistical plots by reading all output data
        in the run directory.  Works for single-job and merged multi-job output.

        Args:
            run_dir: Path to the run directory containing parsed_data/, etc.
        """
        self.logger.info(f"Generating consolidated plots from: {run_dir}")

        plots_dir = os.path.join(run_dir, "plots")
        plotter = StatisticsPlotter(plots_dir)

        pipeline_stats = self._collect_stats_from_output(run_dir)

        created_plots = plotter.create_all_plots(pipeline_stats)

        if created_plots:
            self.logger.info(f"Created {len(created_plots)} statistical plots:")
            for p in created_plots:
                self.logger.info(f"  - {p}")
        else:
            self.logger.info("No plots generated (insufficient data)")

    def save_batch_stats(self, run_dir: str, batch_index: int, context: PipelineContext):
        """
        Save per-batch statistics JSON so the merge job can aggregate them.

        Saved to: <run_dir>/logs/batch_<N>_stats.json

        Args:
            run_dir:     Run directory path
            batch_index: 1-based batch index
            context:     Final pipeline context after execution
        """
        stats = {
            "batch_index": batch_index,
            "summary": context.get_summary(),
            "parsed_files": context.parsed_files,
            "im_files": context.im_files,
            "processed_files": context.processed_files,
        }

        # Add parsing stats if available
        if context.parsing_stats:
            stats["parsing"] = context.parsing_stats.to_dict()

        # Add custom data (mass_calc_stats, etc.)
        for key, value in context.custom_data.items():
            stats[key] = value

        logs_dir = os.path.join(run_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        stats_path = os.path.join(logs_dir, f"batch_{batch_index}_stats.json")

        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)

        self.logger.info(f"Saved batch stats to: {stats_path}")

    def merge_outputs(self, run_dir: str):
        """
        Merge outputs from all batch jobs:
          1. hadd batch histogram ROOT files → single output file
          2. Aggregate stats from batch JSON files
          3. Generate plots

        Args:
            run_dir: Run directory path
        """
        self.logger.info(f"=== Merging outputs from: {run_dir} ===")

        hist_dir = os.path.join(run_dir, "histograms")
        logs_dir = os.path.join(run_dir, "logs")

        # ---- 1. hadd histogram files ----
        batch_roots = sorted(Path(hist_dir).glob("batch_*.root"))
        if batch_roots:
            # Determine merged output filename from config
            hc = self.config.histogram_creation_config
            merged_name = hc.output_filename if hc else "atlas_opendata.root"
            merged_path = os.path.join(hist_dir, merged_name)

            self.logger.info(
                f"Merging {len(batch_roots)} batch histogram files → {merged_name}"
            )

            # Use ROOT's hadd to combine histogram files
            hadd_cmd = ["hadd", "-f", merged_path] + [str(p) for p in batch_roots]
            try:
                result = subprocess.run(
                    hadd_cmd, capture_output=True, text=True, timeout=300
                )
                if result.returncode == 0:
                    self.logger.info(f"hadd succeeded: {merged_path}")
                    # Clean up batch files
                    for bf in batch_roots:
                        bf.unlink()
                        self.logger.debug(f"Removed batch file: {bf}")
                else:
                    self.logger.error(f"hadd failed (rc={result.returncode}): {result.stderr}")
            except FileNotFoundError:
                self.logger.warning(
                    "hadd not found – cannot merge histogram files.  "
                    "Make sure ROOT is in PATH (lsetup)."
                )
            except subprocess.TimeoutExpired:
                self.logger.error("hadd timed out after 300s")
        else:
            self.logger.info("No batch_*.root files found in histograms/ – skipping hadd")

        # ---- 2. Aggregate batch stats ----
        batch_stats_files = sorted(Path(logs_dir).glob("batch_*_stats.json"))
        if batch_stats_files:
            self.logger.info(f"Aggregating stats from {len(batch_stats_files)} batch files")
            aggregated = self._aggregate_batch_stats(batch_stats_files)
            agg_path = os.path.join(logs_dir, "aggregated_stats.json")
            with open(agg_path, "w") as f:
                json.dump(aggregated, f, indent=2, default=str)
            self.logger.info(f"Aggregated stats saved to: {agg_path}")

        # ---- 3. Generate plots ----
        self.generate_plots_from_output(run_dir)

    @staticmethod
    def _aggregate_batch_stats(stats_files: list[Path]) -> dict:
        """
        Aggregate per-batch stats JSON files into a single summary dict.
        """
        batch_stats = []
        for sf in stats_files:
            with open(sf) as f:
                batch_stats.append(json.load(f))

        aggregated = {
            "num_batches": len(batch_stats),
            "batches": [s.get("batch_index") for s in batch_stats],
            "total_parsed_files": sum(len(s.get("parsed_files", [])) for s in batch_stats),
            "total_im_files": sum(len(s.get("im_files", [])) for s in batch_stats),
            "total_processed_files": sum(len(s.get("processed_files", [])) for s in batch_stats),
        }

        # Aggregate parsing stats
        parsing_sums = {}
        for s in batch_stats:
            p = s.get("parsing", {})
            for key in ("total_events", "successful_files", "failed_files",
                        "total_chunks", "total_size_mb"):
                if key in p:
                    parsing_sums[key] = parsing_sums.get(key, 0) + p[key]
        if parsing_sums:
            aggregated["parsing_totals"] = parsing_sums

        # Aggregate timing
        total_time = 0
        for s in batch_stats:
            summary = s.get("summary", {})
            total_time += summary.get("elapsed_time_sec", 0)
        aggregated["total_batch_time_sec"] = total_time

        return aggregated

    # ------------------------------------------------------------------
    # Stats collection from on-disk output
    # ------------------------------------------------------------------

    def _collect_stats_from_output(self, run_dir: str) -> dict:
        """
        Scan all output directories and build a consolidated stats dict
        for the plotter.
        """
        pipeline_stats = {}

        # ----- Parsed data -----
        parsed_dir = os.path.join(run_dir, "parsed_data")
        parsing_stats, particle_stats = self._read_parsed_data_stats(parsed_dir)
        if parsing_stats:
            pipeline_stats['parsing'] = parsing_stats
        if particle_stats:
            pipeline_stats['particles'] = particle_stats

        # ----- Invariant mass arrays -----
        im_dir = os.path.join(run_dir, "im_arrays")
        mass_stats = self._read_im_array_stats(im_dir)
        if mass_stats:
            pipeline_stats['mass_calc'] = mass_stats

        # ----- Post-processed arrays -----
        im_proc_dir = os.path.join(run_dir, "im_arrays_processed")
        post_stats = self._read_post_processing_stats(im_proc_dir)
        if post_stats:
            pipeline_stats['post_processing'] = post_stats

        # ----- Histograms -----
        hist_dir = os.path.join(run_dir, "histograms")
        hist_stats = self._read_histogram_stats(hist_dir)
        if hist_stats:
            pipeline_stats['histograms'] = hist_stats

        return pipeline_stats

    def _read_parsed_data_stats(self, parsed_dir: str):
        """Read all parsed ROOT files and compute consolidated stats."""
        import uproot
        import numpy as np

        if not os.path.isdir(parsed_dir):
            return None, None

        root_files = sorted(Path(parsed_dir).glob("*.root"))
        if not root_files:
            return None, None

        total_events = 0
        total_size_bytes = 0
        particle_counts = {}      # {type: total_count}
        particle_dists = {}       # {type: [per-event counts]}
        events_per_file = []

        for rf in root_files:
            try:
                total_size_bytes += rf.stat().st_size
                with uproot.open(str(rf)) as f:
                    if "events" not in f:
                        continue
                    tree = f["events"]
                    n_events = tree.num_entries
                    total_events += n_events
                    events_per_file.append(n_events)

                    # Discover particle branches (nElectrons, nMuons, …)
                    for branch_name in tree.keys():
                        if branch_name.startswith("n") and branch_name != "nEvents":
                            ptype = branch_name[1:]  # e.g. "Electrons"
                            counts = tree[branch_name].array(library="np")
                            total_for_type = int(np.sum(counts))
                            particle_counts[ptype] = particle_counts.get(ptype, 0) + total_for_type
                            if ptype not in particle_dists:
                                particle_dists[ptype] = []
                            particle_dists[ptype].extend(counts.tolist())
            except Exception as e:
                self.logger.warning(f"Could not read {rf}: {e}")

        avg_per_file = total_events / len(root_files) if root_files else 0

        parsing_stats = {
            'total_files': len(root_files),
            'successful_files': len(root_files),
            'failed_files': 0,
            'total_events': total_events,
            'total_chunks': len(root_files),
            'total_size_mb': total_size_bytes / (1024 * 1024),
            'total_time_sec': 0,   # not available from disk
            'average_events_per_file': avg_per_file,
            'max_memory_mb': 0,
            'timeout_count': 0,
            'error_types': {},
        }

        particle_stats = {
            'particle_counts': particle_counts,
            'distributions': particle_dists,
            'events_per_file': events_per_file,
        }

        return parsing_stats, particle_stats

    def _read_im_array_stats(self, im_dir: str):
        """Read invariant mass array (.npy) files and compute stats.

        Filenames follow the pattern:
            <prefix>_FS_<Xe>_<Xm>_<Xj>_<Xg>_IM_<Ye>_<Ym>_<Yj>_<Yg>.npy
        Example:
            parsed_2024r-pp_batch2_final_FS_2e_2m_4j_2g_IM_2e_2m_4j_2g.npy
        """
        import re
        import numpy as np

        if not os.path.isdir(im_dir):
            return None

        npy_files = sorted(Path(im_dir).glob("*.npy"))
        if not npy_files:
            return None

        total_combinations = 0
        combos_per_object = {}   # e.g. {"Electrons": 12345, "Muons": 678, ...}
        combo_sizes = {}         # e.g. {2: 5, 3: 3, 4: 1} — num particle types involved
        fs_events = {}           # final-state string → total IM values across files
        unique_fs = set()
        unique_im = set()

        fs_im_pattern = re.compile(
            r'_FS_(\d+e_\d+m_\d+j_\d+g)_IM_(\d+e_\d+m_\d+j_\d+g)'
        )
        particle_pattern = re.compile(r'(\d+)([emjg])')
        particle_name_map = {'e': 'Electrons', 'm': 'Muons', 'j': 'Jets', 'g': 'Photons'}

        for nf in npy_files:
            try:
                arr = np.load(str(nf), allow_pickle=True)
                n_entries = len(arr)
                total_combinations += n_entries

                match = fs_im_pattern.search(nf.stem)
                if not match:
                    self.logger.debug(f"Could not parse FS/IM from filename: {nf.name}")
                    continue

                fs_str, im_str = match.groups()
                unique_fs.add(fs_str)
                unique_im.add(im_str)

                # --- Aggregate events per final state ---
                fs_events[fs_str] = fs_events.get(fs_str, 0) + n_entries

                # --- Combos per object type ---
                # Weight by how many particles of each type are in the combination
                # e.g. IM_2e_2m_4j_2g → Electrons+=2*n, Jets+=4*n, etc.
                im_particles = particle_pattern.findall(im_str)
                involved_types = 0
                for count_str, pchar in im_particles:
                    count = int(count_str)
                    if count > 0:
                        involved_types += 1
                        pname = particle_name_map.get(pchar, pchar)
                        combos_per_object[pname] = (
                            combos_per_object.get(pname, 0) + count * n_entries
                        )

                # --- Combination size (number of particle types involved) ---
                combo_sizes[involved_types] = combo_sizes.get(involved_types, 0) + 1

            except Exception as e:
                self.logger.warning(f"Could not read {nf}: {e}")

        return {
            'total_final_states': len(unique_fs),
            'total_combinations': total_combinations,
            'total_events_processed': 0,
            'combinations_per_object': combos_per_object,
            'combination_size_distribution': combo_sizes,
            'events_per_final_state': fs_events,
            'total_time_sec': 0,
        }

    def _read_post_processing_stats(self, proc_dir: str):
        """Read post-processing stats from output directory."""
        if not os.path.isdir(proc_dir):
            return None

        npy_files = sorted(Path(proc_dir).glob("*.npy"))
        if not npy_files:
            return None

        main_count = sum(1 for f in npy_files if 'outlier' not in f.stem)
        outlier_count = sum(1 for f in npy_files if 'outlier' in f.stem)

        return {
            'total_files': len(npy_files),
            'main_files': main_count,
            'outlier_files': outlier_count,
            'total_time_sec': 0,
        }

    def _read_histogram_stats(self, hist_dir: str):
        """Read histogram ROOT files and compute stats."""
        if not os.path.isdir(hist_dir):
            return None

        root_files = sorted(Path(hist_dir).glob("*.root"))
        if not root_files:
            return None

        total_histograms = 0
        main_histograms = 0
        outlier_histograms = 0
        entries_per_histogram = []
        fs_hist_counts = {}

        try:
            import uproot
            for rf in root_files:
                try:
                    with uproot.open(str(rf)) as f:
                        for key in f.keys():
                            name = key.rstrip(";1")
                            total_histograms += 1
                            if "outlier" in name.lower():
                                outlier_histograms += 1
                            else:
                                main_histograms += 1

                            # Try to get entries
                            try:
                                hist = f[key]
                                if hasattr(hist, 'values'):
                                    entries_per_histogram.append(int(sum(hist.values())))
                            except Exception:
                                pass

                            # Parse final state from name
                            # e.g. ROI_mass_e2m2j4g3_cat_2ex_2mx_4jx_3gx_width_10.0
                            import re as _re
                            cat_match = _re.search(r'_cat_([\w_]+?)_width_', name)
                            if cat_match:
                                fs = cat_match.group(1)
                                fs_hist_counts[fs] = fs_hist_counts.get(fs, 0) + 1
                            elif '_cat_' in name:
                                fs = name.split('_cat_')[1]
                                fs_hist_counts[fs] = fs_hist_counts.get(fs, 0) + 1
                except Exception as e:
                    self.logger.warning(f"Could not read histogram file {rf}: {e}")
        except ImportError:
            self.logger.warning("uproot not available for reading histograms")

        if total_histograms == 0:
            return None

        # Get histogram config if available
        hc = self.config.histogram_creation_config
        return {
            'total_histograms': total_histograms,
            'main_histograms': main_histograms,
            'outlier_histograms': outlier_histograms,
            'peaks_detected': 0,
            'peaks_cut': 0,
            'histograms_with_peaks_cut': 0,
            'entries_per_histogram': entries_per_histogram,
            'histograms_per_final_state': fs_hist_counts,
            'bin_width_gev': hc.bin_width_gev if hc else 'N/A',
            'exclude_outliers': hc.exclude_outliers if hc else 'N/A',
            'use_bumpnet_naming': hc.use_bumpnet_naming if hc else 'N/A',
            'output_filename': hc.output_filename if hc else 'N/A',
        }

    # ------------------------------------------------------------------
    # Pipeline internals
    # ------------------------------------------------------------------

    def _create_initial_context(self) -> PipelineContext:
        tasks = self.config.tasks
        
        if tasks.do_parsing:
            initial_state = PipelineState.FETCHING_METADATA
        elif tasks.do_mass_calculating:
            initial_state = PipelineState.MASS_CALCULATION
        elif tasks.do_post_processing:
            initial_state = PipelineState.POST_PROCESSING
        elif tasks.do_histogram_creation:
            initial_state = PipelineState.HISTOGRAM_CREATION
        else:
            initial_state = PipelineState.IDLE
        
        self.logger.info(f"Starting state: {initial_state}")
        return PipelineContext(config=self.config, current_state=initial_state)
    
    def _build_state_machine(self) -> StateMachine:
        self.logger.info("Building state machine with services")
        services = self._create_services()
        handlers = self._create_handlers(services)
        return StateMachine(handlers)
    
    def _create_services(self) -> dict:
        services = {}
        
        if self.config.tasks.do_parsing and self.config.parsing_config:
            pc = self.config.parsing_config
            services['metadata_fetcher'] = MetadataFetcher(
                timeout=pc.fetching_metadata_timeout,
                show_progress=pc.show_progress_bar
            )
            services['metadata_cache'] = MetadataCache(
                cache_path=pc.file_urls_path,
                max_wait_time=300
            )
            services['file_parser'] = FileParser()
            services['event_accumulator'] = EventAccumulator(
                chunk_threshold_bytes=pc.chunk_yield_threshold_bytes
            )
            services['threaded_processor'] = ThreadedFileProcessor(
                file_parser=services['file_parser'],
                max_threads=pc.threads,
                show_progress=pc.show_progress_bar
            )
        
        return services
    
    def _create_handlers(self, services: dict) -> dict:
        handlers = {}
        
        if 'metadata_fetcher' in services:
            handlers[PipelineState.FETCHING_METADATA] = FetchMetadataHandler(
                metadata_fetcher=services['metadata_fetcher'],
                metadata_cache=services['metadata_cache']
            )
        if 'file_parser' in services:
            handlers[PipelineState.PARSING] = ParsingHandler(
                file_parser=services['file_parser'],
                threaded_processor=services['threaded_processor'],
                event_accumulator=services['event_accumulator']
            )
        
        # These handlers have no external service dependencies.
        if self.config.tasks.do_mass_calculating:
            handlers[PipelineState.MASS_CALCULATION] = MassCalculationHandler()
        if self.config.tasks.do_post_processing:
            handlers[PipelineState.POST_PROCESSING] = PostProcessingHandler()
        if self.config.tasks.do_histogram_creation:
            handlers[PipelineState.HISTOGRAM_CREATION] = HistogramCreationHandler()
        
        return handlers
    
    def _log_results(self, context: PipelineContext):
        self.logger.info("=" * 60)
        self.logger.info("Pipeline Execution Summary")
        self.logger.info("=" * 60)
        
        summary = context.get_summary()
        for key, value in summary.items():
            self.logger.info(f"{key:30s}: {value}")
        
        if context.parsing_stats:
            self.logger.info("\nParsing Statistics:")
            for key, value in context.parsing_stats.to_dict().items():
                self.logger.info(f"{key:30s}: {value}")
        
        self.logger.info("=" * 60)
