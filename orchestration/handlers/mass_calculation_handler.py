"""
MassCalculationHandler - Handles invariant mass calculation state.

Reads pre-parsed ROOT files from the parsing stage and runs invariant
mass calculations using the combinatorics and IM calculator modules.
"""

import os
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set

import uproot
import awkward as ak
import numpy as np

from orchestration.context import PipelineContext
from orchestration.states import PipelineState
from .base import StateHandler
from services.storage.sqlite_shards import (
    SqliteArrayShardWriter,
)


class MassCalculationHandler(StateHandler):
    """
    Handler for MASS_CALCULATION state.

    Reads parsed ROOT files (with an 'events' tree), reconstructs the
    awkward-array structure, then uses IMCalculator + combinatorics to
    compute invariant masses and save .npy files.
    """

    def handle(self, context: PipelineContext) -> tuple[PipelineContext, PipelineState]:
        self._log_state_entry(context)

        mc = context.config.mass_calculation_config
        if mc is None:
            self.logger.warning("No mass_calculation_config – skipping")
            return context, self._determine_next_state(context)

        start = time.perf_counter()

        from services.calculations import combinatorics
        from services.calculations.im_calculator import IMCalculator
        from services.pipelines.im_pipeline import process_final_state

        os.makedirs(mc.output_dir, exist_ok=True)

        # ── Build combinations ──
        all_combinations = combinatorics.get_all_combinations(
            list(mc.objects_to_calculate),
            min_particles=mc.min_particles_in_combination,
            max_particles=mc.max_particles_in_combination,
            min_count=mc.min_count_particle_in_combination,
            max_count=mc.max_count_particle_in_combination,
            max_total_particles=mc.max_total_particles_in_combination,
            include_subleading=getattr(mc, 'include_subleading', False),
            max_subleading_index=getattr(mc, 'max_subleading_index', 1),
        )
        self.logger.info(f"Generated {len(all_combinations)} combinations to process")

        # Config dict expected by process_final_state & helpers
        batch_idx = context.config.batch_job_index or 1
        shard_name = f"im_batch_{batch_idx}.sqlite"
        shard_path = os.path.join(mc.output_dir, shard_name)
        if os.path.exists(shard_path):
            os.remove(shard_path)
        sqlite_writer = SqliteArrayShardWriter(shard_path)

        config_dict = {
            "field_to_slice_by": mc.field_to_slice_by,
            "fs_chunk_threshold_bytes": mc.fs_chunk_threshold_bytes,
            "output_mode": "sqlite",
            "sqlite_writer": sqlite_writer,
        }

        # ── Discover parsed ROOT files ──
        parsed_dir = Path(mc.input_dir)
        if context.parsed_files:
            root_files = [Path(f) for f in context.parsed_files if Path(f).exists()]
            self.logger.info(
                f"Using {len(root_files)} file(s) from parsing stage context"
            )
            # Parsing already assigned the correct files — no further splitting needed
        else:
            root_files = sorted(parsed_dir.glob("*.root"))
            self.logger.info(
                f"No context files — reading {len(root_files)} file(s) from {parsed_dir}"
            )
            # Apply batch splitting only when reading from disk
            batch_idx = context.config.batch_job_index
            total_batches = context.config.total_batch_jobs
            if batch_idx is not None and total_batches is not None:
                total_files = len(root_files)
                chunk_size = max(1, total_files // total_batches)
                slice_start = (batch_idx - 1) * chunk_size
                slice_end = total_files if batch_idx == total_batches else slice_start + chunk_size
                root_files = root_files[slice_start:slice_end]
                self.logger.info(
                    f"Batch {batch_idx}/{total_batches}: "
                    f"processing files {slice_start+1}-{slice_end} of {total_files}"
                )

        if not root_files:
            self.logger.warning(f"No parsed ROOT files found in {parsed_dir}")
            sqlite_writer.close()
            return context, self._determine_next_state(context)

        total_created_chunks = 0

        try:
            eligible_final_states = self._find_eligible_final_states(
                root_files, mc, IMCalculator, context
            )
            for root_file_path in root_files:
                try:
                    created = self._process_single_parsed_file(
                        root_file_path,
                        mc.output_dir,
                        all_combinations,
                        config_dict,
                        mc,
                        IMCalculator,
                        process_final_state,
                        eligible_final_states,
                    )
                    if created:
                        total_created_chunks += len(created)
                except Exception as exc:
                    self.logger.error(
                        f"Error processing {root_file_path.name}: {exc}",
                        exc_info=True,
                    )
        finally:
            elapsed = time.perf_counter() - start
            try:
                sqlite_writer.set_metadata("mass_calculation_time_sec", elapsed)
                sqlite_writer.set_metadata("created_chunks", total_created_chunks)
            finally:
                sqlite_writer.close()

        self.logger.info(
            f"Mass calculation complete: {total_created_chunks} IM chunks/signatures "
            f"in {elapsed:.1f}s; shard={shard_path}"
        )

        updated = context.with_im_files([shard_name]).with_custom_data(
            "mass_calc",
            {
                "total_time_sec": elapsed,
                "created_chunks": total_created_chunks,
                "shard": shard_name,
            },
        )
        next_state = self._determine_next_state(updated)
        self._log_state_exit(context, next_state)
        return updated, next_state

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _find_eligible_final_states(
        self,
        root_files: List[Path],
        mc,
        IMCalculator,
        context: PipelineContext,
    ) -> Optional[Set[str]]:
        """Count final states globally before doing invariant-mass calculations.

        ``None`` means prefiltering is intentionally disabled. In distributed
        batch mode an individual job cannot know the population in the other
        shards, so the existing post-processing global threshold remains the
        correctness-preserving fallback.
        """
        threshold = int(mc.min_events_per_fs)
        if threshold <= 1:
            return None

        if (
            context.config.batch_job_index is not None
            and context.config.total_batch_jobs is not None
            and context.config.total_batch_jobs > 1
        ):
            self.logger.info(
                "Skipping pre-calculation final-state threshold in distributed "
                "batch mode; the global threshold will be applied after shards "
                "are combined"
            )
            return None

        self.logger.info(
            "Counting final states across %d parsed file(s) before invariant-mass "
            "calculation (minimum events: %d)",
            len(root_files),
            threshold,
        )
        global_counts: Counter = Counter()
        try:
            for root_file_path in root_files:
                particle_arrays = self._load_particle_arrays(root_file_path)
                if particle_arrays is None:
                    raise RuntimeError(
                        f"could not load final-state counts from {root_file_path.name}"
                    )
                calculator = IMCalculator(
                    particle_arrays,
                    min_events_per_fs=1,
                    min_k=mc.min_count_particle_in_combination,
                    max_k=mc.max_count_particle_in_combination,
                    min_n=mc.min_particles_in_combination,
                    max_n=mc.max_particles_in_combination,
                )
                global_counts.update(calculator.final_state_counts())
        except Exception as exc:
            self.logger.warning(
                "Could not safely complete the final-state count pre-pass (%s); "
                "calculating all states and retaining the post-processing threshold",
                exc,
            )
            return None

        eligible = {
            final_state
            for final_state, count in global_counts.items()
            if count >= threshold
        }
        self.logger.info(
            "Final-state prefilter: %d/%d states have at least %d events; "
            "%d states will be skipped before invariant-mass calculation",
            len(eligible),
            len(global_counts),
            threshold,
            len(global_counts) - len(eligible),
        )
        return eligible

    @staticmethod
    def _reconstruct_particle_arrays(tree) -> ak.Array:
        """
        Reconstruct the nested awkward array structure that IMCalculator
        expects from the flat ROOT branches written by ParsingHandler.

        Input branches look like:
            nElectrons, Electrons_pt, Electrons_eta, Electrons_phi, Electrons_mass
            nMuons, Muons_pt, Muons_eta, Muons_phi
            …

        Output:
            ak.Array with fields Electrons, Muons, Jets, Photons – each a
            jagged array of records with pt, eta, phi (and optionally mass/e).
        """
        branch_names = tree.keys()
        particle_types = []
        for bn in branch_names:
            if bn.startswith("n") and bn != "nEvents":
                ptype = bn[1:]  # e.g. "Electrons"
                particle_types.append(ptype)

        particle_dict = {}
        for ptype in particle_types:
            sub_branches = {}
            for bn in branch_names:
                # Match  Electrons_pt, Electrons_eta, etc.
                prefix = f"{ptype}_"
                if bn.startswith(prefix):
                    field_name = bn[len(prefix):]        # "pt", "eta", …
                    sub_branches[field_name] = tree[bn].array(library="ak")

            if sub_branches:
                particle_dict[ptype] = ak.zip(sub_branches)

        return ak.Array(particle_dict)

    # Branch mapping for raw ATLAS Open Data files (2024r release)
    ATLAS_BRANCH_MAP = {
        "Electrons": "AnalysisElectronsAuxDyn",
        "Muons": "AnalysisMuonsAuxDyn",
        "Jets": "AnalysisJetsAuxDyn",
        "Photons": "AnalysisPhotonsAuxDyn",
        "Taus": "AnalysisTauJetsAuxDyn",
    }

    @classmethod
    def _reconstruct_from_atlas_tree(cls, tree) -> ak.Array:
        """
        Reconstruct particle arrays from a raw ATLAS CollectionTree.

        Branch naming: Analysis{Type}sAuxDyn.{field} (e.g. AnalysisElectronsAuxDyn.pt)
        """
        particle_dict = {}
        for ptype, prefix in cls.ATLAS_BRANCH_MAP.items():
            sub_branches = {}
            for field in ("pt", "eta", "phi"):
                branch_name = f"{prefix}.{field}"
                if branch_name in tree:
                    sub_branches[field] = tree[branch_name].array(library="ak")
            if sub_branches:
                particle_dict[ptype] = ak.zip(sub_branches)

        return ak.Array(particle_dict)

    def _load_particle_arrays(self, root_file_path: Path) -> Optional[ak.Array]:
        """Load either a parsed events tree or a supported raw ATLAS tree."""
        with uproot.open(str(root_file_path)) as f:
            if "events" in f:
                return self._reconstruct_particle_arrays(f["events"])
            if "CollectionTree" in f:
                return self._reconstruct_from_atlas_tree(f["CollectionTree"])

        self.logger.warning(
            f"{root_file_path.name} has no recognised tree – skipping"
        )
        return None

    def _process_single_parsed_file(
        self,
        root_file_path: Path,
        output_dir: str,
        all_combinations: List[Dict[str, int]],
        config_dict: dict,
        mc,
        IMCalculator,
        process_final_state,
        eligible_final_states: Optional[Set[str]] = None,
    ) -> List[str]:
        """Read one parsed ROOT file and compute invariant masses."""
        self.logger.info(f"Reading parsed file: {root_file_path.name}")

        particle_arrays = self._load_particle_arrays(root_file_path)
        if particle_arrays is None:
            return []

        num_events = len(particle_arrays)
        if num_events == 0:
            self.logger.info(f"{root_file_path.name}: empty – skipping")
            return []

        self.logger.info(
            f"{root_file_path.name}: {num_events:,} events loaded "
            f"(particle types: {particle_arrays.fields})"
        )

        # Initialise calculator
        calculator = IMCalculator(
            particle_arrays,
            # The global single-job threshold was applied by the count pre-pass.
            # Keep this at one so each file exposes all locally present states;
            # eligibility is checked against the global count below.
            min_events_per_fs=1,
            min_k=mc.min_count_particle_in_combination,
            max_k=mc.max_count_particle_in_combination,
            min_n=mc.min_particles_in_combination,
            max_n=mc.max_particles_in_combination,
        )

        created_files: List[str] = []

        for cur_fs in calculator.group_by_final_state():
            if (
                eligible_final_states is not None
                and cur_fs not in eligible_final_states
            ):
                continue
            fs_events = calculator.get_events_for_final_state(cur_fs)
            config_dict["sqlite_writer"].record_final_state_count(
                cur_fs, len(fs_events)
            )
            result = process_final_state(
                cur_fs,
                fs_events,
                root_file_path.name,
                all_combinations,
                config_dict,
                output_dir,
                self.logger,
                calculator,
            )
            if result is None:
                continue
            fs_stats, fs_created = result
            if fs_created:
                created_files.extend(fs_created)

        self.logger.info(
            f"{root_file_path.name}: created {len(created_files)} IM array(s)"
        )
        return created_files
