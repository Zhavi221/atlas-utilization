"""
MassCalculationHandler - Handles invariant mass calculation state.

Reads pre-parsed ROOT files from the parsing stage and runs invariant
mass calculations using the combinatorics and IM calculator modules.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import uproot
import awkward as ak
import numpy as np

from orchestration.context import PipelineContext
from orchestration.states import PipelineState
from .base import StateHandler


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

        start = datetime.now()

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
        )
        self.logger.info(f"Generated {len(all_combinations)} combinations to process")

        # Config dict expected by process_final_state & helpers
        config_dict = {
            "field_to_slice_by": mc.field_to_slice_by,
            "fs_chunk_threshold_bytes": mc.fs_chunk_threshold_bytes,
        }

        # ── Discover parsed ROOT files ──
        parsed_dir = Path(mc.input_dir)
        if context.parsed_files:
            root_files = [Path(f) for f in context.parsed_files if Path(f).exists()]
        else:
            root_files = sorted(parsed_dir.glob("*.root"))

        if not root_files:
            self.logger.warning(f"No parsed ROOT files found in {parsed_dir}")
            return context, self._determine_next_state(context)

        # ── Apply batch splitting if configured ──
        batch_idx = context.config.batch_job_index
        total_batches = context.config.total_batch_jobs

        if batch_idx is not None and total_batches is not None:
            total_files = len(root_files)
            chunk_size = total_files // total_batches
            start = (batch_idx - 1) * chunk_size
            end = total_files if batch_idx == total_batches else start + chunk_size
            root_files = root_files[start:end]
            self.logger.info(
                f"Batch {batch_idx}/{total_batches}: "
                f"processing files {start+1}-{end} of {total_files}"
            )

        self.logger.info(
            f"Processing {len(root_files)} parsed file(s) from {parsed_dir}"
        )

        all_created_files: List[str] = []

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
                )
                if created:
                    all_created_files.extend(created)
            except Exception as exc:
                self.logger.error(
                    f"Error processing {root_file_path.name}: {exc}",
                    exc_info=True,
                )

        elapsed = (datetime.now() - start).total_seconds()
        self.logger.info(
            f"Mass calculation complete: {len(all_created_files)} IM arrays "
            f"in {elapsed:.1f}s"
        )

        updated = context.with_im_files(all_created_files)
        next_state = self._determine_next_state(updated)
        self._log_state_exit(context, next_state)
        return updated, next_state

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

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

    def _process_single_parsed_file(
        self,
        root_file_path: Path,
        output_dir: str,
        all_combinations: List[Dict[str, int]],
        config_dict: dict,
        mc,
        IMCalculator,
        process_final_state,
    ) -> List[str]:
        """Read one parsed ROOT file and compute invariant masses."""
        self.logger.info(f"Reading parsed file: {root_file_path.name}")

        with uproot.open(str(root_file_path)) as f:
            if "events" in f:
                tree = f["events"]
                particle_arrays = self._reconstruct_particle_arrays(tree)
            elif "CollectionTree" in f:
                tree = f["CollectionTree"]
                particle_arrays = self._reconstruct_from_atlas_tree(tree)
            else:
                self.logger.warning(
                    f"{root_file_path.name} has no recognised tree – skipping"
                )
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
            min_events_per_fs=mc.min_events_per_fs,
            min_k=mc.min_count_particle_in_combination,
            max_k=mc.max_count_particle_in_combination,
            min_n=mc.min_particles_in_combination,
            max_n=mc.max_particles_in_combination,
        )

        created_files: List[str] = []

        for cur_fs in calculator.group_by_final_state():
            fs_events = calculator.get_events_for_final_state(cur_fs)
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
