"""Regression coverage for parsing-time object vetoes and storage filtering."""

from pathlib import Path
import tempfile
import unittest

import awkward as ak
import numpy as np
import uproot
import yaml

from domain.config import PipelineConfig
from orchestration.handlers.mass_calculation_handler import MassCalculationHandler
from pipeline.executor import PipelineExecutor
from services.calculations.im_calculator import IMCalculator
from services.parsing.event_selection import (
    apply_parsing_event_selection,
    retain_objects_for_storage,
)


ALLOWED = ("Electrons", "Muons", "Jets", "BJets")
EXPLICIT_COUNTS = {
    "jets": {"min": 0, "max": 4},
    "electrons": {"min": 0, "max": 4},
    "muons": {"min": 0, "max": 4},
    "bjets": {"min": 0, "max": 4},
}


def object_events(**collections):
    """Create one-event physics records with the fields cuts expect."""
    return ak.Array(collections)


class ObjectAllowListTests(unittest.TestCase):
    def _select(self, events, *, allowed=ALLOWED, cuts=None):
        return apply_parsing_event_selection(
            events,
            particle_counts=EXPLICIT_COUNTS,
            kinematic_cuts=cuts,
            allowed_objects=allowed,
        )

    def test_tau_veto_rejects_event_instead_of_turning_it_into_jets_only(self):
        events = object_events(
            Jets=[[{"pt": 100.0, "eta": 0.1, "phi": 0.2},
                   {"pt": 80.0, "eta": 0.3, "phi": 0.4}]],
            Taus=[[{"pt": 40.0, "eta": 0.2, "phi": 0.1}]],
        )

        self.assertEqual(len(self._select(events)), 0)

    def test_photon_veto_rejects_event(self):
        events = object_events(
            Jets=[[{"pt": 100.0, "eta": 0.1, "phi": 0.2},
                   {"pt": 80.0, "eta": 0.3, "phi": 0.4}]],
            Photons=[[{"pt": 40.0, "eta": 0.2, "phi": 0.1}]],
        )

        self.assertEqual(len(self._select(events)), 0)

    def test_event_with_only_allowed_objects_survives(self):
        events = object_events(
            Jets=[[{"pt": 100.0, "eta": 0.1, "phi": 0.2},
                   {"pt": 80.0, "eta": 0.3, "phi": 0.4}]],
        )

        self.assertEqual(len(self._select(events)), 1)

    def test_kinematic_filter_runs_before_implicit_excluded_object_veto(self):
        events = object_events(
            Jets=[[{"pt": 100.0, "eta": 0.1, "phi": 0.2}]],
            Taus=[[{"pt": 10.0, "eta": 0.2, "phi": 0.1}]],
        )

        selected = self._select(events, cuts={"taus": {"pt_min": 20.0}})

        self.assertEqual(len(selected), 1)
        self.assertEqual(ak.num(selected.Taus, axis=1).to_list(), [0])

    def test_excluded_branches_are_available_for_selection_then_not_stored(self):
        events = object_events(
            Jets=[[{"pt": 100.0, "eta": 0.1, "phi": 0.2}]],
            Taus=[[]],
            Photons=[[]],
        )

        selected = self._select(events)
        self.assertEqual(set(selected.fields), {"Jets", "Taus", "Photons"})
        stored = retain_objects_for_storage(selected, ALLOWED)
        self.assertEqual(stored.fields, ["Jets"])

    def test_adding_tau_to_allow_list_disables_its_implicit_veto(self):
        events = object_events(
            Jets=[[{"pt": 100.0, "eta": 0.1, "phi": 0.2}]],
            Taus=[[{"pt": 40.0, "eta": 0.2, "phi": 0.1}]],
        )

        selected = self._select(events, allowed=ALLOWED + ("Taus",))
        self.assertEqual(len(selected), 1)
        self.assertEqual(ak.num(selected.Taus, axis=1).to_list(), [1])

    def test_adding_photon_to_allow_list_disables_its_implicit_veto(self):
        events = object_events(
            Jets=[[{"pt": 100.0, "eta": 0.1, "phi": 0.2}]],
            Photons=[[{"pt": 40.0, "eta": 0.2, "phi": 0.1}]],
        )

        selected = self._select(events, allowed=ALLOWED + ("Photons",))
        self.assertEqual(len(selected), 1)
        self.assertEqual(ak.num(selected.Photons, axis=1).to_list(), [1])

    def test_particle_counts_rejects_excluded_tau_and_photon_keys(self):
        for name in ("taus", "photons"):
            config = {
                "tasks": {"do_parsing": True},
                "parsing_task_config": {
                    "output_path": "out", "file_urls_path": "urls", "jobs_logs_path": "logs",
                    "particle_counts": {name: {"min": 0, "max": 2}},
                },
                "mass_calculation_task_config": {"objects_to_calculate": list(ALLOWED)},
            }
            with self.assertRaisesRegex(ValueError, "objects_to_calculate"):
                PipelineConfig.from_dict(config)

    def test_mass_reconstruction_excludes_unstored_physics_objects(self):
        class FakeTree:
            def keys(self):
                return ["nJets", "nTaus", "Jets_pt", "Taus_pt"]

            def __getitem__(self, _name):
                return type("Branch", (), {"array": lambda _self, library: ak.Array([[1.0]])})()

        arrays = MassCalculationHandler._reconstruct_particle_arrays(FakeTree(), ALLOWED)
        self.assertEqual(arrays.fields, ["Jets"])

    def test_final_state_labels_only_include_stored_object_types(self):
        events = object_events(
            Jets=[[{"pt": 100.0, "eta": 0.1, "phi": 0.2},
                   {"pt": 80.0, "eta": 0.3, "phi": 0.4}]],
        )
        calculator = IMCalculator(events, min_events_per_fs=1, min_k=1, max_k=4, min_n=1, max_n=4)

        self.assertEqual(dict(calculator.final_state_counts()), {"2j": 1})

    def test_statistics_exclude_unstored_physics_objects(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir) / "parsed.root"
            with uproot.recreate(root_path) as root_file:
                root_file["events"] = {
                    "nJets": np.array([1, 2]),
                    "nTaus": np.array([5, 8]),
                    "nPhotons": np.array([3, 7]),
                }
            executor = object.__new__(PipelineExecutor)
            _, particle_stats = executor._read_parsed_data_stats(temp_dir, set(ALLOWED))

        self.assertEqual(particle_stats["particle_counts"], {"Jets": 3})

    def test_default_config_keeps_excluded_object_kinematic_templates(self):
        config_path = Path(__file__).parents[1] / "config.yaml"
        config = yaml.safe_load(config_path.read_text())
        cuts = config["parsing_task_config"]["kinematic_cuts"]
        self.assertIn("photons", cuts)
        self.assertIn("taus", cuts)


if __name__ == "__main__":
    unittest.main()
